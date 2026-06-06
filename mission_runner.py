#!/usr/bin/env python3
"""
任务执行器 — 路径规划 → 相对移动 → 校正 一体化。

核心思路:
  1. path_planner 规划绝对坐标路径 + 校正点
  2. 相邻路点间的差转为相对位移，通过 /MoveRelative 执行（不依赖绝对定位）
  3. 在校正点处，用 LiDAR 做角度校正 + 坐标校正

用法（在小车上运行，需 ROS + LiDAR + Flask 服务端）:
    python3 mission_runner.py

依赖:
    path_planner   — plan_path, get_car_position
    Flask 服务端    — /MoveRelative, /SyncYaw, /SetBaseline, /Reset
"""

import sys
import time
import math
import requests


# ============================================================
#  比赛点位（6 个目标点，坐标待填入）
# ============================================================

POINTS = {
    1: (None, None),
    2: (None, None),
    3: (None, None),
    4: (None, None),
    5: (None, None),
    6: (None, None),
}

# ============================================================
#  步骤定义
# ============================================================

STEPS = {
    1: [1, 2, 3],
    2: [4, 5],
    3: [6, 1],
    4: [2, 4, 6],
}


# ============================================================
#  可调参数
# ============================================================

CAR_IP = "10.26.36.227"
CAR_PORT = 5000
MOVE_TIMEOUT = 30
STEP_DELAY = 0.5
NUM_WAYPOINTS = 0                 # 路径中间点个数 (0=自动，不做重采样)
POS_CORRECTION_THRESHOLD = 0.08   # 坐标校正阈值 (m)，LiDAR 与期望差超过此值则纠正
MAX_POS_CORRECTION_RETRIES = 3    # 坐标校正最多重试次数


# ============================================================
#  服务器通信
# ============================================================

class _CarClient:
    """Flask 服务端 HTTP 客户端"""

    def __init__(self, ip=CAR_IP, port=CAR_PORT):
        self.base = f"http://{ip}:{port}"
        self.sess = requests.Session()
        self.tid = 1

    def _post(self, endpoint, payload=None, timeout=MOVE_TIMEOUT):
        if payload is None:
            payload = {}
        body = {**payload, "TaskId": self.tid}
        try:
            r = self.sess.post(f"{self.base}/{endpoint}", json=body, timeout=timeout)
            resp = r.json()
            if resp.get("isSuccess"):
                self.tid += 1
                return True, resp
            if "expectedTaskId" in resp:
                self.tid = resp["expectedTaskId"]
                return self._post(endpoint, payload, timeout)
            return False, resp
        except Exception as e:
            return False, {"error": str(e)}

    def reset(self):
        ok, _ = self._post("Reset", timeout=5)
        self.tid = 1
        return ok

    def set_baseline(self):
        return self._post("SetBaseline")

    def sync_yaw(self):
        return self._post("SyncYaw", timeout=20)

    def move_relative(self, dx, dy):
        """相对位移"""
        return self._post("MoveRelative", {
            "delta_x": dx, "delta_y": dy
        })


# ============================================================
#  核心执行逻辑
# ============================================================

def _move_segment(x1, y1, x2, y2, client):
    """将绝对路点段转为相对位移并执行"""
    dx = x2 - x1
    dy = y2 - y1

    if abs(dx) < 0.001 and abs(dy) < 0.001:
        return True, None

    direction = ""
    if abs(dx) > 0.001 and abs(dy) < 0.001:
        direction = f"dX={dx:+.3f}"
    elif abs(dy) > 0.001 and abs(dx) < 0.001:
        direction = f"dY={dy:+.3f}"
    else:
        direction = f"dX={dx:+.3f}, dY={dy:+.3f}"

    print(f"  → MoveRelative: {direction}")
    ok, resp = client.move_relative(dx, dy)
    return ok, resp


def _do_correction(client, expected_x, expected_y):
    """
    在校正点执行双重校正:
      1) 角度校正 (SetBaseline → SyncYaw)
      2) 坐标校正 (LiDAR vs 期望坐标 → 补相对位移)
    """
    from path_planner import get_car_position

    # ── 1) 角度校正 ──
    print("  → 设定校正基准 (SetBaseline)...")
    client.set_baseline()
    print("  → 角度校正 (SyncYaw)...")
    ok, _ = client.sync_yaw()
    if ok:
        print("  ✓ 角度校正完成")
    else:
        print("  ✗ 角度校正失败")

    # ── 2) 坐标校正 ──
    for attempt in range(1, MAX_POS_CORRECTION_RETRIES + 1):
        cx, cy = get_car_position()
        err_x = expected_x - cx
        err_y = expected_y - cy
        err = math.hypot(err_x, err_y)

        print(f"  → 坐标校正 #{attempt}: 期望({expected_x:.3f},{expected_y:.3f}) "
              f"实际({cx:.3f},{cy:.3f}) 误差{err:.3f}m")

        if err <= POS_CORRECTION_THRESHOLD:
            print(f"  ✓ 坐标已收敛 (误差 {err:.3f}m <= {POS_CORRECTION_THRESHOLD}m)")
            break

        # 发一个相对位移纠正当前位置
        print(f"    补相对位移: ({err_x:+.3f}, {err_y:+.3f})")
        ok, _ = client.move_relative(err_x, err_y)
        if not ok:
            print(f"  ✗ 纠正移动失败")
            break

        time.sleep(0.3)
    else:
        print(f"  ⚠ 达到最大重试次数 ({MAX_POS_CORRECTION_RETRIES})，继续执行")


def run_mission_step(point_ids, client):
    """
    执行一个步骤：依次到达 point_ids 中的每个点位。
    """
    from path_planner import plan_path, get_car_position

    print(f"\n{'=' * 60}")
    print(f"  Step: 依次到达点位 {point_ids}")
    print(f"{'=' * 60}")

    for pid in point_ids:
        tx, ty = POINTS[pid]
        if tx is None or ty is None:
            print(f"\n点位 {pid} 坐标未配置，请输入:")
            try:
                tx = float(input(f"  点位{pid} X: "))
                ty = float(input(f"  点位{pid} Y: "))
            except (ValueError, EOFError):
                print("  输入无效，跳过此点位")
                continue

        # ── 获取当前位置 ──
        cx, cy = get_car_position()
        print(f"\n点位 {pid}: 目标 ({tx:.2f}, {ty:.2f}), 当前位置 ({cx:.3f}, {cy:.3f})")

        if abs(cx - tx) < 0.05 and abs(cy - ty) < 0.05:
            print(f"  已在目标点，跳过")
            continue

        # ── 规划路径（绝对坐标） ──
        result = plan_path(cx, cy, tx, ty, cx, cy, num_waypoints=NUM_WAYPOINTS)
        waypoints = result["waypoints"]
        correction_points = result.get("correction_points", [])
        cp_indices = {p["index"]: p for p in correction_points}

        print(f"  路径: {result['path_name']}, "
              f"距障碍物 {result['obstacle_margin']}m, "
              f"{'✓' if result['safe'] else '⚠'}")
        print(f"  路径点 {len(waypoints)} 个, 校正点 {len(correction_points)} 个")

        # ── 逐段执行 ──
        for i in range(len(waypoints) - 1):
            x1, y1 = waypoints[i]
            x2, y2 = waypoints[i + 1]

            # 当前点是校正点 → 先校正再走
            if i in cp_indices:
                cp = cp_indices[i]
                if cp["safe"]:
                    print(f"\n  ● 校正点 #{i} ({x1:.3f}, {y1:.3f}) type={cp['type']}")
                    _do_correction(client, x1, y1)
                else:
                    print(f"\n  - 校正点 #{i} 有遮挡，跳过校正")

            # 相对位移：Δ = 下一路点 - 当前路点
            ok, resp = _move_segment(x1, y1, x2, y2, client)
            if not ok:
                print(f"  ✗ 移动失败: {resp.get('errorMessage', resp.get('error', '未知'))}")
                return False
            time.sleep(STEP_DELAY)

        # 到达目标点 → 校正
        print(f"\n  ✓ 到达点位 {pid}")
        _do_correction(client, tx, ty)

    return True


# ============================================================
#  主入口
# ============================================================

def main():
    import rospy
    from sensor_msgs.msg import LaserScan
    from path_planner import get_car_position

    # ── 初始化 ──
    rospy.init_node("mission_runner", anonymous=True)
    rospy.wait_for_message("scan", LaserScan, timeout=5.0)

    client = _CarClient()
    client.reset()

    # ── 记录起点 ──
    start_x, start_y = get_car_position()
    print("=" * 60)
    print(f"  起点已记录: X = {start_x:.3f},  Y = {start_y:.3f}")
    print("=" * 60)

    # ── 设定校正基准 ──
    print("\n设定校正基准...")
    client.set_baseline()

    # ── 步骤选择循环 ──
    step_names = {
        1: f"点位 {STEPS[1]}",
        2: f"点位 {STEPS[2]}",
        3: f"点位 {STEPS[3]}",
        4: f"点位 {STEPS[4]}",
    }

    while True:
        print(f"\n{'─' * 40}")
        print("可选步骤:")
        for k, v in step_names.items():
            print(f"  {k}. Step {k}: {v}")
        print("  Q. 退出")
        print(f"{'─' * 40}")

        choice = input("请选择步骤: ").strip()

        if choice.upper() == 'Q':
            break

        try:
            step = int(choice)
            if step not in STEPS:
                print("无效步骤号")
                continue
        except ValueError:
            print("无效输入")
            continue

        run_mission_step(STEPS[step], client)

    # ── 显示终点 vs 起点 ──
    end_x, end_y = get_car_position()
    print(f"\n{'=' * 60}")
    print(f"  任务结束")
    print(f"  起点: ({start_x:.3f}, {start_y:.3f})")
    print(f"  终点: ({end_x:.3f}, {end_y:.3f})")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
