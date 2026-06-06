#!/usr/bin/env python3
"""
任务执行器 — 路径规划 → 校正 → 移动 一体化。

通过命令行选择 step，自动规划避障路径，在可校正点执行位姿校正，
调用服务器端点逐段移动。

用法（在小车上运行，需 ROS + LiDAR + Flask 服务端）:
    python3 mission_runner.py

依赖:
    path_planner   — plan_path_to, get_car_position
    Flask 服务端    — /MoveOnlyX, /MoveOnlyY, /SyncYaw, /SetBaseline, /Reset
"""

import sys
import time
import requests


# ============================================================
#  比赛点位（6 个目标点，坐标待填入）
# ============================================================

POINTS = {
    1: (None, None),    # 点位 1: (x, y)
    2: (None, None),    # 点位 2: (x, y)
    3: (None, None),    # 点位 3: (x, y)
    4: (None, None),    # 点位 4: (x, y)
    5: (None, None),    # 点位 5: (x, y)
    6: (None, None),    # 点位 6: (x, y)
}

# ============================================================
#  步骤定义（每个 step 要依次经过的点位 ID）
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

CAR_IP = "10.26.36.227"     # Flask 服务端 IP
CAR_PORT = 5000             # Flask 服务端端口
MOVE_TIMEOUT = 30           # 单次移动超时 (秒)
STEP_DELAY = 0.5            # 段间等待 (秒)


# ============================================================
#  服务器通信
# ============================================================

class _CarClient:
    """Flask 服务端 HTTP 客户端（内部使用）"""

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

    def move_x(self, target_x, ref_y):
        return self._post("MoveOnlyX", {
            "location_x": target_x, "location_y": ref_y
        })

    def move_y(self, target_y, ref_x):
        return self._post("MoveOnlyY", {
            "location_x": ref_x, "location_y": target_y
        })


# ============================================================
#  核心执行逻辑
# ============================================================

def _move_segment(x1, y1, x2, y2, client):
    """执行一段纯轴对齐移动"""
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    if dx > 0.001 and dy < 0.001:
        print(f"  → MoveOnlyX: X={x2:.3f} (ref Y={y1:.3f})")
        return client.move_x(x2, y1)
    elif dy > 0.001 and dx < 0.001:
        print(f"  → MoveOnlyY: Y={y2:.3f} (ref X={x1:.3f})")
        return client.move_y(y2, x1)
    else:
        print(f"  ⚠ 段不是纯轴对齐，跳过")
        return True, None


def _do_correction(client):
    """执行一次位姿校正"""
    print("  → 位姿校正 (SyncYaw)...")
    ok, _ = client.sync_yaw()
    if ok:
        print("  ✓ 校正完成")
    else:
        print("  ✗ 校正失败")
    return ok


def run_mission_step(point_ids, client):
    """
    执行一个步骤：依次到达 point_ids 中的每个点位。

    对每个目标点：
      1. 获取当前位置
      2. 调用 path_planner 规划避障路径（含校正点）
      3. 沿路径逐段移动，在可校正点执行校正
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

        # ── 规划路径 ──
        result = plan_path(cx, cy, tx, ty, cx, cy)
        waypoints = result["waypoints"]
        correction_points = result.get("correction_points", [])
        cp_indices = {p["index"] for p in correction_points}

        print(f"  路径: {result['path_name']}, "
              f"距障碍物 {result['obstacle_margin']}m, "
              f"{'✓' if result['safe'] else '⚠'}")
        print(f"  路径点 {len(waypoints)} 个, 校正点 {len(correction_points)} 个")

        # ── 逐段执行 ──
        for i in range(len(waypoints) - 1):
            x1, y1 = waypoints[i]
            x2, y2 = waypoints[i + 1]

            # 当前点如果是校正点 → 先校正再走
            if i in cp_indices:
                cpi = next(p for p in correction_points if p["index"] == i)
                if cpi["safe"]:
                    _do_correction(client)
                else:
                    print(f"  - 校正点#{i} 有遮挡，跳过校正")

            ok, resp = _move_segment(x1, y1, x2, y2, client)
            if not ok:
                print(f"  ✗ 移动失败: {resp.get('errorMessage', resp.get('error', '未知'))}")
                return False
            time.sleep(STEP_DELAY)

        # 到达目标点 → 校正
        print(f"  ✓ 到达点位 {pid}")
        _do_correction(client)

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
