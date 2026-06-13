#!/usr/bin/env python3
"""
全流程任务脚本 — 无限循环，输入 1~7 从当前位置前往目标点并执行配套动作。

用法（在小车上运行，需 ROS + LiDAR + Flask 服务端）:
    python3 test_full_pipeline.py

点位来源:
    1~6: target_points.json（规划坐标 + 校正坐标）
    7:   起点（运行时自动记录）

动作来源:
    action_for_each_target.py（rotate / stay / correct）
"""

import sys
import time
import math
import rospy
import requests
import numpy as np
from sensor_msgs.msg import LaserScan

# ============================================================
#  配置
# ============================================================

CAR_IP = "10.152.203.227"
CAR_PORT = 5000
BASE_URL = f"http://{CAR_IP}:{CAR_PORT}"
MOVE_TIMEOUT = 30
STEP_DELAY = 0.3
POS_THRESHOLD = 0.08
MAX_RETRIES = 5

ROOM_X_MAX = 4.5
ROOM_Y_MAX = 8.8

# 每个点位的校正光束索引表达式
_CORRECT_BEAMS = {
    1: ("n//2",   "n//4"),     # 前 + 右
    2: ("n-2",    "n//4"),     # 后 + 右
    3: ("n-2",    "n*3//4"),   # 后 + 左
    4: ("n//2",   "n*3//4"),   # 前 + 左
}


# ============================================================
#  LiDAR 定位
# ============================================================

def _get_beam(index, timeout=5.0):
    data = rospy.wait_for_message("scan", LaserScan, timeout=5.0)
    n = len(data.ranges)
    d = data.ranges[index % n]
    t0 = time.time()
    while d == np.inf:
        if time.time() - t0 > timeout:
            raise TimeoutError(f"LiDAR 光束 {index} 持续 inf，超时 {timeout}s")
        data = rospy.wait_for_message("scan", LaserScan, timeout=5.0)
        n = len(data.ranges)
        d = data.ranges[index % n]
    return d


def _get_n():
    return len(rospy.wait_for_message("scan", LaserScan).ranges)


def car_position():
    """规划坐标 — 前X + 右Y"""
    n = _get_n()
    return _get_beam(n // 2), _get_beam(n // 4)


def _read_correct_beams(pid):
    """读取点位 pid 对应的校正光束原始距离"""
    if pid not in _CORRECT_BEAMS:
        return None, None
    n = _get_n()
    x_expr, y_expr = _CORRECT_BEAMS[pid]
    x = _get_beam(eval(x_expr, {"n": n}))
    y = _get_beam(eval(y_expr, {"n": n}))
    return x, y


# ============================================================
#  Flask 通信
# ============================================================

class _Client:
    def __init__(self):
        self.sess = requests.Session()
        self.tid = 1

    def _post(self, endpoint, payload=None, timeout=MOVE_TIMEOUT):
        if payload is None:
            payload = {}
        body = {**payload, "TaskId": self.tid}
        try:
            r = self.sess.post(f"{BASE_URL}/{endpoint}", json=body, timeout=timeout)
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
        return self._post("MoveRelative", {"delta_x": dx, "delta_y": dy})

    def move_x(self, target_x, ref_y):
        return self._post("MoveOnlyX", {"location_x": target_x, "location_y": ref_y})

    def move_y(self, target_y, ref_x):
        return self._post("MoveOnlyY", {"location_x": ref_x, "location_y": target_y})

    def circle(self, rad_z):
        return self._post("Circle", {"rad_z": rad_z})


# ============================================================
#  移动与校正
# ============================================================

def _move_segment(x1, y1, x2, y2, cli):
    """轴对齐移动一段"""
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    if dx > 0.001 and dy < 0.001:
        return cli.move_x(x2, y1)
    elif dy > 0.001 and dx < 0.001:
        return cli.move_y(y2, x1)
    return True, None


def _do_correction(cli, expected_x, expected_y):
    """坐标校正（用规划坐标前X+右Y做LiDAR闭环）"""
    for attempt in range(1, MAX_RETRIES + 1):
        cx, cy = car_position()
        err = math.hypot(expected_x - cx, expected_y - cy)
        print(f"  坐标校正 #{attempt}: 期望({expected_x:.3f},{expected_y:.3f}) "
              f"实际({cx:.3f},{cy:.3f}) 误差{err:.3f}m")
        if err <= POS_THRESHOLD:
            print("  ✓ 已收敛")
            break
        cdx, cdy = cx - expected_x, cy - expected_y
        cli.move_relative(cdx, cdy)
        time.sleep(0.3)
    else:
        print(f"  ⚠ 达到最大重试")


def _do_target_correction(cli, pid, target_info):
    """用校正光束坐标做精校（仅点位 1~4）"""
    if "correct" not in target_info:
        return
    stored_x, stored_y = target_info["correct"]
    curr_x, curr_y = _read_correct_beams(pid)
    if curr_x is None:
        return

    # 前/后光束 → 绝对 X；左/右光束 → 绝对 Y
    beam_cfg = {
        1: ("front", "right"),
        2: ("rear",  "right"),
        3: ("rear",  "left"),
        4: ("front", "left"),
    }
    x_type, y_type = beam_cfg[pid]

    # 前光束直接是绝对 X，后光束需反转
    abs_curr_x = curr_x if x_type == "front" else ROOM_X_MAX - curr_x
    abs_stored_x = stored_x if x_type == "front" else ROOM_X_MAX - stored_x
    # 右光束直接是绝对 Y，左光束需反转
    abs_curr_y = curr_y if y_type == "right" else ROOM_Y_MAX - curr_y
    abs_stored_y = stored_y if y_type == "right" else ROOM_Y_MAX - stored_y

    cdx = abs_curr_x - abs_stored_x
    cdy = abs_curr_y - abs_stored_y

    print(f"  精校: 存储({stored_x:.3f},{stored_y:.3f}) → 绝对({abs_stored_x:.3f},{abs_stored_y:.3f})")
    print(f"        当前({curr_x:.3f},{curr_y:.3f}) → 绝对({abs_curr_x:.3f},{abs_curr_y:.3f})")
    print(f"        底盘位移: ({cdx:+.3f}, {cdy:+.3f})")

    if abs(cdx) > POS_THRESHOLD or abs(cdy) > POS_THRESHOLD:
        cli.move_relative(cdx, cdy)


# ============================================================
#  表格打印
# ============================================================

def _print_obstacle_table(obstacles):
    """表格形式打印障碍物"""
    if not obstacles:
        print("  障碍物: 无")
        return
    print(f"\n  障碍物 ({len(obstacles)}个):")
    print(f"  {'─' * 40}")
    print(f"  {'#':<4} {'X':<10} {'Y':<10} {'距离':<10}")
    print(f"  {'─' * 40}")
    for i, obs in enumerate(obstacles, 1):
        print(f"  {i:<4} {obs['x']:<10.3f} {obs['y']:<10.3f} {obs['distance']:<10.3f}")
    print(f"  {'─' * 40}")


def _print_waypoint_table(waypoints, correction_points):
    """表格形式打印路径点"""
    cp_indices = {p["index"]: p for p in correction_points}
    print(f"\n  路径点 ({len(waypoints)}个):")
    print(f"  {'─' * 45}")
    print(f"  {'#':<4} {'X':<10} {'Y':<10} {'类型':<8} {'备注'}")
    print(f"  {'─' * 45}")
    for i, (x, y) in enumerate(waypoints):
        if i == 0:
            kind = "起点"
        elif i == len(waypoints) - 1:
            kind = "终点"
        else:
            kind = "中间"

        if i in cp_indices:
            cp = cp_indices[i]
            note = "✓可校正" if cp["safe"] else "⚠有遮挡"
        else:
            note = ""

        print(f"  {i:<4} {x:<10.3f} {y:<10.3f} {kind:<8} {note}")
    print(f"  {'─' * 45}")


# ============================================================
#  主流程
# ============================================================

def main():
    from path_planner import plan_path
    from target_points import load_targets
    from forbidden_zones import load_forbidden_zones
    from action_for_each_target import get_actions

    # ── 初始化 ──
    rospy.init_node("full_pipeline", anonymous=True)
    rospy.wait_for_message("scan", LaserScan, timeout=5.0)

    cli = _Client()
    cli.reset()
    cli.set_baseline()

    # 记录起点
    start_x, start_y = car_position()
    print(f"\n起点已记录: ({start_x:.3f}, {start_y:.3f})")

    # 加载配置
    targets = load_targets()
    if not targets:
        print("未找到 target_points.json，请先运行 target_points.py 设置点位")
        return
    zones = load_forbidden_zones()
    if zones:
        print(f"已加载 {len(zones)} 个禁区")

    print("\n" + "=" * 50)
    print("  输入 1~6 前往目标点，7 返回起点，Q 退出")
    print("=" * 50)

    while True:
        choice = input("\n>>> ").strip()
        if choice.upper() == 'Q':
            print("退出。")
            break

        try:
            num = int(choice)
        except ValueError:
            print("无效输入")
            continue

        # 确定目标坐标
        if num == 7:
            tx, ty = start_x, start_y
            target_info = None
            actions = []
            label = "起点"
        elif 1 <= num <= 6:
            t = targets[num - 1]
            tx, ty = t["plan"]
            target_info = t
            actions = get_actions(num)
            label = f"目标点 {num}"
        else:
            print("无效输入 (1~7)")
            continue

        # 当前位置
        cx, cy = car_position()
        print(f"\n{'─' * 40}")
        print(f"  {label}: ({tx:.2f}, {ty:.2f})  当前: ({cx:.3f}, {cy:.3f})")

        if abs(cx - tx) < 0.05 and abs(cy - ty) < 0.05:
            print("  已在目标点，跳过移动")
        else:
            # 规划
            result = plan_path(cx, cy, tx, ty, cx, cy, forbidden_zones=zones)
            wp = result["waypoints"]
            cp = result.get("correction_points", [])
            cp_indices = {p["index"]: p for p in cp}

            _print_obstacle_table(result.get("obstacles", []))
            _print_waypoint_table(wp, cp)

            # 执行移动
            move_ok = True
            for i in range(len(wp) - 1):
                x1, y1 = wp[i]
                x2, y2 = wp[i + 1]

                # 中间校正点
                if i in cp_indices and cp_indices[i]["safe"]:
                    print(f"\n  ● 校正点 #{i} ({x1:.3f}, {y1:.3f})")
                    _do_correction(cli, x1, y1)

                ok, resp = _move_segment(x1, y1, x2, y2, cli)
                if not ok:
                    print(f"  ✗ 移动失败: {resp.get('errorMessage', resp.get('error', '?'))}")
                    move_ok = False
                    break
                time.sleep(STEP_DELAY)

            if not move_ok:
                print("  移动未完成，跳过后续动作")
                continue

        # ── 到达后按顺序执行动作 ──
        print(f"\n  到达 {label}")

        for step, act in enumerate(actions):
            t = act["type"]
            if t == "correct":
                if target_info and "correct" in target_info:
                    _do_target_correction(cli, num, target_info)
                else:
                    _do_correction(cli, tx, ty)
            elif t == "rotate":
                deg = act.get("value", 0)
                print(f"  → 旋转 {deg}°")
                ok, resp = cli.circle(math.radians(deg))
                if not ok:
                    print(f"  ✗ 旋转失败: {resp}")
            elif t == "stay":
                input("  >>> 按 Enter 继续下一步...")

        print(f"  ✓ {label} 完成")

    cli.sess.close()


if __name__ == "__main__":
    main()
