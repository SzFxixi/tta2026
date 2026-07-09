import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import math
import threading
import rospy
import requests
import numpy as np
from sensor_msgs.msg import LaserScan

# ============================================================
#  参数 — 从 config.yaml 读取
# ============================================================

from utils.config_loader import cfg

CAR_IP = cfg.client.car_ip
CAR_PORT = cfg.client.car_port
BASE_URL = f"http://{CAR_IP}:{CAR_PORT}"
MOVE_TIMEOUT = cfg.client.move_timeout
STEP_DELAY = cfg.client.step_delay
POS_THRESHOLD = cfg.client.pos_threshold
MAX_RETRIES = cfg.client.max_retries
DRIFT_THRESHOLD = cfg.client.drift_threshold
HARD_DRIFT = cfg.client.hard_drift_threshold

ROOM_X_MIN = cfg.room.x_min
ROOM_X_MAX = cfg.room.x_max
ROOM_Y_MIN = cfg.room.y_min
ROOM_Y_MAX = cfg.room.y_max

# 每个点位的校正光束索引表达式
# (X光束, Y光束, 校正轴: "xy"/"x"/"y")
_CORRECT_BEAMS = {
    1: ("n//2",   "n//4",   "xy"),
    2: ("n-2",    "n//4",   "xy"),
    3: ("n-2",    "n*3//4", "xy"),
    4: ("n//2",   "n*3//4", "xy"),
    5: ("n-2",    "n//4",   "x"),
    6: ("n-2",    "n//4",   "x"),
    7: ("n-2",    "n//4",   "xy"),
}


# ============================================================
#  LiDAR 定位（墙壁直线建模）
# ============================================================

from utils.wall_positioning import fit_walls

_walls_cache = None
_walls_cache_time = 0
_CACHE_TTL = 0.05

def _read_walls(walls=None):
    """读一次 LiDAR，拟合墙壁，50ms 内重复调用走缓存。"""
    global _walls_cache, _walls_cache_time
    now = time.time()
    if _walls_cache is not None and (now - _walls_cache_time) < _CACHE_TTL:
        return _walls_cache
    data = rospy.wait_for_message("scan", LaserScan)
    _walls_cache = fit_walls(data, walls=walls)
    _walls_cache_time = now
    return _walls_cache

def car_position():
    """规划坐标 — 前墙距离(X) + 右墙距离(Y)。
    若某墙被遮挡，用对面墙推算。返回 (x, y) 或 (None, None)。"""
    w = _read_walls()
    x = w.get("前墙")
    y = w.get("右墙")
    if x is None:
        rear = w.get("后墙")
        if rear is not None:
            rw = cfg.room.x_max - cfg.room.x_min
            x = rw - rear
    if y is None:
        left = w.get("左墙")
        if left is not None:
            rh = cfg.room.y_max - cfg.room.y_min
            y = rh - left
    return x, y

def get_x():
    """后墙距离 (m)，墙壁建模。"""
    return _read_walls().get("后墙")

def get_y():
    """左墙距离 (m)，墙壁建模。"""
    return _read_walls().get("左墙")

# 保留旧接口用于目标点校正（校正坐标仍用原始光束值，后续迁移）
def _get_beam(index, timeout=None):
    if timeout is None:
        timeout = cfg.client.lidar_beam_timeout
    data = rospy.wait_for_message("scan", LaserScan, timeout=5.0)
    n = len(data.ranges)
    d = data.ranges[index % n]
    t0 = time.time()
    while d == np.inf:
        if time.time() - t0 > timeout:
            return None
        data = rospy.wait_for_message("scan", LaserScan, timeout=5.0)
        n = len(data.ranges)
        d = data.ranges[index % n]
    return d

def _get_n():
    return len(rospy.wait_for_message("scan", LaserScan).ranges)

def _read_correct_beams(pid):
    """读取点位 pid 对应的校正光束原始距离（暂保留旧实现）。"""
    if pid not in _CORRECT_BEAMS:
        return None, None, "xy"
    n = _get_n()
    cfg = _CORRECT_BEAMS[pid]
    x_expr, y_expr = cfg[0], cfg[1]
    axes = cfg[2] if len(cfg) > 2 else "xy"
    x = _get_beam(eval(x_expr, {"n": n}))
    y = _get_beam(eval(y_expr, {"n": n}))
    return x, y, axes


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

    def sync_yaw(self):
        return self._post("SyncYaw", timeout=20)

    def move_relative(self, dx, dy):
        return self._post("MoveRelative", {"delta_x": dx, "delta_y": dy})

    def circle(self, rad_z):
        return self._post("Circle", {"rad_z": rad_z})


# ============================================================
#  移动与校正
# ============================================================

def _point_to_segment_dist(px, py, ax, ay, bx, by):
    """点 (px,py) 到线段 AB 的最短距离"""
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _move_segment_monitored(x1, y1, x2, y2, cli,
                             drift_threshold=DRIFT_THRESHOLD,
                             safety_monitor=None):
    """整段移动 + 后台实时监控。偏移偏大/离障碍物太近时立即打断校正。"""
    seg_start_x, seg_start_y = x1, y1
    yaw_threshold = cfg.server.sync_yaw_threshold_deg

    while True:
        dx_rem = x2 - seg_start_x
        dy_rem = y2 - seg_start_y
        seg_len = math.hypot(dx_rem, dy_rem)
        if seg_len < 0.05:  # 剩余距离足够短，视为完成
            return True, None, False

        result = {"ok": None, "resp": None, "drifted": False, "correct": None, "interrupt": False, "yaw_interrupt": False}

        def _do_move():
            ok, resp = cli.move_relative(-dx_rem, -dy_rem)
            result["ok"] = ok
            result["resp"] = resp

        t = threading.Thread(target=_do_move)
        t.start()

        # 主线程：边走边读 LiDAR
        last_valid_x, last_valid_y = None, None
        while t.is_alive():
            time.sleep(0.15)
            cx, cy = car_position()
            yaw = _read_angle_deviation()

            if cx is None or cy is None:
                continue
            if not _lidar_readings_sane():
                continue

            # 帧间跳变检查：0.15s 内位置不可能跳 > 0.5m
            if last_valid_x is not None and last_valid_y is not None:
                jump = math.hypot(cx - last_valid_x, cy - last_valid_y)
                if jump > 0.5:
                    continue  # 垃圾帧，跳过
            last_valid_x, last_valid_y = cx, cy

            drift = _point_to_segment_dist(cx, cy, seg_start_x, seg_start_y, x2, y2)

            if drift > HARD_DRIFT:
                status = "严重偏移"
            elif drift > drift_threshold:
                status = "偏移偏大"
            elif yaw is not None and abs(yaw) > yaw_threshold:
                status = "偏角过大"
            else:
                status = "正常"

            yaw_str = f"{yaw:.1f}°" if yaw is not None else "-"
            obs_str = safety_monitor.dist_summary(cx, cy) if safety_monitor else ""
            print(f"  X={cx:.3f} Y={cy:.3f}  偏角={yaw_str}  偏离={drift:.3f}m{obs_str}  {status}")

            # 离障碍物太近 → 急停
            if safety_monitor:
                should_stop, stop_reason = safety_monitor.check(cx, cy)
                if should_stop:
                    print(f"  ⚠ {stop_reason}")
                    result["drifted"] = True
                    t.join(timeout=5)
                    break

            if drift > HARD_DRIFT:
                result["drifted"] = True
                t.join(timeout=5)
                break
            elif drift > drift_threshold:
                result["correct"] = (cx, cy)
                result["interrupt"] = True
                t.join(timeout=5)
                break
            elif yaw is not None and abs(yaw) > yaw_threshold:
                result["yaw_interrupt"] = True
                t.join(timeout=5)
                break

        t.join(timeout=30)

        if result["drifted"]:
            return False, {"error": "drifted"}, True

        # 偏角过大 → 立即校正（监控中打断或移动完成后检查）
        yaw = _read_angle_deviation()
        if yaw is not None and abs(yaw) > yaw_threshold:
            print(f"  偏角={yaw:.1f}° 过大，校正中...")
            cli.sync_yaw()

        if result["correct"]:
            cx_before, cy_before = car_position()
            if cx_before is not None and cy_before is not None:
                if abs(dx_rem) > 0.001:
                    cdx, cdy = 0.0, seg_start_y - cy_before   # X 段：校正 Y
                else:
                    cdx, cdy = seg_start_x - cx_before, 0.0   # Y 段：校正 X
                action = "打断校正" if result["interrupt"] else "垂直校正"
                print(f"  {action} 前({cx_before:.3f},{cy_before:.3f}) + ({cdx:+.3f},{cdy:+.3f}) → 期望({cx_before+cdx:.3f},{cy_before+cdy:.3f})  规划线 {'Y' if abs(dx_rem)>0.001 else 'X'}={'='+str(seg_start_y) if abs(dx_rem)>0.001 else '='+str(seg_start_x)}")
                cli.move_relative(-cdx, -cdy)
                time.sleep(0.1)
                cx_after, cy_after = car_position()
                if cx_after is not None:
                    print(f"  校正后实际 ({cx_after:.3f},{cy_after:.3f})")
                    if result["interrupt"]:
                        # 更新起点为校正后位置，继续走剩余距离
                        seg_start_x, seg_start_y = cx_after, cy_after
                        continue  # 回到 while 循环，用剩余距离继续移动

        # 正常完成（未被中断打断）
        if not result["ok"]:
            return False, result["resp"], False
        return result["ok"], result["resp"], False


_start_correct_beams = None  # (rear_beam, right_beam) 起点校正光束原始距离（暂保留）


def _lidar_readings_sane():
    """检查墙壁拟合是否合理：前+后≈房间宽，右+左≈房间高。"""
    w = _read_walls()
    f, r = w.get("前墙"), w.get("后墙")
    ri, l = w.get("右墙"), w.get("左墙")
    tol = cfg.client.sanity_check_tolerance
    if f is None or r is None or ri is None or l is None:
        return False
    rw = cfg.room.x_max - cfg.room.x_min
    rh = cfg.room.y_max - cfg.room.y_min
    return abs(f + r - rw) < tol and abs(ri + l - rh) < tol


def _read_angle_deviation():
    """读取当前偏角（度），归一化到 [-90, 90]。"""
    w = _read_walls()
    yaw = w.get("yaw")
    if yaw is None:
        return None
    # 归一化：>90° 则反向（墙壁法向量可能指反了）
    while yaw > 90:
        yaw -= 180
    while yaw < -90:
        yaw += 180
    return yaw


def _do_correction(cli, expected_x, expected_y):
    """坐标校正（用规划坐标前X+右Y做LiDAR闭环）"""
    # 表头
    print(f"  {'─' * 70}")
    print(f"  {'轮次':<4} {'期望X':<10} {'期望Y':<10} {'实际X':<10} {'实际Y':<10} {'dX':<8} {'dY':<8} {'偏角':<8}")
    print(f"  {'─' * 70}")
    for attempt in range(1, MAX_RETRIES + 1):
        if not _lidar_readings_sane():
            print(f"  ⚠ 前后和或左右和异常，LiDAR 读数不可靠，放弃校正")
            break
        cx, cy = car_position()
        if cx is None or cy is None:
            print(f"  ⚠ LiDAR 无回波，跳过坐标校正")
            return
        angle = _read_angle_deviation()
        dx = expected_x - cx
        dy = expected_y - cy
        err = math.hypot(dx, dy)
        angle_str = f"{angle:.1f}°" if angle is not None else "-"
        print(f"  {attempt:<4} {expected_x:<10.3f} {expected_y:<10.3f} {cx:<10.3f} {cy:<10.3f} {dx:<+8.3f} {dy:<+8.3f} {angle_str:<8}")
        if err <= POS_THRESHOLD:
            print(f"  {'─' * 70}")
            print("  ✓ 已收敛")
            break
        cdx, cdy = cx - expected_x, cy - expected_y
        cli.move_relative(cdx, cdy)
        time.sleep(0.3)
    else:
        print(f"  {'─' * 70}")
        print(f"  ⚠ 达到最大重试")


def _do_target_correction(cli, pid, target_info):
    """用墙壁拟合坐标做精校（替代老的 raw beam 方案）"""
    if "plan" not in target_info:
        return
    expected_x, expected_y = target_info["plan"]

    if not _lidar_readings_sane():
        print("  ⚠ LiDAR不可靠，跳过精校")
        return

    cx, cy = car_position()
    if cx is None or cy is None:
        print("  ⚠ 定位失败，跳过精校")
        return

    cdx = expected_x - cx   # LiDAR 差量：期望 − 当前
    cdy = expected_y - cy

    print(f"  精校: 期望({expected_x:.3f},{expected_y:.3f})")
    print(f"        当前({cx:.3f},{cy:.3f})")
    print(f"        LiDAR差量 ({cdx:+.3f}, {cdy:+.3f}) → chassis发送 ({-cdx:+.3f}, {-cdy:+.3f})")

    if abs(cdx) > POS_THRESHOLD or abs(cdy) > POS_THRESHOLD:
        cli.move_relative(-cdx, -cdy)


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


def _print_waypoint_table(waypoints):
    """表格形式打印路径点"""
    print(f"\n  路径点 ({len(waypoints)}个):")
    print(f"  {'─' * 35}")
    print(f"  {'#':<4} {'X':<10} {'Y':<10} {'类型':<8}")
    print(f"  {'─' * 35}")
    for i, (x, y) in enumerate(waypoints):
        if i == 0:
            kind = "起点"
        elif i == len(waypoints) - 1:
            kind = "终点"
        else:
            kind = "中间"
        print(f"  {i:<4} {x:<10.3f} {y:<10.3f} {kind:<8}")
    print(f"  {'─' * 35}")


# ============================================================
#  主流程
# ============================================================

# 外部控制信号（--external 模式）
_next_point = None
_arm_signal = None  # 外部模式时设为 threading.Event()

_last_good_position = None  # 上一次成功到达的目标坐标，LiDAR异常时兜底


def _execute_point(num, targets, zones, start_pos, cli, obstacles=None):
    """执行单次导航 + 动作，返回 True 成功 / False 失败"""
    from entities.path_planner import plan_path
    from entities.action_for_each_target import get_actions

    start_x, start_y = start_pos

    if num == 7:
        tx, ty = start_x, start_y
        if _start_correct_beams is not None:
            target_info = {"plan": (start_x, start_y), "correct": _start_correct_beams}
            actions = [{"type": "correct"}]
        else:
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
        return False

    global _last_good_position

    cx, cy = car_position()
    if cx is None or cy is None:
        cx, cy = None, None

    if not _lidar_readings_sane() or cx is None:
        if _last_good_position is not None:
            cx, cy = _last_good_position
            print(f"\n  ⚠ LiDAR不可靠，沿用上次坐标: ({cx:.3f}, {cy:.3f})")
        else:
            return False

    print(f"\n{'─' * 40}")
    print(f"  {label}: ({tx:.2f}, {ty:.2f})  当前: ({cx:.3f}, {cy:.3f})")

    if not (abs(cx - tx) < 0.05 and abs(cy - ty) < 0.05):
        print(f"\n  [规划] {label} 路径规划中...")

        # 安全监控 + 先离开危险区，再规划
        from entities.safety_monitor import SafetyMonitor
        safety = SafetyMonitor(cli, car_position, obstacles, zones)
        cx, cy = safety.ensure_safe(cx, cy)

        result = plan_path(cx, cy, tx, ty, cx, cy, forbidden_zones=zones,
                           obstacles=obstacles)
        wp = result["waypoints"]
        print(f"  [规划] {result['path_name']}  段数={len(wp)-1}")

        _print_waypoint_table(wp)

        from utils.visualize_path import visualize_plan
        visualize_plan(wp, obstacles=result.get("obstacles"),
                       zones=zones, title=f"路径规划: {label}", save_path="/tmp/path_plan.png")

        replan_count = 0
        seg_idx = 0
        while seg_idx < len(wp) - 1:
            x1, y1 = wp[seg_idx]
            x2, y2 = wp[seg_idx + 1]
            dx, dy = x2 - x1, y2 - y1
            dist = abs(dx) if abs(dx) > 0.001 else abs(dy)

            if abs(dx) > 0.001:
                print(f"  #{seg_idx}→#{seg_idx+1}  X  {x1:.3f} → {x2:.3f}  ({dist:.2f}m)")
            else:
                print(f"  #{seg_idx}→#{seg_idx+1}  Y  {y1:.3f} → {y2:.3f}  ({dist:.2f}m)")

            ok, resp, need_replan = _move_segment_monitored(x1, y1, x2, y2, cli,
                                                              safety_monitor=safety)

            if need_replan:
                replan_count += 1
                print(f"  ✗ 偏离规划线 → 重规划 #{replan_count}")

                cx, cy = car_position()
                if cx is None or cy is None:
                    print("  ⚠ 无法定位，放弃")
                    return False

                cx, cy = safety.ensure_safe(cx, cy)

                result = plan_path(cx, cy, tx, ty, cx, cy,
                                   forbidden_zones=zones, obstacles=obstacles)
                wp = result["waypoints"]
                seg_idx = 0
                print(f"  [重规划#{replan_count}] {result['path_name']}  段数={len(wp)-1}")
                _print_waypoint_table(wp)
                visualize_plan(wp, obstacles=result.get("obstacles"),
                               zones=zones, title=f"重规划#{replan_count}: {label}",
                               save_path="/tmp/path_plan.png")
                continue

            if not ok:
                print(f"  ✗ {resp.get('errorMessage', resp.get('error', '?'))}")
                return False

            seg_idx += 1
            time.sleep(STEP_DELAY)

    # 执行动作
    for act in actions:
        t = act["type"]
        if t == "correct":
            if num == 7 and not _lidar_readings_sane():
                print("  ⚠ 前后和或左右和异常，跳过起点精校")
                continue
            if target_info and "correct" in target_info:
                _do_target_correction(cli, num, target_info)
            else:
                _do_correction(cli, tx, ty)
        elif t == "move_rel":
            cli.move_relative(act.get("dx", 0), act.get("dy", 0))
        elif t == "rotate":
            cli.circle(act.get("value", 0))
        elif t == "stay":
            if _arm_signal is not None:
                _arm_signal.clear()
                _arm_signal.wait()
            else:
                input("  >>> 按 Enter 继续...")
            sx, sy = car_position()
            yaw = _read_angle_deviation()
            if sx is not None:
                yaw_str = f"{yaw:.1f}°" if yaw is not None else "-"
                print(f"  停留位置: X={sx:.3f} Y={sy:.3f}  偏角={yaw_str}")

    print(f"  ✓ {label} 完成")
    sx, sy = car_position()
    if sx is not None and sy is not None:
        _last_good_position = (sx, sy)
    else:
        _last_good_position = (tx, ty)  # fallback
    return True


def _run_interactive(start_x, start_y, targets, zones, cli, obstacles):
    """交互模式：键盘输入"""
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
        _execute_point(num, targets, zones, (start_x, start_y), cli, obstacles)


def _run_external(start_x, start_y, targets, zones, cli, obstacles):
    """外部模式：HTTP API"""
    import threading as _th
    from flask import Flask as _Flask, request as _req, jsonify as _j
    global _arm_signal, _next_point

    _arm_signal = _th.Event()
    _next_point = None
    _point_event = _th.Event()
    ext_app = _Flask(__name__)

    @ext_app.route('/go', methods=['POST'])
    def _go():
        global _next_point
        data = _req.json or {}
        pt = data.get("point")
        if pt not in (1,2,3,4,5,6,7):
            return _j({"ok": False, "error": f"无效点位 {pt}"}), 400
        _next_point = pt
        _point_event.set()
        return _j({"ok": True, "point": pt})

    @ext_app.route('/continue', methods=['POST'])
    def _cont():
        _arm_signal.set()
        return _j({"ok": True})

    @ext_app.route('/status', methods=['GET'])
    def _stat():
        return _j({"busy": _point_event.is_set() and not _arm_signal.is_set()})

    _th.Thread(target=lambda: ext_app.run(host='0.0.0.0', port=6000), daemon=True).start()
    print("外部模式就绪: /go (飞机)  /continue (机械臂)  /status")

    while True:
        _point_event.clear()
        _point_event.wait()
        num = _next_point
        _next_point = None
        _execute_point(num, targets, zones, (start_x, start_y), cli, obstacles)


def main():
    from entities.target_points import load_targets
    from entities.forbidden_zones import load_forbidden_zones
    external_mode = "--external" in sys.argv

    rospy.init_node("full_pipeline", anonymous=True)
    rospy.wait_for_message("scan", LaserScan, timeout=5.0)

    cli = _Client()
    cli.reset()

    # 从 LiDAR 实测房间尺寸，覆盖 config
    w = _read_walls()
    room_x = w.get("前墙", 0) + w.get("后墙", 0) if w.get("前墙") and w.get("后墙") else 0
    room_y = w.get("右墙", 0) + w.get("左墙", 0) if w.get("右墙") and w.get("左墙") else 0
    if room_x > 0 and room_y > 0:
        cfg.room.x_min = 0.0
        cfg.room.x_max = room_x
        cfg.room.y_min = 0.0
        cfg.room.y_max = room_y
        # 障碍物过滤边界也跟着扩
        cfg.obstacle.filter_x_max = room_x + 0.2
        cfg.obstacle.filter_y_max = room_y + 0.2
        print(f"房间实测: {room_x:.2f}×{room_y:.2f}m  (wall_margin={cfg.path_planning.wall_margin})")
    else:
        print(f"⚠ 房间测量失败，使用 config 默认值")

    start_x, start_y = w.get("前墙"), w.get("右墙")
    global _last_good_position
    _last_good_position = (start_x, start_y)
    if start_x is not None:
        print(f"起点: ({start_x:.3f}, {start_y:.3f})")
    else:
        print("⚠ 起点定位失败")

    global _start_correct_beams
    n = _get_n()
    sr = _get_beam(n - 2)   # rear beam → X
    sry = _get_beam(n // 4)  # right beam → Y
    if sr is not None and sry is not None:
        _start_correct_beams = (sr, sry)
        print(f"起点校正光束: rear={sr:.3f}  right={sry:.3f}")
    else:
        _start_correct_beams = None
        print("⚠ 起点校正光束无回波")

    targets = load_targets()
    if not targets:
        print("未找到 target_points.json，请先运行 target_points.py 设置点位")
        return
    zones = load_forbidden_zones()
    if zones:
        print(f"已加载 {len(zones)} 个禁区")

    # 一次性检测障碍物，后续规划复用
    from entities.obstacle_detector import detect_obstacles

    car_x, car_y = car_position()
    obstacles, _ = detect_obstacles(car_x, car_y, forbidden_zones=zones)

    print(f"检测到 {len(obstacles)} 个障碍物")
    _print_obstacle_table(obstacles)

    if external_mode:
        _run_external(start_x, start_y, targets, zones, cli, obstacles)
    else:
        print("\n" + "=" * 50)
        print("  输入 1~6 前往目标点，7 返回起点，Q 退出")
        print("=" * 50)
        _run_interactive(start_x, start_y, targets, zones, cli, obstacles)

    cli.sess.close()


if __name__ == "__main__":
    main()
