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

ROOM_X_MAX = cfg.room.x_max
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
    """规划坐标 — 前墙距离(X) + 右墙距离(Y)。返回 (x, y) 或 (None, None)"""
    w = _read_walls()
    return w.get("前墙"), w.get("右墙")

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

    def set_baseline(self):
        return self._post("SetBaseline")

    def sync_yaw(self):
        return self._post("SyncYaw", timeout=20)

    def move_relative(self, dx, dy):
        return self._post("MoveRelative", {"delta_x": dx, "delta_y": dy})

    def move_x(self, target_x, ref_y, correct=True):
        return self._post("MoveOnlyX", {"location_x": target_x, "location_y": ref_y, "correct": correct})

    def move_y(self, target_y, ref_x, correct=True):
        return self._post("MoveOnlyY", {"location_x": ref_x, "location_y": target_y, "correct": correct})

    def circle(self, rad_z):
        return self._post("Circle", {"rad_z": rad_z})


# ============================================================
#  移动与校正
# ============================================================

def _move_segment(x1, y1, x2, y2, cli, correct=True):
    """轴对齐移动一段。correct=False 纯开环(不读LiDAR)，correct=True 用LiDAR+闭环"""
    dx, dy = x2 - x1, y2 - y1
    if not correct:
        # 非校正点：纯开环，拆成单轴移动，不碰 LiDAR
        # 注意：底盘坐标系与 LiDAR 坐标系方向相反，delta 需要取反
        if abs(dx) > 0.001:
            ok, resp = cli.move_relative(-dx, 0)
            if not ok:
                return False, resp
        if abs(dy) > 0.001:
            ok, resp = cli.move_relative(0, -dy)
            if not ok:
                return False, resp
        return True, None
    # 校正点：读 LiDAR 算初始距离 + 服务端闭环(有 sanity check)
    if abs(dx) > 0.001 and abs(dy) < 0.001:
        return cli.move_x(x2, y1, correct=True)
    elif abs(dy) > 0.001 and abs(dx) < 0.001:
        return cli.move_y(y2, x1, correct=True)
    return True, None


# 房间尺寸常量
_ROOM_W = cfg.room.x_max - cfg.room.x_min
_ROOM_H = cfg.room.y_max - cfg.room.y_min

_start_correct_beams = None  # (rear_beam, right_beam) 起点校正光束原始距离（暂保留）


def _lidar_readings_sane():
    """检查墙壁拟合是否合理：前+后≈房间宽，右+左≈房间高。"""
    w = _read_walls()
    f, r = w.get("前墙"), w.get("后墙")
    ri, l = w.get("右墙"), w.get("左墙")
    tol = cfg.client.sanity_check_tolerance
    if f is None or r is None or ri is None or l is None:
        return False
    return abs(f + r - _ROOM_W) < tol and abs(ri + l - _ROOM_H) < tol


def _read_angle_deviation():
    """读取当前偏角（度），基于前墙法向量。"""
    w = _read_walls()
    return w.get("yaw")


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
    """用校正光束坐标做精校"""
    if "correct" not in target_info:
        return
    stored_x, stored_y = target_info["correct"]
    curr_x, curr_y, axes = _read_correct_beams(pid)
    if curr_x is None or curr_y is None:
        print("  ⚠ 校正光束无回波，跳过精校")
        return

    # 前/后光束 → 绝对 X；左/右光束 → 绝对 Y
    beam_cfg = {
        1: ("front", "right"),
        2: ("rear",  "right"),
        3: ("rear",  "left"),
        4: ("front", "left"),
        5: ("rear",  "right"),
        6: ("rear",  "right"),
        7: ("rear",  "right"),
    }
    x_type, y_type = beam_cfg[pid]

    # X 方向
    if "x" in axes:
        abs_curr_x = curr_x if x_type == "front" else ROOM_X_MAX - curr_x
        abs_stored_x = stored_x if x_type == "front" else ROOM_X_MAX - stored_x
        cdx = abs_curr_x - abs_stored_x
    else:
        cdx = 0.0
        abs_curr_x, abs_stored_x = 0, 0

    # Y 方向
    if "y" in axes:
        abs_curr_y = curr_y if y_type == "right" else ROOM_Y_MAX - curr_y
        abs_stored_y = stored_y if y_type == "right" else ROOM_Y_MAX - stored_y
        cdy = abs_curr_y - abs_stored_y
    else:
        cdy = 0.0
        abs_curr_y, abs_stored_y = 0, 0

    print(f"  精校: 存储({stored_x:.3f},{stored_y:.3f}) → 绝对({abs_stored_x:.3f},{abs_stored_y:.3f})")
    print(f"        当前({curr_x:.3f},{curr_y:.3f}) → 绝对({abs_curr_x:.3f},{abs_curr_y:.3f})")
    print(f"        底盘位移: ({cdx:+.3f}, {cdy:+.3f}) [校正轴: {axes}]")

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

        # 起终点连线与X/Y轴夹角 < 15° → 偏好远离障碍物的路径(而非最短)
        near_axis_deg = 15
        a = math.degrees(math.atan2(abs(ty - cy), abs(tx - cx)))
        near_axis = a < near_axis_deg or a > 90 - near_axis_deg
        risk_weight = 1.0 if near_axis else 0.0
        if near_axis:
            print(f"  [规划] 起终点近轴(a={a:.1f}°)，启用远离障碍物偏好")

        # 层层降级：A*无解时逐步减小膨胀半径
        base_om = cfg.path_planning.obstacle_margin
        base_ge = cfg.path_planning.grid_expand
        levels = [
            (base_om,       base_ge,       "默认"),
            (base_om * 0.6, base_ge * 0.5, "降级1"),
            (base_om * 0.3, base_ge * 0.2, "降级2"),
            (base_om * 0.1, 0.0,           "降级3(最小)"),
        ]
        result = None
        final_level = "默认"
        for om, ge, level_name in levels:
            if result is not None:
                break
            result = plan_path(cx, cy, tx, ty, cx, cy, forbidden_zones=zones,
                               obstacles=obstacles,
                               obstacle_margin=om, grid_expand=ge,
                               cost_risk_weight=risk_weight)
            if result["path_name"] == "先X后Y(安全无解)" or not result["safe"]:
                print(f"  [规划] {level_name}: margin={om:.2f} expand={ge:.2f} → 无解，尝试降级...")
                if level_name == levels[-1][2]:
                    print(f"  [规划] 降至最小半径仍无解，使用安全路径")
                    final_level = level_name
                else:
                    result = None
            else:
                final_level = level_name
        wp = result["waypoints"]
        cp = result.get("correction_points", [])
        cp_indices = {p["index"]: p for p in cp}
        print(f"  [规划] 路径={result['path_name']}({final_level})  段数={len(wp)-1}  校正点={len(cp)}")

        _print_waypoint_table(wp, cp)

        from utils.visualize_path import visualize_plan
        visualize_plan(wp, correction_points=cp, obstacles=result.get("obstacles"),
                       zones=zones, title=f"路径规划: {label}", save_path="/tmp/path_plan.png")

        for i in range(len(wp) - 1):
            x1, y1 = wp[i]
            x2, y2 = wp[i + 1]
            dx, dy = x2 - x1, y2 - y1
            axis = "X" if abs(dx) > 0.001 else "Y"

            is_correction_point = i in cp_indices and cp_indices[i]["safe"]

            if is_correction_point:
                lidar_ok = _lidar_readings_sane()
                if lidar_ok:
                    print(f"  [校正] 路点#{i} ({x1:.3f},{y1:.3f}) SyncYaw →", end=" ")
                    cli.sync_yaw()
                    _do_correction(cli, x1, y1)
                    # sync_yaw 可能旋转了车身，重新检查 LiDAR 读数
                    lidar_ok = _lidar_readings_sane()
                else:
                    print(f"  [校正] 路点#{i} ({x1:.3f},{y1:.3f}) LiDAR异常，跳过校正")
            else:
                lidar_ok = False

            # 校正点 + LiDAR 读数合理 → 服务端闭环；否则纯开环
            use_correct = lidar_ok
            print(f"  [移动] #{i}→#{i+1} {axis}: {x1:.3f}→{x2:.3f}" if axis=="X" else f"  [移动] #{i}→#{i+1} {axis}: {y1:.3f}→{y2:.3f}", end=" ")
            ok, resp = _move_segment(x1, y1, x2, y2, cli, correct=use_correct)
            if not ok:
                print(f"✗ {resp.get('errorMessage', resp.get('error', '?'))}")
                return False
            angle = _read_angle_deviation()
            angle_str = f"  偏角={angle:.1f}°" if angle is not None else ""
            print(f"✓{angle_str}")
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

    print(f"  ✓ {label} 完成")
    _last_good_position = (tx, ty)
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
    # set_baseline 已废弃（墙壁建模不需要基准），保留调用兼容旧服务端
    cli.set_baseline()

    start_x, start_y = car_position()
    global _last_good_position
    _last_good_position = (start_x, start_y)
    if start_x is not None:
        print(f"\n起点已记录: ({start_x:.3f}, {start_y:.3f})")
    else:
        print("\n⚠ 起点定位失败")

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
