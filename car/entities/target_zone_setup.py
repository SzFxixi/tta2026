#!/usr/bin/env python3
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ROS_PYTHON_PATH = "/opt/ros/noetic/lib/python3/dist-packages"
if os.path.isdir(_ROS_PYTHON_PATH) and _ROS_PYTHON_PATH not in sys.path:
    sys.path.append(_ROS_PYTHON_PATH)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGETS_FILE = os.path.join(BASE_DIR, "target_points.json")
ZONES_FILE = os.path.join(BASE_DIR, "forbidden_zones.json")
NUM_TARGETS = 6

# Per-point forbidden-zone offsets derived from field calibration.
# zone = [point_x + xmin_offset, point_x + xmax_offset,
#         point_y + ymin_offset, point_y + ymax_offset]
ZONE_OFFSETS = {
    1: (-0.934, 0.219, -1.036, -0.198),
    2: (-0.263, 0.580, -1.072, -0.179),
    3: (-0.265, 0.463, 0.239, 1.174),
    4: (-0.973, 0.234, 0.233, 0.967),
    5: (-0.895, -0.101, -0.735, 0.568),
    6: (0.194, 0.837, -0.720, 0.558),
}

_ROS_INITIALIZED = False


def _load_ros():
    import rospy
    from sensor_msgs.msg import LaserScan
    return rospy, LaserScan


def _ensure_ros_ready():
    global _ROS_INITIALIZED
    rospy, LaserScan = _load_ros()
    if not _ROS_INITIALIZED:
        rospy.init_node("target_zone_setup_tool", anonymous=True)
        _ROS_INITIALIZED = True
    rospy.wait_for_message("scan", LaserScan, timeout=5.0)


def _read_position():
    """Target-point coordinate using wall fitting (same as navigation).

    Returns (front_wall_dist, right_wall_dist) — consistent with
    car_position() in run.py, so recorded coordinates match what the
    car actually navigates to.
    """
    from utils.wall_positioning import fit_walls
    from utils.config_loader import cfg

    rospy, LaserScan = _load_ros()
    data = rospy.wait_for_message("scan", LaserScan)
    w = fit_walls(data)

    x = w.get("前墙")
    y = w.get("右墙")

    # Fallback: use opposite wall if direct wall is occluded
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

    if x is None or y is None:
        raise RuntimeError(
            f"墙壁拟合失败：前墙={w.get('前墙')}, 右墙={w.get('右墙')}, "
            f"后墙={w.get('后墙')}, 左墙={w.get('左墙')}")

    return round(x, 3), round(y, 3)


def _read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_targets():
    """Read target point config. Returns point dict list, or None if missing."""
    config = _read_json(TARGETS_FILE)
    if config is None:
        return None
    return config.get("points", [])


def generate_forbidden_zones(points):
    """Generate six rectangular forbidden zones from six target points."""
    if not points or len(points) < NUM_TARGETS:
        raise ValueError(f"need {NUM_TARGETS} target points, got {len(points) if points else 0}")

    zones = []
    for point_id in range(1, NUM_TARGETS + 1):
        point = points[point_id - 1]
        if "plan" not in point or len(point["plan"]) < 2:
            raise ValueError(f"target point {point_id} missing plan coordinate")
        x, y = point["plan"][:2]
        dx_min, dx_max, dy_min, dy_max = ZONE_OFFSETS[point_id]
        zones.append([
            round(x + dx_min, 3),
            round(x + dx_max, 3),
            round(y + dy_min, 3),
            round(y + dy_max, 3),
        ])
    return zones


def save_forbidden_zones(zones):
    _write_json(ZONES_FILE, {"zones": zones})
    print(f"{len(zones)} 个禁区已保存到 {ZONES_FILE}")


def regenerate_forbidden_zones():
    points = load_targets()
    zones = generate_forbidden_zones(points)
    save_forbidden_zones(zones)
    return zones


def list_targets():
    config = _read_json(TARGETS_FILE)
    if config is None or not config.get("points"):
        print("暂无目标点配置")
        return 0
    for i, p in enumerate(config["points"], 1):
        px, py = p["plan"]
        print(f"  目标点 {i}: ({px:.3f},{py:.3f})")
    return len(config["points"])


def list_forbidden_zones():
    config = _read_json(ZONES_FILE)
    if config is None or not config.get("zones"):
        print("暂无禁区配置")
        return 0
    for i, (xmin, xmax, ymin, ymax) in enumerate(config["zones"], 1):
        print(f"  禁区 {i}: X[{xmin:.3f}~{xmax:.3f}] Y[{ymin:.3f}~{ymax:.3f}]")
    return len(config["zones"])


def setup_targets():
    """Interactively record target points and regenerate forbidden zones."""
    _ensure_ros_ready()
    points = []
    print(f"\n目标点坐标设置（共 {NUM_TARGETS} 个）")
    print("每点只记录一个坐标: (X, Y)")
    print("完成后会根据 1~6 目标点自动生成 6 个禁区\n")

    for i in range(1, NUM_TARGETS + 1):
        print(f"目标点 {i}/{NUM_TARGETS}")
        input("  移动到位后按 Enter >>> ")

        x, y = _read_position()
        points.append({"plan": (x, y)})
        print(f"  ✓ 已记录 ({x:.3f},{y:.3f})\n")

    _write_json(TARGETS_FILE, {"points": points})
    print(f"{NUM_TARGETS} 个目标点已保存到 {TARGETS_FILE}")

    zones = generate_forbidden_zones(points)
    save_forbidden_zones(zones)
    return points


def modify_target():
    """Modify a single target point and regenerate forbidden zones."""
    points = load_targets()
    if not points or len(points) < NUM_TARGETS:
        print("目标点未设置完整，请先用 s 全部设置")
        return

    _ensure_ros_ready()
    list_targets()
    try:
        pid = int(input(f"\n  修改哪个目标点 (1~{NUM_TARGETS})? >>> "))
    except ValueError:
        print("  无效输入")
        return
    if pid < 1 or pid > NUM_TARGETS:
        print(f"  目标点编号须在 1~{NUM_TARGETS}")
        return

    print(f"\n目标点 {pid}/{NUM_TARGETS}")
    old_x, old_y = points[pid - 1]["plan"]
    print(f"  旧坐标: ({old_x:.3f}, {old_y:.3f})")
    input("  移动到新位置后按 Enter >>> ")

    x, y = _read_position()
    points[pid - 1]["plan"] = (x, y)
    print(f"  ✓ 已更新 ({old_x:.3f},{old_y:.3f}) → ({x:.3f},{y:.3f})\n")

    _write_json(TARGETS_FILE, {"points": points})
    print(f"目标点 {pid} 已保存")

    zones = generate_forbidden_zones(points)
    save_forbidden_zones(zones)


def cli_main():
    print("目标点/禁区管理工具")
    print("  s  - 重新设置目标点，并自动生成禁区")
    print("  m  - 修改单个目标点，并自动重新生成禁区")
    print("  g  - 根据已有目标点重新生成禁区")
    print("  l  - 列出目标点")
    print("  z  - 列出禁区")
    print("  q  - 退出")

    while True:
        cmd = input("\n>>> ").strip().lower()
        if cmd == "q":
            break
        elif cmd == "s":
            setup_targets()
        elif cmd == "m":
            modify_target()
        elif cmd == "g":
            regenerate_forbidden_zones()
        elif cmd == "l":
            list_targets()
        elif cmd == "z":
            list_forbidden_zones()
        else:
            print("未知命令")


if __name__ == "__main__":
    cli_main()
