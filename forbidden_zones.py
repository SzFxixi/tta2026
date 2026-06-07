#!/usr/bin/env python3
"""
禁区管理模块 — 独立于路径规划，支持交互式设置、保存/加载、追加、删除。

配置文件 forbidden_zones.json 格式:
    {
        "origin_x": 1.446,
        "origin_y": 4.799,
        "zones": [
            [xmin, xmax, ymin, ymax],
            ...
        ]
    }

用法:
    from forbidden_zones import (setup_forbidden_zones, load_forbidden_zones,
                                  add_forbidden_zone, delete_forbidden_zone,
                                  point_in_forbidden, segment_crosses_forbidden)
"""

import json
import os
import math

# 配置文件路径（与本模块同目录）
_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "forbidden_zones.json")


# ============================================================
#  文件读写
# ============================================================

def _read_config():
    if not os.path.exists(_CONFIG_FILE):
        return None
    with open(_CONFIG_FILE, 'r') as f:
        return json.load(f)


def _write_config(config):
    with open(_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


# ============================================================
#  查询
# ============================================================

def load_forbidden_zones():
    """读取禁区配置。返回 (zones, origin_x, origin_y)，无文件时返回 (None, 0, 0)。"""
    config = _read_config()
    if config is None:
        return None, 0.0, 0.0
    return config.get("zones", []), config.get("origin_x", 0.0), config.get("origin_y", 0.0)


def list_forbidden_zones():
    """打印当前已保存的禁区列表。返回禁区数量。"""
    config = _read_config()
    if config is None or not config.get("zones"):
        print("暂无禁区配置")
        return 0
    ox, oy = config["origin_x"], config["origin_y"]
    print(f"禁区原点: ({ox:.3f}, {oy:.3f})")
    for i, (xmin, xmax, ymin, ymax) in enumerate(config["zones"]):
        print(f"  #{i}: X[{xmin:+.3f}~{xmax:+.3f}] Y[{ymin:+.3f}~{ymax:+.3f}]")
    return len(config["zones"])


# ============================================================
#  交互式设置
# ============================================================

def setup_forbidden_zones():
    """
    交互式设置禁区（覆盖已有配置）。移动小车到角点按 Enter，对角点按 Enter。
    按 Q 退出。原点以调用时的小车位置为准。
    """
    from path_planner import get_car_position

    origin_x, origin_y = get_car_position()
    zones = []
    print(f"\n禁区设置 (原点={origin_x:.3f},{origin_y:.3f})")
    print("移动小车到禁区角点，按 Enter 记录；按 Q 退出")

    while True:
        choice = input(">>> ").strip()
        if choice.upper() == 'Q':
            break

        cx1, cy1 = get_car_position()
        rx1, ry1 = cx1 - origin_x, cy1 - origin_y
        print(f"  角点1 绝对:({cx1:.3f},{cy1:.3f})  相对:({rx1:+.3f},{ry1:+.3f})")

        input("  移动到对角点后按 Enter...")
        cx2, cy2 = get_car_position()
        rx2, ry2 = cx2 - origin_x, cy2 - origin_y
        print(f"  角点2 绝对:({cx2:.3f},{cy2:.3f})  相对:({rx2:+.3f},{ry2:+.3f})")

        zones.append((min(rx1, rx2), max(rx1, rx2), min(ry1, ry2), max(ry1, ry2)))
        print(f"  ✓ 禁区 #{len(zones)-1}: X[{zones[-1][0]:.3f}~{zones[-1][1]:.3f}] "
              f"Y[{zones[-1][2]:.3f}~{zones[-1][3]:.3f}]")

    config = {"origin_x": origin_x, "origin_y": origin_y, "zones": zones}
    _write_config(config)
    print(f"共 {len(zones)} 个禁区，已保存\n")
    return zones, origin_x, origin_y


# ============================================================
#  追加
# ============================================================

def add_forbidden_zone():
    """
    向已有配置追加一个禁区。需要 LiDAR 已就绪且小车在原地。
    原点沿用已有配置的 origin，不可更改。
    """
    from path_planner import get_car_position

    config = _read_config()
    if config is None:
        print("暂无禁区配置，请先运行 setup_forbidden_zones()")
        return False

    origin_x, origin_y = config["origin_x"], config["origin_y"]
    zones = config.get("zones", [])
    print(f"追加禁区 (原点={origin_x:.3f},{origin_y:.3f})")

    cx1, cy1 = get_car_position()
    rx1, ry1 = cx1 - origin_x, cy1 - origin_y
    print(f"  角点1 绝对:({cx1:.3f},{cy1:.3f})  相对:({rx1:+.3f},{ry1:+.3f})")

    input("  移动到对角点后按 Enter...")
    cx2, cy2 = get_car_position()
    rx2, ry2 = cx2 - origin_x, cy2 - origin_y
    print(f"  角点2 绝对:({cx2:.3f},{cy2:.3f})  相对:({rx2:+.3f},{ry2:+.3f})")

    zones.append((min(rx1, rx2), max(rx1, rx2), min(ry1, ry2), max(ry1, ry2)))
    config["zones"] = zones
    _write_config(config)
    idx = len(zones) - 1
    print(f"  ✓ 已追加禁区 #{idx}: X[{zones[-1][0]:.3f}~{zones[-1][1]:.3f}] "
          f"Y[{zones[-1][2]:.3f}~{zones[-1][3]:.3f}]\n")
    return True


# ============================================================
#  删除
# ============================================================

def delete_forbidden_zone(index=None):
    """
    删除指定索引的禁区。不传 index 则列出所有禁区让用户选择。
    """
    config = _read_config()
    if config is None or not config.get("zones"):
        print("暂无禁区可删除")
        return False

    zones = config["zones"]

    if index is None:
        list_forbidden_zones()
        try:
            index = int(input("输入要删除的禁区序号: ").strip())
        except (ValueError, EOFError):
            print("取消删除")
            return False

    if index < 0 or index >= len(zones):
        print(f"无效序号: {index}（有效范围 0~{len(zones)-1}）")
        return False

    removed = zones.pop(index)
    config["zones"] = zones
    _write_config(config)
    print(f"已删除禁区 #{index}: X[{removed[0]:.3f}~{removed[1]:.3f}] "
          f"Y[{removed[2]:.3f}~{removed[3]:.3f}]")
    return True


# ============================================================
#  几何检测
# ============================================================

def point_in_forbidden(px, py, forbidden_zones, origin_x=0.0, origin_y=0.0):
    """点 (px,py) 是否在某个禁区内。"""
    if not forbidden_zones:
        return False
    for xmin, xmax, ymin, ymax in forbidden_zones:
        if xmin + origin_x <= px <= xmax + origin_x and \
           ymin + origin_y <= py <= ymax + origin_y:
            return True
    return False


def segment_crosses_forbidden(ax, ay, bx, by, forbidden_zones,
                               origin_x=0.0, origin_y=0.0):
    """线段 AB 是否穿过某个禁区（分段采样检测）。"""
    if not forbidden_zones:
        return False
    seg_len = math.hypot(bx - ax, by - ay)
    if seg_len < 0.001:
        return point_in_forbidden(ax, ay, forbidden_zones, origin_x, origin_y)
    n_samples = max(2, int(seg_len / 0.05))
    for i in range(n_samples + 1):
        t = i / n_samples
        if point_in_forbidden(ax + (bx - ax) * t, ay + (by - ay) * t,
                              forbidden_zones, origin_x, origin_y):
            return True
    return False


# ============================================================
#  命令行入口
# ============================================================

if __name__ == "__main__":
    import sys
    import rospy
    from sensor_msgs.msg import LaserScan
    from path_planner import get_car_position

    rospy.init_node("forbidden_zones_tool", anonymous=True)
    rospy.wait_for_message("scan", LaserScan, timeout=5.0)

    print("禁区管理工具")
    print("  s  - 重新设置（覆盖）")
    print("  l  - 列出禁区")
    print("  a  - 追加禁区")
    print("  d  - 删除禁区")
    print("  q  - 退出")

    while True:
        cmd = input("\n>>> ").strip().lower()
        if cmd == 'q':
            break
        elif cmd == 's':
            setup_forbidden_zones()
        elif cmd == 'l':
            list_forbidden_zones()
        elif cmd == 'a':
            add_forbidden_zone()
        elif cmd == 'd':
            delete_forbidden_zone()
        else:
            print("未知命令")
