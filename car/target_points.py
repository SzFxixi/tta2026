#!/usr/bin/env python3
"""
目标点位坐标管理 — 交互式设置 7 个目标点位的坐标，保存到 target_points.json。
参考 forbidden_zones.py 的设计。

用法:
    python3 target_points.py                # 交互式命令行工具
    from target_points import load_targets  # 在其他模块中加载
"""

import json
import os
import rospy
from sensor_msgs.msg import LaserScan

_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "target_points.json")
NUM_TARGETS = 7


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

def load_targets():
    """读取目标点配置。返回 [(x,y), ...] 列表，无文件时返回 None。"""
    config = _read_config()
    if config is None:
        return None
    return [tuple(p) for p in config.get("points", [])]


def list_targets():
    """打印已保存的目标点坐标。"""
    config = _read_config()
    if config is None or not config.get("points"):
        print("暂无目标点配置")
        return 0
    for i, (x, y) in enumerate(config["points"], 1):
        print(f"  目标点 {i}: ({x:.3f}, {y:.3f})")
    return len(config["points"])


# ============================================================
#  LiDAR 定位
# ============================================================

def _get_car_position():
    from path_planner import get_car_position
    return get_car_position()


# ============================================================
#  交互式设置
# ============================================================

def setup_targets():
    """交互式设置 {NUM_TARGETS} 个目标点坐标（覆盖已有配置）。"""
    points = []
    print(f"\n目标点坐标设置（共 {NUM_TARGETS} 个）")
    print("移动小车到目标位置后按 Enter 记录")

    for i in range(1, NUM_TARGETS + 1):
        input(f"目标点 {i}/{NUM_TARGETS} >>> ")
        x, y = _get_car_position()
        points.append((round(x, 3), round(y, 3)))
        print(f"  ✓ 目标点 {i}: ({x:.3f}, {y:.3f})")

    config = {"points": points}
    _write_config(config)
    print(f"\n{NUM_TARGETS} 个目标点已保存到 {_CONFIG_FILE}\n")
    return points


# ============================================================
#  命令行入口
# ============================================================

if __name__ == "__main__":
    rospy.init_node("target_points_tool", anonymous=True)
    rospy.wait_for_message("scan", LaserScan, timeout=5.0)

    print("目标点管理工具")
    print("  s  - 重新设置（覆盖）")
    print("  l  - 列出目标点")
    print("  q  - 退出")

    while True:
        cmd = input("\n>>> ").strip().lower()
        if cmd == 'q':
            break
        elif cmd == 's':
            setup_targets()
        elif cmd == 'l':
            list_targets()
        else:
            print("未知命令")
