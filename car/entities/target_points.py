#!/usr/bin/env python3
"""
目标点位坐标管理 — 交互式设置 6 个目标点位，每点存储两套坐标：

  规划坐标: 统一使用前光束(X) + 右侧光束(Y) 读取，用于路径规划
  校正坐标: 根据点位位置使用不同光束组合，存原始光束距离（不做房间尺寸减法），
            仅点位 1~4 存储，用于到位后的精校

点位校正光束:
  1: 前光束 + 右侧光束  (同规划)
  2: 后光束 + 右侧光束
  3: 后光束 + 左侧光束
  4: 前光束 + 左侧光束
  5~6: 无校正坐标 (仅规划)

配置文件 target_points.json 格式:
    {
        "points": [
            {"plan": [x1, y1], "correct": [front, right]},
            {"plan": [x2, y2], "correct": [rear, right]},
            ...
        ]
    }

用法:
    python3 target_points.py                # 交互式命令行工具
    from entities.target_points import load_targets  # 在其他模块中加载
"""

import json
import os
import numpy as np
import rospy
from sensor_msgs.msg import LaserScan

_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "target_points.json")
NUM_TARGETS = 6


# ============================================================
#  LiDAR 光束读取
# ============================================================

def _get_beam(index):
    """读取指定索引的光束距离，滤除 inf"""
    data = rospy.wait_for_message("scan", LaserScan)
    n = len(data.ranges)
    d = data.ranges[index % n]
    while d == np.inf:
        data = rospy.wait_for_message("scan", LaserScan)
        n = len(data.ranges)
        d = data.ranges[index % n]
    return d


def _get_n():
    data = rospy.wait_for_message("scan", LaserScan)
    return len(data.ranges)


# ── 规划坐标：统一前X + 右Y ──

def _read_plan():
    """规划坐标 — 始终用前光束和右侧光束"""
    n = _get_n()
    x = _get_beam(n // 2)       # 前
    y = _get_beam(n // 4)       # 右
    return round(x, 3), round(y, 3)


# ── 校正坐标：按点位读取原始光束距离 ──

# 每个点位的 (X光束索引, Y光束索引, 描述)
_CORRECT_BEAMS = {
    1: ("n//2",   "n//4",     "前 + 右"),
    2: ("n-2",    "n//4",     "后 + 右"),
    3: ("n-2",    "n*3//4",   "后 + 左"),
    4: ("n//2",   "n*3//4",   "前 + 左"),
}


def _read_correct(point_id):
    """校正坐标 — 按点位读取原始光束距离，不做减法"""
    n = _get_n()
    x_expr, y_expr, desc = _CORRECT_BEAMS[point_id]
    x_idx = eval(x_expr, {"n": n})
    y_idx = eval(y_expr, {"n": n})
    x = _get_beam(x_idx)
    y = _get_beam(y_idx)
    return round(x, 3), round(y, 3)


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
    """读取目标点配置。返回 point dicts 列表，无文件时返回 None。"""
    config = _read_config()
    if config is None:
        return None
    return config.get("points", [])


def list_targets():
    """打印已保存的目标点坐标。"""
    config = _read_config()
    if config is None or not config.get("points"):
        print("暂无目标点配置")
        return 0
    for i, p in enumerate(config["points"], 1):
        px, py = p["plan"]
        if "correct" in p:
            cx, cy = p["correct"]
            print(f"  目标点 {i}: 规划({px:.3f},{py:.3f})  校正({cx:.3f},{cy:.3f})")
        else:
            print(f"  目标点 {i}: 规划({px:.3f},{py:.3f})  校正(-)")
    return len(config["points"])


# ============================================================
#  交互式设置
# ============================================================

def setup_targets():
    """交互式设置 {NUM_TARGETS} 个目标点坐标（覆盖已有配置）。"""
    points = []
    print(f"\n目标点坐标设置（共 {NUM_TARGETS} 个）")
    print("每点记录两套坐标：规划坐标 + 校正坐标（点1~4）\n")

    for i in range(1, NUM_TARGETS + 1):
        has_correct = i in _CORRECT_BEAMS
        beam_desc = _CORRECT_BEAMS[i][2] if has_correct else "无"

        print(f"目标点 {i}/{NUM_TARGETS}  [校正光束: {beam_desc}]")
        input(f"  移动到位后按 Enter >>> ")

        plan_x, plan_y = _read_plan()

        if has_correct:
            corr_x, corr_y = _read_correct(i)
            points.append({
                "plan": (plan_x, plan_y),
                "correct": (corr_x, corr_y),
            })
            print(f"  ✓ 规划({plan_x:.3f},{plan_y:.3f})  校正({corr_x:.3f},{corr_y:.3f})\n")
        else:
            points.append({"plan": (plan_x, plan_y)})
            print(f"  ✓ 规划({plan_x:.3f},{plan_y:.3f})\n")

    config = {"points": points}
    _write_config(config)
    print(f"{NUM_TARGETS} 个目标点已保存到 {_CONFIG_FILE}\n")
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
