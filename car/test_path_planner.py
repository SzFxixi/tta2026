#!/usr/bin/env python3
"""
路径规划交互测试脚本

用法（在小车上运行，需 ROS + LiDAR）:
    python3 test_path_planner.py

流程:
    1. 自动获取小车当前位置
    2. 输入终点坐标
    3. 输入期望中间点个数
    4. 调用 path_planner 规划路径
    5. 打印路径点（相邻点仅单轴变化）
"""

import sys
import rospy
from sensor_msgs.msg import LaserScan
from path_planner import plan_path_to, get_car_position


def main():
    rospy.init_node("test_path_planner", anonymous=True)
    rospy.wait_for_message("scan", LaserScan, timeout=5.0)

    # ── 显示当前位置 ──
    car_x, car_y = get_car_position()
    print("=" * 50)
    print(f"  当前位置: X = {car_x:.3f},  Y = {car_y:.3f}")
    print("=" * 50)

    # ── 输入终点 ──
    try:
        ex = float(input("\n请输入终点 X 坐标: "))
        ey = float(input("请输入终点 Y 坐标: "))
    except (ValueError, EOFError):
        print("输入无效，退出。")
        sys.exit(1)

    # ── 输入中间点个数 ──
    try:
        num = int(input("请输入期望中间点个数: "))
    except (ValueError, EOFError):
        num = 0

    # ── 规划 ──
    result = plan_path_to(ex, ey, num_waypoints=num)

    # ── 打印结果 ──
    print(f"\n{'=' * 50}")
    print(f"  规划结果")
    print(f"{'=' * 50}")
    print(f"  路径方案:     {result['path_name']}")
    print(f"  路径得分:     {result.get('score', '-')}  (越低越好)")
    print(f"  距障碍物:     {result['obstacle_margin']}m  {'✓' if result['safe'] else '⚠ 危险!'}")
    print(f"  检测到障碍物: {len(result['obstacles'])} 个")
    for obs in result['obstacles']:
        print(f"    - ({obs['x']:.3f}, {obs['y']:.3f})  距离={obs['distance']:.3f}m")

    wp = result["waypoints"]
    cp = result.get("correction_points", [])
    cp_indices = {p["index"] for p in cp}

    print(f"\n  路径点 ({len(wp)} 个, * = 可校正):")
    print(f"  {'─' * 50}")
    print(f"  {'序号':<6} {'X (m)':<12} {'Y (m)':<12} {'备注'}")
    print(f"  {'─' * 50}")
    for i, (x, y) in enumerate(wp):
        if i == 0:
            note = "起点"
        elif i == len(wp) - 1:
            note = "终点"
        else:
            note = ""
        if i in cp_indices:
            cpi = next(p for p in cp if p["index"] == i)
            safe_str = "✓" if cpi["safe"] else "⚠"
            note = (note + " " if note else "") + f"*可校正 {safe_str}"
        print(f"  {i:<6} {x:<12.3f} {y:<12.3f} {note}")
    print(f"  {'─' * 50}")

    # 校正点汇总
    if cp:
        print(f"\n  校正点 ({len(cp)} 个):")
        for p in cp:
            safe_str = "安全" if p["safe"] else "可能有遮挡"
            print(f"    路径点#{p['index']} ({p['x']:.3f}, {p['y']:.3f}) [{p['type']}] — {safe_str}")


if __name__ == "__main__":
    main()
