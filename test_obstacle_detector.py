#!/usr/bin/env python3
"""
障碍物检测独立测试脚本

用法（在小车上运行）:
    python3 test_obstacle_detector.py <x1> <y1> <x2> <y2>

示例 — 安全区矩形对角两点:
    python3 test_obstacle_detector.py 0.5 0.5 3.0 6.0

不传参数时使用默认安全区。
"""

import sys
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from obstacle_detector import detect_obstacles

# ── 小车定位（复用主服务的逻辑） ──

def getx():
    """前方激光距离 → X 坐标"""
    data = rospy.wait_for_message("scan", LaserScan)
    number = len(data.ranges)
    middle = number // 2
    d = data.ranges[middle]
    while d == np.inf:
        data = rospy.wait_for_message("scan", LaserScan)
        d = data.ranges[len(data.ranges) // 2]
    return d


def gety():
    """侧方激光距离 → Y 坐标"""
    data = rospy.wait_for_message("scan", LaserScan)
    number = len(data.ranges)
    qur = number // 4
    d = data.ranges[qur]
    while d == np.inf:
        data = rospy.wait_for_message("scan", LaserScan)
        d = data.ranges[len(data.ranges) // 4]
    return d


# ── 主测试 ──

def main():
    rospy.init_node("test_obstacle_detector", anonymous=True)
    print("等待 LiDAR 数据...")
    rospy.wait_for_message("scan", LaserScan, timeout=10.0)

    # 获取小车当前位置（减去偏移量）
    car_x = getx()
    car_y = gety()

    # 安全区参数
    if len(sys.argv) >= 5:
        x1, y1 = float(sys.argv[1]), float(sys.argv[2])
        x2, y2 = float(sys.argv[3]), float(sys.argv[4])
    else:
        # 默认安全区：X [0.5, 3.6]  Y [0.5, 7.8]
        x1, y1 = 0.5, 0.5
        x2, y2 = 3.6, 7.8

    safe_zone = ((x1, y1), (x2, y2))

    print("=" * 60)
    print(f"小车位置: X={car_x:.3f}  Y={car_y:.3f}")
    print(f"安全区:    ({x1:.1f}, {y1:.1f}) → ({x2:.1f}, {y2:.1f})")
    print("=" * 60)

    obstacles, diag = detect_obstacles(
        safe_zone_p1=safe_zone[0],
        safe_zone_p2=safe_zone[1],
        car_x=car_x,
        car_y=car_y,
    )

    # ── 打印诊断 ──
    print(f"\n雷达诊断:")
    print(f"  总光束: {diag['total_beams']}")
    print(f"  有效光束: {diag['valid_beams']}")
    print(f"  安全区内光束: {diag['inside_zone_beams']}")
    print(f"  突变点数: {diag['jump_count']}")
    print(f"  距离分布: min={diag['dist_min']}  "
          f"p50={diag['dist_p50']}  max={diag['dist_max']}")

    # ── 打印障碍物 ──
    print(f"\n检测到 {len(obstacles)} 个障碍物:")
    if obstacles:
        for i, obs in enumerate(obstacles, 1):
            print(f"  #{i}  坐标: ({obs['x']:.3f}, {obs['y']:.3f})  "
                  f"距离: {obs['distance']:.3f}m  "
                  f"角度: {obs['angle_deg']:.1f}°  "
                  f"光束数: {obs['beam_count']}")
    else:
        print("  （无）")


if __name__ == "__main__":
    main()
