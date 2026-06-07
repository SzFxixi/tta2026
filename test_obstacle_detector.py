#!/usr/bin/env python3
"""
障碍物检测独立测试脚本

用法（在小车上运行）:
    python3 test_obstacle_detector.py
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

    car_x = getx()
    car_y = gety()

    print("=" * 60)
    print(f"小车位置: X={car_x:.3f}  Y={car_y:.3f}")
    print("=" * 60)

    obstacles, diag = detect_obstacles(car_x=car_x, car_y=car_y)

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
