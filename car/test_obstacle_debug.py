#!/usr/bin/env python3
"""障碍物检测调试脚本 — 打印小车位置和检测到的障碍物，验证坐标是否正确。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rospy
from sensor_msgs.msg import LaserScan
from utils.config_loader import cfg
from utils.wall_positioning import fit_walls
from entities.obstacle_detector import detect_obstacles
from entities.forbidden_zones import load_forbidden_zones

rospy.init_node("obs_debug", anonymous=True)
rospy.wait_for_message("scan", LaserScan, timeout=5.0)

zones = load_forbidden_zones()
print(f"禁区数量: {len(zones) if zones else 0}")

while True:
    data = rospy.wait_for_message("scan", LaserScan, timeout=5.0)
    walls = fit_walls(data)
    car_x = walls.get("前墙")
    car_y = walls.get("右墙")
    car_rear = walls.get("后墙")
    car_left = walls.get("左墙")
    yaw = walls.get("yaw")

    print(f"\n{'='*50}")
    print(f"小车定位:")
    print(f"  前墙(X)={car_x:.3f}m  右墙(Y)={car_y:.3f}m")
    print(f"  后墙={car_rear:.3f}m  左墙={car_left:.3f}m")
    print(f"  偏角={yaw:.1f}°" if yaw else "  偏角=无")
    print(f"  房间尺寸验证: 前+后={car_x+car_rear:.3f}m (应≈{cfg.room.x_max-cfg.room.x_min:.1f})  右+左={car_y+car_left:.3f}m (应≈{cfg.room.y_max-cfg.room.y_min:.1f})")

    obstacles, diag = detect_obstacles(car_x, car_y, forbidden_zones=zones)
    print(f"\n障碍物 ({len(obstacles)}个):")
    if obstacles:
        for i, o in enumerate(obstacles, 1):
            print(f"  #{i}: X={o['x']:.3f}  Y={o['y']:.3f}  距离={o['distance']:.3f}m  角度={o['angle_deg']:.1f}°  光束数={o['beam_count']}")
    else:
        print("  无")
    print(f"诊断: 总光束={diag.get('total_beams')}  有效={diag.get('valid_beams')}  跳变={diag.get('jump_count')}  过滤={diag.get('filtered_out')}  禁区过滤={diag.get('filtered_forbidden')}")

    input("\n按 Enter 刷新, Q 退出 > ")
