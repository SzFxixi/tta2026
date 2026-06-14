#!/usr/bin/env python3
# LiDAR 工具函数

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan


LIDAR_SAMPLES = 5


def getsum(samples=LIDAR_SAMPLES):
    """前后激光距离均值之和（多次采样平均）"""
    front_sum = 0.0
    rear_sum = 0.0

    for _ in range(samples):
        data = rospy.wait_for_message("scan", LaserScan)
        n = len(data.ranges)
        middle = n // 2

        front = data.ranges[middle]
        rear = data.ranges[n - 2]

        # 滤除 inf（无回波）
        while front == np.inf or rear == np.inf:
            data = rospy.wait_for_message("scan", LaserScan)
            n = len(data.ranges)
            front = data.ranges[n // 2]
            rear = data.ranges[n - 2]

        front_sum += front
        rear_sum += rear

    return front_sum / samples + rear_sum / samples
