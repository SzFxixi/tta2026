#!/usr/bin/env python3
# 基于 LiDAR 前后距离和 + 底盘偏航角进行车身角度纠正

import socket
import time
import rospy
import numpy as np
import math
from sensor_msgs.msg import LaserScan


# 可调参数
DEVIATION_LARGE = 6.0
DEVIATION_SMALL = 4.0
YAW_DRIFT_THRESHOLD = 3.0
CORRECTION_STEP = 8.0
CORRECTION_SLEEP = 0.5
MAX_ITERATIONS = 20
LIDAR_SAMPLES = 5
CHASSIS_SOCKET_TIMEOUT = 3.0
MOVE_TIMEOUT = 10.0
WHEEL_SPEED_THRESHOLD = 10.0


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


class PoseCorrector:
    """位姿校正器：set_baseline → get_deviation → correct_if_needed"""

    def __init__(self, chassis_ip="192.168.42.2", chassis_port=40923):
        self._ip = chassis_ip
        self._port = chassis_port
        self._sock = None
        self.baseline_distance = 0.0
        self.baseline_yaw = 0.0

    def connect(self):
        """建立底盘 TCP 连接，初始化 ROS 节点"""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((self._ip, self._port))
        self._sock.send("command;".encode("utf-8"))
        self._sock.recv(1024)
        self._sock.settimeout(CHASSIS_SOCKET_TIMEOUT)
        try:
            rospy.init_node("pose_corrector", anonymous=True)
        except rospy.ROSException:
            pass
        rospy.wait_for_message("scan", LaserScan, timeout=5.0)

    def disconnect(self):
        """关闭底盘连接"""
        if self._sock:
            try:
                self._sock.shutdown(socket.SHUT_WR)
            except Exception:
                pass
            self._sock.close()
            self._sock = None

    def _send_chassis(self, cmd: str):
        if not cmd.endswith(";"):
            cmd += ";"
        self._sock.send(cmd.encode("utf-8"))
        while True:
            try:
                self._sock.recv(4096)
                break
            except socket.timeout:
                break
        time.sleep(1.0)
        start = time.time()
        while True:
            if time.time() - start > MOVE_TIMEOUT:
                break
            self._sock.send("chassis speed ?;".encode("utf-8"))
            try:
                result = self._sock.recv(1024).decode("utf-8").split(" ")
                _, _, _, a, b, c, d = map(float, result[0:7])
                if a < WHEEL_SPEED_THRESHOLD and b < WHEEL_SPEED_THRESHOLD \
                   and c < WHEEL_SPEED_THRESHOLD and d < WHEEL_SPEED_THRESHOLD:
                    break
            except Exception:
                pass

    def _get_yaw(self) -> float:
        """读取底盘当前偏航角 (度)"""
        self._sock.send("chassis position ?;".encode("utf-8"))
        result = self._sock.recv(1024).decode("utf-8").split(" ")
        _, _, yaw = map(float, result[:3])
        return yaw

    def set_baseline(self):
        self.baseline_distance = getsum()
        self.baseline_yaw = self._get_yaw()

    def _compute_lidar_angle(self) -> float:
        """arccos(baseline/current) 反推车身偏角"""
        s = getsum()
        if s <= 0:
            return 0.0
        t = self.baseline_distance / s
        if t > 1.0:
            t = 1.0  # 浮点误差保护
        angle = math.degrees(math.acos(t))
        return angle

    def get_deviation(self) -> float:
        """检测当前车身相对基准的偏角。返回偏角(度)，基准未设定返回 -1。"""
        if self.baseline_distance == 0.0:
            return -1.0
        lidar_angle = self._compute_lidar_angle()
        current_yaw = self._get_yaw()
        yaw_drift = abs(current_yaw - self.baseline_yaw)
        if yaw_drift > 180:
            yaw_drift = 360 - yaw_drift
        return lidar_angle

    def correct_if_needed(self) -> bool:
        """检测偏角是否超过阈值，超过则逐步校正直到收敛"""
        if self.baseline_distance == 0.0:
            return False

        corrected = False
        for iteration in range(1, MAX_ITERATIONS + 1):
            lidar_angle = self._compute_lidar_angle()
            current_yaw = self._get_yaw()

            if current_yaw > 90:
                current_yaw -= 180
            yaw_drift = abs(current_yaw - self.baseline_yaw)
            if yaw_drift > 180:
                yaw_drift = 360 - yaw_drift

            need_correct_large = lidar_angle > DEVIATION_LARGE
            need_correct_small = (lidar_angle > DEVIATION_SMALL
                                  and yaw_drift > YAW_DRIFT_THRESHOLD)
            if not (need_correct_large or need_correct_small):
                break

            direction = -1.0 if (current_yaw - self.baseline_yaw) > 0 else 1.0
            step = CORRECTION_STEP * direction
            self._send_chassis(f"chassis move z {step}")
            corrected = True
            time.sleep(CORRECTION_SLEEP)

        return corrected
