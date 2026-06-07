#!/usr/bin/env python3
"""
位姿校正模块 — 基于 LiDAR 前后距离和 + 底盘偏航角进行车身角度纠正。

解耦自 CarControlServiceFlask.py，可独立测试：
    python3 pose_correction.py

三大功能：
    1. set_baseline()    — 在当前位置设定基准（记录前后墙距离和 + 底盘偏航角）
    2. get_deviation()   — 检测当前车身相对基准的偏角
    3. correct_if_needed() — 分析偏角是否超过阈值，超过则执行校正
"""

import socket
import time
import rospy
import numpy as np
import math
from sensor_msgs.msg import LaserScan


# ============================================================
#  可调参数
# ============================================================

# ── 校正触发阈值 ──
DEVIATION_LARGE = 6.0       # LiDAR 偏角超过此值 → 无条件校正 (度)
DEVIATION_SMALL = 4.0       # LiDAR 偏角超过此值，且底盘 yaw 也确认漂移 → 校正 (度)
YAW_DRIFT_THRESHOLD = 3.0   # 底盘 yaw 漂移量超过此值，配合 SMALL 阈值确认 (度)

# ── 校正执行 ──
CORRECTION_STEP = 8.0       # 每次校正旋转角度 (度)
CORRECTION_SLEEP = 0.5      # 每次校正后等待 (秒)
MAX_ITERATIONS = 20         # 单次校正最多循环次数，防止死循环

# ── LiDAR 采样 ──
LIDAR_SAMPLES = 5           # getsum() 对前后距离各取多少帧做平均

# ── 底盘通信 ──
CHASSIS_SOCKET_TIMEOUT = 3.0       # socket recv 超时 (秒)
MOVE_TIMEOUT = 10.0                # 等待轮子停转的超时 (秒)
WHEEL_SPEED_THRESHOLD = 10.0       # 四轮速度全低于此值视为已停转


# ============================================================
#  LiDAR 工具函数
# ============================================================

def getsum(samples=LIDAR_SAMPLES):
    """
    取前后激光距离的均值之和（多次采样取平均，抑制噪声）。

    前激光 = 扫描正前方 (middle 光束)
    后激光 = 扫描正后方 (倒数第 2 光束)

    返回:
        float: 前墙距离 + 后墙距离
    """
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


# ============================================================
#  PoseCorrector 类
# ============================================================

class PoseCorrector:
    """
    位姿校正器。

    用法:
        pc = PoseCorrector("192.168.42.2", 40923)
        pc.connect()
        pc.set_baseline()           # 1. 设定基准

        # ... 小车移动、旋转 ...

        dev = pc.get_deviation()    # 2. 检测偏角
        if dev is not None:
            print(f"当前偏角: {dev:.1f}°")

        ok = pc.correct_if_needed() # 3. 超过阈值则校正
        pc.disconnect()
    """

    def __init__(self, chassis_ip="192.168.42.2", chassis_port=40923):
        self._ip = chassis_ip
        self._port = chassis_port
        self._sock = None

        # 基准值（由 set_baseline() 设定）
        self.baseline_distance = 0.0   # 基准前后距离和
        self.baseline_yaw = 0.0        # 基准底盘偏航角 (度)

    # ── 连接 / 断开 ──

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

    # ── 底盘通信底层 ──

    def _send_chassis(self, cmd: str):
        """发送底盘指令并等待运动完成"""
        if not cmd.endswith(";"):
            cmd += ";"
        self._sock.send(cmd.encode("utf-8"))

        # 等待 ACK
        while True:
            try:
                self._sock.recv(4096)
                break
            except socket.timeout:
                break

        time.sleep(1.0)

        # 轮询轮速直到停转或超时
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

    # ════════════════════════════════════════════════════════
    #  功能一：设定基准
    # ════════════════════════════════════════════════════════

    def set_baseline(self):
        """记录基准：前后墙距离和 + 底盘偏航角"""
        self.baseline_distance = getsum()
        self.baseline_yaw = self._get_yaw()

    # ════════════════════════════════════════════════════════
    #  功能二：检测偏角
    # ════════════════════════════════════════════════════════

    def _compute_lidar_angle(self) -> float:
        """
        用 LiDAR 前后距离和反推车身相对墙面的偏角。

        原理:
            baseline = 车正对墙时的前后距离和
            current  = 车倾斜时的前后距离和
            t = baseline / current
            angle = arccos(t)    ← 车身与墙面法线的夹角

        返回:
            float: 偏角 (度)，恒为非负值
        """
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

    # ════════════════════════════════════════════════════════
    #  功能三：分析与校正
    # ════════════════════════════════════════════════════════

    def correct_if_needed(self) -> bool:
        """检测偏角是否超过阈值，超过则逐步校正直到收敛。返回 True=执行了校正。"""
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
