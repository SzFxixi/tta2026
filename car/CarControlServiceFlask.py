#!/usr/bin/env python3
# coding=utf-8
from flask import Flask, request, jsonify
import socket
import sys
import time
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose2D

import signal
import os
import math

def getx():
    data = rospy.wait_for_message("scan", LaserScan)
    number = len(data.ranges)
    middle = number // 2
    lidarx = data.ranges[middle]
    while lidarx == np.inf:
        data = rospy.wait_for_message("scan", LaserScan)
        number = len(data.ranges)
        middle = number // 2
        lidarx = data.ranges[middle]
    return lidarx

def gety():
    data = rospy.wait_for_message("scan", LaserScan)
    number = len(data.ranges)
    qur = number // 4
    lidary = data.ranges[qur]
    while lidary == np.inf:
        data = rospy.wait_for_message("scan", LaserScan)
        number = len(data.ranges)
        qur = number // 4
        lidary = data.ranges[qur]
    return lidary

def getsum():
    lidar_x_average = 0
    lidar_x__average = 0
    for i in range(5):
        data = rospy.wait_for_message("scan", LaserScan)
        number = len(data.ranges)
        middle = number // 2
        lidarx = data.ranges[middle]
        lidarx_ = data.ranges[number-2]
        while lidarx_ == np.inf or lidarx== np.inf:
            data = rospy.wait_for_message("scan", LaserScan)
            number = len(data.ranges)
            middle = number // 2
            lidarx = data.ranges[middle]
            lidarx_ = data.ranges[number-2]
        lidar_x_average +=  lidarx
        lidar_x__average += lidarx_
    lidar_x_average /= 5
    lidar_x__average /= 5
    return lidar_x__average + lidar_x_average

def signal_handler(sig, frame):
    print('收到终止信号，正在关闭资源...')
    try:
        car.Shutdown()
    except Exception as e:
        print(f"关闭小车连接时出错: {e}")

    try:
        rospy.signal_shutdown("程序终止")
    except Exception as e:
        print(f"关闭ROS节点时出错: {e}")

    print('资源已关闭，退出程序')
    os._exit(0)

class CarService:
    channel: socket.socket
    address: tuple

    def __init__(self, ip: str, port: int, x_offset: float, y_offset: float):
        self.channel = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.address = (ip, port)
        self.x_offset = x_offset
        self.y_offset = y_offset

    def StartUp(self):
        self.channel.connect(self.address)
        self.channel.send("command;".encode('utf-8'))
        self.channel.recv(1024).decode('utf-8').split(' ')
        self.channel.settimeout(3)
        rospy.init_node("lidar_data")
        rospy.wait_for_message("scan", LaserScan)
        self.distance = getsum()
        self.initialYaw = self.getYaw()
        print("CarService ready")

    def Shutdown(self):
        self.channel.shutdown(socket.SHUT_WR)
        self.channel.close()

    def Move(self, cmd: str):
        if not cmd.endswith(';'):
            cmd = cmd + ';'
        self.channel.send(cmd.encode('utf-8'))
        while True:
            try:
                self.channel.recv(4096)
                break
            except socket.timeout:
                break
        time.sleep(1)
        start_time = time.time()
        while True:
            if time.time() - start_time > 10:  # 超时保护放在 try 外，recv 失败也能触发
                break
            self.channel.send("chassis speed ?;".encode('utf-8'))
            try:
                result = self.channel.recv(1024).decode('utf-8').split(' ')
                x, y, z, a, b, c, d = map(float, result[0:7])
                if a < 10 and b < 10 and c < 10 and d < 10:
                    break
            except Exception:
                pass

    def getYaw(self):
        self.channel.send("chassis position ?;".encode("utf-8"))
        result = self.channel.recv(1024).decode('utf-8').split(' ')
        _,_,yaw = map(float,result[:3])
        return yaw

    def set_baseline(self):
        """保存当前前后距离和以及偏航角，作为后续纠偏的基准"""
        self.distance = getsum()
        self.initialYaw = self.getYaw()

    def SyncYaw(self):
        for iteration in range(10):
            self.channel.send("chassis position ?;".encode("utf-8"))
            result = self.channel.recv(1024).decode('utf-8').split(' ')
            try:
                _,_,yaw = map(float,result[:3])
                theat = self.return_theat()
                if yaw > 90:
                    yaw = yaw - 180
                if theat > 15:
                    if yaw - self.initialYaw > 0:
                        theat = theat * (-1)
                    step = max(2.5, min(abs(theat), 10.0)) * (theat / abs(theat))
                    self.channel.send(f"chassis move z {step};".encode('utf-8'))
                    time.sleep(0.5)
                    try:
                        self.channel.recv(1024)
                    except Exception:
                        pass
                else:
                    break
            except Exception:
                pass

    def return_theat(self):
        s = getsum()
        t = self.distance / s
        if t > 1:
            t = 1
        return np.arccos(t) * 180 / 3.1415926


if __name__ == "__main__":

    signal.signal(signal.SIGINT, signal_handler)

    app = Flask(__name__)
    car = CarService("192.168.42.2", 40923, 0, 0)
    car.StartUp()
    target_pub = rospy.Publisher("/target", Pose2D, queue_size=5)
    CurrentTaskID = 0

    # ========== 原有端点 ==========
    @app.route('/Circle', methods=['POST'])
    def circle():
        try:
            global CurrentTaskID
            data = request.json
            rad_z = data['rad_z']
            task_id_gotten = data['TaskId']

            if task_id_gotten != CurrentTaskID + 1:
                error_response = {
                    "isSuccess": False,
                    "errorCode": -1,
                    "errorMessage": "Task is not consistent",
                    "expectedTaskId": CurrentTaskID + 1
                }
                return jsonify(error_response), 400

            cmd = f"chassis move x 0 y 0 z {rad_z};"
            car.Move(cmd)

            CurrentTaskID += 1
            success_response = {
                "isSuccess": True,
                "currentTaskId": CurrentTaskID
            }

            return jsonify(success_response)
        except Exception as e:
            error_response = {
                "isSuccess": False,
                "errorCode": -1,
                "errorMessage": str(e),
            }
            return jsonify(error_response), 500

    @app.route('/Move', methods=['POST'])
    def move():
        try:
            global CurrentTaskID
            data = request.json
            location = [data['location_x'], data['location_y']]
            task_id_gotten = data['TaskId']

            if task_id_gotten != CurrentTaskID + 1:
                error_response = {
                    "isSuccess": False,
                    "errorCode": -1,
                    "errorMessage": "Task is not consistent",
                    "expectedTaskId": CurrentTaskID + 1
                }
                return jsonify(error_response), 400

            target_pub.publish(Pose2D(x=float(location[0]), y=float(location[1]), theta=0))
            attempts = 0

            while(abs(float(location[0]) - (getx() - car.x_offset)) > 0.08 and attempts < 5):
                move_x = (getx() - car.x_offset) - float(location[0])
                cmd = f"chassis move x {move_x} y 0 z 0;"
                car.Move(cmd)
                attempts += 1
            attempts = 0
            while(abs(float(location[1]) - (gety() - car.y_offset)) > 0.08 and attempts < 5):
                move_y = ((gety() - car.y_offset) - float(location[1]))
                cmd = f"chassis move x 0 y {move_y} z 0;"
                car.Move(cmd)
                attempts += 1
            CurrentTaskID += 1
            success_response = {
                "isSuccess": True,
                "currentTaskId": CurrentTaskID
            }

            return jsonify(success_response)
        except Exception as e:
            error_response = {
                "isSuccess": False,
                "errorCode": -1,
                "errorMessage": str(e),
            }
            return jsonify(error_response), 500

    @app.route('/MoveRelative', methods=['POST'])
    def moverelative():
        try:
            global CurrentTaskID
            data = request.json
            delta_x = float(data['delta_x'])
            delta_y = float(data['delta_y'])
            step    = float(data.get('step', 0.0))
            task_id_gotten = data['TaskId']

            if task_id_gotten != CurrentTaskID + 1:
                error_response = {
                    "isSuccess": False,
                    "errorCode": -1,
                    "errorMessage": "Task is not consistent",
                    "expectedTaskId": CurrentTaskID + 1
                }
                return jsonify(error_response), 400

            total_dist = math.sqrt(delta_x**2 + delta_y**2)

            if step > 0 and total_dist > 1e-6:
                target_pub.publish(Pose2D(x=delta_x, y=delta_y, theta=0))
                n_steps      = max(1, round(total_dist / step))
                dx_step      = delta_x / n_steps
                dy_step      = delta_y / n_steps
                for _ in range(n_steps):
                    car.Move(f"chassis move x {dx_step:.3f} y {dy_step:.3f} z 0;")
            else:
                cmd = f"chassis move x {delta_x} y {delta_y} z 0;"
                car.Move(cmd)

            CurrentTaskID += 1
            success_response = {
                "isSuccess": True,
                "currentTaskId": CurrentTaskID
            }

            return jsonify(success_response)
        except Exception as e:
            error_response = {
                "isSuccess": False,
                "errorCode": -1,
                "errorMessage": str(e),
            }
            return jsonify(error_response), 500

    @app.route('/MoveOnlyY', methods=['POST'])
    def moveonlyy():
        try:
            global CurrentTaskID
            data = request.json
            location = [data['location_x'], data['location_y']]
            task_id_gotten = data['TaskId']

            if task_id_gotten != CurrentTaskID + 1:
                error_response = {
                    "isSuccess": False,
                    "errorCode": -1,
                    "errorMessage": "Task is not consistent",
                    "expectedTaskId": CurrentTaskID + 1
                }
                return jsonify(error_response), 400

            target_pub.publish(Pose2D(x=float(location[0]), y=float(location[1]), theta=0))
            move_x = (getx() - car.x_offset) - float(location[0])
            cmd = f"chassis move x {move_x} y 0 z 0;"
            car.Move(cmd)
            attempts = 0
            while(abs(float(location[1]) - (gety() - car.y_offset)) > 0.08 and attempts < 5):
                move_y = ((gety() - car.y_offset) - float(location[1]))
                cmd = f"chassis move x 0 y {move_y} z 0;"
                car.Move(cmd)
                attempts += 1
            CurrentTaskID += 1
            success_response = {
                "isSuccess": True,
                "currentTaskId": CurrentTaskID
            }

            return jsonify(success_response)
        except Exception as e:
            error_response = {
                "isSuccess": False,
                "errorCode": -1,
                "errorMessage": str(e),
            }
            return jsonify(error_response), 500

    @app.route('/MoveOnlyX', methods=['POST'])
    def moveonlyx():
        try:
            global CurrentTaskID
            data = request.json
            location = [data['location_x'], data['location_y']]
            task_id_gotten = data['TaskId']

            if task_id_gotten != CurrentTaskID + 1:
                error_response = {
                    "isSuccess": False,
                    "errorCode": -1,
                    "errorMessage": "Task is not consistent",
                    "expectedTaskId": CurrentTaskID + 1
                }
                return jsonify(error_response), 400

            target_pub.publish(Pose2D(x=float(location[0]), y=float(location[1]), theta=0))
            move_x = ((getx() - car.x_offset) - float(location[0]))
            cmd = f"chassis move x {move_x} y 0 z 0;"
            car.Move(cmd)
            attempts = 0
            while(abs(float(location[0]) - (getx() - car.x_offset)) > 0.08 and attempts < 5):
                move_x = (getx() - car.x_offset) - float(location[0])
                cmd = f"chassis move x {move_x} y 0 z 0;"
                car.Move(cmd)
                attempts += 1
            CurrentTaskID += 1
            success_response = {
                "isSuccess": True,
                "currentTaskId": CurrentTaskID
            }

            return jsonify(success_response)
        except Exception as e:
            error_response = {
                "isSuccess": False,
                "errorCode": -1,
                "errorMessage": str(e),
            }
            return jsonify(error_response), 500

    @app.route('/MoveLongDistance', methods=['POST'])
    def move_long_distance():
        try:
            global CurrentTaskID
            data = request.json
            location = [data['location_x'], data['location_y']]
            task_id_gotten = data['TaskId']

            if task_id_gotten != CurrentTaskID + 1:
                error_response = {
                    "isSuccess": False,
                    "errorCode": -1,
                    "errorMessage": "Task is not consistent",
                    "expectedTaskId": CurrentTaskID + 1
                }
                return jsonify(error_response), 400
            target_pub.publish(Pose2D(x=float(location[0]), y=float(location[1]), theta=0))
            move_x = ( (getx() - car.x_offset) - float(location[0])) / 3
            for i in range(3):
                cmd = f"chassis move x {move_x} y 0 z 0;"
                car.Move(cmd)

            attempts = 0
            while(abs(float(location[0]) - (getx() - car.x_offset)) > 0.00001 and attempts < 5):
                move_x = (getx() - car.x_offset) - float(location[0])
                cmd = f"chassis move x {move_x} y 0 z 0;"
                car.Move(cmd)
                attempts += 1
            attempts = 0
            while(abs(float(location[1]) - (gety() - car.y_offset)) > 0.00001 and attempts < 5):
                move_y = (gety() - car.y_offset) - float(location[1])
                cmd = f"chassis move x 0 y {move_y} z 0;"
                car.Move(cmd)
                attempts += 1

            CurrentTaskID += 1
            success_response = {
                "isSuccess": True,
                "currentTaskId": CurrentTaskID
            }
            return jsonify(success_response)
        except Exception as e:
            error_response = {
                "isSuccess": False,
                "errorCode": -1,
                "errorMessage": str(e),
            }
            return jsonify(error_response), 500

    # ========== 新增端点：姿态纠正与基准设定 ==========
    @app.route('/SyncYaw', methods=['POST'])
    def sync_yaw():
        """基于雷达前后距离和纠正车头方向"""
        try:
            global CurrentTaskID
            data = request.json
            task_id_gotten = data['TaskId']

            if task_id_gotten != CurrentTaskID + 1:
                error_response = {
                    "isSuccess": False,
                    "errorCode": -1,
                    "errorMessage": "TaskId mismatch",
                    "expectedTaskId": CurrentTaskID + 1
                }
                return jsonify(error_response), 400

            car.SyncYaw()
            CurrentTaskID += 1
            return jsonify({"isSuccess": True, "currentTaskId": CurrentTaskID})
        except Exception as e:
            return jsonify({"isSuccess": False, "errorCode": -1, "errorMessage": str(e)}), 500

    @app.route('/SetBaseline', methods=['POST'])
    def set_baseline():
        """重新记录当前前后距离和与偏航角作为纠偏基准"""
        try:
            global CurrentTaskID
            data = request.json
            task_id_gotten = data['TaskId']

            if task_id_gotten != CurrentTaskID + 1:
                error_response = {
                    "isSuccess": False,
                    "errorCode": -1,
                    "errorMessage": "TaskId mismatch",
                    "expectedTaskId": CurrentTaskID + 1
                }
                return jsonify(error_response), 400

            car.set_baseline()
            CurrentTaskID += 1
            return jsonify({"isSuccess": True, "currentTaskId": CurrentTaskID})
        except Exception as e:
            return jsonify({"isSuccess": False, "errorCode": -1, "errorMessage": str(e)}), 500

    @app.route('/ShutDown', methods=['POST'])
    def shut_down():
        try:
            global CurrentTaskID
            data = request.json

            success_response = {
                "isSuccess": True,
                "currentTaskId": CurrentTaskID
            }
            car.Shutdown()
            CurrentTaskID = 0
            return jsonify(success_response)
        except Exception as e:
            error_response = {
                "isSuccess": False,
                "errorCode": -1,
                "errorMessage": str(e),
            }
            return jsonify(error_response), 500

    @app.route('/Reset', methods=['POST'])
    def reset():
        try:
            global CurrentTaskID

            success_response = {
                "isSuccess": True,
                "currentTaskId": CurrentTaskID
            }

            CurrentTaskID = 0
            return jsonify(success_response)
        except Exception as e:
            error_response = {
                "isSuccess": False,
                "errorCode": -1,
                "errorMessage": str(e),
            }
            return jsonify(error_response), 500

    try:
        app.run(host='0.0.0.0', port=5000)
    finally:
        car.Shutdown()
        rospy.signal_shutdown("程序终止")