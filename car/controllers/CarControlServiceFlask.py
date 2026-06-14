#!/usr/bin/env python3
# coding=utf-8
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flask import Flask, request, jsonify
import socket
import time
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose2D

import signal
import os
import math

from utils.config_loader import cfg
from utils.wall_positioning import fit_walls

# 房间尺寸
_ROOM_W = cfg.room.x_max - cfg.room.x_min
_ROOM_H = cfg.room.y_max - cfg.room.y_min

_walls_cache = None
_walls_cache_time = 0
_CACHE_TTL = 0.05  # 50ms 内复用

def _read_walls(walls=None):
    """读一次 LiDAR，拟合墙壁，50ms 内重复调用走缓存。"""
    global _walls_cache, _walls_cache_time
    now = time.time()
    if _walls_cache is not None and (now - _walls_cache_time) < _CACHE_TTL:
        return _walls_cache
    data = rospy.wait_for_message("scan", LaserScan)
    _walls_cache = fit_walls(data, walls=walls)
    _walls_cache_time = now
    return _walls_cache

def getx():
    """前墙垂直距离 (m)。"""
    return _read_walls(walls=["前"]).get("前墙")

def gety():
    """右墙垂直距离 (m)。"""
    return _read_walls(walls=["右"]).get("右墙")

def get_x():
    """后墙垂直距离 (m)。"""
    return _read_walls(walls=["后"]).get("后墙")

def get_y():
    """左墙垂直距离 (m)。"""
    return _read_walls(walls=["左"]).get("左墙")

def getsum():
    """前后墙距离之和 (m)。"""
    w = _read_walls(walls=["前", "后"])
    f, r = w.get("前墙"), w.get("后墙")
    return (f + r) if (f is not None and r is not None) else None

def getsum_y():
    """左右墙距离之和 (m)。"""
    w = _read_walls(walls=["右", "左"])
    r, l = w.get("右墙"), w.get("左墙")
    return (r + l) if (r is not None and l is not None) else None

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
        """已废弃 — 墙壁建模不需要基准。保留兼容旧 API。"""
        pass

    def SyncYaw(self):
        """基于墙壁法向量的偏航角校正。"""
        for iteration in range(cfg.server.sync_yaw_max_iterations):
            w = _read_walls()
            yaw = w.get("yaw")
            if yaw is None:
                break
            if abs(yaw) < cfg.server.sync_yaw_threshold_deg:
                break
            # 旋转方向：yaw 为正表示车头偏左，需要右转（负 z）
            step = max(cfg.server.sync_yaw_step_min_deg,
                       min(abs(yaw), cfg.server.sync_yaw_step_max_deg))
            step = step * (-1 if yaw > 0 else 1)
            self.channel.send(f"chassis move z {step};".encode('utf-8'))
            try:
                self.channel.recv(1024)
            except Exception:
                pass
            time.sleep(0.5)

    def return_theat(self):
        """返回当前偏航角（度），基于墙壁法向量。"""
        return _read_walls().get("yaw")

    def readings_sane(self):
        """检查墙壁拟合是否合理：前+后≈房间宽，右+左≈房间高。"""
        w = _read_walls()
        f, r = w.get("前墙"), w.get("后墙")
        ri, l = w.get("右墙"), w.get("左墙")
        tol = cfg.client.sanity_check_tolerance
        x_ok = (f is not None and r is not None and abs(f + r - _ROOM_W) < tol)
        y_ok = (ri is not None and l is not None and abs(ri + l - _ROOM_H) < tol)
        return x_ok and y_ok


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

            gx = getx()
            if gx is None:
                print("[MoveOnlyY] LiDAR X 无回波，仅发开环指令")
                cmd = f"chassis move x 0 y 0 z 0;"
                car.Move(cmd)
            else:
                move_x = (gx - car.x_offset) - float(location[0])
                cmd = f"chassis move x {move_x} y 0 z 0;"
                car.Move(cmd)

            do_correct = data.get('correct', True)
            if do_correct and car.readings_sane():
                attempts = 0
                while(abs(float(location[1]) - (gety() - car.y_offset)) > 0.08 and attempts < 5):
                    gy = gety()
                    if gy is None:
                        break
                    move_y = (gy - car.y_offset) - float(location[1])
                    cmd = f"chassis move x 0 y {move_y} z 0;"
                    car.Move(cmd)
                    attempts += 1
            elif do_correct and not car.readings_sane():
                print("[MoveOnlyY] LiDAR 读数异常，跳过闭环校正")
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

            gx = getx()
            if gx is None:
                print("[MoveOnlyX] LiDAR X 无回波，仅发开环指令")
                cmd = f"chassis move x 0 y 0 z 0;"
                car.Move(cmd)
            else:
                move_x = (gx - car.x_offset) - float(location[0])
                cmd = f"chassis move x {move_x} y 0 z 0;"
                car.Move(cmd)

            do_correct = data.get('correct', True)
            if do_correct and car.readings_sane():
                attempts = 0
                while(abs(float(location[0]) - (getx() - car.x_offset)) > 0.08 and attempts < 5):
                    gx2 = getx()
                    if gx2 is None:
                        break
                    move_x = (gx2 - car.x_offset) - float(location[0])
                    cmd = f"chassis move x {move_x} y 0 z 0;"
                    car.Move(cmd)
                    attempts += 1
            elif do_correct and not car.readings_sane():
                print("[MoveOnlyX] LiDAR 读数异常，跳过闭环校正")
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