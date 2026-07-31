#!/usr/bin/env python3
# coding=utf-8
"""
PC端总控 — 协调无人机与小车完成比赛全流程
===========================================
运行方式:
    python MasterController.py --config configs/rescue_config.json --car-url http://192.168.43.8:6001

流程:
    1. 无人机起飞 → 巡检4点 → 装货区降落
    2. 小车同步出发取货 → 装货区汇合
    3. 装货：小车机械臂放置物资到无人机平台
    4. 无人机飞目标救援点 → 等小车 → 降落
    5. 小车放置物资
    6. 双方返航
"""

import sys
import os
import time
import json
import threading
import requests
import argparse

# Add drone directory to path
DRONE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DRONE_DIR)

from Controllers.DroneNavigator import DroneNavigator
from Entities.RescuePointManager import RescuePointManager
from Entities.Waypoint import Waypoint
from Utils.JsonHelper import JsonHelper


class CarHttpClient:
    """小车 HTTP 客户端：调用小车端的阶段接口"""
    
    def __init__(self, base_url):
        self.base_url = base_url.rstrip('/')
        self.sess = requests.Session()
    
    def _post(self, endpoint, data=None, timeout=None):
        url = f"{self.base_url}{endpoint}"
        try:
            r = self.sess.post(url, json=data or {}, timeout=timeout or 300)
            return r
        except Exception as e:
            print(f"[CarClient] POST {endpoint} 失败: {e}")
            return None
    
    def _get(self, endpoint, timeout=5):
        url = f"{self.base_url}{endpoint}"
        try:
            r = self.sess.get(url, timeout=timeout)
            return r
        except Exception as e:
            print(f"[CarClient] GET {endpoint} 失败: {e}")
            return None
    
    def is_busy(self):
        r = self._get('/status')
        if r and r.status_code == 200:
            return r.json().get('busy', False)
        # 网络异常时返回 None（区别于 False），避免误判小车就绪
        return None
    
    def wait_idle(self, timeout=None):
        """等待小车空闲。timeout 为 None 时无限等待。"""
        print("[Master] 等待小车就绪...")
        t0 = time.time()
        while True:
            busy = self.is_busy()
            if busy is False:
                print("[Master] 小车就绪")
                return True
            if timeout and (time.time() - t0) > timeout:
                print("[Master] 等待小车超时")
                return False
            if busy is None:
                print("[Master] 无法连接小车，3秒后重试...")
                time.sleep(3)
            else:
                time.sleep(3)
    
    def _send_phase(self, endpoint, data=None, label=""):
        """发送 phase 指令到小车。车端异步执行，202/200 即成功。
        409=正忙则等2秒重试，网络错误有限重试。"""
        net_fails = 0
        while net_fails < 10:
            r = self._post(endpoint, data=data, timeout=5)
            if r is not None:
                code = r.status_code
                if code in (200, 202):
                    print(f"[Master] 触发小车: {label} (HTTP {code})")
                    return True
                if code == 409:
                    phase = r.json().get('phase', 'unknown')
                    print(f"[Master] 小车正忙({phase})，2秒后重试...")
                    time.sleep(2)
                    net_fails = 0
                    continue
            net_fails += 1
            print(f"[Master] 触发{label}网络异常({net_fails}/10)，3秒后重试...")
            time.sleep(3)
        print(f"[Master] 触发{label}失败，放弃")
        return False

    def start_pickup(self):
        return self._send_phase('/phase/pickup', label="取货")

    def start_loading(self):
        return self._send_phase('/phase/loading', label="装货")

    def start_rescue(self, rescue_point):
        return self._send_phase('/phase/rescue', data={"rescue_point": rescue_point}, label=f"救援(点{rescue_point})")

    def start_return(self):
        return self._send_phase('/phase/return', label="返航")


class MasterController:
    """PC端总控，协调无人机和小车全流程"""
    
    def __init__(self, config, car_url):
        self.config = config
        self.drone = DroneNavigator(config)
        self.car = CarHttpClient(car_url)
        
        # 救援点管理
        rescue_points = config.get("rescue_points", [])
        self.rescue_points = RescuePointManager(rescue_points)
        
        # 输出配置
        output = config.get("output", {})
        self.output_folder = os.path.abspath(output.get("folder", "output"))
        self.csv_filename = output.get("csv_filename", "rescue_levels.csv")
        os.makedirs(self.output_folder, exist_ok=True)
        
        # home 配置
        hp = config.get("home_point", {})
        self.home_point = (float(hp.get("x", 0.7)), float(hp.get("y", 1.3)),
                           float(hp.get("z", 1.5)))
        self.home_rotate_yaw = float(hp.get("rotate_yaw", 0.0))
        
        self.landing_offset = float(config.get("landing_offset", 0.04))
        self.preview_h = float(config.get("landing_preview_height", 0.8))
    
    # ---- 无人机辅助方法（复用 RescueController 的逻辑） ----
    
    def _save_h_result(self, frame, det, prefix: str):
        """保存 H 识别结果（原始图像 + YOLO 检测 JSON）到 output。"""
        import json
        import cv2
        img_path = os.path.join(self.output_folder, f'{prefix}_h.jpg')
        cv2.imwrite(img_path, frame)
        if det.get('h_candidate'):
            result = {
                'h_detected': True,
                'confidence': det['h_candidate']['confidence'],
                'box': det['h_candidate']['box'],
            }
            json_path = os.path.join(self.output_folder, f'{prefix}_h.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"[Master] H识别结果已保存: {json_path}")

    def _servo_h_loop(self, rotation=0.0, max_search=3, max_servo=None, servo_tolerance=None, save_prefix=None):
        """云台朝下搜索 H 并迭代伺服居中。
        找不到 H 时自动拔高 0.2m 重试，再降回原高度。"""
        if servo_tolerance is None:
            servo_tolerance = (self.drone.servo_tolerance_scan
                               if save_prefix and 'home' in str(save_prefix)
                               else self.drone.servo_tolerance_land)
        if max_servo is None:
            max_servo = self.drone.servo_iters
        # Home 点伺服加速
        _speed = float(self.config.get('home_servo_speed', 0.6)) if save_prefix and 'home' in str(save_prefix) else 1.0
        _orig_stl = self.drone.servo_settle
        _orig_rd = self.drone.servo_read
        _orig_slp = self.drone.servo_sleep
        _stl = _orig_stl * _speed
        _rd = _orig_rd * _speed
        _slp = _orig_slp * _speed
        settle_extra = self.drone.servo_settle_extra

        def _try_servo():
            for _ in range(max_search):
                frame = self.drone._capture_fresh_frame(
                    settle=_stl*1.5 + settle_extra, drain_first=True)
                if frame is None:
                    continue
                if abs(rotation) > 0.1:
                    frame = self.drone._rotate_frame(frame, rotation)
                det = self.drone.detect_all(frame)
                if det['h_candidate']:
                    if self.drone._stream_broken:
                        print("[Master] 断流后首帧有H，冲洗确认...")
                        self.drone._stream_broken = False
                        t0 = time.time()
                        while time.time() - t0 < 3.0:
                            self.drone.camera.read()
                        time.sleep(1)
                        confirm = self.drone._capture_fresh_frame(
                            settle=_stl + settle_extra,
                            read_time=_rd, drain_first=True)
                        if confirm is not None:
                            if abs(rotation) > 0.1:
                                confirm = self.drone._rotate_frame(confirm, rotation)
                            recheck = self.drone.detect_all(confirm)
                            if recheck['h_candidate'] is not None:
                                det = recheck
                                frame = confirm
                            else:
                                continue
                        else:
                            continue
                    self.drone._reset_servo_memory()
                    for __ in range(max_servo):
                        moved = self.drone._servo_toward_h(det['h_candidate']['box'], frame.shape, rotation=rotation, servo_tolerance=servo_tolerance)
                        if not moved:
                            break
                        time.sleep(_slp)
                        frame = self.drone._capture_fresh_frame(settle=_stl + settle_extra,
                                                                 read_time=_rd, drain_first=True)
                        if frame is None:
                            break
                        if abs(rotation) > 0.1:
                            frame = self.drone._rotate_frame(frame, rotation)
                        det = self.drone.detect_all(frame)
                        if det['h_candidate'] is None:
                            break
                    # 伺服后若 H 丢了，补搜一次
                    if det['h_candidate'] is None:
                        recover = self.drone._capture_fresh_frame(
                            settle=_stl + settle_extra,
                            read_time=_rd, drain_first=True)
                        if recover is not None:
                            if abs(rotation) > 0.1:
                                recover = self.drone._rotate_frame(recover, rotation)
                            det = self.drone.detect_all(recover)
                            if det['h_candidate'] is not None:
                                frame = recover
                    # ── 角度校正：伺服后测量 H 偏转并旋转无人机 ──
                    if det['h_candidate'] is not None:
                        angle_frame = self.drone._capture_fresh_frame(
                            settle=_stl + settle_extra, read_time=_rd, drain_first=True)
                        if angle_frame is not None:
                            if abs(rotation) > 0.1:
                                angle_frame = self.drone._rotate_frame(angle_frame, rotation)
                            recheck = self.drone.detect_all(angle_frame)
                            if recheck['h_candidate'] is not None:
                                applied = self.drone.correct_h_rotation(
                                    angle_frame, recheck['h_candidate']['box'],
                                    rotation=rotation)
                                if abs(applied) > 0.5:
                                    print(f"[Master] 旋转校正 {applied:.1f}°, 重新伺服...")
                                    time.sleep(_slp)
                                    frame = self.drone._capture_fresh_frame(
                                        settle=_stl + settle_extra, read_time=_rd, drain_first=True)
                                    if frame is not None:
                                        if abs(rotation) > 0.1:
                                            frame = self.drone._rotate_frame(frame, rotation)
                                        det2 = self.drone.detect_all(frame)
                                        if det2['h_candidate'] is not None:
                                            for _ in range(self.drone.servo_iters):
                                                moved = self.drone._servo_toward_h(
                                                    det2['h_candidate']['box'], frame.shape,
                                                    rotation=rotation)
                                                if not moved:
                                                    break
                                                time.sleep(_slp)
                                                frame = self.drone._capture_fresh_frame(
                                                    settle=_stl + settle_extra, read_time=_rd)
                                                if frame is None:
                                                    break
                                                if abs(rotation) > 0.1:
                                                    frame = self.drone._rotate_frame(frame, rotation)
                                                det2 = self.drone.detect_all(frame)
                                                if det2['h_candidate'] is None:
                                                    break
                    if save_prefix and det.get('h_candidate'):
                        self._save_h_result(frame, det, save_prefix)
                    return True
            return False

        # 第一次尝试
        if _try_servo():
            return True

        # 持续升高直到找到 H 或达到上限
        original_z = self.drone.drone.state['z']
        max_z = self.drone.h_search_max_height
        step = self.drone.h_search_step_height
        current_z = original_z + step
        while current_z <= max_z:
            print(f"[Master] 升高至 {current_z:.1f}m 搜索H...")
            self.drone.move_to(self.drone.drone.state['x'], self.drone.drone.state['y'], current_z)
            if _try_servo():
                print(f"[Master] 降回 {original_z:.1f}m")
                self.drone.move_to(self.drone.drone.state['x'], self.drone.drone.state['y'], original_z)
                self.drone.drone.state['z'] = original_z
                time.sleep(_slp)
                self.drone._capture_fresh_frame(settle=_stl, drain_first=True)
                if _try_servo():
                    return True
            current_z += step

        print(f"[Master] 升至 {max_z:.1f}m 仍未找到H，降回 {original_z:.1f}m")
        self.drone.move_to(self.drone.drone.state['x'], self.drone.drone.state['y'], original_z)
        self.drone.drone.state['z'] = original_z
        return False
    
    def _servo_and_land(self, rotation=0.0):
        """标准降落：找 H 伺服 → 前移 → 降落"""
        self.drone._rotate_gimbal_with_recovery(-90)
        self._servo_h_loop(rotation=rotation)
        print(f"[Master] 前移 {self.landing_offset:.2f}m 后降落")
        self.drone.move_to(self.drone.drone.state['x'] + self.landing_offset,
                           self.drone.drone.state['y'],
                           self.drone.drone.state['z'])
        self.drone._rotate_gimbal_with_recovery(0)
        saved = dict(self.drone.drone.state)
        self.drone.land()
        time.sleep(17)
        return saved
    
    def _find_waypoint(self, name):
        return next((wp for wp in self.drone.waypoints if wp.name == name), None)
    
    def _select_target_waypoint(self, results):
        """选择目标救援点（优先 target_grade最高置信度 > unknown兜底 > target_waypoint_name > target_labels > 第一个成功点）"""
        target_grade = self.config.get('target_grade')
        if target_grade is not None:
            candidates = []
            for point_name, result in results.items():
                if result.get('success') and str(result.get('grade', '')).lower() == str(target_grade).lower():
                    candidates.append((point_name, result))
            if candidates:
                best = max(candidates, key=lambda x: x[1].get('confidence', 0))
                print(f"[Master] 目标等级{target_grade}: {len(candidates)}个候选 → "
                      f"选 {best[0]} (conf={best[1]['confidence']:.3f})")
                return best[0]
            # 未找到目标等级 → 优先选 unknown
            for point_name, result in results.items():
                if result.get('success') and str(result.get('grade', '')).lower() == 'unknown':
                    print(f"[Master] 无等级{target_grade} → 选未知点: {point_name}")
                    return point_name
        target_name = self.config.get('target_waypoint_name')
        if target_name and target_name in results and results[target_name].get('success'):
            return target_name
        target_labels = self.config.get('target_labels', [])
        if isinstance(target_labels, str):
            target_labels = [target_labels]
        for point_name, result in results.items():
            if result.get('success') and str(result.get('raw_label', '')).lower() in [str(l).lower() for l in target_labels if l]:
                return point_name
        for point_name, result in results.items():
            if result.get('success'):
                return point_name
        return None
    
    def _fill_unknown_grades(self, results):
        """巡检后按 1×1级 + 1×2级 + 2×3级 原则填补 unknown。"""
        grade_counts = {'1': 0, '2': 0, '3': 0}
        unknown_points = []
        for name, r in results.items():
            if not r.get('success'):
                continue
            g = str(r.get('grade', 'unknown'))
            if g in grade_counts:
                grade_counts[g] += 1
            else:
                unknown_points.append(name)

        if not unknown_points:
            return

        expected = {'1': 1, '2': 1, '3': 2}
        missing = []
        for g in ['1', '2', '3']:
            need = max(0, expected[g] - grade_counts[g])
            missing.extend([g] * need)

        print(f"[Master] 查漏补缺: 已知{grade_counts}, "
              f"unknown={len(unknown_points)}个, 需补{missing}")

        for i, point_name in enumerate(unknown_points):
            if i < len(missing):
                fill_grade = missing[i]
                results[point_name]['grade'] = fill_grade
                results[point_name]['raw_label'] = f'filled_{fill_grade}'
                results[point_name]['confidence'] = 0.0
                print(f"[Master]   {point_name}: unknown → {fill_grade}级 (推断)")
            else:
                print(f"[Master]   {point_name}: unknown 保留 (超出需补数)")

    def _update_rescue_results(self, results):
        for point_name, result in results.items():
            if result["success"]:
                self.rescue_points.set_result(
                    point_name,
                    grade=result["grade"],
                    confidence=result["confidence"],
                    image_path=result["image_path"],
                )
            else:
                print(f"[Master] 警告: {point_name} 扫描失败 — {result.get('reason', 'unknown')}")
    
    def _write_csv(self):
        path = os.path.join(self.output_folder, "rescue_levels.csv")
        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(self.rescue_points.to_csv_rows())
        print(f"[Master] 等级文件已输出: {path}")
    
    # ---- 主流程 ----
    
    def execute_full_mission(self):
        """执行完整的比赛全流程"""
        print("\n" + "=" * 60)
        print("  比赛全流程总控")
        print("  巡检 → 取货装货 → 投送 → 放置 → 返航")
        print("=" * 60)
        
        results = {}
        target_name = None
        target_waypoint = None
        
        try:
            # ============ 阶段 1+2 并行：无人机巡检 + 小车取货 ============
            print("\n" + "=" * 60)
            print("  [阶段1+2] 无人机巡检 + 小车取货（并行）")
            print("=" * 60)
            
            # 用容器在线程间传递 drone 扫描结果
            drone_result = {"results": None, "error": None}
            
            def drone_scan():
                try:
                    drone_result["results"] = self.drone.scan_waypoints()
                except Exception as e:
                    drone_result["error"] = str(e)
            
            # 启动无人机巡检（后台线程）
            drone_thread = threading.Thread(target=drone_scan, daemon=True)
            drone_thread.start()
            
            # 等待无人机起飞后再触发小车出发取货（轮询确认，不是盲等）
            print("[Master] 等待无人机起飞（最多30秒）...")
            takeoff_timeout = 30
            t0 = time.time()
            while time.time() - t0 < takeoff_timeout:
                if self.drone.drone.taken_off:
                    print("[Master] 无人机已起飞，触发小车取货")
                    break
                if drone_result["error"]:
                    print(f"[Master] 无人机起飞失败: {drone_result['error']}")
                    drone_thread.join()
                    return False
                time.sleep(1)
            else:
                print("[Master] 无人机起飞超时，终止任务")
                return False
            
            # 触发小车取货（导航点5 + 视觉抓取 + 导航点6）
            self.car.start_pickup()
            
            # 等待无人机巡检完成
            print("[Master] 等待无人机巡检完成...")
            drone_thread.join()
            
            if drone_result["error"]:
                print(f"[Master] 无人机巡检异常: {drone_result['error']}")
                return False
            
            results = drone_result["results"]
            self._fill_unknown_grades(results)
            self._update_rescue_results(results)
            self._write_csv()
            saved_loading = dict(self.drone.drone.state)

            # 断言自身位置在装货区
            la = self.drone.loading_area
            print(f"[Master] 断言位置: 装货区=({la.x:.2f},{la.y:.2f},{la.z:.2f})")
            self.drone.drone.state['x'] = la.x
            self.drone.drone.state['y'] = la.y
            self.drone.drone.state['z'] = la.z
            
            target_name = self._select_target_waypoint(results)
            if target_name is None:
                print("[Master] 未能确定目标救援点，终止任务")
                return False
            target_waypoint = self._find_waypoint(target_name)
            print(f"[Master] 目标救援点: {target_name}")
            print(f"[Master] 无人机已降落装货区")
            
            # 等待小车完成取货并到达装货区
            print("[Master] 等待小车到达装货区...")
            self.car.wait_idle()
            print("[Master] 小车已到达装货区")
            
            # ============ 阶段 3：装货 ============
            print("\n" + "=" * 60)
            print("  [阶段3] 装货区 — 小车给无人机装货")
            print("=" * 60)
            
            # 触发小车执行装货（机械臂放置到无人机平台）
            self.car.start_loading()
            print("[Master] 等待小车完成装货...")
            self.car.wait_idle()
            print("[Master] 装货完成")
            
            # ============ 阶段 4：投送 ============
            print("\n" + "=" * 60)
            print("  [阶段4] 无人机飞往救援点")
            print("=" * 60)
            
            rot = target_waypoint.rotation + target_waypoint.rotation_offset
            
            # 从装货区起飞
            self.drone.reset()
            time.sleep(1)
            if not self.drone.takeoff():
                print("[Master] 装货区起飞失败")
                return False
            time.sleep(6)
            self.drone.drone.state['z'] = 1.2  # 起飞后实际高度, 确保move_to抬升0.3m
            for _ in range(3):
                if self.drone.move_to(self.drone.drone.state['x'], self.drone.drone.state['y'], 1.5):
                    break
                print("[Master] 升至1.5m失败, 重试...")
                time.sleep(2)
            self.drone.drone.state['x'] = saved_loading['x']
            self.drone.drone.state['y'] = saved_loading['y']

            # 后退（撤销降落前移）
            back = -float(self.config.get("back_offset", 0.09))
            print(f"[Master] 后退 {back:.2f}m")
            self.drone.move_to(self.drone.drone.state['x'] + back,
                               self.drone.drone.state['y'],
                               self.drone.drone.state['z'])
            
            # 旋转 180° 回正
            if not self.drone.rotate_yaw(180):
                print("[Master] 旋转180° 失败")
                return False
            if abs(self.home_rotate_yaw) > 0.1:
                self.drone.rotate_yaw(self.home_rotate_yaw)

            # 飞往 home_point 并 H 伺服对齐
            hx, hy, hz = self.home_point
            print(f"[Master] 飞往 home ({hx}, {hy}, {hz})...")
            if not self.drone.move_to(hx, hy, hz):
                return False
            print("[Master] home H 伺服对齐...")
            self.drone._rotate_gimbal_with_recovery(-90)
            self._servo_h_loop(save_prefix="home")

            # 对齐 H 后前移 landing_offset
            self.drone.move_to(self.drone.drone.state['x'] + self.landing_offset,
                               self.drone.drone.state['y'],
                               self.drone.drone.state['z'])
            self.drone._rotate_gimbal_with_recovery(0)

            print(f"[Master] 断言原点 (0, 0, {hz})")
            self.drone.drone.state['x'] = 0.0
            self.drone.drone.state['y'] = 0.0
            self.drone.drone.state['z'] = hz
            
            # 飞往目标救援点
            print(f"[Master] 飞往 {target_name} ({target_waypoint.x}, {target_waypoint.y}, {target_waypoint.z})")
            self.drone.move_to(target_waypoint.x, target_waypoint.y, target_waypoint.z)
            
            # H 伺服
            print("[Master] 目标点 H 伺服...")
            self.drone._rotate_gimbal_with_recovery(-90)
            self._servo_h_loop(rotation=target_waypoint.rotation, save_prefix="rescue")

            # 旋转无人机朝向（不加伺服，前面已居中）
            if abs(rot) > 0.1:
                print(f"[Master] 旋转 {rot}°")
                self.drone.rotate_yaw(rot)
                time.sleep(self.drone.servo_sleep)
                self._servo_h_loop()

            # [已禁用预降] 旋转对齐后直接前移+降落
            # print(f"[Master] 预降至 {self.preview_h:.1f}m")
            # self.drone.move_to(self.drone.drone.state['x'], self.drone.drone.state['y'], self.preview_h)
            # time.sleep(2)
            print("[Master] 旋转对齐后直接降落...")
            self.drone.move_to(self.drone.drone.state['x'] + self.landing_offset,
                               self.drone.drone.state['y'],
                               self.drone.drone.state['z'])
            self.drone._rotate_gimbal_with_recovery(0)
            landing_state = dict(self.drone.drone.state)
            self.drone.land()
            time.sleep(17)

            # 断言目标点位置
            print(f"[Master] 断言位置: {target_name}=({target_waypoint.x:.2f},{target_waypoint.y:.2f},{target_waypoint.z:.2f})")
            self.drone.drone.state['x'] = target_waypoint.x
            self.drone.drone.state['y'] = target_waypoint.y
            self.drone.drone.state['z'] = target_waypoint.z

            # 降落后再触发小车去救援点
            rescue_point_num = int(target_name.replace("救援点", ""))
            print(f"[Master] 无人机已降落，触发小车前往救援点{rescue_point_num}")
            self.car.start_rescue(rescue_point_num)

            # 等待小车完成救援区放置
            print("[Master] 等待小车完成救援区放置...")
            self.car.wait_idle()
            print("[Master] 小车救援区放置完成")
            
            # ============ 阶段 5：地面放置 ============
            print("\n" + "=" * 60)
            print("  [阶段5] 小车放置物资")
            print("=" * 60)
            
            # 注意：rescue 阶段在 car 端已包含抓取+放置，这里可能已经完成了
            # 根据现有 car_arm_integration.py 的 phase3_rescue 逻辑，
            # 它在救援点执行：抓取 → 平移 → 放置
            # 但这里无人机降落时物资还在无人机平台上
            # 小车需要从无人机平台抓取物资并放置
            # 实际上 phase3_rescue 已经包含了这个逻辑
            # 所以此时小车应该已经完成了（wait_idle 已返回）
            print("[Master] 物资放置完成")
            
            # ============ 阶段 6：返航 ============
            print("\n" + "=" * 60)
            print("  [阶段6] 双方返航")
            print("=" * 60)
            
            # 触发小车返航
            self.car.start_return()
            
            # 无人机返航
            self.drone.reset()
            time.sleep(1)
            if not self.drone.takeoff():
                print("[Master] 救援点起飞失败")
                return False
            time.sleep(6)
            self.drone.drone.state['z'] = 1.2  # 起飞后实际高度, 确保move_to抬升0.3m
            for _ in range(3):
                if self.drone.move_to(self.drone.drone.state['x'], self.drone.drone.state['y'], 1.5):
                    break
                print("[Master] 升至1.5m失败, 重试...")
                time.sleep(2)
            self.drone.drone.state['x'] = landing_state['x']
            self.drone.drone.state['y'] = landing_state['y']

            if abs(rot) > 0.1:
                print(f"[Master] 旋转 -{rot}° 恢复朝向")
                self.drone.rotate_yaw(-rot)
            
            # 飞 home 校准
            hx, hy, hz = self.home_point
            self.drone.move_to(hx, hy, hz)
            self.drone._rotate_gimbal_with_recovery(-90)
            self._servo_h_loop(save_prefix="home")
            self.drone.move_to(self.drone.drone.state['x'] + self.landing_offset,
                               self.drone.drone.state['y'],
                               self.drone.drone.state['z'])
            self.drone._rotate_gimbal_with_recovery(0)
            self.drone.drone.state['x'] = 0.0
            self.drone.drone.state['y'] = 0.0
            self.drone.drone.state['z'] = hz

            print("[Master] home点 H对齐降落...")
            self.drone.land()
            time.sleep(17)
            
            # 等待小车返航完成
            print("[Master] 等待小车返航...")
            self.car.wait_idle()
            
            print("\n" + "=" * 60)
            print("  全流程完成！")
            print("=" * 60)
            return True
            
        except KeyboardInterrupt:
            print("\n[Master] 用户中断")
            return False
        except Exception as e:
            print(f"\n[Master] 异常: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def shutdown(self):
        """清理资源"""
        print("[Master] 清理中...")
        # DroneNavigator 的 land() 已经释放了摄像头
        try:
            self.drone.drone.land()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description='比赛全流程总控（PC端）')
    parser.add_argument('--config', type=str, required=True, help='配置文件 JSON 路径')
    parser.add_argument('--car-url', type=str, default='http://192.168.43.8:6001',
                        help='小车 HTTP 服务地址')
    args = parser.parse_args()
    
    config = JsonHelper.load_json(args.config)
    
    controller = MasterController(config, args.car_url)
    
    try:
        success = controller.execute_full_mission()
        if success:
            print("[Master] 任务成功完成")
        else:
            print("[Master] 任务未完成")
    except KeyboardInterrupt:
        print("\n[Master] 用户中断")
    finally:
        controller.shutdown()


if __name__ == '__main__':
    main()
