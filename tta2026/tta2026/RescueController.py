import csv
import os
import time
from typing import Any, Dict, Optional

from Controllers.DroneNavigator import DroneNavigator
from Entities.RescuePointManager import RescuePointManager
from Utils.JsonHelper import JsonHelper


class RescueController:
    """顶层编排器 — 协调无人机巡检、数据汇总和 CSV 输出。小车部分预留接口。"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # 输出配置
        output = config.get("output", {})
        self.output_folder = os.path.abspath(output.get("folder", "output"))
        self.csv_filename = output.get("csv_filename", "rescue_levels.csv")
        os.makedirs(self.output_folder, exist_ok=True)

        # 无人机巡检子系统
        self.drone = DroneNavigator(config)

        # 救援点数据管理
        rescue_points = config.get("rescue_points", [])
        self.rescue_points = RescuePointManager(rescue_points)

        # 小车预留（后续接入）
        self.car = None

        # 原点（home）配置
        hp = config.get("home_point", {})
        self.home_point = (float(hp.get("x", 0.7)), float(hp.get("y", 1.3)),
                           float(hp.get("z", 1.5)))
        self.home_rotate_yaw = float(hp.get("rotate_yaw", 0.0))

    def execute_scan_mission(self) -> bool:
        """执行一次完整的无人机巡检扫描任务。返回是否全部扫描成功。"""
        print("[RescueController] 开始无人机巡检任务...")

        # 1. 执行扫描
        results = self.drone.scan_waypoints()

        # 1.5 查漏补缺
        self._fill_unknown_grades(results)

        # 2. 汇总结果到 RescuePointManager
        for point_name, result in results.items():
            if result["success"]:
                self.rescue_points.set_result(
                    point_name,
                    grade=result["grade"],
                    confidence=result["confidence"],
                    image_path=result["image_path"],
                )
            else:
                print(f"[RescueController] 警告: {point_name} 扫描失败 — {result.get('reason', 'unknown')}")

        # 3. 输出 CSV
        self._write_csv()
        print(f"[RescueController] 等级文件已输出: {os.path.join(self.output_folder, self.csv_filename)}")

        # 4. 返回是否全部成功
        all_ok = self.rescue_points.all_scanned()
        print(f"[RescueController] 巡检{'全部完成' if all_ok else '部分完成'} — {self.rescue_points.summary()}")
        return all_ok

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
            print(f"[RescueController] H识别结果已保存: {json_path}")

    def _servo_h_loop(self, rotation: float = 0.0, max_search: int = 3, max_servo: int = None,
                      servo_tolerance: float = None, save_prefix: str = None):
        """内部辅助：云台朝下搜索 H 并迭代伺服居中。
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
                if frame is None: continue
                if abs(rotation) > 0.1:
                    frame = self.drone._rotate_frame(frame, rotation)
                det = self.drone.detect_all(frame)
                if det['h_candidate']:
                    if self.drone._stream_broken:
                        print("[RescueController] 断流后首帧有H，冲洗确认...")
                        self.drone._stream_broken = False
                        t0 = time.time()
                        while time.time() - t0 < 3.0:
                            self.drone.camera.read()
                        time.sleep(1)
                        confirm = self.drone._capture_fresh_frame(settle=_stl + settle_extra, read_time=_rd, drain_first=True)
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
                        moved = self.drone._servo_toward_h(det['h_candidate']['box'], frame.shape,
                                                           rotation=rotation, servo_tolerance=servo_tolerance)
                        if not moved: break
                        time.sleep(_slp)
                        frame = self.drone._capture_fresh_frame(settle=_stl + settle_extra,
                                                                 read_time=_rd, drain_first=True)
                        if frame is None: break
                        if abs(rotation) > 0.1:
                            frame = self.drone._rotate_frame(frame, rotation)
                        det = self.drone.detect_all(frame)
                        if det['h_candidate'] is None: break
                    # 伺服后若 H 丢了，补搜一次
                    if det['h_candidate'] is None:
                        recover = self.drone._capture_fresh_frame(settle=_stl + settle_extra,
                                                                   read_time=_rd, drain_first=True)
                        if recover is not None:
                            if abs(rotation) > 0.1:
                                recover = self.drone._rotate_frame(recover, rotation)
                            det = self.drone.detect_all(recover)
                            if det['h_candidate'] is not None:
                                frame = recover
                    # 角度校正：捕获新帧检测H偏转
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
                                    print(f"[RescueController] 旋转校正 {applied:.1f}°, 重新伺服...")
                                    time.sleep(_slp)
                                    re_frame = self.drone._capture_fresh_frame(
                                        settle=_stl + settle_extra, read_time=_rd, drain_first=True)
                                    if re_frame is not None:
                                        if abs(rotation) > 0.1:
                                            re_frame = self.drone._rotate_frame(re_frame, rotation)
                                        re_det = self.drone.detect_all(re_frame)
                                        if re_det['h_candidate'] is not None:
                                            for _ in range(self.drone.servo_iters):
                                                moved = self.drone._servo_toward_h(
                                                    re_det['h_candidate']['box'], re_frame.shape,
                                                    rotation=rotation)
                                                if not moved:
                                                    break
                                                time.sleep(_slp)
                                                re_frame = self.drone._capture_fresh_frame(
                                                    settle=_stl + settle_extra,
                                                    read_time=_rd)
                                                if re_frame is None:
                                                    break
                                                if abs(rotation) > 0.1:
                                                    re_frame = self.drone._rotate_frame(re_frame, rotation)
                                                re_det = self.drone.detect_all(re_frame)
                                                if re_det['h_candidate'] is None:
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
            print(f"[RescueController] 升高至 {current_z:.1f}m 搜索H...")
            self.drone.move_to(self.drone.drone.state['x'], self.drone.drone.state['y'], current_z)
            if _try_servo():
                print(f"[RescueController] 降回 {original_z:.1f}m")
                self.drone.move_to(self.drone.drone.state['x'], self.drone.drone.state['y'], original_z)
                self.drone.drone.state['z'] = original_z
                # 降回原高度后再次伺服
                time.sleep(_slp)
                self.drone._capture_fresh_frame(settle=_stl, drain_first=True)  # 冲洗旧帧
                if _try_servo():
                    return True
            current_z += step

        print(f"[RescueController] 升至 {max_z:.1f}m 仍未找到H，降回 {original_z:.1f}m")
        self.drone.move_to(self.drone.drone.state['x'], self.drone.drone.state['y'], original_z)
        self.drone.drone.state['z'] = original_z
        return False

    def _servo_and_land(self, rotation: float = 0.0):
        """标准降落：找 H 伺服 → 前移 → 降落。"""
        self.drone._rotate_gimbal_with_recovery(-90)
        self._servo_h_loop(rotation=rotation)
        offset = float(self.config.get('landing_offset', 0.04))
        print(f"[RescueController] 前移 {offset:.2f}m 后降落")
        self.drone.move_to(self.drone.drone.state['x'] + offset, self.drone.drone.state['y'], self.drone.drone.state['z'])
        self.drone._rotate_gimbal_with_recovery(0)
        saved = dict(self.drone.drone.state)
        self.drone.land()
        time.sleep(17)
        return saved

    def execute_drone_full_test(self) -> bool:
        """无人机全流程测试（无小车）：
        巡检4点 → 装货区降落(180°) → 起飞回原点 → 飞目标点 →
        伺服 → 旋转(rotation) → 预降至(landing_preview_height) → 伺服 → 降落 →
        起飞 → 旋转(-rotation) → 回原点 → 伺服 → 降落。"""
        import time
        print("[RescueController] ====== 无人机全流程测试 ======")

        results = self.drone.scan_waypoints()
        self._fill_unknown_grades(results)
        self._update_rescue_results(results)
        self._write_csv()
        time.sleep(16)  # 装货区降落后等稳定

        # 断言自身位置在装货区
        la = self.drone.loading_area
        saved_loading = dict(self.drone.drone.state)
        print(f"[RescueController] 断言位置: 装货区=({la.x:.2f},{la.y:.2f},{la.z:.2f})")
        self.drone.drone.state['x'] = la.x
        self.drone.drone.state['y'] = la.y
        self.drone.drone.state['z'] = la.z

        target_name = self._select_target_waypoint(results)
        if target_name is None:
            return False
        target_waypoint = self._find_waypoint(target_name)
        rot = target_waypoint.rotation + target_waypoint.rotation_offset
        return_altitude = float(self.config.get('return_altitude', 1.5))
        preview_h = float(self.config.get('landing_preview_height', 0.8))
        print(f"[RescueController] 目标: {target_name}, rotation={rot}°")

        # ---- 起飞回原点 ----
        print("[RescueController] --- 起飞回原点 ---")
        self.drone.reset(); time.sleep(1)
        if not self.drone.takeoff():
            print("[RescueController] 装货区起飞失败")
            return False
        time.sleep(6)
        # 起飞后实际高度约1.2m，重置state.z确保move_to计算正确的抬升量
        self.drone.drone.state['z'] = 1.2
        for _ in range(3):
            if self.drone.move_to(self.drone.drone.state['x'], self.drone.drone.state['y'], 1.5):
                break
            print("[RescueController] 升至1.5m失败, 重试...")
            time.sleep(2)
        self.drone.drone.state['x'] = saved_loading['x']
        self.drone.drone.state['y'] = saved_loading['y']

        # 1. 先后退 2×landing_offset（撤销降落时的前移）
        back = -float(self.config.get('back_offset', 0.09))
        print(f"[RescueController] 先后退 {back:.2f}m")
        self.drone.move_to(self.drone.drone.state['x'] + back,
                           self.drone.drone.state['y'],
                           self.drone.drone.state['z'])

        # 2. 旋转 180° 回正
        if not self.drone.rotate_yaw(180):
            print("[RescueController] 旋转180° 失败")
            return False

        # 2b. home 额外旋转
        if abs(self.home_rotate_yaw) > 0.1:
            self.drone.rotate_yaw(self.home_rotate_yaw)

        # 3. 飞往 home_point 并 H 伺服对齐
        hx, hy, hz = self.home_point
        print(f"[RescueController] 飞往 home ({hx}, {hy}, {hz})...")
        if not self.drone.move_to(hx, hy, hz):
            return False
        print("[RescueController] home H 伺服对齐...")
        self.drone._rotate_gimbal_with_recovery(-90)
        self._servo_h_loop(save_prefix="home")

        # 对齐 H 后前移 landing_offset
        offset = float(self.config.get('landing_offset', 0.04))
        print(f"[RescueController] 前移 {offset:.2f}m")
        self.drone.move_to(self.drone.drone.state['x'] + offset,
                           self.drone.drone.state['y'],
                           self.drone.drone.state['z'])
        self.drone._rotate_gimbal_with_recovery(0)

        # 4. 断言当前位置为 (0, 0, z)，重新校准坐标系
        print(f"[RescueController] 断言原点 (0, 0, {hz})")
        self.drone.drone.state['x'] = 0.0
        self.drone.drone.state['y'] = 0.0
        self.drone.drone.state['z'] = hz

        # ---- 飞目标点 → 伺服 → 旋转 → 预降 → 伺服 → 降落 ----
        print(f"[RescueController] --- 飞目标点 {target_name} ---")
        self.drone.move_to(target_waypoint.x, target_waypoint.y, target_waypoint.z)
        self.drone._rotate_gimbal_with_recovery(-90)
        self._servo_h_loop(rotation=target_waypoint.rotation, save_prefix="rescue")
        if abs(rot) > 0.1:
            print(f"[RescueController] 旋转 {rot}°")
            self.drone.rotate_yaw(rot)
            time.sleep(self.drone.servo_sleep)
            self._servo_h_loop()

        # [已禁用预降] print(f"[RescueController] 预降至 {preview_h:.1f}m")
        # [已禁用预降] self.drone.move_to(self.drone.drone.state['x'], self.drone.drone.state['y'], preview_h)
        # [已禁用预降] time.sleep(2)
        # 旋转对齐后伺服+前移+降落
        offset = float(self.config.get('landing_offset', 0.04))
        self.drone.move_to(self.drone.drone.state['x'] + offset, self.drone.drone.state['y'], self.drone.drone.state['z'])
        self.drone._rotate_gimbal_with_recovery(0)
        landing_state = dict(self.drone.drone.state)
        self.drone.land()
        time.sleep(17)

        # ---- 起飞 → 旋转 -rotation → 飞home → 伺服 → 降落 ----
        print("[RescueController] --- 起飞返航 ---")
        self.drone.reset(); time.sleep(1)
        self.drone.takeoff()
        time.sleep(6)
        # 起飞后实际高度约1.2m，重置state.z确保move_to计算正确的抬升量
        self.drone.drone.state['z'] = 1.2
        for _ in range(3):
            if self.drone.move_to(self.drone.drone.state['x'], self.drone.drone.state['y'], 1.5):
                break
            print("[RescueController] 升至1.5m失败, 重试...")
            time.sleep(2)
        self.drone.drone.state['x'] = landing_state['x']
        self.drone.drone.state['y'] = landing_state['y']
        if abs(rot) > 0.1:
            print(f"[RescueController] 旋转 -{rot}° 恢复朝向")
            self.drone.rotate_yaw(-rot)

        # 先飞 home 校准坐标系
        hx, hy, hz = self.home_point
        print(f"[RescueController] 飞往 home ({hx}, {hy}, {hz})...")
        self.drone.move_to(hx, hy, hz)
        print("[RescueController] home H 伺服对齐...")
        self.drone._rotate_gimbal_with_recovery(-90)
        self._servo_h_loop(save_prefix="home")
        self.drone.move_to(self.drone.drone.state['x'] + float(self.config.get('landing_offset', 0.04)),
                           self.drone.drone.state['y'],
                           self.drone.drone.state['z'])
        self.drone._rotate_gimbal_with_recovery(0)
        print(f"[RescueController] 断言原点 (0, 0, {hz})")
        self.drone.drone.state['x'] = 0.0
        self.drone.drone.state['y'] = 0.0
        self.drone.drone.state['z'] = hz

        print("[RescueController] --- home点 降落 ---")
        self.drone.land()
        time.sleep(17)

        print("[RescueController] ====== 全流程测试完成 =====")
        return True

    def execute_half_mission(self, target_name: str) -> bool:
        """后半程测试（从装货区起飞开始）。
        参数 target_name 如 \"救援点1\"。"""
        import time
        print(f"[RescueController] ====== 后半程测试 → {target_name} ======")

        target_waypoint = self._find_waypoint(target_name)
        if target_waypoint is None:
            print(f"[RescueController] 未找到航点: {target_name}")
            return False

        rot = target_waypoint.rotation + target_waypoint.rotation_offset
        return_altitude = float(self.config.get('return_altitude', 1.5))
        preview_h = float(self.config.get('landing_preview_height', 0.8))

        # 断言自身位置在装货区
        la = self.drone.loading_area
        print(f"[RescueController] 断言位置: 装货区=({la.x:.2f},{la.y:.2f},{la.z:.2f})")
        self.drone.drone.state['x'] = la.x
        self.drone.drone.state['y'] = la.y
        self.drone.drone.state['z'] = la.z

        # ---- 起飞回原点 ----
        print("[RescueController] --- 起飞回原点 ---")
        self.drone.reset(); time.sleep(1)
        if not self.drone.takeoff():
            print("[RescueController] 装货区起飞失败")
            return False
        time.sleep(6)
        # 起飞后实际高度约1.2m，重置state.z确保move_to计算正确的抬升量
        self.drone.drone.state['z'] = 1.2
        for _ in range(3):
            if self.drone.move_to(self.drone.drone.state['x'], self.drone.drone.state['y'], 1.5):
                break
            print("[RescueController] 升至1.5m失败, 重试...")
            time.sleep(2)
        self.drone.drone.state['x'] = la.x
        self.drone.drone.state['y'] = la.y

        back = -float(self.config.get('back_offset', 0.09))
        print(f"[RescueController] 先后退 {back:.2f}m")
        self.drone.move_to(self.drone.drone.state['x'] + back,
                           self.drone.drone.state['y'],
                           self.drone.drone.state['z'])

        if not self.drone.rotate_yaw(180):
            print("[RescueController] 旋转180° 失败")
            return False
        if abs(self.home_rotate_yaw) > 0.1:
            self.drone.rotate_yaw(self.home_rotate_yaw)

        hx, hy, hz = self.home_point
        print(f"[RescueController] 飞往 home ({hx}, {hy}, {hz})...")
        if not self.drone.move_to(hx, hy, hz):
            return False
        print("[RescueController] home H 伺服对齐...")
        self.drone._rotate_gimbal_with_recovery(-90)
        self._servo_h_loop(save_prefix="home")
        offset = float(self.config.get('landing_offset', 0.04))
        print(f"[RescueController] 前移 {offset:.2f}m")
        self.drone.move_to(self.drone.drone.state['x'] + offset,
                           self.drone.drone.state['y'],
                           self.drone.drone.state['z'])
        self.drone._rotate_gimbal_with_recovery(0)
        print(f"[RescueController] 断言原点 (0, 0, {hz})")
        self.drone.drone.state['x'] = 0.0
        self.drone.drone.state['y'] = 0.0
        self.drone.drone.state['z'] = hz

        # ---- 飞目标点 ----
        print(f"[RescueController] --- 飞目标点 {target_name} ---")
        self.drone.move_to(target_waypoint.x, target_waypoint.y, target_waypoint.z)
        self.drone._rotate_gimbal_with_recovery(-90)
        self._servo_h_loop(rotation=target_waypoint.rotation, save_prefix="rescue")

        if abs(rot) > 0.1:
            print(f"[RescueController] 旋转 {rot}°")
            self.drone.rotate_yaw(rot)
            time.sleep(self.drone.servo_sleep)
            self._servo_h_loop()

        # [已禁用预降] print(f"[RescueController] 预降至 {preview_h:.1f}m")
        # [已禁用预降] self.drone.move_to(self.drone.drone.state['x'], self.drone.drone.state['y'], preview_h)
        # [已禁用预降] time.sleep(2)
        offset = float(self.config.get('landing_offset', 0.04))
        self.drone.move_to(self.drone.drone.state['x'] + offset, self.drone.drone.state['y'], self.drone.drone.state['z'])
        self.drone._rotate_gimbal_with_recovery(0)
        landing_state = dict(self.drone.drone.state)
        self.drone.land()
        time.sleep(17)

        # ---- 返航 ----
        print("[RescueController] --- 起飞返航 ---")
        self.drone.reset(); time.sleep(1)
        self.drone.takeoff()
        time.sleep(6)
        # 起飞后实际高度约1.2m，重置state.z确保move_to计算正确的抬升量
        self.drone.drone.state['z'] = 1.2
        for _ in range(3):
            if self.drone.move_to(self.drone.drone.state['x'], self.drone.drone.state['y'], 1.5):
                break
            print("[RescueController] 升至1.5m失败, 重试...")
            time.sleep(2)
        self.drone.drone.state['x'] = landing_state['x']
        self.drone.drone.state['y'] = landing_state['y']
        if abs(rot) > 0.1:
            print(f"[RescueController] 旋转 -{rot}° 恢复朝向")
            self.drone.rotate_yaw(-rot)

        hx, hy, hz = self.home_point
        print(f"[RescueController] 飞往 home ({hx}, {hy}, {hz})...")
        self.drone.move_to(hx, hy, hz)
        self.drone._rotate_gimbal_with_recovery(-90)
        self._servo_h_loop(save_prefix="home")
        self.drone.move_to(self.drone.drone.state['x'] + float(self.config.get('landing_offset', 0.04)),
                           self.drone.drone.state['y'],
                           self.drone.drone.state['z'])
        self.drone._rotate_gimbal_with_recovery(0)
        print(f"[RescueController] 断言原点 (0, 0, {hz})")
        self.drone.drone.state['x'] = 0.0
        self.drone.drone.state['y'] = 0.0
        self.drone.drone.state['z'] = hz

        print("[RescueController] --- home点 降落 ---")
        self.drone.land()
        time.sleep(17)

        print("[RescueController] ====== 后半程测试完成 =====")
        return True

    def execute_delivery_mission(self) -> bool:
        """完整任务流程：
        无人机巡检 → 装货 → 投送 → 物资放置 → 返航。
        与小车通过 CarController.wait_for_signal 同步。"""
        print("[RescueController] ====== 完整任务开始 ======")

        # ==================== 阶段 1：巡检 ====================
        print("[RescueController] --- 阶段 1: 无人机巡检 ---")
        results = self.drone.scan_waypoints()
        self._fill_unknown_grades(results)
        self._update_rescue_results(results)
        self._write_csv()
        saved_loading = dict(self.drone.drone.state)  # 保存装货区降落前坐标
        time.sleep(17)  # 装货区降落后等稳定
        # scan_waypoints 末尾已在装货区旋转 180° 并降落

        # 断言自身位置在装货区
        la = self.drone.loading_area
        print(f"[RescueController] 断言位置: 装货区=({la.x:.2f},{la.y:.2f},{la.z:.2f})")
        self.drone.drone.state['x'] = la.x
        self.drone.drone.state['y'] = la.y
        self.drone.drone.state['z'] = la.z

        target_name = self._select_target_waypoint(results)
        if target_name is None:
            print("[RescueController] 未能确定目标航点，任务终止")
            return False
        print(f"[RescueController] 目标航点: {target_name}")

        # ==================== 阶段 2：装货 ====================
        print("[RescueController] --- 阶段 2: 小车取货装货 ---")

        # 2a. 无人机通知小车去取货
        print("[RescueController] 通知小车: 前往取货区取物资")
        if not self.wait_for_car_signal('go_fetch'):
            return False

        # 2b. 等待小车取货完成并回到装货区
        print("[RescueController] 等待小车回到装货区并装载完成...")
        if not self.wait_for_car_signal('loaded'):
            return False

        # 2c. 小车装完货后，控制小车返回初始位置
        print("[RescueController] 通知小车: 返回初始位置")
        if self.car is not None:
            self.car.move_to(0.0, 0.0)
        else:
            print("[RescueController] 小车未接入，跳过小车移动")

        # ==================== 阶段 3：无人机出征 ====================
        print("[RescueController] --- 阶段 3: 无人机携带物资出征 ---")

        self.drone.reset()
        time.sleep(1)
        if not self.drone.takeoff():
            return False
        time.sleep(6)
        # 起飞后实际高度约1.2m，重置state.z确保move_to计算正确的抬升量
        self.drone.drone.state['z'] = 1.2
        for _ in range(3):
            if self.drone.move_to(self.drone.drone.state['x'], self.drone.drone.state['y'], 1.5):
                break
            print("[RescueController] 升至1.5m失败, 重试...")
            time.sleep(2)
        self.drone.drone.state['x'] = saved_loading['x']
        self.drone.drone.state['y'] = saved_loading['y']

        # 1. 先后退 2×landing_offset
        back = -float(self.config.get('back_offset', 0.09))
        print(f"[RescueController] 先后退 {back:.2f}m")
        self.drone.move_to(self.drone.drone.state['x'] + back,
                           self.drone.drone.state['y'],
                           self.drone.drone.state['z'])

        # 2. 旋转 180° 回正
        print("[RescueController] 旋转 180° 恢复起始朝向")
        if not self.drone.rotate_yaw(180):
            return False
        if abs(self.home_rotate_yaw) > 0.1:
            self.drone.rotate_yaw(self.home_rotate_yaw)

        # 3. 飞往 home_point 并 H 伺服对齐，断言原点
        hx, hy, hz = self.home_point
        return_altitude = float(self.config.get('return_altitude', 1.5))
        print(f"[RescueController] 飞往 home ({hx}, {hy}, {hz})...")
        if not self.drone.move_to(hx, hy, hz):
            return False
        print("[RescueController] home H 伺服对齐...")
        self.drone._rotate_gimbal_with_recovery(-90)
        self._servo_h_loop(save_prefix="home")
        offset = float(self.config.get('landing_offset', 0.04))
        print(f"[RescueController] 前移 {offset:.2f}m")
        self.drone.move_to(self.drone.drone.state['x'] + offset,
                           self.drone.drone.state['y'],
                           self.drone.drone.state['z'])
        self.drone._rotate_gimbal_with_recovery(0)
        print(f"[RescueController] 断言原点 (0, 0, {hz})")
        self.drone.drone.state['x'] = 0.0
        self.drone.drone.state['y'] = 0.0
        self.drone.drone.state['z'] = hz

        # 飞往目标救援点
        target_waypoint = self._find_waypoint(target_name)
        if target_waypoint is None:
            return False
        rot = target_waypoint.rotation + target_waypoint.rotation_offset
        preview_h = float(self.config.get('landing_preview_height', 0.8))
        print(f"[RescueController] 飞往目标: {target_waypoint.name} ({target_waypoint.x}, {target_waypoint.y}, {target_waypoint.z}) rotation={rot}°")
        if not self.drone.move_to(target_waypoint.x, target_waypoint.y, target_waypoint.z):
            return False

        # 目标点: 伺服 → 旋转 → 预降 → 伺服 → 降落
        print("[RescueController] 目标点 H 对齐...")
        self.drone._rotate_gimbal_with_recovery(-90)
        self._servo_h_loop(rotation=target_waypoint.rotation, save_prefix="rescue")

        if abs(rot) > 0.1:
            print(f"[RescueController] 旋转 {rot}°")
            self.drone.rotate_yaw(rot)
            time.sleep(self.drone.servo_sleep)
            self._servo_h_loop()

        # [已禁用预降] print(f"[RescueController] 预降至 {preview_h:.1f}m")
        # [已禁用预降] self.drone.move_to(self.drone.drone.state['x'], self.drone.drone.state['y'], preview_h)
        # [已禁用预降] time.sleep(2)
        offset = float(self.config.get('landing_offset', 0.04))
        self.drone.move_to(self.drone.drone.state['x'] + offset, self.drone.drone.state['y'], self.drone.drone.state['z'])
        self.drone._rotate_gimbal_with_recovery(0)
        landing_state = dict(self.drone.drone.state)
        self.drone.land()
        time.sleep(17)

        # ==================== 阶段 4：地面物资放置 ====================
        print("[RescueController] --- 阶段 4: 小车放置物资 ---")

        # 4a. 通知小车来救援点
        print("[RescueController] 通知小车: 前往救援点取物资并放置")
        if not self.wait_for_car_signal('come_to_target'):
            return False

        # 4b. 等待小车完成物资放置
        print("[RescueController] 等待小车放置物资完成...")
        if not self.wait_for_car_signal('delivery_done'):
            return False

        # 4c. 小车返回初始位置
        print("[RescueController] 通知小车: 返回初始位置")
        if self.car is not None:
            self.car.move_to(0.0, 0.0)

        # ==================== 阶段 5：返航 ====================
        print("[RescueController] --- 阶段 5: 无人机返航 ---")
        self.drone.reset()
        time.sleep(1)
        if not self.drone.takeoff():
            return False
        time.sleep(6)
        # 起飞后实际高度约1.2m，重置state.z确保move_to计算正确的抬升量
        self.drone.drone.state['z'] = 1.2
        for _ in range(3):
            if self.drone.move_to(self.drone.drone.state['x'], self.drone.drone.state['y'], 1.5):
                break
            print("[RescueController] 升至1.5m失败, 重试...")
            time.sleep(2)
        self.drone.drone.state['x'] = landing_state['x']
        self.drone.drone.state['y'] = landing_state['y']
        if abs(rot) > 0.1:
            print(f"[RescueController] 旋转 -{rot}° 恢复朝向")
            self.drone.rotate_yaw(-rot)

        # 先飞 home 校准坐标系
        hx, hy, hz = self.home_point
        print(f"[RescueController] 飞往 home ({hx}, {hy}, {hz})...")
        self.drone.move_to(hx, hy, hz)
        print("[RescueController] home H 伺服对齐...")
        self.drone._rotate_gimbal_with_recovery(-90)
        self._servo_h_loop(save_prefix="home")
        self.drone.move_to(self.drone.drone.state['x'] + float(self.config.get('landing_offset', 0.04)),
                           self.drone.drone.state['y'],
                           self.drone.drone.state['z'])
        self.drone._rotate_gimbal_with_recovery(0)
        print(f"[RescueController] 断言原点 (0, 0, {hz})")
        self.drone.drone.state['x'] = 0.0
        self.drone.drone.state['y'] = 0.0
        self.drone.drone.state['z'] = hz

        print("[RescueController] home点 降落...")
        self.drone.land()
        time.sleep(17)

        print("[RescueController] ====== 完整任务完成 =====")
        return True

    def _fill_unknown_grades(self, results: Dict[str, Dict[str, Any]]) -> None:
        """巡检后按 1×1级 + 1×2级 + 2×3级 原则填补 unknown。"""
        grade_counts = {'1': 0, '2': 0, '3': 0}
        unknown_points: list[str] = []
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
        missing: list[str] = []
        for g in ['1', '2', '3']:
            need = max(0, expected[g] - grade_counts[g])
            missing.extend([g] * need)

        print(f"[RescueController] 查漏补缺: 已知{grade_counts}, "
              f"unknown={len(unknown_points)}个, 需补{missing}")

        for i, point_name in enumerate(unknown_points):
            if i < len(missing):
                fill_grade = missing[i]
                results[point_name]['grade'] = fill_grade
                results[point_name]['raw_label'] = f'filled_{fill_grade}'
                results[point_name]['confidence'] = 0.0
                print(f"[RescueController]   {point_name}: unknown → {fill_grade}级 (推断)")
            else:
                print(f"[RescueController]   {point_name}: unknown 保留 (超出需补数)")

    def _update_rescue_results(self, results: Dict[str, Dict[str, Any]]) -> None:
        for point_name, result in results.items():
            if result["success"]:
                self.rescue_points.set_result(
                    point_name,
                    grade=result["grade"],
                    confidence=result["confidence"],
                    image_path=result["image_path"],
                )
            else:
                print(f"[RescueController] 警告: {point_name} 扫描失败 — {result.get('reason', 'unknown')}")

    def wait_for_car_signal(self, signal_name: str, timeout: Optional[float] = None) -> bool:
        if self.car is None:
            print(f"[RescueController] 小车控制器未设置，默认认为信号已到达: {signal_name}")
            return True
        return self.car.wait_for_signal(signal_name, timeout)

    def _find_waypoint(self, name: str):
        return next((wp for wp in self.drone.waypoints if wp.name == name), None)

    def _select_target_waypoint(self, results: Dict[str, Dict[str, Any]]) -> str | None:
        # 优先级 1: target_grade — 按等级数字匹配，多个时取置信度最高的
        target_grade = self.config.get('target_grade')
        if target_grade is not None:
            candidates = []  # (point_name, result)
            for point_name, result in results.items():
                if result.get('success') and str(result.get('grade', '')).lower() == str(target_grade).lower():
                    candidates.append((point_name, result))
            if candidates:
                best = max(candidates, key=lambda x: x[1].get('confidence', 0))
                print(f"[RescueController] 目标等级{target_grade}: {len(candidates)}个候选 → "
                      f"选 {best[0]} (conf={best[1]['confidence']:.3f})")
                return best[0]
            # 未找到目标等级 → 优先选 unknown
            for point_name, result in results.items():
                if result.get('success') and str(result.get('grade', '')).lower() == 'unknown':
                    print(f"[RescueController] 无等级{target_grade} → 选未知点: {point_name}")
                    return point_name

        # 优先级 2: target_waypoint_name — 直接指定航点名称
        target_name = self.config.get('target_waypoint_name')
        if target_name and target_name in results and results[target_name].get('success'):
            return target_name

        # 优先级 3: target_labels — 按灾害类型名匹配
        target_labels = self.config.get('target_labels', [])
        if isinstance(target_labels, str):
            target_labels = [target_labels]
        target_labels = [str(label).lower() for label in target_labels if label]

        if target_labels:
            for point_name, result in results.items():
                if result.get('success') and str(result.get('raw_label', '')).lower() in target_labels:
                    return point_name

        # 默认：选取第一个成功扫描到的点
        for point_name, result in results.items():
            if result.get('success'):
                return point_name

        return None

    def _write_csv(self) -> None:
        path = os.path.join(self.output_folder, self.csv_filename)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(self.rescue_points.to_csv_rows())

    # ---- 后续扩展接口 ----

    def set_car_controller(self, car) -> None:
        self.car = car
