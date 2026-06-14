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

    def execute_scan_mission(self) -> bool:
        """执行一次完整的无人机巡检扫描任务。返回是否全部扫描成功。"""
        print("[RescueController] 开始无人机巡检任务...")

        # 1. 执行扫描
        results = self.drone.scan_waypoints()

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

    def execute_delivery_mission(self) -> bool:
        """完整任务流程：
        无人机巡检 → 装货 → 投送 → 物资放置 → 返航。
        与小车通过 CarController.wait_for_signal 同步。"""
        print("[RescueController] ====== 完整任务开始 ======")

        # ==================== 阶段 1：巡检 ====================
        print("[RescueController] --- 阶段 1: 无人机巡检 ---")
        results = self.drone.scan_waypoints()
        self._update_rescue_results(results)
        self._write_csv()
        # scan_waypoints 末尾已在装货区旋转 180° 并降落

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

        if not self.drone.takeoff():
            return False

        # 撤销装货区着陆时的 180° 旋转，恢复起始朝向
        print("[RescueController] 旋转 180° 恢复起始朝向")
        self.drone.rotate_yaw(180)

        # 回到原点对齐坐标系
        return_altitude = float(self.config.get('return_altitude', 1.2))
        print(f"[RescueController] 返回原点 (0, 0, {return_altitude}) 校准坐标系")
        if not self.drone.move_to(0.0, 0.0, return_altitude):
            return False

        # 飞往目标救援点
        target_waypoint = self._find_waypoint(target_name)
        if target_waypoint is None:
            return False
        print(f"[RescueController] 飞往目标: {target_waypoint.name} ({target_waypoint.x}, {target_waypoint.y}, {target_waypoint.z})")
        if not self.drone.move_to(target_waypoint.x, target_waypoint.y, target_waypoint.z):
            return False

        # 目标点 H 对齐后降落
        print("[RescueController] 目标点 H 对齐并降落")
        self.drone._rotate_gimbal_with_recovery(-90)
        for attempt in range(3):
            frame = self.drone._capture_fresh_frame(settle=3.0)
            if frame is None:
                continue
            all_det = self.drone.detect_all(frame)
            if all_det['h_candidate'] is not None:
                self.drone._servo_toward_h(all_det['h_candidate']['box'], frame.shape)
                break
        self.drone._rotate_gimbal_with_recovery(0)
        if not self.drone.land():
            return False

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
        if not self.drone.takeoff():
            return False
        if not self.drone.move_to(0.0, 0.0, return_altitude):
            return False
        if not self.drone.land():
            return False

        print("[RescueController] ====== 完整任务完成 ======")
        return True

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

    def wait_for_car_signal(self, signal_name: str, timeout: float | None = None) -> bool:
        if self.car is None:
            print(f"[RescueController] 小车控制器未设置，默认认为信号已到达: {signal_name}")
            return True
        return self.car.wait_for_signal(signal_name, timeout)

    def _find_waypoint(self, name: str):
        return next((wp for wp in self.drone.waypoints if wp.name == name), None)

    def _select_target_waypoint(self, results: Dict[str, Dict[str, Any]]) -> str | None:
        # 优先级 1: target_grade — 按等级数字("1"/"2"/"3")匹配
        target_grade = self.config.get('target_grade')
        if target_grade is not None:
            for point_name, result in results.items():
                if result.get('success') and str(result.get('grade', '')).lower() == str(target_grade).lower():
                    return point_name

        # 优先级 2: target_waypoint_name — 直接指定航点名称
        target_name = self.config.get('target_waypoint_name')
        if target_name and target_name in results:
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
