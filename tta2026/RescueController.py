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
        """执行完整任务：扫描、等待小车信号、前往目标点着陆、再等待返回信号并回到原点。"""
        print("[RescueController] 开始完整交付任务...")

        results = self.drone.scan_waypoints()
        self._update_rescue_results(results)
        self._write_csv()

        target_name = self._select_target_waypoint(results)
        if target_name is None:
            print("[RescueController] 未能确定目标航点，无法继续执行交付任务")
            return False
        print(f"[RescueController] 目标航点已选定: {target_name}")

        print("[RescueController] 已在降落区等待小车信号，准备前往目标点")
        if not self.wait_for_car_signal('go_to_target'):
            print("[RescueController] 等待小车信号前往目标超时或失败")
            return False

        if not self.drone.takeoff():
            print("[RescueController] 起飞失败，无法前往目标点")
            return False

        target_waypoint = self._find_waypoint(target_name)
        if target_waypoint is None:
            print(f"[RescueController] 未找到对应的航点数据: {target_name}")
            return False

        print(f"[RescueController] 起飞前往目标航点 {target_waypoint.name} ({target_waypoint.x}, {target_waypoint.y}, {target_waypoint.z})")
        if not self.drone.move_to(target_waypoint.x, target_waypoint.y, target_waypoint.z):
            print(f"[RescueController] 飞往目标航点 {target_waypoint.name} 失败")
            return False

        print(f"[RescueController] 已到达目标航点 {target_waypoint.name}，执行降落")
        if not self.drone.land():
            print("[RescueController] 目标点降落失败")
            return False

        print("[RescueController] 目标点已着陆，等待小车信号返回原点")
        if not self.wait_for_car_signal('return_home'):
            print("[RescueController] 等待小车信号返回原点超时或失败")
            return False

        print("[RescueController] 收到返回信号，准备回到原点")
        if not self.drone.takeoff():
            print("[RescueController] 起飞失败，无法返回原点")
            return False

        return_altitude = float(self.config.get('return_altitude', 1.2))
        if not self.drone.move_to(0.0, 0.0, return_altitude):
            print("[RescueController] 返回原点飞行失败")
            return False

        print("[RescueController] 已到达原点，准备降落")
        if not self.drone.land():
            print("[RescueController] 原点降落失败")
            return False

        print("[RescueController] 完整任务执行完成")
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
        target_name = self.config.get('target_waypoint_name')
        if target_name and target_name in results:
            return target_name

        target_labels = self.config.get('target_labels', [])
        if isinstance(target_labels, str):
            target_labels = [target_labels]
        target_labels = [str(label).lower() for label in target_labels if label]

        if target_labels:
            for point_name, result in results.items():
                if result.get('success') and str(result.get('raw_label', '')).lower() in target_labels:
                    return point_name

        target_grade = self.config.get('target_grade')
        if target_grade is not None:
            for point_name, result in results.items():
                if result.get('success') and str(result.get('grade', '')).lower() == str(target_grade).lower():
                    return point_name

        # 默认选取第一个成功扫描到的点
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
