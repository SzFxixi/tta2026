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

    def _write_csv(self) -> None:
        path = os.path.join(self.output_folder, self.csv_filename)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(self.rescue_points.to_csv_rows())

    # ---- 后续扩展接口 ----

    def set_car_controller(self, car) -> None:
        self.car = car
