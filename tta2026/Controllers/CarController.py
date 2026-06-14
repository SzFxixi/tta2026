from typing import Any, Dict, Optional

from Clients.CarClient import CarController as CarClient


class CarController:
    """小车控制器 — 包装真实底盘 HTTP 客户端 (CarClient)，
    提供统一接口: move_to / rotate / grasp / release / wait_for_signal。"""

    def __init__(self, config: Dict[str, Any]):
        car_config = config.get("car", {})
        self.enabled = bool(car_config.get("enabled", False))
        self.client: Optional[CarClient] = None

        if self.enabled:
            ip = car_config.get("ip", "10.203.94.227")
            port = int(car_config.get("port", 5000))
            self.client = CarClient(ip=ip, port=port)
            print(f"[CarController] 已连接小车: {ip}:{port}")

    # ── 底盘移动 ──

    def move_to(self, x: float, y: float) -> bool:
        if self.client is None:
            print(f"[CarController] 未启用, 模拟移动: ({x:.2f}, {y:.2f})")
            return True
        print(f"[CarController] 小车移动到: ({x:.2f}, {y:.2f})")
        return self.client.Move(x, y)

    def rotate(self, degrees: float) -> bool:
        if self.client is None:
            print(f"[CarController] 未启用, 模拟旋转: {degrees}°")
            return True
        return self.client.Circle(degrees)

    # ── 机械臂（桩，待硬件就绪）──

    def grasp(self) -> bool:
        if self.client is None:
            print("[CarController] 模拟: 抓取物资")
            return True
        print("[CarController] 抓取物资")
        return True

    def release(self) -> bool:
        if self.client is None:
            print("[CarController] 模拟: 释放物资")
            return True
        print("[CarController] 释放物资")
        return True

    # ── 信号同步 ──

    def wait_for_signal(self, signal_name: str, timeout: float | None = None) -> bool:
        if self.client is None:
            print(f"[CarController] 小车未接入, 默认通过: {signal_name}")
            return True
        print(f"[CarController] 等待小车信号: {signal_name}")
        return True

    # ── 生命周期 ──

    def shutdown(self) -> None:
        if self.client is not None:
            self.client.Shutdown()
            print("[CarController] 小车已关闭")
