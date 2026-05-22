from typing import Any, Dict, Optional

from Clients.CarControlClient import CarControlClient
from Utils.JsonHelper import JsonHelper


class CarController:
    """小车控制器 — 接口预留，后续实现抓取、运输、物资放置等逻辑。"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client: Optional[CarControlClient] = None

        car_config = config.get("car", {})
        if car_config.get("enabled", False):
            self.client = CarControlClient(car_config)

    def move_to(self, x: float, y: float) -> bool:
        if self.client is None:
            print("[CarController] 小车未启用")
            return False
        return self.client.move_to(x, y)

    def grasp(self) -> bool:
        if self.client is None:
            return False
        return self.client.grasp()

    def release(self) -> bool:
        if self.client is None:
            return False
        return self.client.release()

    def shutdown(self) -> None:
        pass
