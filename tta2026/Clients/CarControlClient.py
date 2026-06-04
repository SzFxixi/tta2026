from typing import Any, Dict


class CarControlClient:
    """小车底层控制客户端 — 接口预留，后续填充 HTTP/串口实现。"""

    def __init__(self, config: Dict[str, Any]):
        self.enabled = bool(config.get("enabled", False))
        self.control_url = config.get("control_url", "")

    def move_to(self, x: float, y: float) -> bool:
        if not self.enabled:
            print(f"[CarControlClient] 模拟: move_to({x}, {y})")
            return True
        raise NotImplementedError("小车实际控制尚未实现")

    def grasp(self) -> bool:
        if not self.enabled:
            print("[CarControlClient] 模拟: grasp()")
            return True
        raise NotImplementedError("抓取尚未实现")

    def release(self) -> bool:
        if not self.enabled:
            print("[CarControlClient] 模拟: release()")
            return True
        raise NotImplementedError("释放尚未实现")

    def wait_for_signal(self, signal_name: str, timeout: float | None = None) -> bool:
        """等待小车发来的信号。实现时只需在这里对接实际通信接口。"""
        if not self.enabled:
            print(f"[CarControlClient] 模拟: wait_for_signal({signal_name}, timeout={timeout})")
            return True
        raise NotImplementedError("等待小车信号尚未实现")
