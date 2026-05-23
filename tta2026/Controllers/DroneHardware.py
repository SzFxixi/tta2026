"""
无人机硬件抽象层。

定义所有飞控操作的统一接口。当前包含：
- MockDroneHardware: 模拟实现，用于地面站测试
- 后续扩展: PWMDroneHardware (GPIO 直驱), MavlinkDroneHardware (Pixhawk)
"""

import time
from abc import ABC, abstractmethod


class DroneHardwareBase(ABC):
    """飞控硬件抽象基类。"""

    @abstractmethod
    def connect(self) -> bool:
        """建立硬件连接，返回是否成功。"""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """断开硬件连接。"""
        ...

    @abstractmethod
    def arm(self) -> bool:
        """解锁电机。"""
        ...

    @abstractmethod
    def disarm(self) -> bool:
        """锁定电机。"""
        ...

    @abstractmethod
    def takeoff(self, height_m: float = 1.5) -> bool:
        """起飞到指定高度 (米)。"""
        ...

    @abstractmethod
    def land(self) -> bool:
        """降落并锁定。"""
        ...

    @abstractmethod
    def move_by(self, dx: float, dy: float, dz: float,
                speed_x: float, speed_y: float, speed_z: float,
                duration_ms: int) -> bool:
        """
        增量移动。坐标系: x=前, y=左, z=上 (相对无人机机头方向)。

        参数:
            dx, dy, dz: 位移量 (米)
            speed_x, speed_y, speed_z: 速度分量 (m/s)
            duration_ms: 持续时长 (毫秒)，0 表示由硬件自行决定
        """
        ...

    @abstractmethod
    def rotate(self, rate_dps: float, duration_ms: int) -> bool:
        """
        偏航旋转。正值 = 顺时针 (从上方看)。

        参数:
            rate_dps: 旋转速率 (度/秒)
            duration_ms: 持续时长 (毫秒)
        """
        ...

    @abstractmethod
    def set_gimbal(self, pitch_deg: float, roll_deg: float = 0.0,
                   yaw_deg: float = 0.0) -> bool:
        """设置云台角度 (度)。"""
        ...

    @abstractmethod
    def get_state(self) -> dict:
        """返回当前状态: {armed, flying, x, y, z, yaw, battery, ...}。"""
        ...


class MockDroneHardware(DroneHardwareBase):
    """模拟飞控 — 仅打印日志，用于 PC 端开发调试。"""

    def __init__(self):
        self._armed = False
        self._flying = False
        self._x = 0.0
        self._y = 0.0
        self._z = 0.0
        self._yaw = 0.0

    def connect(self) -> bool:
        print("[MockHW] 连接成功")
        return True

    def disconnect(self) -> None:
        print("[MockHW] 已断开")

    def arm(self) -> bool:
        print("[MockHW] 解锁电机")
        self._armed = True
        return True

    def disarm(self) -> bool:
        print("[MockHW] 锁定电机")
        self._armed = False
        self._flying = False
        return True

    def takeoff(self, height_m: float = 1.5) -> bool:
        print(f"[MockHW] 起飞 → {height_m}m")
        self._armed = True
        self._flying = True
        self._z = height_m
        time.sleep(0.5)
        return True

    def land(self) -> bool:
        print(f"[MockHW] 降落 (从 z={self._z:.1f}m)")
        self._flying = False
        self._z = 0.0
        time.sleep(0.5)
        return True

    def move_by(self, dx: float, dy: float, dz: float,
                speed_x: float, speed_y: float, speed_z: float,
                duration_ms: int) -> bool:
        dur_s = duration_ms / 1000.0
        self._x += dx
        self._y += dy
        self._z += dz
        print(f"[MockHW] 移动 Δ=({dx:.2f},{dy:.2f},{dz:.2f}) "
              f"速度=({speed_x:.2f},{speed_y:.2f},{speed_z:.2f}) 持续={dur_s:.1f}s")
        time.sleep(dur_s)
        return True

    def rotate(self, rate_dps: float, duration_ms: int) -> bool:
        dur_s = duration_ms / 1000.0
        angle = rate_dps * dur_s
        self._yaw = (self._yaw + angle) % 360
        print(f"[MockHW] 旋转 {rate_dps}°/s × {dur_s:.1f}s → 偏航={self._yaw:.0f}°")
        time.sleep(dur_s)
        return True

    def set_gimbal(self, pitch_deg: float, roll_deg: float = 0.0,
                   yaw_deg: float = 0.0) -> bool:
        print(f"[MockHW] 云台 pitch={pitch_deg}° roll={roll_deg}° yaw={yaw_deg}°")
        return True

    def get_state(self) -> dict:
        return {
            "armed": self._armed,
            "flying": self._flying,
            "x": self._x,
            "y": self._y,
            "z": self._z,
            "yaw": self._yaw,
            "battery": 99.0,
            "gps": 0,
        }
