from dataclasses import dataclass


@dataclass
class Waypoint:
    name: str
    x: float
    y: float
    z: float
    rotation: float = 0.0
    gimbal_pitch: float = -90.0  # 云台俯仰角（度）
