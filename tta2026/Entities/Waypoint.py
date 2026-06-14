from dataclasses import dataclass


@dataclass
class Waypoint:
    name: str
    x: float
    y: float
    z: float
    rotation: float = 0.0
    gimbal_pitch: float = -90.0
    rotate_to: float = 0.0  # 到达后顺时针旋转角度（度）
