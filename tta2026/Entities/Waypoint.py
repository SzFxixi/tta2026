from dataclasses import dataclass


@dataclass
class Waypoint:
    name: str
    x: float
    y: float
    z: float
    rotation: float = 0.0
    gimbal_pitch: float = -90.0
    rotate_to: float = 0.0
    rotation_offset: float = 0.0  # 物理旋转时的额外偏移（度），不影响图像识别
