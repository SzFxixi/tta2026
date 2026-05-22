from dataclasses import dataclass


@dataclass
class Waypoint:
    name: str
    x: float
    y: float
    z: float
