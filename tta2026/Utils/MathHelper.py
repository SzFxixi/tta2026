import math


class MathHelper:
    """坐标变换和基础数学工具。"""

    @staticmethod
    def rotate_axis(x: float, y: float, angle: float) -> tuple[float, float]:
        """旋转坐标系。angle 弧度制，逆时针为正。"""
        x_new = x * math.cos(angle) - y * math.sin(angle)
        y_new = x * math.sin(angle) + y * math.cos(angle)
        return x_new, y_new

    @staticmethod
    def sign_of(x: float) -> int:
        if x > 0:
            return 1
        elif x < 0:
            return -1
        return 0

    @staticmethod
    def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    @staticmethod
    def standardize(x: float, y: float, z: float, standard: float = 1.0) -> tuple[float, float, float, float]:
        """将向量标准化到指定长度，返回 (x, y, z, 原长度)。"""
        length = (x ** 2 + y ** 2 + z ** 2) ** 0.5
        if abs(standard) < 1e-7:
            return 0.0, 0.0, 0.0, length
        if length < 1e-7:
            return 0.0, 0.0, 0.0, 0.0
        rate = standard / length
        return x * rate, y * rate, z * rate, length
