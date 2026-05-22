import json
import math
import time
from typing import Any, Dict

from Utils.MathHelper import MathHelper


class DroneControlClient:
    """无人机运动 SDK — 支持模拟/HTTP 控制、指数退避重试、偏航感知的坐标系变换。"""

    def __init__(self, config: Dict[str, Any]):
        self.enabled = bool(config.get("enabled", False))
        self.control_url = config.get("control_url", "")

        init_pos = config.get("initial_position", {})
        self.state = {
            "x": float(init_pos.get("x", 0.0)),
            "y": float(init_pos.get("y", 0.0)),
            "z": float(init_pos.get("z", 0.0)),
            "yaw": float(init_pos.get("yaw", 0.0)),  # 偏航角，度，0=正前方
        }
        self.taken_off = False

        # 速度和最小时间阈值
        speed_cfg = config.get("speed", {})
        self.speed_translate = float(speed_cfg.get("translate", 1.0))  # m/s
        self.speed_rotate = float(speed_cfg.get("rotate", 30.0))  # 度/s

        threshold_cfg = config.get("threshold", {})
        self.threshold_translate = float(threshold_cfg.get("translate", 200))  # ms 最小移动时间
        self.threshold_rotate = float(threshold_cfg.get("rotate", 200))  # ms 最小旋转时间

        self.max_retries = int(config.get("max_retries", 10))

    # ------------------------------------------------------------------
    # 底层通信（指数退避重试，超限返回 False）
    # ------------------------------------------------------------------

    def _send_command(self, command: str, payload: Dict[str, Any]) -> bool:
        if not self.enabled:
            print(f"[DroneControlClient] 模拟: {command} {payload}")
            return True

        if not self.control_url:
            print("[DroneControlClient] 未配置 control_url，转为模拟模式")
            return True

        import urllib.request

        request_data = json.dumps({"command": command, "payload": payload}).encode("utf-8")
        retry_count = 0
        max_delay = 5.0

        while retry_count < self.max_retries:
            try:
                req = urllib.request.Request(
                    self.control_url,
                    data=request_data,
                    headers={"Content-Type": "application/json"},
                )
                response = urllib.request.urlopen(req, timeout=10)
                text = response.read().decode("utf-8")
                print(f"[DroneControlClient] 返回: {text}")
                return True
            except Exception as exc:
                retry_count += 1
                if retry_count >= self.max_retries:
                    print(f"[DroneControlClient] 致命错误: {command} 重试 {retry_count} 次后仍失败，已放弃。最后错误: {exc}")
                    return False
                delay = min(0.5 * (2 ** retry_count), max_delay)
                print(f"[DroneControlClient] 发送失败 (第{retry_count}/{self.max_retries}次): {exc}，{delay:.1f}s 后重试")
                time.sleep(delay)

        return False

    # ------------------------------------------------------------------
    # 基本飞控
    # ------------------------------------------------------------------

    def reset(self) -> bool:
        return self._send_command("reset", {})

    def takeoff(self) -> bool:
        if self.taken_off:
            print("[DroneControlClient] 已起飞")
            return True

        success = self._send_command("takeoff", {})
        if success:
            self.taken_off = True
            self.state["z"] = max(self.state["z"], 1.2)
        return success

    def land(self) -> bool:
        if not self.taken_off:
            print("[DroneControlClient] 已着陆")
            return True

        success = self._send_command("land", {})
        if success:
            self.taken_off = False
            self.state["z"] = 0.0
        return success

    # ------------------------------------------------------------------
    # 平移（偏航感知）
    # ------------------------------------------------------------------

    def move_to(self, x: float, y: float, z: float) -> bool:
        """移动到全局坐标 (x, y, z)，内部处理偏航旋转和速度/阈值。"""
        dx = x - self.state["x"]
        dy = y - self.state["y"]
        dz = z - self.state["z"]

        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        if distance < 0.001:
            return True

        # 将全局偏移转换到无人机坐标系（考虑偏航角）
        yaw_rad = math.radians(self.state["yaw"])
        relative_x, relative_y = MathHelper.rotate_axis(dx, dy, -yaw_rad)

        # 速度和时间阈值
        translate_speed = self.speed_translate
        duration_ms = distance / translate_speed * 1000
        while duration_ms < self.threshold_translate:
            translate_speed /= 2
            duration_ms *= 2

        # 标准化速度分量
        speed_x, speed_y, speed_z, _ = MathHelper.standardize(
            relative_x, relative_y, dz, translate_speed
        )
        duration_ms = int(duration_ms)

        success = self._send_command("move", {
            "dx": relative_x,
            "dy": relative_y,
            "dz": dz,
            "speed_x": speed_x,
            "speed_y": speed_y,
            "speed_z": speed_z,
            "duration": duration_ms,
        })

        if success:
            self.state["x"] = x
            self.state["y"] = y
            self.state["z"] = z
            print(f"[DroneControlClient] 位置: ({x:.2f}, {y:.2f}, {z:.2f})")

        time.sleep(0.5)  # 补偿图传延迟
        return success

    # ------------------------------------------------------------------
    # 旋转（速度/阈值控制）
    # ------------------------------------------------------------------

    def rotate_yaw(self, angle: float) -> bool:
        """旋转偏航角 angle 度（正值=顺时针从上方看）。"""
        if abs(angle) <= 1.0:
            return True

        rotate_speed = self.speed_rotate * MathHelper.sign_of(angle)
        duration_ms = abs(angle) / abs(rotate_speed) * 1000

        while duration_ms < self.threshold_rotate:
            rotate_speed /= 2
            duration_ms *= 2

        duration_ms = int(duration_ms)

        success = self._send_command("rotate", {
            "rate": rotate_speed,
            "duration": duration_ms,
        })

        if success:
            self.state["yaw"] = (self.state["yaw"] + angle) % 360
            print(f"[DroneControlClient] 偏航: {self.state['yaw']:.1f}°")

        time.sleep(0.5)  # 补偿图传延迟
        return success

    # ------------------------------------------------------------------
    # 云台
    # ------------------------------------------------------------------

    def rotate_gimbal(self, pitch: float) -> bool:
        """旋转云台俯仰角 pitch 度。"""
        success = self._send_command("gimbal", {"pitch": pitch})
        if success:
            print(f"[DroneControlClient] 云台: pitch={pitch}°")
        return success
