import json
import math
import time
from typing import Any, Dict

import urllib.request

from Utils.MathHelper import MathHelper


class DroneControlClient:
    """无人机运动 SDK — HTTP PUT → PlaneServer (PSDK)，指数退避重试 + taskId + 偏航感知坐标变换。"""

    def __init__(self, config: Dict[str, Any]):
        self.enabled = bool(config.get("enabled", False))
        self.ip = config.get("ip", "192.168.31.100")
        self.port = int(config.get("port", 18080))
        self.base_url = f"http://{self.ip}:{self.port}"
        self.task_id = 0
        self.task_history: Dict[int, Dict[str, Any]] = {}
        self.max_retries = int(config.get("max_retries", 10))
        self.max_backoff = 5.0

        init_pos = config.get("initial_position", {})
        self.state = {
            "x": float(init_pos.get("x", 0.0)),
            "y": float(init_pos.get("y", 0.0)),
            "z": float(init_pos.get("z", 0.0)),
            "yaw": float(init_pos.get("yaw", 0.0)),
        }
        self.taken_off = False

        speed_cfg = config.get("speed", {})
        self.speed_translate = float(speed_cfg.get("translate", 1.0))
        self.speed_rotate = float(speed_cfg.get("rotate", 30.0))

        threshold_cfg = config.get("threshold", {})
        self.threshold_translate = float(threshold_cfg.get("translate", 200))
        self.threshold_rotate = float(threshold_cfg.get("rotate", 200))

    # ------------------------------------------------------------------
    # 底层通信（HTTP PUT → PlaneServer，指数退避重试 + taskId）
    # ------------------------------------------------------------------

    def _send_command(self, endpoint: str, payload: Dict[str, Any]) -> bool:
        if not self.enabled:
            print(f"[DroneControlClient] 模拟: PUT /{endpoint} {payload}")
            return True

        self.task_id += 1
        payload["taskId"] = self.task_id
        self.task_history[self.task_id] = {"endpoint": endpoint, "payload": dict(payload)}
        url = f"{self.base_url}/{endpoint}"
        request_data = json.dumps(payload).encode("utf-8")
        retry_count = 0

        while retry_count < self.max_retries:
            try:
                req = urllib.request.Request(url, data=request_data, method="PUT",
                                             headers={"Content-Type": "application/json"})
                response = urllib.request.urlopen(req, timeout=10)
                text = response.read().decode("utf-8")
                print(f"[DroneControlClient] /{endpoint} → {text}")
                try:
                    result = json.loads(text)
                    if not result.get("isSuccess", False):
                        self.task_id -= 1  # 失败回滚
                        print(f"[DroneControlClient] PlaneServer 拒绝: {result.get('errorMessage', 'unknown')}")
                        return False
                except json.JSONDecodeError:
                    pass
                return True
            except Exception as exc:
                retry_count += 1
                if retry_count >= self.max_retries:
                    self.task_id -= 1  # 最终失败回滚
                    print(f"[DroneControlClient] 致命错误: /{endpoint} 重试 {retry_count} 次后仍失败: {exc}")
                    return False
                delay = min(0.5 * (2 ** retry_count), self.max_backoff)
                print(f"[DroneControlClient] /{endpoint} 失败 (第{retry_count}/{self.max_retries}次): {exc}，{delay:.1f}s 后重试")
                time.sleep(delay)

        return False

    def health_check(self) -> bool:
        """检查 PlaneServer 是否可达（尝试 /Reset）。"""
        try:
            url = f"{self.base_url}/Reset"
            req = urllib.request.Request(url, data=b"{}", method="PUT",
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=3)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # 飞控
    # ------------------------------------------------------------------

    def reset(self) -> bool:
        ok = self._send_command("Reset", {})
        if ok:
            self.task_id = 0
            self.task_history.clear()
        return ok

    def takeoff(self) -> bool:
        if self.taken_off:
            print("[DroneControlClient] 已起飞")
            return True

        # 尝试 PSDK StartTakeoff
        ok = self._send_command("Takeoff", {})
        if ok:
            self.taken_off = True
            self.state["z"] = max(self.state["z"], 1.2)
            return True

        # StartTakeoff 失败 → 用速度指令垂直上升
        print("[DroneControlClient] StartTakeoff 被拒，尝试速度起飞...")
        target_z = 1.5
        dz = target_z - self.state["z"]
        if dz < 0.5:
            dz = 1.5

        # 低速上升，避免触发避障急停
        ascent_speed = 0.5
        duration_ms = int(dz / ascent_speed * 1000)
        ok = self._send_command("Translate", {"x": 0.0, "y": 0.0, "z": ascent_speed, "time": duration_ms})
        if ok:
            self.taken_off = True
            self.state["z"] = target_z
            print(f"[DroneControlClient] 速度起飞成功，到达 {target_z}m")
            time.sleep(1)
            return True

        return False

    def land(self) -> bool:
        if not self.taken_off:
            print("[DroneControlClient] 已着陆")
            return True
        ok = self._send_command("Landing", {})
        if ok:
            self.taken_off = False
            self.state["z"] = 0.0
        return ok

    # ------------------------------------------------------------------
    # 平移（偏航感知 + 速度/阈值）
    # ------------------------------------------------------------------

    def move_to(self, x: float, y: float, z: float) -> bool:
        dx = x - self.state["x"]
        dy = y - self.state["y"]
        dz = z - self.state["z"]
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        if distance < 0.001:
            return True

        yaw_rad = math.radians(self.state["yaw"])
        relative_x, relative_y = MathHelper.rotate_axis(dx, dy, -yaw_rad)

        translate_speed = self.speed_translate
        duration_ms = distance / translate_speed * 1000
        while duration_ms < self.threshold_translate:
            translate_speed /= 2
            duration_ms *= 2

        speed_x, speed_y, speed_z, _ = MathHelper.standardize(relative_x, relative_y, dz, translate_speed)
        duration_ms = int(duration_ms)

        ok = self._send_command("Translate", {"x": speed_x, "y": speed_y, "z": speed_z, "time": duration_ms})
        if ok:
            self.state["x"] = x
            self.state["y"] = y
            self.state["z"] = z
            print(f"[DroneControlClient] 位置: ({x:.2f}, {y:.2f}, {z:.2f})")
        time.sleep(0.5)
        return ok

    # ------------------------------------------------------------------
    # 旋转
    # ------------------------------------------------------------------

    def rotate_yaw(self, angle: float) -> bool:
        if abs(angle) <= 1.0:
            return True

        rotate_speed = self.speed_rotate * MathHelper.sign_of(angle)
        duration_ms = abs(angle) / abs(rotate_speed) * 1000
        while duration_ms < self.threshold_rotate:
            rotate_speed /= 2
            duration_ms *= 2
        duration_ms = int(duration_ms)

        ok = self._send_command("Rotate", {"yawRate": rotate_speed, "time": duration_ms})
        if ok:
            self.state["yaw"] = (self.state["yaw"] + angle) % 360
            print(f"[DroneControlClient] 偏航: {self.state['yaw']:.1f}°")
        time.sleep(0.5)
        return ok

    # ------------------------------------------------------------------
    # 云台
    # ------------------------------------------------------------------

    def rotate_gimbal(self, pitch: float) -> bool:
        ok = self._send_command("RotateGimbal", {"pitch": pitch})
        if ok:
            print(f"[DroneControlClient] 云台: pitch={pitch}°")
        time.sleep(0.5)
        return ok
