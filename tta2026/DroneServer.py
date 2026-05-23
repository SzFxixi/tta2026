"""
无人机 HTTP 控制服务端 — 部署在树莓派上。

接收 PC 地面站发来的飞控命令，转交给硬件抽象层执行。

启动方式:
    python DroneServer.py --port 5000          # 模拟模式 (PC 测试)
    python DroneServer.py --port 5000 --real   # 真实硬件模式 (树莓派)

API:
    POST /drone_command  {"command": "...", "payload": {...}}  → {"ok": true, ...}
    GET  /drone_status                                         → {"armed": ..., "x": ..., ...}
    GET  /health                                               → {"status": "ok"}
"""

import argparse
import json
import sys
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler

from Controllers.DroneHardware import DroneHardwareBase, MockDroneHardware


# ── 硬件工厂 ────────────────────────────────────────────────

def create_hardware(real: bool) -> DroneHardwareBase:
    if real:
        # TODO: 替换为真实硬件驱动，例如:
        #   from Controllers.PWMDroneHardware import PWMDroneHardware
        #   return PWMDroneHardware()
        raise NotImplementedError(
            "真实硬件驱动尚未实现。请根据你的飞控类型补全。\n"
            "  如果用的是 PWM/GPIO 直驱 → 实现 PWMDroneHardware\n"
            "  如果用的是 Pixhawk     → 实现 MavlinkDroneHardware\n"
            "  如果用的是 DJI SDK      → 实现 DJIDroneHardware"
        )
    return MockDroneHardware()


# ── 命令路由 ─────────────────────────────────────────────────

class CommandHandler:
    """将 HTTP 命令分发到硬件操作。"""

    def __init__(self, hw: DroneHardwareBase):
        self.hw = hw
        self._taken_off = False

    def handle(self, command: str, payload: dict) -> dict:
        method = getattr(self, f'_cmd_{command}', None)
        if method is None:
            return {"ok": False, "error": f"未知命令: {command}"}

        try:
            result = method(payload)
            result["ok"] = True
            return result
        except Exception as e:
            traceback.print_exc()
            return {"ok": False, "error": str(e)}

    def _cmd_reset(self, p: dict) -> dict:
        self.hw.disarm()
        self._taken_off = False
        return {"message": "已复位"}

    def _cmd_takeoff(self, p: dict) -> dict:
        if self._taken_off:
            return {"message": "已起飞，跳过"}
        self.hw.connect()
        self.hw.arm()
        if not self.hw.takeoff():
            return {"ok": False, "error": "起飞失败"}
        self._taken_off = True
        return {"message": "起飞成功"}

    def _cmd_land(self, p: dict) -> dict:
        if not self._taken_off:
            return {"message": "已着陆，跳过"}
        self.hw.land()
        self.hw.disarm()
        self._taken_off = False
        return {"message": "着陆成功"}

    def _cmd_move(self, p: dict) -> dict:
        return {
            "message": self.hw.move_by(
                dx=float(p.get("dx", 0)),
                dy=float(p.get("dy", 0)),
                dz=float(p.get("dz", 0)),
                speed_x=float(p.get("speed_x", 1.0)),
                speed_y=float(p.get("speed_y", 1.0)),
                speed_z=float(p.get("speed_z", 1.0)),
                duration_ms=int(p.get("duration", 1000)),
            )
        }

    def _cmd_rotate(self, p: dict) -> dict:
        return {
            "message": self.hw.rotate(
                rate_dps=float(p.get("rate", 30)),
                duration_ms=int(p.get("duration", 1000)),
            )
        }

    def _cmd_gimbal(self, p: dict) -> dict:
        return {
            "message": self.hw.set_gimbal(
                pitch_deg=float(p.get("pitch", 0)),
                roll_deg=float(p.get("roll", 0)),
                yaw_deg=float(p.get("yaw", 0)),
            )
        }

    def status(self) -> dict:
        return self.hw.get_state()


# ── HTTP 服务器 ──────────────────────────────────────────────

class DroneHTTPHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        print(f"[DroneServer] {args[0]}")

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/drone_status":
            self._send_json(self.server.cmd_handler.status())
        elif self.path == "/health":
            self._send_json({"status": "ok"})
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/drone_command":
            self._send_json({"error": "not found"}, 404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        command = body.get("command", "")
        payload = body.get("payload", {})

        print(f"[DroneServer] 收到命令: {command} {payload}")
        result = self.server.cmd_handler.handle(command, payload)
        self._send_json(result)


# ── 入口 ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="无人机 HTTP 控制服务端")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--real", action="store_true", help="使用真实硬件 (默认模拟)")
    args = parser.parse_args()

    hw = create_hardware(args.real)
    handler = CommandHandler(hw)

    server = HTTPServer(("0.0.0.0", args.port), DroneHTTPHandler)
    server.cmd_handler = handler  # type: ignore

    print(f"[DroneServer] 启动在 0.0.0.0:{args.port}")
    print(f"[DroneServer] 模式: {'真实硬件' if args.real else '模拟 (Mock)'}")
    print(f"[DroneServer] 等待 PC 地面站连接...")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[DroneServer] 服务停止")
        hw.disconnect()
        server.shutdown()


if __name__ == "__main__":
    main()
