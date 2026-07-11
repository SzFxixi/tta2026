#!/usr/bin/env python3
# coding=utf-8
"""
小车 + 机械臂 联调脚本（重写版）
================================
- /go 只做导航+correct（action_for_each_target.py 已简化为仅 correct）
- rotate / move_rel / 机械臂动作 由本脚本统一调度
- 取货区：中间找 → 向左找 → 向右两次找 → 找到就抓
- 机械臂用策略层：VisualServoGrasp.execute_grasp() / PlacingStrategy.place_object()
- 小车底盘直接调 CarControlServiceFlask（/Circle /MoveRelative），自动同步 TaskId

流程: 取货区(点5) → 装货区(点6) → 救援点(点N) → 返回(点7)

运行（在小车端）:
    bash start.sh
    python3 car_arm_integration.py --arm-port /dev/ttyACM1 --camera-id 11
    python3 car_arm_integration.py --skip-arm          # 仅导航
"""

import sys
import os
import time
import subprocess
import requests
import signal

CAR_DIR = os.path.dirname(os.path.abspath(__file__))
# car 和 lerobot 都在 lsx10 下（同级目录）
LSX10_DIR = os.path.dirname(CAR_DIR)
LEROBOT_DIR = os.path.join(LSX10_DIR, "lerobot")
VGS_DIR = os.path.join(LEROBOT_DIR, "visual_grasping_system")
sys.path.insert(0, VGS_DIR)

# 救援点参数（从原 action_for_each_target.py 提取）
RESCUE_PARAMS = {
    1: {"rotate": 90,  "move_rel": (0, -0.4), "rotate_back": -90},
    2: {"rotate": 90,  "move_rel": (0, -0.2), "rotate_back": -90},
    3: {"rotate": -90, "move_rel": (0, 0.2),  "rotate_back": 90},
    4: {"rotate": -90, "move_rel": (0, -0.2), "rotate_back": 90},
}

# 取货区搜索序列：(标签, dx, dy)
# 方向和原 adjust_x 一致：左=Y+0.2, 右=Y-0.2
# 中间找 → 向左0.2 → 向右0.4（从左位到右位）
PICKUP_SEARCH = [
    ("中间",     0,  0),
    ("向左",     0,  0.2),
    ("向右两次", 0, -0.4),
]


# ═══════════════════════════════════════════════
#  CarControlServiceFlask 客户端（直接调底盘）
# ═══════════════════════════════════════════════

class CarFlaskClient:
    """直接调 CarControlServiceFlask，用试探+重试自动同步 TaskId"""

    def __init__(self, base_url):
        self.base_url = base_url
        self.tid = 1
        self.sess = requests.Session()

    def _post(self, endpoint, payload, timeout=30):
        for _ in range(3):
            body = {**payload, "TaskId": self.tid}
            try:
                r = self.sess.post(f"{self.base_url}/{endpoint}", json=body, timeout=timeout)
                resp = r.json()
                if resp.get("isSuccess"):
                    self.tid += 1
                    return True, resp
                if "expectedTaskId" in resp:
                    self.tid = resp["expectedTaskId"]
                    continue
                return False, resp
            except Exception as e:
                return False, {"error": str(e)}
        return False, {"error": "TaskId 同步失败"}

    def circle(self, deg):
        """旋转（度数，和原 action 序列一致：90/180/-90）"""
        return self._post("Circle", {"rad_z": deg})

    def move_relative(self, dx, dy):
        """相对平移"""
        return self._post("MoveRelative", {"delta_x": dx, "delta_y": dy})


# ═══════════════════════════════════════════════
#  run.py --external 客户端
# ═══════════════════════════════════════════════

class RunClient:
    def __init__(self, url):
        self.url = url

    def go(self, point):
        try:
            r = requests.post(f"{self.url}/go", json={"point": point}, timeout=5)
            return r.status_code == 200
        except Exception as e:
            print(f"[NAV] /go 异常: {e}")
            return False

    def busy(self):
        try:
            r = requests.get(f"{self.url}/status", timeout=3)
            return r.json().get("busy", False)
        except Exception:
            return False

    def _wait_busy_true(self, timeout=10):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.busy():
                return True
            time.sleep(0.2)
        return False

    def _wait_idle(self, timeout=180):
        t0 = time.time()
        false_since = None
        while time.time() - t0 < timeout:
            if not self.busy():
                if false_since is None:
                    false_since = time.time()
                elif time.time() - false_since > 2.0:
                    return True
            else:
                false_since = None
            time.sleep(0.5)
        return False

    def go_and_wait(self, point, timeout=180):
        """导航+correct，等完成返回 True"""
        if not self.go(point):
            return False
        self._wait_busy_true(timeout=10)
        return self._wait_idle(timeout=timeout)


# ═══════════════════════════════════════════════
#  联调主类
# ═══════════════════════════════════════════════

class CarArmIntegration:
    def __init__(self, arm_port, camera_id, rescue_point, skip_arm=False,
                 car_flask_url="http://192.168.43.8:5000",
                 run_url="http://localhost:6000"):
        self.arm_port = arm_port
        self.camera_id = camera_id
        self.rescue_point = rescue_point
        self.skip_arm = skip_arm
        self.car = CarFlaskClient(car_flask_url)
        self.run = RunClient(run_url)
        self.arm = None
        self.camera = None
        self.flow = None
        self.run_proc = None

    # ── 启动 ──

    def start_run(self):
        run_script = os.path.join(CAR_DIR, "controllers", "run.py")
        self.run_proc = subprocess.Popen(
            [sys.executable, run_script, "--external"],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        print("[INIT] run.py --external 启动中...")
        for _ in range(30):
            try:
                r = requests.get(f"{self.run.url}/status", timeout=2)
                if r.status_code == 200:
                    print("[INIT] run.py 就绪")
                    return True
            except Exception:
                pass
            time.sleep(2)
        print("[ERROR] run.py 启动超时")
        return False

    def init_arm(self):
        """完全复用 competition_flow.py main() 的初始化代码"""
        if self.skip_arm:
            print("[INIT] 跳过机械臂")
            return
        from soarm101_sdk_urdf import SOARM101Controller
        from wrist_camera import WristCamera
        from competition_flow import (
            CompetitionFlow, CONFIG_DIR, _load_sys_cfg
        )

        # 读取 system_config.yaml（和 competition_flow.py 完全一致）
        sys_cfg = _load_sys_cfg()
        arm_cfg = sys_cfg.get('arm', {})
        cam_cfg = sys_cfg.get('camera', {})

        port = self.arm_port or arm_cfg.get('port', 'COM18')
        camera_id = self.camera_id or cam_cfg.get('camera_id', 1)

        urdf_rel = arm_cfg.get(
            'urdf_path', '../SO-ARM100/Simulation/SO101/so101_new_calib.urdf'
        )
        urdf_path = os.path.join(CONFIG_DIR, urdf_rel)
        if not os.path.exists(urdf_path):
            urdf_path = None
            print("[WARN] URDF未找到")

        print("=" * 60)
        print("比赛自动化流程")
        print("=" * 60)

        self.arm = SOARM101Controller(port, urdf_path=urdf_path)
        self.camera = WristCamera(camera_id=camera_id)

        if not self.arm.connect():
            print("[ERROR] 机械臂连接失败")
            self.arm = None
            return

        if not self.camera.is_ready():
            print("[WARN] 摄像头未就绪")

        self.flow = CompetitionFlow(self.arm, self.camera)
        self.flow.show_config()
        print("[INIT] 机械臂就绪")

    # ── 拓展坞 USB 重连（解决 ROS/LiDAR 启动后串口断开问题）──

    def recover_arm(self):
        """start_run() 后重连机械臂，应对拓展坞 USB 总线重置"""
        if self.skip_arm or self.arm is None:
            return False
        print("[ARM] 小车初始化完成，重连机械臂（拓展坞 USB 可能被重置）...")

        # 先断开
        try:
            self.arm.disconnect()
        except Exception:
            pass
        time.sleep(2)

        # 等串口设备重新出现
        for _ in range(15):
            if os.path.exists(self.arm_port):
                break
            print(f"  [ARM] 等待串口 {self.arm_port} ...")
            time.sleep(1)

        if not os.path.exists(self.arm_port):
            print(f"[ERROR] 串口 {self.arm_port} 未恢复")
            self.arm = None
            return False

        # 重连
        if not self.arm.connect():
            print("[ERROR] 机械臂重连失败")
            self.arm = None
            return False

        # 确保扭矩
        try:
            for i in range(6):
                self.arm.bus.enable_torque(i + 1, True)
            self.arm._torque_on = True
            print("[ARM] 扭矩已重新使能")
        except Exception as e:
            print(f"[ARM] 拉扭矩异常: {e}")

        # 重建 CompetitionFlow（因为 arm 内部状态已刷新）
        from competition_flow import CompetitionFlow
        self.flow = CompetitionFlow(self.arm, self.camera)
        self.flow.show_config()
        print("[ARM] 机械臂重连完成")
        return True

    # ── 机械臂辅助 ──

    def _to_travel_pose(self):
        """切换到安全移动姿态（小车移动前调用）"""
        if not self.flow or self.flow.travel_pose is None:
            return
        self.flow._go_to_pose_keep_gripper(self.flow.travel_pose, "安全移动姿态")

    def _from_travel_pose(self, target_pose, label):
        """从安全姿态切换回工作姿态"""
        if not self.flow or target_pose is None:
            return
        self.flow._go_to_pose_keep_gripper(target_pose, label)

    def _go_pose(self, pose, label):
        if self.flow and pose:
            self.flow._go_to_pose_keep_gripper(pose, label)

    # ── Phase 1: 取货区 ──

    def phase1_pickup(self):
        """取货区：单姿态配合小车左右移动搜索"""
        print("\n" + "=" * 60)
        print("  [Phase 1] 取货区 (点5)")
        print("=" * 60)

        # 出发前切换到取货姿态（不用安全姿态）
        self._go_pose(self.flow.pickup_poses[0], "取货姿态")
        if not self.run.go_and_wait(5):
            print("[NAV] 取货区导航失败")
            return False

        if self.skip_arm:
            print("[ARM] 跳过机械臂")
            return True

        for label, dx, dy in PICKUP_SEARCH:
            print(f"\n[ARM] 搜索位置: {label}")
            if dx != 0 or dy != 0:
                self.car.move_relative(dx, dy)
                time.sleep(1)

            # 移动到唯一取货姿态
            self._go_pose(self.flow.pickup_poses[0], "取货姿态")

            # 多次检测蓝色物块（避免太快跳过）
            for attempt in range(3):
                frame = self.camera.get_frame()
                if frame is None:
                    print(f"  [ARM] {label} 尝试{attempt+1}: 摄像头未就绪")
                    time.sleep(0.3)
                    continue
                cube = self.camera.detect_blue_cube(frame)
                if cube is None:
                    print(f"  [ARM] {label} 尝试{attempt+1}: 未找到蓝色物块")
                    time.sleep(0.3)
                    continue

                print(f"  [ARM] {label} 尝试{attempt+1}: 检测到蓝色物块!")
                if self.flow.pickup_grasp.execute_grasp(show_display=True):
                    print("[Phase 1] ✓ 取货完成")
                    return True
                print(f"  [ARM] {label}: 抓取失败，重试下一个位置")

            print(f"  [ARM] {label}: 3次均未找到")

        print("[Phase 1] ✗ 取货失败")
        return False

    # ── Phase 2: 装货区 ──

    def phase2_loading(self):
        """装货区：导航 → 旋转180 → 放置 → 旋转回"""
        print("\n" + "=" * 60)
        print("  [Phase 2] 装货区 (点6)")
        print("=" * 60)

        self._to_travel_pose()
        if not self.run.go_and_wait(6):
            print("[NAV] 装货区导航失败")
            return False

        if self.skip_arm:
            print("[ARM] 跳过机械臂")
            return True

        self.car.circle(180)
        time.sleep(1)

        self._go_pose(self.flow.loading_place_pose, "装货区")
        self.flow.loading_placer.place_object(show_display=True)

        self.car.circle(-180)
        time.sleep(1)
        print("[Phase 2] ✓ 装货完成")
        return True

    # ── Phase 3: 救援区 ──

    def phase3_rescue(self):
        """救援区：导航 → 旋转 → 抓取 → 平移 → 放置 → 旋转回"""
        point = self.rescue_point
        params = RESCUE_PARAMS[point]
        print("\n" + "=" * 60)
        print(f"  [Phase 3] 救援区 (点{point})")
        print("=" * 60)

        self._to_travel_pose()
        if not self.run.go_and_wait(point):
            print("[NAV] 救援区导航失败")
            return False

        if self.skip_arm:
            print("[ARM] 跳过机械臂")
            return True

        # 旋转
        self.car.circle(params["rotate"])
        time.sleep(1)

        # 抓取
        self._go_pose(self.flow.rescue_pickup_pose, "救援抓取")
        frame = self.camera.get_frame()
        if frame is not None:
            cube = self.camera.detect_blue_cube(frame)
            if cube is None:
                self.flow.rescue_grasp.search_for_object(show_display=True)
        self.flow.rescue_grasp.execute_grasp(show_display=True)

        # 平移
        dx, dy = params["move_rel"]
        self._to_travel_pose()
        self.car.move_relative(dx, dy)
        time.sleep(1)

        # 放置
        self._go_pose(self.flow.rescue_place_pose, "救援放置")
        self.flow.rescue_placer.place_object(show_display=True)

        # 旋转回
        self.car.circle(params["rotate_back"])
        time.sleep(1)
        print("[Phase 3] ✓ 救援完成")
        return True

    # ── Phase 4: 返回起点 ──

    def phase4_return(self):
        print("\n" + "=" * 60)
        print("  [Phase 4] 返回起点 (点7)")
        print("=" * 60)
        self._to_travel_pose()
        if self.run.go_and_wait(7):
            print("[Phase 4] ✓ 返回完成")
            return True
        print("[Phase 4] ✗ 返回失败")
        return False

    # ── 完整流程 ──

    def run_mission(self):
        print("\n" + "=" * 60)
        print("  小车 + 机械臂 联调")
        print("  取货区(点5) → 装货区(点6) → 救援区(点N) → 返回(点7)")
        print("=" * 60)

        results = []
        for name, fn in [
            ("Phase 1 取货区", self.phase1_pickup),
            ("Phase 2 装货区", self.phase2_loading),
            ("Phase 3 救援区", self.phase3_rescue),
            ("Phase 4 返回",   self.phase4_return),
        ]:
            try:
                ok = fn()
            except Exception as e:
                print(f"[ERROR] {name} 异常: {e}")
                ok = False
            results.append((name, ok))

        print("\n" + "=" * 60)
        print("  流程汇总")
        print("=" * 60)
        for name, ok in results:
            print(f"  {'✓' if ok else '✗'} {name}")
        print("=" * 60)

    # ── 清理 ──

    def shutdown(self):
        print("\n[SHUTDOWN] 清理中...")
        if self.arm:
            try:
                self.arm.disconnect()
            except Exception:
                pass
        if self.camera:
            try:
                self.camera.release()
            except Exception:
                pass
        if self.run_proc:
            try:
                self.run_proc.send_signal(signal.SIGINT)
                self.run_proc.wait(timeout=5)
            except Exception:
                try:
                    self.run_proc.terminate()
                    self.run_proc.wait(timeout=3)
                except Exception:
                    self.run_proc.kill()
        print("[SHUTDOWN] 完成")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="小车+机械臂联调")
    parser.add_argument("--rescue-point", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--arm-port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--camera-id", type=int, default=11)
    parser.add_argument("--skip-arm", action="store_true")
    parser.add_argument("--car-flask", type=str, default="http://192.168.43.8:5000")
    parser.add_argument("--run-host", type=str, default="localhost:6000")
    args = parser.parse_args()

    integration = CarArmIntegration(
        arm_port=args.arm_port,
        camera_id=args.camera_id,
        rescue_point=args.rescue_point,
        skip_arm=args.skip_arm,
        car_flask_url=args.car_flask,
        run_url=f"http://{args.run_host}",
    )

    try:
        # 先连机械臂（避免 ROS/LiDAR 启动后串口冲突）
        integration.init_arm()
        if not integration.start_run():
            return
        # 拓展坞 USB 总线可能被 ROS/LiDAR 重置，重连机械臂
        integration.recover_arm()
        integration.run_mission()
    except KeyboardInterrupt:
        print("\n[INTERRUPT] 用户中断")
    finally:
        integration.shutdown()


if __name__ == "__main__":
    main()
