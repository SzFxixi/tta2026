import os
import time
import math
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

from Clients.DroneControlClient import DroneControlClient
from Entities.Waypoint import Waypoint
from Utils.CameraSource import CameraSource
from Utils.HAngleDetector import HAngleDetector
from Utils.JsonHelper import JsonHelper
from Utils.MathHelper import MathHelper


class DroneNavigator:
    """无人机自主巡检导航器。"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.camera_source = config.get('camera_source', '0')
        self.yolo_weights = config.get('yolo_weights', 'yolov8n.pt')
        self.h_weights = config.get('yolo_weights_h', '') or self.yolo_weights
        self.grade_weights = config.get('yolo_weights_grade', '') or self.yolo_weights
        self.confidence = float(config.get('confidence', 0.25))
        self.output_folder = os.path.abspath(config.get('output_folder', 'output'))
        os.makedirs(self.output_folder, exist_ok=True)

        print(f"[DroneNavigator] 加载 H 模型: {self.h_weights}")
        print(f"[DroneNavigator] 加载等级模型: {self.grade_weights}")
        self.h_model = YOLO(self.h_weights)
        self.grade_model = YOLO(self.grade_weights)

        self.drone = DroneControlClient(config.get('drone', {}))
        ffmpeg_opts = config.get('camera_ffmpeg_opts', {})
        self.camera = CameraSource(
            self.camera_source,
            ffmpeg_opts=ffmpeg_opts,
            loop=config.get('camera_loop', True),
            listen=config.get('listen', False),
            listen_fps=config.get('listen_fps', 30),
        )
        self.waypoints = self._load_waypoints(config.get('waypoints', []))
        self.waypoint_frame = config.get('waypoint_frame', 'world')
        # 搜索参数：在已知大致点位周围做小范围搜索以精确定位目标
        self.search_step = float(config.get('search_step', 0.1))
        self.servo_max_attempts = int(config.get('servo_max_attempts', 3))
        self.h_marker_size = float(config.get('h_marker_size', 0.15))
        self.h_search_max_height = float(config.get('h_search_max_height', 2.5))
        self.h_search_step_height = float(config.get('h_search_step_height', 0.2))
        self.servo_settle_extra = float(config.get('servo_settle_extra', 0.0))
        self.servo_tolerance_scan = float(config.get('servo_tolerance_scan', 0.02))
        self.servo_tolerance_land = float(config.get('servo_tolerance_land', 0.005))
        self.servo_min_step = float(config.get('servo_min_step', 0.0))  # 0=不启用
        self.servo_max_cumulative = float(config.get('servo_max_cumulative', 1.5))
        self.servo_consecutive_limit = int(config.get('servo_consecutive_limit', 3))
        # 伺服悬停时间控制参数
        self.servo_settle = float(config.get('servo_settle', 1.0))
        self.servo_read = float(config.get('servo_read', 1.0))
        self.servo_sleep = float(config.get('servo_sleep', 0.8))
        self.servo_iters = int(config.get('servo_iters', 3))
        self.angle_sleep = float(config.get('angle_sleep', 0.8))
        self.grade_retry_max = int(config.get('grade_retry_max', 20))
        self.grade_retry_interval = float(config.get('grade_retry_interval', 1.0))
        # ── H 角度校正参数 ──
        self.h_angle_enabled = bool(config.get('h_angle_correction_enabled', True))
        self.h_angle_conf_threshold = float(config.get('h_angle_conf_threshold', 0.20))
        self.h_angle_sign = float(config.get('h_angle_sign', 1.0))          # +1 或 -1 翻转方向
        self.h_angle_max_correction = float(config.get('h_angle_max_correction', 30.0))  # 单次最大修正角度
        self.h_angle_scale = float(config.get('h_angle_scale', 1.0))       # 修正比例（偏小→>1.0）
        self.h_angle_debug = bool(config.get('h_angle_debug', False))       # 保存角度调试图
        self.show_camera = bool(config.get('show_camera', False))
        la = config.get('loading_area', {})
        self.loading_area = Waypoint('装货区', float(la.get('x', 0)), float(la.get('y', 0)),
                                      float(la.get('z', 1.5)),
                                      gimbal_pitch=float(la.get('gimbal_pitch', -90.0)),
                                      rotate_to=float(la.get('rotate_to', 0.0)))
        self.landing_offset = float(config.get('landing_offset', 0.1))
        self.landing_preview_height = float(config.get('landing_preview_height', 0.8))
        self._stream_broken = False

        self.h_label = config.get('h_label', 'H')
        raw_grade_labels = config.get('grade_labels', [])
        self.grade_labels = raw_grade_labels if isinstance(raw_grade_labels, list) else [raw_grade_labels]
        self.grade_distance_scale = float(config.get('grade_distance_scale', 2.0))
        # 模型类别名 → 等级数字 的映射: {"1": [...], "2": [...], "3": [...]}
        raw_mapping = config.get('grade_mapping', {})
        self.grade_mapping: Dict[str, str] = {}
        for level, labels in raw_mapping.items():
            if isinstance(labels, list):
                for label in labels:
                    self.grade_mapping[label.lower()] = str(level)
            else:
                self.grade_mapping[str(labels).lower()] = str(level)

        self._reset_servo_memory()

    def _map_type_to_grade(self, raw_label: str) -> str:
        """将模型输出的灾害类型映射为救援等级 (\"1\", \"2\", \"3\")。未映射的返回原标签。"""
        return self.grade_mapping.get(raw_label.lower(), raw_label)

    def _reset_servo_memory(self) -> None:
        """重置伺服方向记忆（每个新航点开始前调用）。"""
        self._servo_prev_dx = 0.0
        self._servo_prev_dy = 0.0
        self._servo_consecutive_same = 0
        self._servo_cumulative = 0.0

    def _load_waypoints(self, waypoints_data: Any) -> List[Waypoint]:
        if not waypoints_data:
            return [
                Waypoint('scan_point_1', 0.5, 0.0, 1.5),
                Waypoint('scan_point_2', 0.5, 0.5, 1.5),
                Waypoint('scan_point_3', 0.0, 0.5, 1.5),
                Waypoint('scan_point_4', -0.5, 0.0, 1.5),
            ]
        if isinstance(waypoints_data, str):
            waypoints_data = JsonHelper.load_json(waypoints_data)
        return [
            Waypoint(name=item.get('name', f'point_{i+1}'),
                     x=float(item['x']), y=float(item['y']), z=float(item['z']),
                     rotation=float(item.get('rotation', 0.0)),
                     gimbal_pitch=float(item.get('gimbal_pitch', -90.0)),
                     rotate_to=float(item.get('rotate_to', 0.0)),
                     rotation_offset=float(item.get('rotation_offset', 0.0)))
            for i, item in enumerate(waypoints_data)
        ]

    def _apply_waypoint_frame(self, yaw: float) -> None:
        """将 body-frame waypoint 转换为 world-frame waypoint，基于起始朝向。"""
        if self.waypoint_frame != 'body' or abs(yaw) < 1e-6:
            return
        print(f"[DroneNavigator] 基于当前朝向将 body-frame waypoints 转换到 world-frame，base_yaw={yaw:.1f}°")
        for waypoint in self.waypoints:
            old_x, old_y = waypoint.x, waypoint.y
            waypoint.x, waypoint.y = MathHelper.rotate_axis(old_x, old_y, math.radians(yaw))
            print(f"[DroneNavigator] {waypoint.name}: body=({old_x:.2f},{old_y:.2f}) -> world=({waypoint.x:.2f},{waypoint.y:.2f})")
        if self.loading_area:
            old_x, old_y = self.loading_area.x, self.loading_area.y
            self.loading_area.x, self.loading_area.y = MathHelper.rotate_axis(old_x, old_y, math.radians(yaw))
            print(f"[DroneNavigator] 装货区: body=({old_x:.2f},{old_y:.2f}) -> world=({self.loading_area.x:.2f},{self.loading_area.y:.2f})")

    def takeoff(self) -> bool:
        print('[DroneNavigator] 开始起飞')
        return self.drone.takeoff()

    def land(self) -> bool:
        print('[DroneNavigator] 开始降落')
        success = self.drone.land()
        self.camera.release()
        return success

    def move_to(self, x: float, y: float, z: float) -> bool:
        return self.drone.move_to(x, y, z)

    def rotate_yaw(self, angle: float) -> bool:
        return self.drone.rotate_yaw(angle)

    def reset(self) -> bool:
        return self.drone.reset()

    def test_single_waypoint(self, waypoint: Waypoint) -> None:
        """专项测试：起飞 → 飞到指定航点 → 伺服居中 H → 降落。"""
        print(f'=== 测试 {waypoint.name} ===')
        self.drone.reset()
        time.sleep(1)
        if not self.takeoff():
            print('起飞失败')
            return
        time.sleep(6)
        time.sleep(3)
        for retry in range(3):
            if self.drone.move_to(0.0, 0.0, 1.5):
                break
            print(f'升至1.5m失败(第{retry+1}次)，重试...')
            time.sleep(2)
        self.drone.state['x'] = 0.0
        self.drone.state['y'] = 0.0
        self.drone.state['z'] = 1.5

        print(f'飞往 {waypoint.name} ({waypoint.x}, {waypoint.y}, {waypoint.z})...')
        self.move_to(waypoint.x, waypoint.y, waypoint.z)

        # 单点测试：物理旋转机身代替图像旋转
        if waypoint.rotation and abs(waypoint.rotation) > 0.1:
            print(f'  物理旋转 {waypoint.rotation}°')
            self.drone.rotate_yaw(waypoint.rotation)

        self._rotate_gimbal_with_recovery(waypoint.gimbal_pitch)
        self._stream_broken = False

        # 垂直搜索
        h_found = False
        current_z = waypoint.z
        while current_z <= self.h_search_max_height:
            print(f'  搜索高度 {current_z:.1f}m')
            if current_z > waypoint.z + 0.01:
                self.move_to(waypoint.x, waypoint.y, current_z)

            frame = self._capture_fresh_frame(settle=self.servo_settle*2 if abs(current_z-waypoint.z)<0.01 else self.servo_settle*1.5,
                                               read_time=self.servo_read*2 if abs(current_z-waypoint.z)<0.01 else self.servo_read*1.5)
            all_det = self.detect_all(frame)
            h = all_det['h_candidate']
            g_label = all_det['grade_info'].get('label', 'unknown')
            print(f'  H={h["label"] if h else "none"}, 等级={g_label}')
            if h is None:
                current_z += self.h_search_step_height
                continue
            for i in range(self.servo_iters):
                moved = self._servo_toward_h(h['box'], frame.shape, servo_tolerance=self.servo_tolerance_scan)
                if not moved: break
                time.sleep(self.servo_sleep)
                frame = self._capture_fresh_frame()
                h = self.detect_all(frame)['h_candidate']
                if h is None: break
            # ── 角度校正 ──
            if h is not None:
                angle_frame = self._capture_fresh_frame(read_time=self.servo_read*1.5)
                if angle_frame is not None:
                    recheck = self.detect_all(angle_frame)
                    if recheck['h_candidate'] is not None:
                        applied = self.correct_h_rotation(
                            angle_frame, recheck['h_candidate']['box'])
                        if abs(applied) > 0.5:
                            time.sleep(self.servo_sleep)
                            post_rot = self._capture_fresh_frame()
                            if post_rot is not None:
                                redet = self.detect_all(post_rot)
                                if redet['h_candidate'] is not None:
                                    for _ in range(self.servo_iters):
                                        moved = self._servo_toward_h(
                                            redet['h_candidate']['box'], post_rot.shape,
                                            servo_tolerance=self.servo_tolerance_scan)
                                        if not moved: break
                                        time.sleep(self.servo_sleep)
                                        post_rot = self._capture_fresh_frame()
                                        if post_rot is None: break
                                        redet = self.detect_all(post_rot)
                                        if redet['h_candidate'] is None: break
            h_found = True
            break

        if not h_found:
            print('未找到 H')

        # 降回原高度并再次伺服
        if current_z > waypoint.z + 0.01:
            self.move_to(self.drone.state['x'], self.drone.state['y'], waypoint.z)
            time.sleep(self.servo_sleep)
            down_frame = self._capture_fresh_frame(read_time=self.servo_read*1.5)
            down_det = self.detect_all(down_frame)
            if down_det['h_candidate'] is not None:
                print('  降低后再次伺服 H')
                for _ in range(self.servo_iters):
                    moved = self._servo_toward_h(down_det['h_candidate']['box'], down_frame.shape,
                                                  servo_tolerance=self.servo_tolerance_scan)
                    if not moved: break
                    time.sleep(self.servo_sleep)
                    down_frame = self._capture_fresh_frame()
                    down_det = self.detect_all(down_frame)
                    if down_det['h_candidate'] is None: break

        # ── 等级重试：取最新帧检测，未检出则持续拍照 ──
        grade_frame = self._capture_fresh_frame(read_time=self.servo_read*1.5, drain_first=True)
        if grade_frame is not None:
            grade_det = self.detect_all(grade_frame)
            grade_info = grade_det['grade_info']
        else:
            grade_info = {'label': 'unknown', 'confidence': 0.0}

        if grade_info.get('label', 'unknown') == 'unknown':
            print(f'  等级未检出, 持续拍照重试 (最多{self.grade_retry_max}次)...')
            best_retry_grade = None
            best_retry_conf = 0.0
            for retry_i in range(self.grade_retry_max):
                time.sleep(self.grade_retry_interval)
                retry_frame = self._capture_fresh_frame(drain_first=True)
                if retry_frame is None:
                    continue
                retry_det = self.detect_all(retry_frame)
                retry_grade = retry_det['grade_info']
                if retry_grade.get('label', 'unknown') != 'unknown':
                    rc = retry_grade.get('confidence', 0)
                    if rc > best_retry_conf:
                        best_retry_grade = retry_grade
                        best_retry_conf = rc
                        mapped = self._map_type_to_grade(retry_grade['label'])
                        print(f'  重试#{retry_i+1} 检出: {retry_grade["label"]} → {mapped}级 conf={rc:.3f} (当前最佳)')
                        if rc >= 0.90 and retry_i >= 2:
                            print(f'  高置信度+3次重试, 提前结束')
                            break
                    else:
                        print(f'  重试#{retry_i+1} 检出: {retry_grade["label"]} conf={rc:.3f} (不优于最佳conf={best_retry_conf:.3f})')
                if (retry_i + 1) % 5 == 0:
                    print(f'  重试#{retry_i+1}/{self.grade_retry_max} (最佳: '
                          f'{best_retry_grade["label"] if best_retry_grade else "none"} conf={best_retry_conf:.3f})')
            if best_retry_grade is not None:
                grade_info = best_retry_grade
                print(f'  重试结束, 最终: {best_retry_grade["label"]} conf={best_retry_conf:.3f}')
            else:
                print(f'  {self.grade_retry_max}次重试完毕, 等级仍为unknown')
        else:
            mapped = self._map_type_to_grade(grade_info['label'])
            print(f'  等级识别: {grade_info["label"]} → {mapped}级 conf={grade_info["confidence"]:.3f}')

        self._rotate_gimbal_with_recovery(0)
        # 角度校正
        if h_found and all_det.get('h_candidate'):
            self.correct_h_rotation(frame, all_det['h_candidate']['box'],
                                     rotation=waypoint.rotation)
        if waypoint.rotate_to and abs(waypoint.rotate_to) > 0.1:
            print(f'伺服后机身旋转 {waypoint.rotate_to}°')
            self.drone.rotate_yaw(waypoint.rotate_to)
        # 断言位置
        print(f'断言位置: {waypoint.name}=({waypoint.x:.2f},{waypoint.y:.2f},{waypoint.z:.2f})')
        self.drone.state['x'] = waypoint.x
        self.drone.state['y'] = waypoint.y
        self.drone.state['z'] = waypoint.z
        time.sleep(1)
        self.land()

    def capture_frame(self) -> Optional[Any]:
        for _ in range(5):
            success, frame = self.camera.read()
            if success and frame is not None:
                return frame
            time.sleep(0.5)
        print("[DroneNavigator] 警告: 5 次尝试仍无法获取画面帧")
        return None

    # ------------------------------------------------------------------
    # H 角度检测与旋转校正
    # ------------------------------------------------------------------

    def measure_h_angle(self, frame: Any, h_box: List[float],
                        debug: bool = False) -> Dict[str, Any]:
        """
        测量 H 标志在画面中的旋转角度。
        返回 HAngleDetector.detect_angle() 的结果字典。
        """
        return HAngleDetector.detect_angle(frame, h_box, debug=debug)

    def correct_h_rotation(self, frame: Any, h_box: List[float],
                           rotation: float = 0.0) -> float:
        """
        测量 H 偏转角度并旋转无人机进行校正。
        rotation: 画面预旋转角度（度），用于坐标补偿。
        返回实际执行的旋转角度（度），0 表示未校正。
        """
        if not self.h_angle_enabled:
            return 0.0

        debug = self.h_angle_debug
        result = self.measure_h_angle(frame, h_box, debug=debug)

        angle = result.get('angle')
        confidence = result.get('confidence', 0.0)
        method = result.get('method', 'unknown')

        print(f"[DroneNavigator] H 角度检测: angle={angle}, "
              f"conf={confidence:.2f}, method={method}, "
              f"lines={result.get('num_lines', 0)}")

        # ── 保存角度检测调试图 ──
        if debug and result.get('debug_frame') is not None:
            ts = time.strftime('%H%M%S')
            path = os.path.join(self.output_folder,
                                f'h_angle_debug_{ts}.jpg')
            cv2.imwrite(path, result['debug_frame'])
            if result.get('edges') is not None:
                edge_path = os.path.join(self.output_folder,
                                         f'h_angle_edges_{ts}.jpg')
                cv2.imwrite(edge_path, result['edges'])

        if angle is None or confidence < self.h_angle_conf_threshold:
            print(f"[DroneNavigator] H 角度检测置信度不足 ({confidence:.2f} < "
                  f"{self.h_angle_conf_threshold}), 跳过旋转校正")
            return 0.0

        # 限幅
        correction = angle * self.h_angle_sign * self.h_angle_scale
        correction = max(-self.h_angle_max_correction,
                         min(self.h_angle_max_correction, correction))

        if abs(correction) < 1.0:
            print(f"[DroneNavigator] H 角度偏差 {correction:.1f}° < 1°, 跳过校正")
            return 0.0

        print(f"[DroneNavigator] H 角度偏差 {angle:.1f}°"
              f" → 校正旋转 {correction:.1f}° (sign={self.h_angle_sign}, scale={self.h_angle_scale})")
        self.drone.rotate_yaw(correction)
        time.sleep(self.angle_sleep)
        return correction

    # ------------------------------------------------------------------
    # 新搜索算法：单帧双模型 + 视觉伺服 + 螺旋展开
    # ------------------------------------------------------------------

    def detect_all(self, frame: Any) -> Dict[str, Any]:
        """一帧同时跑 H 和等级两个模型，返回合并的检测结果。"""
        # 亮度修正
        orig_mean = frame.mean()
        enhanced = cv2.convertScaleAbs(frame, alpha=1.3, beta=10)
        print(f"[DroneNavigator] 亮度增强: {orig_mean:.0f} → {enhanced.mean():.0f}")

        h_detection = self.detect_frame(enhanced, self.h_model)
        h_candidate = self.find_best_h(h_detection)

        grade_detection = self.detect_frame(enhanced, self.grade_model)
        grade_info: Dict[str, Any] = {'label': 'unknown', 'confidence': 0.0, 'box': [], 'distance': float('inf')}
        if h_candidate is not None:
            grade_info = self.find_grade_near_h(h_candidate['box'], grade_detection)
        else:
            # H 不存在时，从 grade_objects 中取置信度最高的作为等级
            grade_objects = grade_detection.get('objects', [])
            if grade_objects:
                best = max(grade_objects, key=lambda obj: obj.get('confidence', 0))
                grade_info = {
                    'label': best['label'],
                    'confidence': best['confidence'],
                    'box': best['box'],
                    'distance': 0.0,
                }

        return {
            'h_candidate': h_candidate,
            'h_objects': h_detection.get('objects', []),
            'grade_info': grade_info,
            'grade_objects': grade_detection.get('objects', []),
        }

    def _servo_toward_h(self, h_box: List[float], frame_shape: tuple, rotation: float = 0.0,
                        servo_tolerance: float = 0.02) -> bool:
        """
        视觉伺服：根据 H 在（已旋转）画面中的像素偏移，微移无人机使 H 靠近画面中心。
        rotation: 画面被旋转的角度（度），用于修正坐标系方向。
        返回 True 表示执行了移动，False 表示未移动（已居中/被拦截/过期帧）。
        """
        height, width = frame_shape[:2]
        cx = width / 2.0
        cy = height / 2.0

        x1, y1, x2, y2 = h_box
        h_cx = (x1 + x2) / 2.0
        h_cy = (y1 + y2) / 2.0
        h_pixel_size = max(x2 - x1, y2 - y1)

        dx_px = h_cx - cx
        dy_px = h_cy - cy

        if h_pixel_size < 0.5:
            return False
        meters_per_pixel = self.h_marker_size / h_pixel_size

        # 基础映射（0° 旋转时）：上→前(+x)，右→正y方向
        drone_dx = -dy_px * meters_per_pixel
        drone_dy = +dx_px * meters_per_pixel

        # 画面被旋转过，伺服方向反向旋转补偿
        if abs(rotation) > 0.1:
            rot_rad = math.radians(-rotation)
            drone_dx, drone_dy = MathHelper.rotate_axis(drone_dx, drone_dy, rot_rad)

        offset_m = (drone_dx ** 2 + drone_dy ** 2) ** 0.5

        # 最小步长：低于此值放大（加快移动速度）
        if self.servo_min_step > 0 and servo_tolerance < offset_m < self.servo_min_step:
            scale = self.servo_min_step / offset_m
            drone_dx *= scale
            drone_dy *= scale
            offset_m = self.servo_min_step
            print(f"[DroneNavigator] 偏移 {offset_m/scale:.3f}m < 最小步长 {self.servo_min_step}m，放大至 {offset_m:.3f}m")

        if offset_m < servo_tolerance:
            print(f"[DroneNavigator] 视觉伺服: H 已居中 (offset={offset_m:.3f}m)")
            self._reset_servo_memory()
            return False

        # ── 过期帧检测 ──
        prev_offset = (self._servo_prev_dx ** 2 + self._servo_prev_dy ** 2) ** 0.5
        if prev_offset > 0.001:
            change_ratio = abs(offset_m - prev_offset) / max(offset_m, prev_offset)
            if change_ratio < 0.15:
                self._servo_consecutive_same += 1
                print(f"[DroneNavigator] 偏移量几乎未变 ({offset_m:.3f}m vs {prev_offset:.3f}m, "
                      f"变化{change_ratio:.1%}) → 疑似过期帧 (x{self._servo_consecutive_same})")
                if self._servo_consecutive_same >= self.servo_consecutive_limit:
                    print(f"[DroneNavigator] 连续 {self._servo_consecutive_same} 次过期帧，但继续伺服...")
            else:
                self._servo_consecutive_same = 0

        self._servo_prev_dx = drone_dx
        self._servo_prev_dy = drone_dy

        # ── 累计位移上限 ──
        self._servo_cumulative += offset_m
        if self._servo_cumulative > self.servo_max_cumulative:
            print(f"[DroneNavigator] 累计伺服 {self._servo_cumulative:.2f}m"
                  f" > {self.servo_max_cumulative}m 上限，但继续伺服...")

        print(f"[DroneNavigator] H 偏移 ({dx_px:.0f}, {dy_px:.0f})px rot={rotation}°"
              f" → 移动({drone_dx:+.3f}, {drone_dy:+.3f})m"
              f" [累计{self._servo_cumulative:.2f}m]")
        return self.drone.move_to(
            self.drone.state['x'] + drone_dx,
            self.drone.state['y'] + drone_dy,
            self.drone.state['z'],
        )

    def _next_spiral_offset(self, attempt: int) -> Tuple[float, float]:
        """螺旋展开：中心 → 十字 → 对角，逐步扩大搜索半径。"""
        if attempt <= 0:
            return 0.0, 0.0
        step = self.search_step * ((attempt + 1) // 2)
        offsets = [
            (step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step),
            (step, step), (-step, -step), (step, -step), (-step, step),
        ]
        idx = (attempt - 1) % len(offsets)
        return offsets[idx]

    def _rotate_frame(self, frame: Any, angle: float) -> Any:
        """将图像旋转指定角度（度），顺时针为正。"""
        if abs(angle) < 0.1:
            return frame
        h, w = frame.shape[:2]
        center = (w / 2, h / 2)
        matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
        rotated = cv2.warpAffine(frame, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0))
        print(f"[DroneNavigator] 画面旋转 {angle}°")
        return rotated

    def _is_stream_alive(self) -> bool:
        """检查 ffmpeg 监听进程是否存活。"""
        return (self.camera._process is not None and self.camera._process.poll() is None)

    def _capture_fresh_frame(self, settle: float = None, read_time: float = None,
                             drain_first: bool = False) -> Optional[Any]:
        """等无人机稳定 settle 秒，持续读帧，返回最后一帧。
        无限重试 + ffmpeg 健康检查。断流后重连时多冲洗旧帧。
        drain_first=True 时先排空帧队列，再阻塞等待真正的新帧。"""
        if settle is None:
            settle = self.servo_settle
        if read_time is None:
            read_time = self.servo_read
        time.sleep(settle)

        # ── 排空缓冲帧 ──
        if drain_first:
            drained = self.camera.drain_queue()
            if drained > 0:
                print(f"[DroneNavigator] 已排空 {drained} 帧，阻塞等待新帧...")
                ok, _ = self.camera.read_blocking(timeout=5.0)
                if not ok:
                    print("[DroneNavigator] 阻塞等待超时，回退到循环读取")

        last_proc = self.camera._process
        while True:
            last_frame = None
            read_count = 0
            start = time.time()
            while time.time() - start < read_time:
                success, frame = self.camera.read()
                if success and frame is not None:
                    last_frame = frame
                    read_count += 1
            # CameraSource 后台自愈重启过（进程引用变了）→ 标记断流
            if self.camera._process is not last_proc:
                self._stream_broken = True
                last_proc = self.camera._process
            if last_frame is not None and read_count >= 3:
                print(f"[DroneNavigator] 已捕获最新帧 ({last_frame.shape[1]}x{last_frame.shape[0]}), {read_count} 帧")
                return last_frame
            if last_frame is not None:
                print(f"[DroneNavigator] 等待稳定: 仅 {read_count} 帧, 等 2s...")
            else:
                self._stream_broken = True
                if not self._is_stream_alive():
                    print("[DroneNavigator] ffmpeg 已死，重启中...")
                    self.camera.reconnect_stream()
                else:
                    print(f"[DroneNavigator] 未获取到画面, 等 2s 重试...")
            time.sleep(2)

    def scan_single_waypoint(self, waypoint: Waypoint) -> Dict[str, Any]:
        """
        单航点扫描：飞到航点 → 找到 H 并居中 → 识别等级 → 不再移动。
        最多 servo_max_attempts 轮。
        """
        # 飞到航点：增加重试与恢复逻辑，遇到失败时尝试 reset 后重试
        print(f"[DroneNavigator] 目标: {waypoint.name} ({waypoint.x:.2f}, {waypoint.y:.2f}, {waypoint.z:.2f}), 当前state: ({self.drone.state['x']:.2f}, {self.drone.state['y']:.2f}, {self.drone.state['z']:.2f}), yaw={self.drone.state['yaw']:.1f}°")
        moved = False
        
        for mv_try in range(3):
            if self.drone.move_to(waypoint.x, waypoint.y, waypoint.z):
                moved = True
                break
            print(f"[DroneNavigator] 警告: 到达 {waypoint.name} 移动失败 (尝试 {mv_try+1}/3)，尝试重置飞控并重试")
            self.drone.reset()
            time.sleep(1)
        if not moved:
            return {'success': False, 'reason': 'move_failed'}

        if not self._rotate_gimbal_with_recovery(waypoint.gimbal_pitch):
            return {'success': False, 'reason': 'gimbal_failed'}

        self._stream_broken = False

        # 垂直搜索：同高度反复尝试取帧，取到但无 H 才升高
        h_found = False
        current_z = waypoint.z
        while current_z <= self.h_search_max_height:
            print(f"[DroneNavigator] {waypoint.name} 搜索高度 {current_z:.1f}m")
            if current_z > waypoint.z + 0.01:
                self.drone.move_to(waypoint.x, waypoint.y, current_z)

            # 取帧（内部无限重试 + ffmpeg 健康检查）
            frame = self._capture_fresh_frame(
                settle=self.servo_settle*2 if abs(current_z - waypoint.z) < 0.01 else self.servo_settle*1.5,
                read_time=self.servo_read*2 if abs(current_z - waypoint.z) < 0.01 else self.servo_read*1.5,
                drain_first=True)

            if waypoint.rotation:
                frame = self._rotate_frame(frame, waypoint.rotation)

            h, w = frame.shape[:2]
            all_detections = self.detect_all(frame)
            h_candidate = all_detections['h_candidate']
            grade_info = all_detections['grade_info']
            print(f"[DroneNavigator] 画面 {w}x{h} → H={h_candidate['label'] if h_candidate else 'none'}, 等级候选={grade_info.get('label','unknown')}")

            prefix = f'scan_{waypoint.name}_{current_z:.1f}'
            combined = {'objects': all_detections['grade_objects'] + all_detections['h_objects']}
            image_path, annotated = self.annotate_and_save(frame, combined, prefix)
            self._preview(annotated)

            if h_candidate is not None:
                if self._stream_broken:
                    print(f"[DroneNavigator] {waypoint.name} 断流后首帧有H，冲洗重取确认...")
                    self._stream_broken = False
                    t0 = time.time()
                    while time.time() - t0 < 3.0:
                        self.camera.read()
                    time.sleep(1)
                    confirm = self._capture_fresh_frame(drain_first=True)
                    if confirm is not None:
                        if waypoint.rotation:
                            confirm = self._rotate_frame(confirm, waypoint.rotation)
                        recheck = self.detect_all(confirm)
                        if recheck['h_candidate'] is not None:
                            h_candidate = recheck['h_candidate']
                            grade_info = recheck['grade_info']
                            image_path, _ = self.annotate_and_save(confirm, {'objects': recheck['grade_objects'] + recheck['h_objects']}, f'scan_{waypoint.name}_{current_z:.1f}_confirmed')
                            print(f"[DroneNavigator] {waypoint.name} 确认帧仍有H，通过")
                        else:
                            print(f"[DroneNavigator] {waypoint.name} 确认帧无H，丢弃继续搜索")
                            continue
                    else:
                        print(f"[DroneNavigator] {waypoint.name} 确认帧采集失败，丢弃继续搜索")
                        continue
                h_found = True
                break

            print(f"[DroneNavigator] {waypoint.name} 未检测到 H，升高 {self.h_search_step_height:.1f}m")
            current_z += self.h_search_step_height

        if not h_found:
            self._rotate_gimbal_with_recovery(0)
            return {'success': False, 'reason': 'not_found'}

        # 找到 H → 迭代伺服直到居中
        for servo_iter in range(self.servo_iters):
            moved = self._servo_toward_h(h_candidate['box'], frame.shape, rotation=waypoint.rotation,
                                          servo_tolerance=self.servo_tolerance_scan)
            if not moved:
                print(f"[DroneNavigator] {waypoint.name} H 已居中 (迭代{servo_iter+1}次)")
                break
            time.sleep(self.servo_sleep)
            frame = self._capture_fresh_frame(drain_first=True)
            if waypoint.rotation:
                frame = self._rotate_frame(frame, waypoint.rotation)
            re_det = self.detect_all(frame)
            re_h = re_det['h_candidate']
            if re_h is None:
                break
            h_candidate = re_h
            # 伺服过程中若检测到有效等级，更新 grade_info（避免伺服后的居中帧漏检）
            if grade_info.get('label', 'unknown') == 'unknown':
                servo_grade = re_det.get('grade_info', {})
                if servo_grade.get('label', 'unknown') != 'unknown':
                    grade_info = servo_grade
                    print(f"[DroneNavigator] {waypoint.name} 伺服中捕获等级: {grade_info['label']} conf={grade_info['confidence']:.2f}")

        print(f"[DroneNavigator] {waypoint.name} 伺服完成，等稳定后识别等级...")

        # ── 角度校正：测量 H 偏转 → 无人机旋转对齐 ──
        rotation_applied = 0.0
        if self.h_angle_enabled and h_candidate is not None:
            angle_frame = self._capture_fresh_frame(read_time=self.servo_read*1.5)
            if angle_frame is not None:
                if waypoint.rotation:
                    angle_frame = self._rotate_frame(angle_frame, waypoint.rotation)
                # 重新检测 H，确保 box 是最新的
                recheck = self.detect_all(angle_frame)
                if recheck['h_candidate'] is not None:
                    rotation_applied = self.correct_h_rotation(
                        angle_frame, recheck['h_candidate']['box'],
                        rotation=waypoint.rotation)
                    if abs(rotation_applied) > 0.5:
                        # 旋转后画面可能偏移，重新伺服一次
                        print(f"[DroneNavigator] {waypoint.name} 旋转校正 {rotation_applied:.1f}°, 重新伺服...")
                        time.sleep(self.servo_sleep)
                        frame = self._capture_fresh_frame()
                        if frame is not None:
                            if waypoint.rotation:
                                frame = self._rotate_frame(frame, waypoint.rotation)
                            re_det2 = self.detect_all(frame)
                            if re_det2['h_candidate'] is not None:
                                for _ in range(self.servo_iters):
                                    moved = self._servo_toward_h(
                                        re_det2['h_candidate']['box'],
                                        frame.shape, rotation=waypoint.rotation,
                                        servo_tolerance=self.servo_tolerance_scan)
                                    if not moved:
                                        break
                                    time.sleep(self.servo_sleep)
                                    frame = self._capture_fresh_frame()
                                    if frame is None:
                                        break
                                    if waypoint.rotation:
                                        frame = self._rotate_frame(frame, waypoint.rotation)
                                    re_det2 = self.detect_all(frame)
                                    if re_det2['h_candidate'] is None:
                                        break
                                h_candidate = re_det2['h_candidate']

        final_grade = grade_info
        final_path = image_path

        grade_frame = self._capture_fresh_frame(read_time=self.servo_read*1.5, drain_first=True)
        if grade_frame is not None:
            if waypoint.rotation:
                grade_frame = self._rotate_frame(grade_frame, waypoint.rotation)
            centered_all = self.detect_all(grade_frame)
            centered_grade = centered_all['grade_info']
            if centered_grade.get('label', 'unknown') != 'unknown':
                prev_c = final_grade.get('confidence', 0) if final_grade.get('label', 'unknown') != 'unknown' else 0
                if centered_grade.get('confidence', 0) >= prev_c:
                    final_grade = centered_grade
                    print(f"[DroneNavigator] {waypoint.name} 居中帧等级: {centered_grade['label']} conf={centered_grade['confidence']:.3f} (优于前次)")
                else:
                    print(f"[DroneNavigator] {waypoint.name} 居中帧等级: {centered_grade['label']} conf={centered_grade['confidence']:.3f} (不优于前次conf={prev_c:.3f})")
            else:
                print(f"[DroneNavigator] 居中帧未检测到等级，回退用伺服前: {final_grade.get('label','unknown')}")
            grade_combined = {'objects': centered_all['grade_objects'] + centered_all['h_objects']}
            final_path, grade_annotated = self.annotate_and_save(grade_frame, grade_combined, f'{prefix}_centered')
            self._preview(grade_annotated)
            print(f"[DroneNavigator] {waypoint.name} 居中后: 等级={final_grade.get('label','unknown')}")

        # 降回原高度并再次伺服
        if current_z > waypoint.z + 0.01:
            print(f"[DroneNavigator] {waypoint.name} 原地降回 {waypoint.z:.1f}m")
            self.drone.move_to(self.drone.state['x'], self.drone.state['y'], waypoint.z)
            time.sleep(self.servo_sleep)

            # 降低后再次检测 H 并伺服
            down_frame = self._capture_fresh_frame(read_time=self.servo_read*1.5, drain_first=True)
            if down_frame is not None:
                if waypoint.rotation:
                    down_frame = self._rotate_frame(down_frame, waypoint.rotation)
                down_det = self.detect_all(down_frame)
                if down_det['h_candidate'] is not None:
                    print(f"[DroneNavigator] {waypoint.name} 降低后再次伺服 H")
                    for _ in range(self.servo_iters):
                        moved = self._servo_toward_h(down_det['h_candidate']['box'], down_frame.shape,
                                                      rotation=waypoint.rotation,
                                                      servo_tolerance=self.servo_tolerance_scan)
                        if not moved:
                            break
                        time.sleep(self.servo_sleep)
                        down_frame = self._capture_fresh_frame(drain_first=True)
                        if down_frame is None:
                            break
                        if waypoint.rotation:
                            down_frame = self._rotate_frame(down_frame, waypoint.rotation)
                        down_det = self.detect_all(down_frame)
                        if down_det['h_candidate'] is None:
                            break
                        down_det_h = down_det['h_candidate']
                    if down_det['grade_info'].get('label', 'unknown') != 'unknown':
                        down_grade = down_det['grade_info']
                        prev_c = final_grade.get('confidence', 0) if final_grade.get('label', 'unknown') != 'unknown' else 0
                        if down_grade.get('confidence', 0) >= prev_c:
                            final_grade = down_grade
                            print(f"[DroneNavigator] {waypoint.name} 降回帧等级: {down_grade['label']} conf={down_grade['confidence']:.3f} (优于前次)")
                        else:
                            print(f"[DroneNavigator] {waypoint.name} 降回帧等级: {down_grade['label']} conf={down_grade['confidence']:.3f} (不优于前次conf={prev_c:.3f})")

        # ── 等级重试：仍未检出则持续拍照识别 ──
        if final_grade.get('label', 'unknown') == 'unknown':
            print(f"[DroneNavigator] {waypoint.name} 等级未检出, 持续拍照重试 (最多{self.grade_retry_max}次)...")
            best_retry_grade = None
            best_retry_conf = 0.0
            for retry_i in range(self.grade_retry_max):
                time.sleep(self.grade_retry_interval)
                retry_frame = self._capture_fresh_frame(drain_first=True)
                if retry_frame is None:
                    continue
                if waypoint.rotation:
                    retry_frame = self._rotate_frame(retry_frame, waypoint.rotation)
                retry_det = self.detect_all(retry_frame)
                retry_grade = retry_det['grade_info']
                if retry_grade.get('label', 'unknown') != 'unknown':
                    rc = retry_grade.get('confidence', 0)
                    if rc > best_retry_conf:
                        best_retry_grade = retry_grade
                        best_retry_conf = rc
                        grade_combined = {'objects': retry_det['grade_objects'] + retry_det['h_objects']}
                        final_path, _ = self.annotate_and_save(retry_frame, grade_combined,
                                                               f'{prefix}_retry{retry_i+1}')
                        print(f"[DroneNavigator] {waypoint.name} 重试#{retry_i+1} 检出: "
                              f"{retry_grade['label']} conf={rc:.3f} (当前最佳)")
                        # 高置信度 + 至少3次重试 → 可提前结束
                        if rc >= 0.90 and retry_i >= 2:
                            print(f"[DroneNavigator] {waypoint.name} 高置信度+3次重试, 提前结束")
                            break
                    else:
                        print(f"[DroneNavigator] {waypoint.name} 重试#{retry_i+1} 检出: "
                              f"{retry_grade['label']} conf={rc:.3f} (不优于最佳conf={best_retry_conf:.3f})")
                if (retry_i + 1) % 5 == 0:
                    print(f"[DroneNavigator] {waypoint.name} 重试#{retry_i+1}/{self.grade_retry_max} (最佳: "
                          f"{best_retry_grade['label'] if best_retry_grade else 'none'} "
                          f"conf={best_retry_conf:.3f})")
            if best_retry_grade is not None:
                final_grade = best_retry_grade
                print(f"[DroneNavigator] {waypoint.name} 重试结束, 最终: {final_grade['label']} conf={final_grade['confidence']:.3f}")
            else:
                print(f"[DroneNavigator] {waypoint.name} {self.grade_retry_max}次重试完毕, 等级仍为unknown")

        self._rotate_gimbal_with_recovery(0)
        if waypoint.rotate_to and abs(waypoint.rotate_to) > 0.1:
            print(f"[DroneNavigator] {waypoint.name} 伺服后机身旋转 {waypoint.rotate_to}°")
            self.drone.rotate_yaw(waypoint.rotate_to)

        # 断言当前位置 = 救援点坐标
        self.drone.state['x'] = waypoint.x
        self.drone.state['y'] = waypoint.y
        self.drone.state['z'] = waypoint.z
        print(f"[DroneNavigator] 断言位置: {waypoint.name}=({waypoint.x:.2f},{waypoint.y:.2f},{waypoint.z:.2f})")

        return {
            'success': True,
            'detection': {'objects': all_detections['h_objects']},
            'h_detection': h_candidate,
            'grade': final_grade,
            'image_path': final_path,
        }

    def detect_frame(self, frame: Any, model: Any) -> Dict[str, Any]:
        results = model.predict(frame, verbose=False, conf=self.confidence)
        detection = {'objects': [], 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}
        if not results:
            return detection

        boxes = getattr(results[0], 'boxes', None)
        if boxes is None:
            return detection

        for box in boxes:
            xyxy = box.xyxy.tolist()[0]
            confidence = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
            class_id = int(box.cls[0]) if hasattr(box, 'cls') else -1
            label = model.names.get(class_id, str(class_id))
            detection['objects'].append({
                'label': label,
                'confidence': confidence,
                'box': [float(x) for x in xyxy],
            })

        detection['level'] = detection['objects'][0]['label'] if detection['objects'] else 'unknown'
        return detection

    def find_best_h(self, detection: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        candidates = [
            obj for obj in detection['objects']
            if obj['label'] == self.h_label or self.h_label.lower() in obj['label'].lower()
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: item['confidence'])

    def find_grade_near_h(self, h_box: List[float], detection: Dict[str, Any]) -> Dict[str, Any]:
        if not detection['objects']:
            return {'label': 'unknown', 'confidence': 0.0, 'box': [], 'distance': float('inf')}

        hx1, hy1, hx2, hy2 = h_box
        h_cx = (hx1 + hx2) / 2.0
        h_cy = (hy1 + hy2) / 2.0
        h_size = max(hx2 - hx1, hy2 - hy1)
        max_distance = h_size * self.grade_distance_scale

        selected = None
        best_score = float('inf')

        for obj in detection['objects']:
            if self.grade_labels and obj['label'] not in self.grade_labels:
                print(f"[DroneNavigator] 跳过非等级候选: {obj['label']}")
                continue
            
            print(f"[DroneNavigator] 评估等级候选: {obj['label']} at {obj['box']} with confidence {obj['confidence']:.2f}")
            ox1, oy1, ox2, oy2 = obj['box']
            o_cx = (ox1 + ox2) / 2.0
            o_cy = (oy1 + oy2) / 2.0
            distance = ((o_cx - h_cx) ** 2 + (o_cy - h_cy) ** 2) ** 0.5
            if distance <= max_distance and distance < best_score:
                selected = obj.copy()
                best_score = distance
                selected['distance'] = distance

        if selected is not None:
            return selected

        return {'label': 'unknown', 'confidence': 0.0, 'box': [], 'distance': float('inf')}

    def annotate_and_save(self, frame: Any, detection: Dict[str, Any], prefix: str):
        filename = os.path.join(self.output_folder, f'{prefix}.jpg')
        annotated = frame.copy()
        for obj in detection['objects']:
            x1, y1, x2, y2 = [int(v) for v in obj['box']]
            label = obj['label']
            confidence = obj['confidence']
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f'{label}:{confidence:.2f}', (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(filename, annotated)
        return filename, annotated

    def _preview(self, frame: Any, title: str = "Drone Camera") -> None:
        """如果 show_camera 为 True，弹窗显示当前帧。"""
        if not self.show_camera:
            return
        cv2.imshow(title, frame)
        cv2.waitKey(1)

    def _close_preview(self) -> None:
        if self.show_camera:
            cv2.destroyAllWindows()

    def _rotate_gimbal_with_recovery(self, pitch: float) -> bool:
        if self.drone.rotate_gimbal(pitch):
            return True
        print("[DroneNavigator] 云台旋转失败，尝试重置 taskId 并重试")
        self.drone.reset()
        time.sleep(1)
        return self.drone.rotate_gimbal(pitch)

    def scan_waypoints(self) -> Dict[str, Dict[str, Any]]:
        """执行四点位巡检，返回 {waypoint_name: {grade, confidence, image_path, success}}。"""
        results: Dict[str, Dict[str, Any]] = {}

        print("[DroneNavigator] 重置飞控状态...")
        self.drone.reset()
        time.sleep(1)

        if not self.takeoff():
            raise RuntimeError('无人机起飞失败')

        print("[DroneNavigator] 等待起飞完成 & 状态稳定...")
        time.sleep(5)

        # 升到1.5m并断言（确认起飞完成再升，失败则重试）
        time.sleep(3)
        for retry in range(3):
            if self.drone.move_to(0.0, 0.0, 1.5):
                break
            print(f"[DroneNavigator] 升至1.5m失败(第{retry+1}次)，重试...")
            time.sleep(2)
        self.drone.state['x'] = 0.0
        self.drone.state['y'] = 0.0
        self.drone.state['z'] = 1.5
        print("[DroneNavigator] 断言原点 at 1.5m")

        posture = self.drone.get_posture()
        base_yaw = float(posture.get('yaw', 0.0))
        print(f"[DroneNavigator] 起始 yaw={base_yaw:.1f}°, waypoint_frame={self.waypoint_frame}")
        if self.waypoint_frame == 'body':
            self._apply_waypoint_frame(base_yaw)

        for waypoint in self.waypoints:
            print(f"[DroneNavigator] 扫描 {waypoint.name}: ({waypoint.x}, {waypoint.y}, {waypoint.z})")
            print(f"[DroneNavigator] 当前无人机状态: {self.drone.state}")
            result = self.scan_single_waypoint(waypoint)

            if result.get('success'):
                grade_info = result.get('grade', {})
                raw_label = grade_info.get('label', 'unknown')
                mapped_grade = self._map_type_to_grade(raw_label)
                results[waypoint.name] = {
                    'success': True,
                    'grade': mapped_grade,
                    'raw_label': raw_label,
                    'confidence': grade_info.get('confidence', 0.0),
                    'image_path': result.get('image_path', ''),
                }
                print(f"[DroneNavigator] {waypoint.name} → {raw_label} → 等级={mapped_grade}")
            else:
                results[waypoint.name] = {
                    'success': False,
                    'grade': 'unknown',
                    'confidence': 0.0,
                    'image_path': '',
                    'reason': result.get('reason', 'not_found'),
                }
                print(f"[DroneNavigator] {waypoint.name} 未找到: {result.get('reason', 'not_found')}")

            time.sleep(1.0)

        la = self.loading_area
        print(f"[DroneNavigator] 扫描完毕，前往装货区 ({la.x}, {la.y}, {la.z})...")
        self.drone.move_to(la.x, la.y, la.z)

        # ── 1. 旋转 180° ──
        print("[DroneNavigator] 装货区: 旋转 180°...")
        self.drone.rotate_yaw(183)
        time.sleep(self.angle_sleep)

        # ── 2. 云台朝下，H 检测并伺服 ──
        print(f"[DroneNavigator] 装货区: H 对齐...")
        self._rotate_gimbal_with_recovery(self.loading_area.gimbal_pitch)
        for attempt in range(self.servo_max_attempts):
            frame = self._capture_fresh_frame(read_time=self.servo_read*1.5)
            if frame is None:
                continue
            all_detections = self.detect_all(frame)
            h_candidate = all_detections['h_candidate']
            if h_candidate is not None and self._stream_broken:
                self._stream_broken = False
                t0 = time.time()
                while time.time() - t0 < 3.0:
                    self.camera.read()
                time.sleep(1)
                confirm = self._capture_fresh_frame()
                if confirm is not None:
                    confirm_det = self.detect_all(confirm)
                    h_candidate = confirm_det['h_candidate'] if confirm_det['h_candidate'] is not None else None
                else:
                    h_candidate = None
            if h_candidate is None:
                if attempt > 0:
                    ox, oy = self._next_spiral_offset(attempt)
                    self.drone.move_to(la.x + ox, la.y + oy, la.z)
                continue
            for servo_iter in range(self.servo_iters):
                moved = self._servo_toward_h(h_candidate['box'], frame.shape,
                                              servo_tolerance=self.servo_tolerance_land)
                if not moved:
                    break
                time.sleep(self.servo_sleep)
                frame = self._capture_fresh_frame()
                if frame is None:
                    break
                re_h = self.detect_all(frame)['h_candidate']
                if re_h is None:
                    break
                h_candidate = re_h
            break

        # ── 角度校正：装货区 H ──
        if self.h_angle_enabled and h_candidate is not None:
            angle_frame = self._capture_fresh_frame(read_time=self.servo_read*1.5)
            if angle_frame is not None:
                recheck = self.detect_all(angle_frame)
                if recheck['h_candidate'] is not None:
                    applied = self.correct_h_rotation(
                        angle_frame, recheck['h_candidate']['box'])
                    if abs(applied) > 0.5:
                        time.sleep(self.servo_sleep)
                        frame = self._capture_fresh_frame()
                        if frame is not None:
                            re_det2 = self.detect_all(frame)
                            if re_det2['h_candidate'] is not None:
                                for _ in range(self.servo_iters):
                                    moved = self._servo_toward_h(
                                        re_det2['h_candidate']['box'], frame.shape,
                                        servo_tolerance=self.servo_tolerance_land)
                                    if not moved:
                                        break
                                    time.sleep(self.servo_sleep)
                                    frame = self._capture_fresh_frame()
                                    if frame is None:
                                        break
                                    re_det2 = self.detect_all(frame)
                                    if re_det2['h_candidate'] is None:
                                        break
                                h_candidate = re_det2['h_candidate']

        # [已禁用预降] ── 5. 预览高度伺服（跳过）──
        # [已禁用预降] ── 6. 角度校正（跳过）──

        # ── 5. 旋转 → 断言位置 → 前移 → 降落 ──
        if la.rotate_to and abs(la.rotate_to) > 0.1:
            print(f"[DroneNavigator] 装货区 伺服后机身旋转 {la.rotate_to}°")
            self.drone.rotate_yaw(la.rotate_to)
        # 断言装货区位置
        print(f"[DroneNavigator] 断言位置: 装货区=({la.x:.2f},{la.y:.2f},{la.z:.2f})")
        self.drone.state['x'] = la.x
        self.drone.state['y'] = la.y
        self.drone.state['z'] = la.z
        print(f"[DroneNavigator] 装货区: 前移 {self.landing_offset}m 后降落")
        self.drone.move_to(
            self.drone.state['x'] + self.landing_offset,
            self.drone.state['y'],
            self.drone.state['z'],
        )
        self._rotate_gimbal_with_recovery(0)
        time.sleep(1)
        self._close_preview()
        self.land()
        return results

    def detect_image_file(self, image_path: str) -> Dict[str, Any]:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f'无法读取图像: {image_path}')
        base = os.path.splitext(os.path.basename(image_path))[0]

        result = self.detect_all(image)
        combined = {'objects': result['grade_objects'] + result['h_objects']}
        self.annotate_and_save(image, combined, base)

        raw_label = result['grade_info'].get('label', 'unknown')
        mapped = self._map_type_to_grade(raw_label)
        print(f'检测结果: {raw_label} → {mapped}级')
        return result

    def stream_test(self) -> None:
        """实时拉流并显示 H + 等级检测画面，按 Q 退出。不控制无人机。"""
        print(f"[DroneNavigator] 开始视频流测试: {self.camera_source}")
        cv2.namedWindow("Stream Test - H & Grade Detection", cv2.WINDOW_KEEPRATIO)

        frame_count = 0
        no_frame_count = 0
        first_frame = True
        try:
            while True:
                frame = self.capture_frame()
                if frame is None:
                    no_frame_count += 1
                    if no_frame_count == 1:
                        print("[DroneNavigator] 等待视频流...")
                    time.sleep(0.5)
                    # 也处理窗口事件，防止灰屏
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    continue

                no_frame_count = 0
                frame_count += 1

                # 首帧出现时自动适配窗口大小
                if first_frame:
                    first_frame = False
                    h, w = frame.shape[:2]
                    print(f"[DroneNavigator] 收到视频流，分辨率: {w}x{h}")

                # 同时检测 H 和等级
                all_det = self.detect_all(frame)
                h_candidate = all_det['h_candidate']
                grade_info = all_det['grade_info']

                # 显示
                display = frame.copy()
                for obj in all_det['h_objects']:
                    x1, y1, x2, y2 = [int(v) for v in obj['box']]
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(display, f'H {obj["confidence"]:.2f}', (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                for obj in all_det['grade_objects']:
                    x1, y1, x2, y2 = [int(v) for v in obj['box']]
                    cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(display, f'{obj["label"]} {obj["confidence"]:.2f}', (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                raw_label = grade_info.get("label", "unknown")
                mapped_grade = self._map_type_to_grade(raw_label)
                status_text = f'Frame #{frame_count} | Grade: {raw_label} -> {mapped_grade}'
                if h_candidate is not None:
                    status_text += f' | H found'
                cv2.putText(display, status_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("Stream Test - H & Grade Detection", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[DroneNavigator] 用户退出视频流测试")
                    break

        except KeyboardInterrupt:
            print("[DroneNavigator] 视频流测试被中断")
        finally:
            cv2.destroyAllWindows()
            self.camera.release()
            print(f"[DroneNavigator] 视频流测试结束，共处理 {frame_count} 帧")
