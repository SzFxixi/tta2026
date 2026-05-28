import os
import time
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
from ultralytics import YOLO

from Clients.DroneControlClient import DroneControlClient
from Entities.Waypoint import Waypoint
from Utils.CameraSource import CameraSource
from Utils.JsonHelper import JsonHelper


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
        # 搜索参数：在已知大致点位周围做小范围搜索以精确定位目标
        self.search_max_attempts = int(config.get('search_max_attempts', 8))
        self.search_step = float(config.get('search_step', 0.2))
        self.grade_search_attempts = int(config.get('grade_search_attempts', 5))
        self.grade_search_step = float(config.get('grade_search_step', 0.2))
        self.servo_max_attempts = int(config.get('servo_max_attempts', 3))
        self.h_marker_size = float(config.get('h_marker_size', 0.15))
        self.show_camera = bool(config.get('show_camera', False))
        la = config.get('loading_area', {})
        self.loading_area = Waypoint('装货区', float(la.get('x', 0)), float(la.get('y', 0)),
                                      float(la.get('z', 1.5)))
        self.landing_offset = float(config.get('landing_offset', 0.1))

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

    def _map_type_to_grade(self, raw_label: str) -> str:
        """将模型输出的灾害类型映射为救援等级 (\"1\", \"2\", \"3\")。未映射的返回原标签。"""
        return self.grade_mapping.get(raw_label.lower(), raw_label)

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
                     gimbal_pitch=float(item.get('gimbal_pitch', -90.0)))
            for i, item in enumerate(waypoints_data)
        ]

    def takeoff(self) -> bool:
        print('[DroneNavigator] 开始起飞')
        return self.drone.takeoff()

    def land(self) -> bool:
        print('[DroneNavigator] 开始降落')
        success = self.drone.land()
        self.camera.release()
        return success

    def capture_frame(self) -> Optional[Any]:
        for _ in range(5):
            success, frame = self.camera.read()
            if success and frame is not None:
                return frame
            time.sleep(0.5)
        print("[DroneNavigator] 警告: 5 次尝试仍无法获取画面帧")
        return None

    def _reset_yaw(self) -> None:
        """检测无人机当前 yaw，偏离超过 5° 则旋转回 0°。"""
        posture = self.drone.get_posture()
        current_yaw = float(posture.get("yaw", 0.0))
        if abs(current_yaw) > 3:
            print(f"[DroneNavigator] 偏航修正: {current_yaw:.1f}° → 0°")
            self.drone.rotate_yaw(-current_yaw)
        else:
            print(f"[DroneNavigator] yaw={current_yaw:.1f}°, 无需修正")

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

        return {
            'h_candidate': h_candidate,
            'h_objects': h_detection.get('objects', []),
            'grade_info': grade_info,
            'grade_objects': grade_detection.get('objects', []),
        }

    def _servo_toward_h(self, h_box: List[float], frame_shape: tuple) -> bool:
        """
        视觉伺服：根据 H 在画面中的像素偏移，微移无人机使 H 靠近画面中心。
        返回 True 表示已发起移动，False 表示偏移量太小无需移动。
        """
        height, width = frame_shape[:2]
        cx = width / 2.0
        cy = height / 2.0

        x1, y1, x2, y2 = h_box
        h_cx = (x1 + x2) / 2.0
        h_cy = (y1 + y2) / 2.0
        h_pixel_size = max(x2 - x1, y2 - y1)

        # 像素偏移
        dx_px = h_cx - cx   # 正值=H在画面右侧
        dy_px = h_cy - cy   # 正值=H在画面下方

        # 像素 → 米换算：已知 H 物理尺寸 / H 像素尺寸
        if h_pixel_size < 0.5:
            return False
        meters_per_pixel = self.h_marker_size / h_pixel_size

        # 坐标映射（云台朝下）：
        #  画面上方(-dy_px) = 近处 = 机身前方(+x)
        #  画面下方(+dy_px) = 远处 = 机身后方(-x)
        #  画面右边(+dx_px) = 机身右侧  = 机身 -y
        drone_dx = -dy_px * meters_per_pixel   # 上下反号 → 前后
        drone_dy = -dx_px * meters_per_pixel   # 左右反号 → 左右
        offset_m = (drone_dx ** 2 + drone_dy ** 2) ** 0.5

        # 偏移小于 5cm 认为已居中
        if offset_m < 0.05:
            print(f"[DroneNavigator] 视觉伺服: H 已居中 (offset={offset_m:.3f}m)")
            return False

        print(f"[DroneNavigator] H 偏移 ({dx_px:.0f}, {dy_px:.0f})px → 移动 前{drone_dx:+.3f}m 右{drone_dy:+.3f}m")
        return self.drone.move_to(
            self.drone.state['x'] + drone_dx,
            self.drone.state['y'] + drone_dy,
            self.drone.state['z'],
        )

    def _next_spiral_offset(self, attempt: int) -> tuple[float, float]:
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

    def _capture_fresh_frame(self, settle: float = 3.0, read_time: float = 2.0) -> Optional[Any]:
        """等无人机稳定 settle 秒，然后持续读帧 read_time 秒，返回最后一帧。"""
        time.sleep(settle)
        last_frame = None
        start = time.time()
        while time.time() - start < read_time:
            success, frame = self.camera.read()
            if success and frame is not None:
                last_frame = frame
        if last_frame is not None:
            print(f"[DroneNavigator] 已捕获最新帧 ({last_frame.shape[1]}x{last_frame.shape[0]})")
        else:
            print("[DroneNavigator] 警告: 未获取到画面")
        return last_frame

    def scan_single_waypoint(self, waypoint: Waypoint) -> Dict[str, Any]:
        """
        单航点扫描：飞到航点 → 找到 H 并居中 → 识别等级 → 不再移动。
        最多 servo_max_attempts 轮。
        """
        if not self.drone.move_to(waypoint.x, waypoint.y, waypoint.z):
            return {'success': False, 'reason': 'move_failed'}

        self.drone.rotate_gimbal(waypoint.gimbal_pitch)
        print(f"[DroneNavigator] 到达 {waypoint.name}，云台={waypoint.gimbal_pitch}°，等稳定后取最新帧...")

        for attempt in range(self.servo_max_attempts):
            # 非首轮：螺旋展开微移
            if attempt > 0:
                ox, oy = self._next_spiral_offset(attempt)
                self.drone.move_to(waypoint.x + ox, waypoint.y + oy, waypoint.z)

            # 等稳定 → 持续读到最新帧
            frame = self._capture_fresh_frame(settle=4.0 if attempt == 0 else 3.0)
            if frame is None:
                continue

            if waypoint.rotation:
                frame = self._rotate_frame(frame, waypoint.rotation)

            h, w = frame.shape[:2]
            all_detections = self.detect_all(frame)
            h_candidate = all_detections['h_candidate']
            grade_info = all_detections['grade_info']
            print(f"[DroneNavigator] 画面 {w}x{h} → H={h_candidate['label'] if h_candidate else 'none'}, 等级候选={grade_info.get('label','unknown')}")

            prefix = f'scan_{waypoint.name}_{attempt}'
            combined = {'objects': all_detections['grade_objects'] + all_detections['h_objects']}
            image_path, annotated = self.annotate_and_save(frame, combined, prefix)
            self._preview(annotated)

            # H 没找到 → 下一轮螺旋展开
            if h_candidate is None:
                print(f"[DroneNavigator] {waypoint.name} 第{attempt}轮: 未检测到 H")
                continue

            # 找到 H → 伺服居中 → 停稳后再拍一张识别等级
            self._servo_toward_h(h_candidate['box'], frame.shape)
            print(f"[DroneNavigator] {waypoint.name} 已居中 H，等稳定后识别等级...")

            final_grade = grade_info
            final_path = image_path

            grade_frame = self._capture_fresh_frame(settle=3.0)
            if grade_frame is not None:
                if waypoint.rotation:
                    grade_frame = self._rotate_frame(grade_frame, waypoint.rotation)
                centered_all = self.detect_all(grade_frame)
                final_grade = centered_all['grade_info']
                grade_combined = {'objects': centered_all['grade_objects'] + centered_all['h_objects']}
                final_path, grade_annotated = self.annotate_and_save(grade_frame, grade_combined, f'{prefix}_centered')
                self._preview(grade_annotated)
                print(f"[DroneNavigator] {waypoint.name} 居中后: 等级={final_grade.get('label','unknown')}")

            self.drone.rotate_gimbal(0)
            return {
                'success': True,
                'detection': {'objects': all_detections['h_objects']},
                'h_detection': h_candidate,
                'grade': final_grade,
                'image_path': final_path,
            }

        self.drone.rotate_gimbal(0)
        return {'success': False, 'reason': 'not_found'}

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


    def scan_waypoints(self) -> Dict[str, Dict[str, Any]]:
        """执行四点位巡检，返回 {waypoint_name: {grade, confidence, image_path, success}}。"""
        results: Dict[str, Dict[str, Any]] = {}

        print("[DroneNavigator] 重置飞控状态...")
        self.drone.reset()
        time.sleep(1)

        if not self.takeoff():
            raise RuntimeError('无人机起飞失败')

        print("[DroneNavigator] 等待起飞完成 & 状态稳定...")
        time.sleep(8)

        for waypoint in self.waypoints:
            print(f"[DroneNavigator] 扫描 {waypoint.name}: ({waypoint.x}, {waypoint.y}, {waypoint.z})")
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
            self._reset_yaw()

        la = self.loading_area
        print(f"[DroneNavigator] 扫描完毕，前往装货区 ({la.x}, {la.y}, {la.z})...")
        self.drone.move_to(la.x, la.y, la.z)

        # 装货区归航：检测 H，居中后往前微移再降落
        print(f"[DroneNavigator] 装货区归航: 检测 H 并精确对准...")
        self.drone.rotate_gimbal(-90)
        for attempt in range(self.servo_max_attempts):
            frame = self._capture_fresh_frame(settle=4.0 if attempt == 0 else 3.0)
            if frame is None:
                continue

            all_detections = self.detect_all(frame)
            h_candidate = all_detections['h_candidate']
            if h_candidate is None:
                if attempt > 0:
                    ox, oy = self._next_spiral_offset(attempt)
                    self.drone.move_to(la.x + ox, la.y + oy, la.z)
                print(f"[DroneNavigator] 装货区归航 第{attempt}轮: 未检测到 H")
                continue

            self._servo_toward_h(h_candidate['box'], frame.shape)
            print(f"[DroneNavigator] 装货区: 已对准 H，前移 {self.landing_offset}m 后降落")
            self.drone.move_to(
                self.drone.state['x'] + self.landing_offset,
                self.drone.state['y'],
                self.drone.state['z'],
            )
            break

        self.drone.rotate_gimbal(0)
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
