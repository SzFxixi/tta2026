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

        self.h_label = config.get('h_label', 'H')
        raw_grade_labels = config.get('grade_labels', [])
        self.grade_labels = raw_grade_labels if isinstance(raw_grade_labels, list) else [raw_grade_labels]
        self.grade_distance_scale = float(config.get('grade_distance_scale', 2.0))
       # 模型类别名 → 等级数字 的映射: {"1": ["earthquake", "fire"], "2": ["leak_water"], "3": ["collapse"]}
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
            Waypoint(name=item.get('name', f'point_{i+1}'), x=float(item['x']), y=float(item['y']), z=float(item['z']))
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
        return None

    # ------------------------------------------------------------------
    # 新搜索算法：单帧双模型 + 视觉伺服 + 螺旋展开
    # ------------------------------------------------------------------

    def detect_all(self, frame: Any) -> Dict[str, Any]:
        """一帧同时跑 H 和等级两个模型，返回合并的检测结果。"""
        h_detection = self.detect_frame(frame, self.h_model)
        grade_detection = self.detect_frame(frame, self.grade_model)

        h_candidate = self.find_best_h(h_detection)
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
        dx_px = h_cx - cx
        dy_px = h_cy - cy

        # 像素 → 米换算：已知 H 物理尺寸 / H 像素尺寸
        if h_pixel_size < 1:
            return False
        meters_per_pixel = self.h_marker_size / h_pixel_size

        dx_m = dx_px * meters_per_pixel
        dy_m = dy_px * meters_per_pixel
        offset_m = (dx_m ** 2 + dy_m ** 2) ** 0.5

        # 偏移小于 5cm 认为已居中
        if offset_m < 0.05:
            print(f"[DroneNavigator] 视觉伺服: H 已居中 (offset={offset_m:.3f}m)")
            return False

        print(f"[DroneNavigator] 视觉伺服: H 偏移 ({dx_px:.0f}, {dy_px:.0f})px → 移动 ({dx_m:.3f}, {dy_m:.3f})m")
        return self.drone.move_to(
            self.drone.state['x'] + dx_m,
            self.drone.state['y'] + dy_m,
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

    def scan_single_waypoint(self, waypoint: Waypoint) -> Dict[str, Any]:
        """
        单航点扫描：飞到航点 → 视觉伺服闭环找到 H → 识别等级。
        最多 servo_max_attempts 轮。
        """
        if not self.drone.move_to(waypoint.x, waypoint.y, waypoint.z):
            return {'success': False, 'reason': 'move_failed'}

        for attempt in range(self.servo_max_attempts):
            # 非首轮：螺旋展开微移
            if attempt > 0:
                ox, oy = self._next_spiral_offset(attempt)
                self.drone.move_to(waypoint.x + ox, waypoint.y + oy, waypoint.z)
                time.sleep(0.3)

            frame = self.capture_frame()
            if frame is None:
                continue

            all_detections = self.detect_all(frame)
            h_candidate = all_detections['h_candidate']
            grade_info = all_detections['grade_info']

            prefix = f'scan_{waypoint.name}_{attempt}'
            combined = {'objects': all_detections['grade_objects'] + all_detections['h_objects']}
            image_path = self.annotate_and_save(frame, combined, prefix)

            print(f"[DroneNavigator] {waypoint.name} 第{attempt}轮: H={h_candidate['label'] if h_candidate else 'none'}, 等级={grade_info['label']}")

            # 情况 A：H 和等级都找到了 → 返回
            if h_candidate is not None and grade_info.get('label', 'unknown') != 'unknown':
                return {
                    'success': True,
                    'detection': {'objects': all_detections['h_objects']},
                    'h_detection': h_candidate,
                    'grade': grade_info,
                    'image_path': image_path,
                }

            # 情况 B：找到 H 但没找到等级 → 视觉伺服靠近 H
            if h_candidate is not None and grade_info.get('label', 'unknown') == 'unknown':
                self._servo_toward_h(h_candidate['box'], frame.shape)
                time.sleep(0.3)
                continue

            # 情况 C：H 也没找到 → 下一轮螺旋展开
            time.sleep(0.3)

        # 所有轮次后最后试一次：飞到航点正上方再拍
        if not self.drone.move_to(waypoint.x, waypoint.y, waypoint.z):
            return {'success': False, 'reason': 'not_found'}
        frame = self.capture_frame()
        if frame is not None:
            all_detections = self.detect_all(frame)
            h_candidate = all_detections['h_candidate']
            grade_info = all_detections['grade_info']
            if h_candidate is not None:
                return {
                    'success': True,
                    'detection': {'objects': all_detections['h_objects']},
                    'h_detection': h_candidate,
                    'grade': grade_info,
                    'image_path': '',
                }

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

    def annotate_and_save(self, frame: Any, detection: Dict[str, Any], prefix: str) -> str:
        filename = os.path.join(self.output_folder, f'{prefix}.jpg')
        annotated = frame.copy()
        for obj in detection['objects']:
            x1, y1, x2, y2 = [int(v) for v in obj['box']]
            label = obj['label']
            confidence = obj['confidence']
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f'{label}:{confidence:.2f}', (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(filename, annotated)
        return filename


    def scan_waypoints(self) -> Dict[str, Dict[str, Any]]:
        """执行四点位巡检，返回 {waypoint_name: {grade, confidence, image_path, success}}。"""
        results: Dict[str, Dict[str, Any]] = {}
        if not self.takeoff():
            raise RuntimeError('无人机起飞失败')

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

        self.land()
        return results

    def detect_image_file(self, image_path: str) -> Dict[str, Any]:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f'无法读取图像: {image_path}')
        detection = self.detect_frame(image, self.grade_model)
        self.annotate_and_save(image, detection, os.path.splitext(os.path.basename(image_path))[0])
        return detection

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
