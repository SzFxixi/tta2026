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
        self.camera = CameraSource(self.camera_source, ffmpeg_opts=ffmpeg_opts, loop=config.get('camera_loop', True))
        self.waypoints = self._load_waypoints(config.get('waypoints', []))
        # 搜索参数：在已知大致点位周围做小范围搜索以精确定位目标
        self.search_max_attempts = int(config.get('search_max_attempts', 8))
        self.search_step = float(config.get('search_step', 0.2))
        self.grade_search_attempts = int(config.get('grade_search_attempts', 5))
        self.grade_search_step = float(config.get('grade_search_step', 0.2))

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

    def search_and_scan_at(self, waypoint: Waypoint) -> Dict[str, Any]:
        """
        在给定的近似航点周围做本地搜索，直到检测到 H 并完成等级识别。

        返回值: dict 包含 'success'(bool), 'h_detection', 'grade', 'image_path'
        """
        if not self.drone.move_to(waypoint.x, waypoint.y, waypoint.z):
            return {'success': False, 'reason': 'move_failed'}

        offsets = [
            (0.0, 0.0),
            (self.search_step, 0.0),
            (-self.search_step, 0.0),
            (0.0, self.search_step),
            (0.0, -self.search_step),
            (self.search_step, self.search_step),
            (-self.search_step, -self.search_step),
        ]

        attempt = 0
        while attempt < self.search_max_attempts:
            for ox, oy in offsets:
                target_x = waypoint.x + ox
                target_y = waypoint.y + oy
                if not self.drone.move_to(target_x, target_y, waypoint.z):
                    continue

                frame = self.capture_frame()
                if frame is None:
                    continue

                h_detection = self.detect_frame(frame, self.h_model)
                h_candidate = self.find_best_h(h_detection)
                prefix = f'scan_{waypoint.name}_{attempt}_{int(ox*100)}_{int(oy*100)}'
                h_image_path = self.annotate_and_save(frame, h_detection, prefix + '_h')

                if not h_candidate:
                    print(f"[DroneNavigator] {waypoint.name} 搜索尝试 {attempt}, 偏移 ({ox}, {oy}) → 未检测到 H")
                    time.sleep(0.5)
                    continue

                grade_detection = self.detect_frame(frame, self.grade_model)
                grade_info = self.find_grade_near_h(h_candidate['box'], grade_detection)
                grade_image_path = h_image_path


                print(f"[DroneNavigator] {waypoint.name} 搜索尝试 {attempt}, 偏移 ({ox}, {oy}) → H={h_candidate['label']}({h_candidate['confidence']:.2f}), 等级检测={grade_detection}, 等级候选={grade_info['label']}({grade_info.get('confidence', 0.0):.2f})")
                if grade_info['label'] == 'unknown':
                    grade_info, grade_image_path = self.search_grade_nearby(waypoint, h_candidate)

                return {
                    'success': True,
                    'detection': h_detection,
                    'h_detection': h_candidate,
                    'grade': grade_info,
                    'image_path': grade_image_path,
                }
            attempt += 1

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

    def search_grade_nearby(self, waypoint: Waypoint, h_candidate: Dict[str, Any]) -> tuple[Dict[str, Any], str]:
        offsets = [
            (0.0, 0.0),
            (self.grade_search_step, 0.0),
            (-self.grade_search_step, 0.0),
            (0.0, self.grade_search_step),
            (0.0, -self.grade_search_step),
        ]
        grade_image_path = ''
        for attempt in range(self.grade_search_attempts):
            for ox, oy in offsets:
                target_x = waypoint.x + ox
                target_y = waypoint.y + oy
                if not self.drone.move_to(target_x, target_y, waypoint.z):
                    continue

                frame = self.capture_frame()
                if frame is None:
                    continue

                grade_detection = self.detect_frame(frame, self.grade_model)
                grade_info = self.find_grade_near_h(h_candidate['box'], grade_detection)
                prefix = f'grade_{waypoint.name}_{attempt}_{int(ox*100)}_{int(oy*100)}'
                grade_image_path = self.annotate_and_save(frame, grade_detection, prefix)

                if grade_info['label'] != 'unknown':
                    return grade_info, grade_image_path

                time.sleep(0.5)
        return {'label': 'unknown', 'confidence': 0.0, 'box': [], 'distance': float('inf')}, grade_image_path

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
            result = self.search_and_scan_at(waypoint)

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
        cv2.namedWindow("Stream Test - H & Grade Detection", cv2.WINDOW_NORMAL)

        frame_count = 0
        try:
            while True:
                frame = self.capture_frame()
                if frame is None:
                    print("[DroneNavigator] 无法获取帧，等待重试...")
                    time.sleep(1.0)
                    continue

                frame_count += 1

                # H 检测
                h_detection = self.detect_frame(frame, self.h_model)
                h_candidate = self.find_best_h(h_detection)

                # 等级检测
                grade_info: Dict[str, Any] = {"label": "unknown", "confidence": 0.0}
                if h_candidate is not None:
                    grade_detection = self.detect_frame(frame, self.grade_model)
                    grade_info = self.find_grade_near_h(h_candidate["box"], grade_detection)
                    if grade_info.get("label", "unknown") == "unknown":
                        grade_detection_full = self.detect_frame(frame, self.grade_model)
                        for obj in grade_detection_full.get("objects", []):
                            if obj.get("label", "unknown") != "unknown":
                                grade_info = obj
                                break

                # 显示
                display = frame.copy()
                for obj in h_detection.get("objects", []):
                    x1, y1, x2, y2 = [int(v) for v in obj["box"]]
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(display, f'H {obj["confidence"]:.2f}', (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

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
