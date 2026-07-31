#!/usr/bin/env python3
# coding=utf-8
"""
离线测试脚本：识别指定文件夹内所有图像的等级
============================================
用法:
    python test_grade_images.py --folder <图像文件夹路径> [--config configs/rescue_config.json] [--output test_output]

输出:
    - 控制台: 每张图的识别结果
    - {output}/result.csv: 汇总表格
    - {output}/*.jpg:  带标注的图像
"""

import os
import sys
import csv
import argparse
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from Utils.JsonHelper import JsonHelper


def load_config(config_path: str) -> Dict[str, Any]:
    return JsonHelper.load_json(config_path)


def build_grade_mapping(raw_mapping: Dict[str, Any]) -> Dict[str, str]:
    """构造 {label_lower: level_str} 映射，与 DroneNavigator 逻辑一致。"""
    mapping: Dict[str, str] = {}
    for level, labels in raw_mapping.items():
        if isinstance(labels, list):
            for label in labels:
                mapping[label.lower()] = str(level)
        else:
            mapping[str(labels).lower()] = str(level)
    return mapping


def detect_frame(frame: np.ndarray, model: YOLO, confidence: float) -> Dict[str, Any]:
    """对单帧运行 YOLO 推理，返回检测结果。"""
    results = model.predict(frame, verbose=False, conf=confidence)
    detection: Dict[str, Any] = {
        'objects': [],
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
    }
    if not results:
        return detection
    boxes = getattr(results[0], 'boxes', None)
    if boxes is None:
        return detection
    for box in boxes:
        xyxy = box.xyxy.tolist()[0]
        conf_val = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
        class_id = int(box.cls[0]) if hasattr(box, 'cls') else -1
        label = model.names.get(class_id, str(class_id))
        detection['objects'].append({
            'label': label,
            'confidence': conf_val,
            'box': [float(x) for x in xyxy],
        })
    return detection


def find_best_h(detection: Dict[str, Any], h_label: str) -> Optional[Dict[str, Any]]:
    candidates = [
        obj for obj in detection['objects']
        if obj['label'] == h_label or h_label.lower() in obj['label'].lower()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item['confidence'])


def find_grade_near_h(h_box: List[float], grade_detection: Dict[str, Any],
                      grade_labels: List[str], grade_distance_scale: float) -> Dict[str, Any]:
    """在 H 附近寻找等级目标，与 DroneNavigator.find_grade_near_h 逻辑一致。"""
    if not grade_detection['objects']:
        return {'label': 'unknown', 'confidence': 0.0, 'box': [], 'distance': float('inf')}

    hx1, hy1, hx2, hy2 = h_box
    h_cx = (hx1 + hx2) / 2.0
    h_cy = (hy1 + hy2) / 2.0
    h_size = max(hx2 - hx1, hy2 - hy1)
    max_distance = h_size * grade_distance_scale

    selected = None
    best_score = float('inf')

    for obj in grade_detection['objects']:
        if grade_labels and obj['label'] not in grade_labels:
            continue
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


def annotate_and_save(frame: np.ndarray, h_candidate: Optional[Dict],
                      grade_info: Dict, grade_objects: List[Dict],
                      output_dir: str, filename: str) -> str:
    """保存带标注的图像。"""
    annotated = frame.copy()

    # 画 H
    if h_candidate:
        x1, y1, x2, y2 = [int(v) for v in h_candidate['box']]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(annotated, f"H:{h_candidate['confidence']:.2f}",
                    (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # 画等级目标
    for obj in grade_objects:
        x1, y1, x2, y2 = [int(v) for v in obj['box']]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"{obj['label']}:{obj['confidence']:.2f}",
                    (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 在左上角写最终等级
    level_text = f"Grade: {grade_info.get('label', 'unknown')}"
    cv2.putText(annotated, level_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, annotated)
    return out_path


def process_image(image_path: str, h_model: YOLO, grade_model: YOLO,
                  config: Dict[str, Any]) -> Dict[str, Any]:
    """对单张图像执行等级识别，返回结果字典。"""
    frame = cv2.imread(image_path)
    if frame is None:
        return {'filename': os.path.basename(image_path), 'error': '无法读取图像'}

    # 亮度增强（与 detect_all 一致）
    enhanced = cv2.convertScaleAbs(frame, alpha=1.3, beta=10)

    confidence = float(config.get('confidence', 0.6))
    h_label = config.get('h_label', 'H')
    grade_labels = config.get('grade_labels', [])
    grade_labels = grade_labels if isinstance(grade_labels, list) else [grade_labels]
    grade_distance_scale = float(config.get('grade_distance_scale', 2.0))
    grade_mapping = build_grade_mapping(config.get('grade_mapping', {}))

    # 检测 H
    h_detection = detect_frame(enhanced, h_model, confidence)
    h_candidate = find_best_h(h_detection, h_label)

    # 检测等级
    grade_detection = detect_frame(enhanced, grade_model, confidence)

    # 确定等级
    if h_candidate is not None:
        grade_info = find_grade_near_h(h_candidate['box'], grade_detection,
                                        grade_labels, grade_distance_scale)
    else:
        grade_objects = grade_detection.get('objects', [])
        if grade_objects:
            best = max(grade_objects, key=lambda obj: obj.get('confidence', 0))
            grade_info = {
                'label': best['label'],
                'confidence': best['confidence'],
                'box': best['box'],
                'distance': 0.0,
            }
        else:
            grade_info = {'label': 'unknown', 'confidence': 0.0, 'box': [], 'distance': float('inf')}

    raw_label = grade_info.get('label', 'unknown')
    mapped_grade = grade_mapping.get(raw_label.lower(), raw_label)

    return {
        'filename': os.path.basename(image_path),
        'h_detected': h_candidate is not None,
        'h_label': h_candidate['label'] if h_candidate else 'none',
        'h_confidence': h_candidate['confidence'] if h_candidate else 0.0,
        'raw_label': raw_label,
        'grade': mapped_grade,
        'grade_confidence': grade_info.get('confidence', 0.0),
        'grade_objects': grade_detection.get('objects', []),
        'h_candidate': h_candidate,
        'frame': frame,
    }


def main():
    parser = argparse.ArgumentParser(description='离线测试：批量识别图像等级')
    parser.add_argument('--folder', type=str, required=True, help='图像文件夹路径')
    parser.add_argument('--config', type=str, default='configs/rescue_config.json', help='配置文件路径')
    parser.add_argument('--output', type=str, default='test_output', help='输出目录 (默认: test_output)')
    parser.add_argument('--no-annotate', action='store_true', help='不保存标注图像')
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"[ERROR] 文件夹不存在: {args.folder}")
        sys.exit(1)

    config = load_config(args.config)

    # 模型路径
    h_weights = config.get('yolo_weights_h', '') or config.get('yolo_weights', 'yolov8n.pt')
    grade_weights = config.get('yolo_weights_grade', '') or config.get('yolo_weights', 'yolov8n.pt')

    print(f"加载 H 模型: {h_weights}")
    h_model = YOLO(h_weights)
    print(f"加载等级模型: {grade_weights}")
    grade_model = YOLO(grade_weights)

    # 等级映射 (用于结果显示)
    grade_mapping = build_grade_mapping(config.get('grade_mapping', {}))

    # 输出目录
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # 收集图像
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = sorted([
        os.path.join(args.folder, f) for f in os.listdir(args.folder)
        if f.lower().endswith(exts)
    ])

    if not image_files:
        print(f"[ERROR] 文件夹内没有图像文件: {args.folder}")
        sys.exit(1)

    print(f"\n找到 {len(image_files)} 张图像，开始识别...\n")
    header = f"{'文件名':<28} {'H':<8} {'等级':<6} {'置信度':<8} {'所有检测'} "
    print(header)
    print("-" * 90)

    results = []
    for img_path in image_files:
        result = process_image(img_path, h_model, grade_model, config)
        results.append(result)

        fname = result['filename']
        if 'error' in result:
            print(f"{fname:<28} {'ERROR':<8} {'-':<6} {'-':<8} {result['error']}")
            continue

        h_str = result['h_label'] if result['h_detected'] else 'none'

        # 汇总所有检测到的等级标签及其置信度
        grade_objs = result.get('grade_objects', [])
        all_detected = []
        for obj in grade_objs:
            lbl = obj['label']
            cf = obj['confidence']
            mapped_lvl = grade_mapping.get(lbl.lower(), '?')
            all_detected.append(f"{lbl}(L{mapped_lvl}/{cf:.2f})")
        detected_summary = ', '.join(all_detected) if all_detected else '(无)'

        print(f"{fname:<28} {h_str:<8} {result['grade']:<6} {result['grade_confidence']:<8.3f} {detected_summary}")

        # 保存标注图像
        if not args.no_annotate:
            name_base = os.path.splitext(fname)[0]
            annotate_and_save(
                result['frame'], result['h_candidate'],
                {'label': result['grade'], 'confidence': result['grade_confidence']},
                result['grade_objects'], output_dir,
                f"{name_base}_annotated.jpg"
            )

    # 写入 CSV
    csv_path = os.path.join(output_dir, 'result.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['文件名', 'H检测', 'H标签', 'H置信度', '原始标签', '等级', '等级置信度'])
        for r in results:
            if 'error' in r:
                writer.writerow([r['filename'], '', '', '', '', '', r['error']])
            else:
                writer.writerow([
                    r['filename'], r['h_detected'], r['h_label'],
                    f"{r['h_confidence']:.3f}", r['raw_label'],
                    r['grade'], f"{r['grade_confidence']:.3f}",
                ])

    # ── 统计汇总 ──
    grade_mapping = build_grade_mapping(config.get('grade_mapping', {}))
    valid = [r for r in results if 'error' not in r]

    # 按等级统计
    grade_count: Dict[str, int] = {}
    grade_confs: Dict[str, List[float]] = {}
    for r in valid:
        g = r['grade']
        grade_count[g] = grade_count.get(g, 0) + 1
        if r['grade_confidence'] > 0:
            grade_confs.setdefault(g, []).append(r['grade_confidence'])

    # 按原始标签统计 (所有检测到的对象)
    label_count: Dict[str, int] = {}
    label_confs: Dict[str, List[float]] = {}
    for r in valid:
        for obj in r.get('grade_objects', []):
            lbl = obj['label']
            label_count[lbl] = label_count.get(lbl, 0) + 1
            label_confs.setdefault(lbl, []).append(obj['confidence'])

    print(f"\n{'='*60}")
    print(f"  📊 识别统计: {len(valid)}/{len(results)} 张")
    print(f"{'='*60}")

    # 等级分布
    print(f"\n  ┌─ 等级分布 ─────────────────────────────┐")
    for g in ['1', '2', '3', 'unknown']:
        cnt = grade_count.get(g, 0)
        confs = grade_confs.get(g, [])
        avg_c = f"{sum(confs)/len(confs):.3f}" if confs else '-'
        bar = '█' * max(1, cnt)
        print(f"  │ 等级{g}: {cnt:>3} 张  {bar:<10} 均置信度={avg_c}  │")
    print(f"  └──────────────────────────────────────────┘")

    # 原始标签分布 (按等级分组)
    print(f"\n  ┌─ 各标签检测详情 ────────────────────────┐")
    level_order = ['1', '2', '3', 'unknown']
    for lvl in level_order:
        lvl_labels = [lb for lb, lv in grade_mapping.items() if lv == lvl]
        shown = False
        for lbl in lvl_labels:
            cnt = label_count.get(lbl, 0)
            if cnt > 0:
                confs = label_confs.get(lbl, [])
                avg_c = f"{sum(confs)/len(confs):.3f}" if confs else '-'
                max_c = f"{max(confs):.3f}" if confs else '-'
                if not shown:
                    print(f"  │  [等级{lvl}]")
                    shown = True
                print(f"  │    {lbl:<25} {cnt:>3}次  均{avg_c}  最高{max_c}")
        # 未映射的标签
        for lbl, cnt in label_count.items():
            if lbl not in grade_mapping and cnt > 0:
                confs = label_confs.get(lbl, [])
                avg_c = f"{sum(confs)/len(confs):.3f}" if confs else '-'
                print(f"  │  [?] {lbl:<25} {cnt:>3}次  均{avg_c}  (未映射)")
    print(f"  └──────────────────────────────────────────┘")

    print(f"\n  CSV 汇总: {csv_path}")
    if not args.no_annotate:
        print(f"  标注图像: {output_dir}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
