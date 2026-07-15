#!/usr/bin/env python3
# coding=utf-8
"""
H 角度检测离线测试脚本
用法:
    python test_h_angle.py --image path/to/image.jpg
    python test_h_angle.py --image path/to/image.jpg --debug

用于在不连接无人机的情况下测试 HAngleDetector 的角度检测效果，
辅助调优 Canny / Hough 参数。
"""
import argparse, cv2, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultralytics import YOLO
from Utils.HAngleDetector import HAngleDetector

def main():
    parser = argparse.ArgumentParser(description='H angle detection test')
    parser.add_argument('--image', required=True, help='test image path')
    parser.add_argument('--model', default='yolov8n_h.pt', help='H model weights')
    parser.add_argument('--conf', type=float, default=0.3, help='YOLO confidence')
    parser.add_argument('--debug', action='store_true', help='show debug images')
    args = parser.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        print(f'Cannot read: {args.image}')
        return
    print(f'Image: {frame.shape[1]}x{frame.shape[0]}')

    # YOLO detect H
    print(f'Loading model: {args.model} ...')
    model = YOLO(args.model)
    results = model.predict(frame, verbose=False, conf=args.conf)
    boxes = getattr(results[0], 'boxes', None)

    if boxes is None or len(boxes) == 0:
        print('No H detected by YOLO!')
        # Show original image anyway
        cv2.imshow('No H detected', frame)
        cv2.waitKey(0)
        return

    best = max(boxes, key=lambda b: float(b.conf[0]))
    xyxy = best.xyxy.tolist()[0]
    conf_val = float(best.conf[0])
    print(f'YOLO H detected: box={xyxy}, conf={conf_val:.2f}')

    # Draw YOLO bbox
    disp = frame.copy()
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(disp, f'H {conf_val:.2f}', (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Detect angle
    result = HAngleDetector.detect_angle(frame, xyxy, debug=args.debug)
    angle = result.get('angle')
    confidence = result.get('confidence', 0)
    method = result.get('method', 'unknown')
    num_lines = result.get('num_lines', 0)

    print(f'Angle: {angle} deg')
    print(f'Confidence: {confidence:.2f}')
    print(f'Method: {method}')
    print(f'Lines: {num_lines}')

    # Show results
    if args.debug and result.get('debug_frame') is not None:
        cv2.imshow('HAngleDetector Debug', result['debug_frame'])
    if args.debug and result.get('edges') is not None:
        cv2.imshow('HAngleDetector Edges', result['edges'])

    # Annotate main image
    if angle is not None:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        import math
        rad = math.radians(angle)
        indicator_len = max(x2 - x1, y2 - y1) // 2
        dx = int(indicator_len * math.sin(rad))
        dy = int(-indicator_len * math.cos(rad))
        cv2.arrowedLine(disp, (cx, cy), (cx + dx, cy + dy),
                        (0, 0, 255), 3, tipLength=0.3)
        cv2.putText(disp, f'Angle: {angle:.1f} deg (conf={confidence:.2f})',
                    (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 255), 2)

    cv2.imshow('H Angle Detection Test', disp)
    print('Press any key to close...')
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
