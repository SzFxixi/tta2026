#!/usr/bin/env python3
# coding=utf-8
"""H flag angle detector using edge detection + Hough lines."""
import cv2
import numpy as np
import math
from typing import Dict, List, Optional


class HAngleDetector:
    CANNY_LOW = 40
    CANNY_HIGH = 120
    CLAHE_CLIP = 2.0
    BLUR_KERNEL = 3
    HOUGH_RHO = 1
    HOUGH_THETA = np.pi / 180
    HOUGH_THRESHOLD = 25
    HOUGH_MIN_LINE_LENGTH = 15
    HOUGH_MAX_LINE_GAP = 8
    MARGIN_RATIO = 0.35
    ANGLE_SEARCH_HALF = 40
    HIST_SMOOTH_KERNEL = 7
    MIN_LINE_COUNT = 3
    MIN_CONFIDENCE = 0.15

    @classmethod
    def detect_angle(cls, frame: np.ndarray, h_box: List[float],
                     margin_ratio: float = None, debug: bool = False) -> Dict:
        """Detect rotation angle of H marker from YOLO bbox region.
        Returns dict with keys: angle, confidence, num_lines, method."""
        if margin_ratio is None:
            margin_ratio = cls.MARGIN_RATIO
        img_h, img_w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in h_box]
        bw, bh = x2 - x1, y2 - y1
        margin_w = int(bw * margin_ratio)
        margin_h = int(bh * margin_ratio)
        cx1 = max(0, x1 - margin_w)
        cy1 = max(0, y1 - margin_h)
        cx2 = min(img_w, x2 + margin_w)
        cy2 = min(img_h, y2 + margin_h)
        if cx2 - cx1 < 15 or cy2 - cy1 < 15:
            return {'angle': None, 'confidence': 0.0, 'num_lines': 0, 'method': 'crop_too_small'}
        crop = frame[cy1:cy2, cx1:cx2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=cls.CLAHE_CLIP, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        gray = cv2.filter2D(gray, -1, sharpen_kernel)
        gray = cv2.GaussianBlur(gray, (cls.BLUR_KERNEL, cls.BLUR_KERNEL), 0)
        edges = cv2.Canny(gray, cls.CANNY_LOW, cls.CANNY_HIGH)
        lines = cv2.HoughLinesP(edges, cls.HOUGH_RHO, cls.HOUGH_THETA,
                                cls.HOUGH_THRESHOLD,
                                minLineLength=cls.HOUGH_MIN_LINE_LENGTH,
                                maxLineGap=cls.HOUGH_MAX_LINE_GAP)
        debug_img = crop.copy() if debug else None
        if lines is None or len(lines) < cls.MIN_LINE_COUNT:
            return cls._fallback_min_area_rect(crop, gray, edges, debug)
        angles_weighted = []
        for line in lines:
            x1l, y1l, x2l, y2l = line[0]
            dx = float(x2l - x1l)
            dy = float(y2l - y1l)
            length = math.sqrt(dx * dx + dy * dy)
            if length < 8:
                continue
            angle = math.degrees(math.atan2(dy, dx)) % 180.0
            angles_weighted.append((angle, length))
            if debug:
                cv2.line(debug_img, (x1l, y1l), (x2l, y2l), (0, 255, 0), 2)
        if len(angles_weighted) < cls.MIN_LINE_COUNT:
            return cls._fallback_min_area_rect(crop, gray, edges, debug)
        histogram = np.zeros(180, dtype=np.float64)
        for angle, weight in angles_weighted:
            b0 = int(angle) % 180
            histogram[b0] += weight
        hist_smooth = cls._smooth_histogram(histogram, cls.HIST_SMOOTH_KERNEL)
        peak_angle, peak_val = cls._find_vertical_peak(hist_smooth, cls.ANGLE_SEARCH_HALF)
        mean_val = np.mean(hist_smooth) + 1e-6
        confidence = min(1.0, peak_val / (mean_val * 2.5))
        if confidence < cls.MIN_CONFIDENCE:
            return cls._fallback_min_area_rect(crop, gray, edges, debug)
        deviation = cls._normalize_deviation(peak_angle)
        if debug:
            cls._draw_debug_overlay(debug_img, deviation, confidence, peak_angle, len(lines))
        return {'angle': deviation, 'confidence': confidence,
                'num_lines': len(lines), 'method': 'hough',
                'debug_frame': debug_img if debug else None,
                'edges': edges if debug else None}

    @staticmethod
    def _smooth_histogram(hist: np.ndarray, kernel_size: int) -> np.ndarray:
        pad = kernel_size * 2
        extended = np.concatenate([hist[-pad:], hist, hist[:pad]])
        kernel = cv2.getGaussianKernel(kernel_size, -1)
        smoothed = np.convolve(extended, kernel.flatten(), mode='same')
        return smoothed[pad:pad + len(hist)]

    @classmethod
    def _find_vertical_peak(cls, hist: np.ndarray, search_half: int):
        n = len(hist)
        center = 90
        start = (center - search_half) % n
        end = (center + search_half) % n
        if start <= end:
            region = hist[start:end]
            idx = int(np.argmax(region))
            peak_angle = float(start + idx)
            peak_val = float(region[idx])
        else:
            region = np.concatenate([hist[start:], hist[:end]])
            idx = int(np.argmax(region))
            if idx < len(hist[start:]):
                peak_angle = float(start + idx)
            else:
                peak_angle = float(idx - len(hist[start:]))
            peak_val = float(region[idx])
        return peak_angle % 180.0, peak_val

    @staticmethod
    def _normalize_deviation(peak_angle: float) -> float:
        deviation = peak_angle - 90.0
        deviation = (deviation + 90.0) % 180.0 - 90.0
        if deviation > 45:
            deviation -= 90.0
        elif deviation < -45:
            deviation += 90.0
        return round(deviation, 2)

    @staticmethod
    def _draw_debug_overlay(img, deviation, confidence, raw_angle, num_lines):
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        cv2.putText(img, f'Deviation: {deviation:.1f} deg', (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(img, f'Confidence: {confidence:.2f}', (8, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(img, f'Raw Angle: {raw_angle:.1f} deg', (8, 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(img, f'Lines: {num_lines}', (8, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        indicator_len = min(w, h) // 3
        rad = math.radians(deviation)
        dx = int(indicator_len * math.sin(rad))
        dy = int(-indicator_len * math.cos(rad))
        cv2.arrowedLine(img, (cx, cy), (cx + dx, cy + dy),
                        (0, 0, 255), 3, tipLength=0.3)
        cv2.line(img, (cx, cy - indicator_len), (cx, cy + indicator_len),
                 (255, 0, 0), 1, cv2.LINE_AA)

    @classmethod
    def _fallback_min_area_rect(cls, crop, gray, edges, debug):
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 15, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        otsu_closed = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=2)
        adaptive_closed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=2)
        best_angle = None
        best_confidence = 0.0
        best_box = None
        for binary_img in [otsu_closed, adaptive_closed]:
            contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            valid_contours = [c for c in contours if cv2.contourArea(c) > 30]
            if not valid_contours:
                continue
            max_contour = max(valid_contours, key=cv2.contourArea)
            area = cv2.contourArea(max_contour)
            crop_area = crop.shape[0] * crop.shape[1]
            if area < crop_area * 0.03:
                continue
            rect = cv2.minAreaRect(max_contour)
            (cx_r, cy_r), (w_r, h_r), angle_r = rect
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            if w_r < h_r:
                vertical_angle = angle_r + 90.0
            else:
                vertical_angle = angle_r
            deviation = cls._normalize_deviation(vertical_angle)
            area_ratio = area / max(crop_area, 1)
            conf = min(0.5, area_ratio * 3.0)
            if conf > best_confidence:
                best_confidence = conf
                best_angle = deviation
                best_box = box
        if best_angle is None:
            if debug:
                debug_img = crop.copy()
                cv2.putText(debug_img, 'FAIL: no contour found', (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                return {'angle': None, 'confidence': 0.0, 'num_lines': 0,
                        'method': 'fallback_failed', 'debug_frame': debug_img}
            return {'angle': None, 'confidence': 0.0, 'num_lines': 0, 'method': 'fallback_failed'}
        if debug and best_box is not None:
            debug_img = crop.copy()
            cv2.drawContours(debug_img, [best_box], 0, (0, 255, 0), 2)
            cv2.putText(debug_img, f'Fallback: {best_angle:.1f} deg', (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.putText(debug_img, f'Conf: {best_confidence:.2f}', (8, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            return {'angle': best_angle, 'confidence': best_confidence,
                    'num_lines': 0, 'method': 'fallback_minrect', 'debug_frame': debug_img}
        return {'angle': best_angle, 'confidence': best_confidence,
                'num_lines': 0, 'method': 'fallback_minrect'}
