#!/usr/bin/env python3
"""
墙壁直线建模定位 — 纯计算模块（不碰 ROS，由调用方传入 scan）。

用法:
    from utils.wall_positioning import fit_walls
    scan = rospy.wait_for_message("scan", LaserScan)
    result = fit_walls(scan)
    # result = {"前墙": 2.30, "右墙": 4.80, "后墙": 2.50, "左墙": 5.10, "yaw": 2.0}
"""

import numpy as np
import math


def fit_walls(scan, sample_count=200, filter_obstacles=True, walls=None):
    """
    从一次 LiDAR 扫描中拟合墙壁，返回各墙垂直距离。

    参数:
        scan:             sensor_msgs/LaserScan 消息
        sample_count:     均匀采样点数
        filter_obstacles: 是否调用障碍物检测过滤
        walls:            需要拟合的墙列表，如 ["前","右"]。None=全部四壁。
    返回:
        dict: {"前墙": dist_m, "右墙": dist_m, "后墙": dist_m, "左墙": dist_m, "yaw": deg}
        拟合失败的墙键值为 None
    """
    if walls is None:
        walls = ["前", "右", "后", "左"]
    need_front = "前" in walls
    need_right = "右" in walls
    need_rear  = "后" in walls
    need_left  = "左" in walls

    # ── 1. 均匀下采样 ──
    n_beams = len(scan.ranges)
    stride = max(1, n_beams // sample_count)
    indices = list(range(0, n_beams, stride))[:sample_count]

    angles = scan.angle_min + np.array(indices) * scan.angle_increment
    ranges = np.array([scan.ranges[i] for i in indices], dtype=np.float64)

    is_valid = (ranges > scan.range_min) & (ranges < scan.range_max) & np.isfinite(ranges)

    # ── 2. 障碍物过滤（复用传入的 scan，不自读） ──
    if filter_obstacles:
        try:
            from entities.obstacle_detector import get_obstacle_beam_mask
            obs_mask_full = get_obstacle_beam_mask(0, 0, scan=scan)
            obs_mask = np.array([obs_mask_full[i] for i in indices], dtype=bool)
            is_valid = is_valid & (~obs_mask)
        except Exception:
            pass

    if is_valid.sum() < 20:
        return {"前墙": None, "右墙": None, "后墙": None, "左墙": None, "yaw": None}

    # ── 3. 笛卡尔坐标 ──
    xs = np.where(is_valid, ranges * np.cos(angles), np.nan)
    ys = np.where(is_valid, ranges * np.sin(angles), np.nan)

    # ── 4. 墙角检测 + 分割 ──
    corner_indices = _find_corners(xs, ys, is_valid, min_sep=len(ranges) // 4)

    valid_idx_map = np.cumsum(is_valid) - 1
    valid_corners = []
    for ci in corner_indices:
        if is_valid[ci]:
            valid_corners.append(valid_idx_map[ci])
        else:
            j = (ci + 1) % len(ranges)
            while j != ci and not is_valid[j]:
                j = (j + 1) % len(ranges)
            if j != ci:
                valid_corners.append(valid_idx_map[j])

    valid_points = np.column_stack((xs, ys))[is_valid]

    if len(valid_corners) >= 4:
        groups = _split_by_corners(valid_points, sorted(valid_corners))
    else:
        groups = _split_by_sectors(angles[is_valid], valid_points)

    # ── 5. RANSAC 只拟合需要的墙 ──
    result = {"前墙": None, "右墙": None, "后墙": None, "左墙": None, "yaw": None}

    for g in groups:
        cx, cy = g[:, 0].mean(), g[:, 1].mean()
        label_en = _wall_label(cx, cy)
        label_cn = {"前墙": "前", "右墙": "右", "后墙": "后", "左墙": "左"}[label_en]
        if label_cn not in walls:
            continue
        line = _ransac_fit_line(g)
        if line is not None:
            result[label_en] = abs(line[2])

    # ── 6. 偏航角（仅当前墙需要时） ──
    if need_front:
        for g in groups:
            cx, cy = g[:, 0].mean(), g[:, 1].mean()
            if _wall_label(cx, cy) == "前墙":
                line = _ransac_fit_line(g)
                if line is not None:
                    a, b = line[0], line[1]
                    result["yaw"] = math.degrees(math.atan2(b, a))
                break

    return result


# ============================================================
#  内部函数
# ============================================================

def _ransac_fit_line(points, max_iters=80, inlier_thresh=0.05, min_inliers=5):
    n = len(points)
    if n < 2:
        return None
    best_inliers = []
    best_line = None
    for _ in range(max_iters):
        i, j = np.random.choice(n, 2, replace=False)
        p1, p2 = points[i], points[j]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        if dx * dx + dy * dy < 1e-12:
            continue
        norm = math.hypot(dx, dy)
        a, b = dy / norm, -dx / norm
        c = -(a * p1[0] + b * p1[1])
        if c > 0:
            a, b, c = -a, -b, -c
        dists = np.abs(a * points[:, 0] + b * points[:, 1] + c)
        mask = dists < inlier_thresh
        if mask.sum() > len(best_inliers):
            best_inliers = np.where(mask)[0]
            best_line = (a, b, c)
    if best_line is None or len(best_inliers) < min_inliers:
        return None
    inlier_pts = points[best_inliers]
    centroid = inlier_pts.mean(axis=0)
    _, _, vh = np.linalg.svd(inlier_pts - centroid)
    a, b = vh[-1]
    c = -(a * centroid[0] + b * centroid[1])
    if c > 0:
        a, b, c = -a, -b, -c
    norm = math.hypot(a, b)
    return (a / norm, b / norm, c / norm, len(best_inliers))


def _find_corners(xs, ys, is_valid, min_sep=30):
    n = len(xs)
    valid_idx = np.where(is_valid)[0]
    n_valid = len(valid_idx)
    if n_valid < 4:
        return []
    gaps = []
    for vi in range(n_valid):
        vi_next = (vi + 1) % n_valid
        fi, ti = valid_idx[vi], valid_idx[vi_next]
        span = (ti - fi) if ti > fi else (n - fi + ti)
        inf_count = span - 1
        dx, dy = xs[ti] - xs[fi], ys[ti] - ys[fi]
        gap = math.hypot(dx, dy)
        mid = (fi + span // 2) % n
        gaps.append((mid, gap + inf_count * 3.0))  # score = gap + inf_bonus
    gaps.sort(key=lambda x: -x[1])
    selected = []
    for mid, _ in gaps:
        if all(min(abs(mid - s), n - abs(mid - s)) >= min_sep for s in selected):
            selected.append(mid)
        if len(selected) >= 4:
            break
    return sorted(selected)


def _split_by_corners(points, corner_indices):
    n = len(points)
    groups, prev = [], corner_indices[-1]
    for ci in corner_indices:
        g = points[prev + 1:ci + 1] if ci > prev else np.vstack((points[prev + 1:], points[:ci + 1]))
        prev = ci
        if len(g) >= 8:
            groups.append(g)
    return groups


def _split_by_sectors(angles, points):
    masks = [
        (angles >= -math.pi / 4) & (angles <= math.pi / 4),
        (angles >= -3 * math.pi / 4) & (angles <= -math.pi / 4),
        (angles >= 3 * math.pi / 4) | (angles <= -3 * math.pi / 4),
        (angles >= math.pi / 4) & (angles <= 3 * math.pi / 4),
    ]
    return [points[m] for m in masks if m.sum() >= 8]


def _wall_label(cx, cy):
    ang = math.degrees(math.atan2(cy, cx))
    if -45 <= ang <= 45:       return "前墙"
    elif 45 < ang <= 135:      return "左墙"
    elif ang > 135 or ang <= -135: return "后墙"
    else:                       return "右墙"
