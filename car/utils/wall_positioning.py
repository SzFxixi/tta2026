import numpy as np
import math
import time

# 跳变检测结果缓存，避免 detect_obstacles 和 fit_walls 重复计算同一帧 LiDAR
_beam_cache = None
_beam_cache_time = 0
_BEAM_CACHE_TTL = 0.05

def fit_walls(scan, sample_count=200, filter_obstacles=True, walls=None):
    if walls is None:
        walls = ["前", "右", "后", "左"]
    need_front = "前" in walls
    need_right = "右" in walls
    need_rear  = "后" in walls
    need_left  = "左" in walls

    n_beams = len(scan.ranges)
    stride = max(1, n_beams // sample_count)
    indices = list(range(0, n_beams, stride))[:sample_count]

    angles = scan.angle_min + np.array(indices) * scan.angle_increment
    ranges = np.array([scan.ranges[i] for i in indices], dtype=np.float64)

    is_valid = (ranges > scan.range_min) & (ranges < scan.range_max) & np.isfinite(ranges)

    if filter_obstacles:
        try:
            obs_mask_full = get_obstacle_beam_mask(0, 0, scan=scan)
            obs_mask = np.array([obs_mask_full[i] for i in indices], dtype=bool)
            is_valid = is_valid & (~obs_mask)
        except Exception:
            pass

    if is_valid.sum() < 20:
        return {"前墙": None, "右墙": None, "后墙": None, "左墙": None, "yaw": None}

    xs = np.where(is_valid, ranges * np.cos(angles), np.nan)
    ys = np.where(is_valid, ranges * np.sin(angles), np.nan)

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

    result = {"前墙": None, "右墙": None, "后墙": None, "左墙": None, "yaw": None}
    _lines = {}

    for g in groups:
        cx, cy = g[:, 0].mean(), g[:, 1].mean()
        label_en = _wall_label(cx, cy)
        label_cn = {"前墙": "前", "右墙": "右", "后墙": "后", "左墙": "左"}[label_en]
        if label_cn not in walls:
            continue
        line = _ransac_fit_line(g)
        if line is not None:
            result[label_en] = abs(line[2])
            _lines[label_en] = line

    def _yaw_from_normal(a, b):
        return math.degrees(math.atan2(b, a))

    def _normalize_angle(deg):
        while deg > 180:
            deg -= 360
        while deg < -180:
            deg += 360
        return deg

    MIN_INLIERS = 10

    yaw_front = None
    yaw_left = None

    if "前墙" in _lines and _lines["前墙"][3] >= MIN_INLIERS:
        a, b = _lines["前墙"][0], _lines["前墙"][1]
        yaw_front = _yaw_from_normal(a, b)

    if "左墙" in _lines and _lines["左墙"][3] >= MIN_INLIERS:
        a, b = _lines["左墙"][0], _lines["左墙"][1]
        raw = _yaw_from_normal(a, b)
        yaw_left = _normalize_angle(raw + 90)

    if yaw_front is not None and yaw_left is not None:
        if abs(_normalize_angle(yaw_front - yaw_left)) < 10:
            result["yaw"] = yaw_front
        else:
            better = "前墙" if _lines["前墙"][3] >= _lines["左墙"][3] else "左墙"
            result["yaw"] = yaw_front if better == "前墙" else yaw_left
    elif yaw_front is not None:
        result["yaw"] = yaw_front
    elif yaw_left is not None:
        result["yaw"] = yaw_left

    result["_lines"] = _lines
    return result

#  内部函数
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
        gaps.append((mid, gap + inf_count * 3.0))
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

def get_obstacle_beam_mask(car_x, car_y, scan=None, jump_threshold=None):
    
    import rospy
    from sensor_msgs.msg import LaserScan
    from entities.obstacle_detector import _analyze_beams, _detect_jumps, _expand_obstacle_beams
    from utils.config_loader import cfg as _cfg

    if scan is None:
        scan = rospy.wait_for_message("scan", LaserScan, timeout=_cfg.obstacle.lidar_timeout)
    if jump_threshold is None:
        jump_threshold = _cfg.obstacle.jump_threshold

    hit_dist, hit_angles, valid_mask, n_beams, valid_count = \
        _analyze_beams(scan, car_x, car_y)

    if valid_count == 0:
        obstacle_beam = np.zeros(n_beams, dtype=bool)
        global _beam_cache, _beam_cache_time
        _beam_cache = (hit_dist, hit_angles, valid_mask, n_beams, valid_count, obstacle_beam)
        _beam_cache_time = time.time()
        return obstacle_beam

    jump_indices = _detect_jumps(hit_dist, valid_mask, n_beams, jump_threshold)
    if len(jump_indices) < 2:
        obstacle_beam = np.zeros(n_beams, dtype=bool)
        _beam_cache = (hit_dist, hit_angles, valid_mask, n_beams, valid_count, obstacle_beam)
        _beam_cache_time = time.time()
        return obstacle_beam

    obstacle_beam = _expand_obstacle_beams(
        hit_dist, valid_mask, n_beams, jump_indices, jump_threshold)
    obstacle_beam = np.array(obstacle_beam, dtype=bool)

    # 缓存完整结果（含 jump_beam），供 detect_obstacles 复用
    _beam_cache = (hit_dist, hit_angles, valid_mask, n_beams, valid_count, obstacle_beam)
    _beam_cache_time = time.time()
    return obstacle_beam

def get_cached_beam_analysis():
    
    if _beam_cache is not None and (time.time() - _beam_cache_time) < _BEAM_CACHE_TTL:
        return _beam_cache
    return None
