#!/usr/bin/env python3
# 从 LiDAR 距离突变 + 墙壁偏差双重检测障碍物

import rospy
import numpy as np
import math
from sensor_msgs.msg import LaserScan

#  参数 — 从 config.yaml 读取
from utils.config_loader import cfg

LIDAR_TIMEOUT = cfg.obstacle.lidar_timeout
JUMP_THRESHOLD = cfg.obstacle.jump_threshold
EXPAND_JUMP_RATIO = cfg.obstacle.expand_jump_ratio
CLUSTER_MIN_BEAMS = cfg.obstacle.cluster_min_beams
CLUSTER_MAX_GAP = cfg.obstacle.cluster_max_gap
CROSS_ZERO_MERGE_MARGIN = cfg.obstacle.cross_zero_merge_margin
WALL_DEVIATION_THRESHOLD = cfg.obstacle.wall_deviation_threshold
CORNER_MARGIN_RAD = math.radians(cfg.obstacle.corner_margin_deg)
ROOM_X_MIN = cfg.obstacle.filter_x_min
ROOM_X_MAX = cfg.obstacle.filter_x_max
ROOM_Y_MIN = cfg.obstacle.filter_y_min
ROOM_Y_MAX = cfg.obstacle.filter_y_max

#  工具函数
def _compute_diag(n_beams, valid_count, jump_count, all_dists):
    """组装诊断字典"""
    n = len(all_dists)
    return {
        "total_beams": n_beams,
        "valid_beams": valid_count,
        "jump_count": jump_count,
        "dist_min":   round(all_dists[0], 3) if n else 0,
        "dist_p10":   round(all_dists[n // 10], 3) if n > 10 else 0,
        "dist_p50":   round(all_dists[n // 2], 3) if n else 0,
        "dist_p90":   round(all_dists[n * 9 // 10], 3) if n > 10 else 0,
        "dist_max":   round(all_dists[-1], 3) if n else 0,
    }

#  第一步：逐束分析
def _analyze_beams(data, car_x, car_y):
    """遍历所有激光光束，标记有效性和命中点坐标。"""
    n_beams = len(data.ranges)
    angle_min = data.angle_min
    angle_inc = data.angle_increment

    hit_dist, hit_angles, valid_mask = [], [], []

    for i in range(n_beams):
        d = data.ranges[i]
        if d == np.inf or d < data.range_min or d > data.range_max:
            valid_mask.append(False)
            hit_dist.append(None)
            hit_angles.append(None)
            continue
        valid_mask.append(True)
        hit_dist.append(d)
        angle = angle_min + i * angle_inc
        hit_angles.append(angle)

    valid_count = sum(valid_mask)
    return hit_dist, hit_angles, valid_mask, n_beams, valid_count

#  第二步：突变检测
def _detect_jumps(hit_dist, valid_mask, n_beams, jump_threshold):
    
    jump_marks = [False] * n_beams
    prev_valid = None
    for i in range(n_beams):
        if not valid_mask[i]:
            continue
        if prev_valid is not None:
            if abs(hit_dist[i] - hit_dist[prev_valid]) > jump_threshold:
                jump_marks[prev_valid] = True
        prev_valid = i

    # 环形闭合：首尾有效光束之间也检查
    first_valid = next((i for i in range(n_beams) if valid_mask[i]), None)
    if prev_valid is not None and first_valid is not None and prev_valid != first_valid:
        if abs(hit_dist[first_valid] - hit_dist[prev_valid]) > jump_threshold:
            jump_marks[prev_valid] = True

    jump_indices = [i for i, m in enumerate(jump_marks) if m]
    return jump_indices

#  第三步：种子扩展
def _expand_obstacle_beams(hit_dist, valid_mask, n_beams,
                            jump_indices, jump_threshold):
    
    obstacle_beam = [False] * n_beams
    jump_limit = jump_threshold * EXPAND_JUMP_RATIO

    for jump_idx in jump_indices:
        # 找到突变两侧最近的相邻有效光束
        prev_v = next((jump_idx - d) % n_beams for d in range(1, n_beams)
                      if valid_mask[(jump_idx - d) % n_beams])
        next_v = (jump_idx + 1) % n_beams
        while not valid_mask[next_v]:
            next_v = (next_v + 1) % n_beams

        d_prev, d_next = hit_dist[prev_v], hit_dist[next_v]

        # 选择距离更近的一侧作为种子
        if d_next < d_prev:
            seed = next_v
        else:
            seed = prev_v

        # 向后扩展
        cur = seed
        while True:
            obstacle_beam[cur] = True
            pc = (cur - 1) % n_beams
            if not valid_mask[pc]:
                break
            if abs(hit_dist[pc] - hit_dist[cur]) > jump_limit:
                break
            cur = pc

        # 向前扩展
        cur = seed
        while True:
            obstacle_beam[cur] = True
            nc = (cur + 1) % n_beams
            if not valid_mask[nc]:
                break
            if abs(hit_dist[nc] - hit_dist[cur]) > jump_limit:
                break
            cur = nc

    return obstacle_beam

#  第四步：聚类
def _cluster_beams(obstacle_beam, n_beams, cluster_min_beams):
    """将标记的障碍物光束按连续性分组为簇"""
    obs_indices = [i for i, is_obs in enumerate(obstacle_beam) if is_obs]
    if not obs_indices:
        return []

    clusters = []
    current = [obs_indices[0]]
    for i in range(1, len(obs_indices)):
        if obs_indices[i] - obs_indices[i - 1] <= CLUSTER_MAX_GAP:
            current.append(obs_indices[i])
        else:
            if len(current) >= cluster_min_beams:
                clusters.append(current)
            current = [obs_indices[i]]
    if len(current) >= cluster_min_beams:
        clusters.append(current)

    # 跨 0° 合并
    if len(clusters) >= 2:
        first, last = clusters[0], clusters[-1]
        if first[0] < CROSS_ZERO_MERGE_MARGIN and last[-1] > n_beams - CROSS_ZERO_MERGE_MARGIN:
            merged = last + [idx + n_beams for idx in first]
            clusters = clusters[1:-1]
            if len(merged) >= cluster_min_beams:
                clusters.append(merged)

    return clusters

#  第五步：计算障碍物坐标
def _compute_obstacles(clusters, hit_dist, hit_angles, car_x, car_y, n_beams):
    """每个簇取中间光束，计算障碍物绝对坐标"""
    obstacles = []
    for cluster in clusters:
        mid = cluster[len(cluster) // 2] % n_beams
        d, angle = hit_dist[mid], hit_angles[mid]
        obstacles.append({
            "x": round(car_x - d * math.cos(angle), 3),
            "y": round(car_y + d * math.sin(angle), 3),
            "distance": round(d, 3),
            "angle_deg": round(math.degrees(angle), 1),
            "beam_count": len(cluster),
        })
    return obstacles

#  房间边界过滤
def _filter_out_of_bounds(obstacles):
    """过滤坐标超出房间有效范围的障碍物（视为噪声）。"""
    kept = []
    dropped = 0
    for obs in obstacles:
        if ROOM_X_MIN <= obs['x'] <= ROOM_X_MAX and ROOM_Y_MIN <= obs['y'] <= ROOM_Y_MAX:
            kept.append(obs)
        else:
            dropped += 1
    return kept, dropped

#  禁区过滤
def _filter_forbidden_zones(obstacles, forbidden_zones):
    """过滤落入禁区内的障碍物（禁区内的不是障碍物，是墙/平台）。"""
    if not forbidden_zones:
        return obstacles, 0
    from entities.forbidden_zones import point_in_forbidden
    kept = []
    dropped = 0
    for obs in obstacles:
        if point_in_forbidden(obs['x'], obs['y'], forbidden_zones):
            dropped += 1
        else:
            kept.append(obs)
    return kept, dropped

#  主入口
# 墙壁角度分区边界（弧度），与 wall_positioning._split_by_sectors 一致
_SECTOR_BOUNDARIES = [-3*math.pi/4, -math.pi/4, math.pi/4, 3*math.pi/4]

def _beam_to_wall(angle_rad):
    """按光束角度判断应该打中哪面墙。返回墙壁标签或 None（墙角区）。"""
    # 标准化到 [-π, π]
    while angle_rad > math.pi:
        angle_rad -= 2 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2 * math.pi

    # 检查是否在墙角边缘区内
    for b in _SECTOR_BOUNDARIES:
        if abs(angle_rad - b) < CORNER_MARGIN_RAD:
            return None
    # 也检查 ±π 边界（后墙的跨越点）
    if abs(angle_rad) > math.pi - CORNER_MARGIN_RAD:
        return None

    if -math.pi/4 <= angle_rad <= math.pi/4:
        return "前墙"
    elif math.pi/4 < angle_rad <= 3*math.pi/4:
        return "左墙"
    elif angle_rad > 3*math.pi/4 or angle_rad < -3*math.pi/4:
        return "后墙"
    else:
        return "右墙"

def _detect_by_wall_deviation(hit_dist, hit_angles, valid_mask, n_beams,
                               wall_lines, deviation_threshold):
    
    obstacle_beam = np.zeros(n_beams, dtype=bool)

    for i in range(n_beams):
        if not valid_mask[i]:
            continue

        angle = hit_angles[i]
        label = _beam_to_wall(angle)
        if label is None or label not in wall_lines:
            continue

        a, b, c = wall_lines[label][:3]

        # 光束方向与法向量的点积：< 0 说明光束射向墙壁
        dot = a * math.cos(angle) + b * math.sin(angle)
        if dot >= 0:
            continue

        d_expected = -c / dot
        d_actual = hit_dist[i]

        if d_expected - d_actual > deviation_threshold:
            obstacle_beam[i] = True

    return obstacle_beam

#  主入口
def detect_obstacles(car_x, car_y,
                     forbidden_zones=None,
                     jump_threshold=JUMP_THRESHOLD,
                     cluster_min_beams=CLUSTER_MIN_BEAMS):
    
    data = rospy.wait_for_message("scan", LaserScan, timeout=LIDAR_TIMEOUT)

    # 用同一帧 LiDAR 做墙壁建模（内部会跑 get_obstacle_beam_mask 并缓存结果）
    try:
        from utils.wall_positioning import fit_walls, get_cached_beam_analysis
        walls = fit_walls(data, filter_obstacles=True)
        wall_lines = walls.get("_lines", {})
        cached = get_cached_beam_analysis()
    except Exception:
        wall_lines = {}
        cached = None

    if cached is not None:
        hit_dist, hit_angles, valid_mask, n_beams, valid_count, jump_beam = cached
        jump_beam = list(jump_beam)
    else:
        hit_dist, hit_angles, valid_mask, n_beams, valid_count = \
            _analyze_beams(data, car_x, car_y)

        if valid_count == 0:
            return [], {"total_beams": n_beams, "valid_beams": 0}

        jump_indices = _detect_jumps(hit_dist, valid_mask, n_beams, jump_threshold)
        jump_beam = _expand_obstacle_beams(
            hit_dist, valid_mask, n_beams, jump_indices, jump_threshold) \
            if len(jump_indices) >= 2 else [False] * n_beams

    if valid_count == 0:
        return [], {"total_beams": n_beams, "valid_beams": 0}

    # ── 墙壁偏差检测 ──
    wall_dev_beam = _detect_by_wall_deviation(
        hit_dist, hit_angles, valid_mask, n_beams,
        wall_lines, WALL_DEVIATION_THRESHOLD) if wall_lines else None

    # ── 融合 ──
    if wall_dev_beam is not None:
        obstacle_beam = [jump_beam[i] or wall_dev_beam[i] for i in range(n_beams)]
    else:
        obstacle_beam = jump_beam

    all_dists = sorted([hit_dist[i] for i in range(n_beams) if valid_mask[i]])
    diag = _compute_diag(n_beams, valid_count, sum(jump_beam), all_dists)
    if wall_dev_beam is not None:
        diag["wall_dev_beams"] = int(wall_dev_beam.sum())

    if not any(obstacle_beam):
        return [], diag

    clusters = _cluster_beams(obstacle_beam, n_beams, cluster_min_beams)
    if not clusters:
        return [], diag

    obstacles = _compute_obstacles(clusters, hit_dist, hit_angles, car_x, car_y, n_beams)
    obstacles, dropped = _filter_out_of_bounds(obstacles)
    diag['filtered_out'] = dropped
    obstacles, filtered_fz = _filter_forbidden_zones(obstacles, forbidden_zones)
    diag['filtered_forbidden'] = filtered_fz
    return obstacles, diag

# ── 独立测试入口 ──
if __name__ == "__main__":
    import sys
    rospy.init_node("obs_test", anonymous=True)
    rospy.wait_for_message("scan", LaserScan, timeout=10.0)
    print("障碍物检测模块就绪")
