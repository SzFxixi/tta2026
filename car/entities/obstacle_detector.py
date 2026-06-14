#!/usr/bin/env python3
# 从 LiDAR 距离突变中检测障碍物

import rospy
import numpy as np
import math
from sensor_msgs.msg import LaserScan


# ============================================================
#  参数 — 从 config.yaml 读取
# ============================================================

from utils.config_loader import cfg

LIDAR_TIMEOUT = cfg.obstacle.lidar_timeout
JUMP_THRESHOLD = cfg.obstacle.jump_threshold
EXPAND_JUMP_RATIO = cfg.obstacle.expand_jump_ratio
CLUSTER_MIN_BEAMS = cfg.obstacle.cluster_min_beams
CLUSTER_MAX_GAP = cfg.obstacle.cluster_max_gap
CROSS_ZERO_MERGE_MARGIN = cfg.obstacle.cross_zero_merge_margin
ROOM_X_MIN = cfg.obstacle.filter_x_min
ROOM_X_MAX = cfg.obstacle.filter_x_max
ROOM_Y_MIN = cfg.obstacle.filter_y_min
ROOM_Y_MAX = cfg.obstacle.filter_y_max


# ============================================================
#  工具函数
# ============================================================

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


# ============================================================
#  第一步：逐束分析
# ============================================================

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


# ============================================================
#  第二步：突变检测
# ============================================================

def _detect_jumps(hit_dist, valid_mask, n_beams, jump_threshold):
    """
    检测相邻有效光束距离差超过固定阈值的突变点。

    返回:
        jump_indices
    """
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


# ============================================================
#  第三步：种子扩展
# ============================================================

def _expand_obstacle_beams(hit_dist, valid_mask, n_beams,
                            jump_indices, jump_threshold):
    """
    从每个突变点出发，向两侧扩展出完整的障碍物光束集合。

    扩展规则：从突变点的较近一侧出发，向两侧延伸。
    遇到相邻光束距离差超过 jump_limit 时停止。
    """
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


# ============================================================
#  第四步：聚类
# ============================================================

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


# ============================================================
#  第五步：计算障碍物坐标
# ============================================================

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


# ============================================================
#  房间边界过滤
# ============================================================

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


# ============================================================
#  禁区过滤
# ============================================================

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


# ============================================================
#  主入口
# ============================================================

def detect_obstacles(car_x, car_y,
                     forbidden_zones=None,
                     jump_threshold=JUMP_THRESHOLD,
                     cluster_min_beams=CLUSTER_MIN_BEAMS):
    """通过 LiDAR 距离突变检测障碍物。返回 (obstacles, diagnostics)。"""
    data = rospy.wait_for_message("scan", LaserScan, timeout=LIDAR_TIMEOUT)

    hit_dist, hit_angles, valid_mask, n_beams, valid_count = \
        _analyze_beams(data, car_x, car_y)

    if valid_count == 0:
        return [], {"total_beams": n_beams, "valid_beams": 0}

    jump_indices = _detect_jumps(hit_dist, valid_mask, n_beams, jump_threshold)

    all_dists = sorted([hit_dist[i] for i in range(n_beams) if valid_mask[i]])
    diag = _compute_diag(n_beams, valid_count, len(jump_indices), all_dists)

    if len(jump_indices) < 2:
        return [], diag

    obstacle_beam = _expand_obstacle_beams(
        hit_dist, valid_mask, n_beams, jump_indices, jump_threshold)

    clusters = _cluster_beams(obstacle_beam, n_beams, cluster_min_beams)
    if not clusters:
        return [], diag

    obstacles = _compute_obstacles(clusters, hit_dist, hit_angles, car_x, car_y, n_beams)
    obstacles, dropped = _filter_out_of_bounds(obstacles)
    diag['filtered_out'] = dropped
    obstacles, filtered_fz = _filter_forbidden_zones(obstacles, forbidden_zones)
    diag['filtered_forbidden'] = filtered_fz
    return obstacles, diag


def get_obstacle_beam_mask(car_x, car_y, jump_threshold=JUMP_THRESHOLD):
    """
    返回每个光束是否为障碍物的布尔掩码（True = 障碍物光束）。

    管线：分析 → 突变检测 → 扩展 → 返回逐束掩码（不做聚类/过滤）。
    用于墙壁建模时排除障碍物光束。
    """
    data = rospy.wait_for_message("scan", LaserScan, timeout=LIDAR_TIMEOUT)
    hit_dist, hit_angles, valid_mask, n_beams, valid_count = \
        _analyze_beams(data, car_x, car_y)

    if valid_count == 0:
        return np.zeros(n_beams, dtype=bool)

    jump_indices = _detect_jumps(hit_dist, valid_mask, n_beams, jump_threshold)
    if len(jump_indices) < 2:
        return np.zeros(n_beams, dtype=bool)

    obstacle_beam = _expand_obstacle_beams(
        hit_dist, valid_mask, n_beams, jump_indices, jump_threshold)
    return np.array(obstacle_beam, dtype=bool)


# ── 独立测试入口 ──
if __name__ == "__main__":
    import sys
    rospy.init_node("obs_test", anonymous=True)
    rospy.wait_for_message("scan", LaserScan, timeout=10.0)
    print("障碍物检测模块就绪")
