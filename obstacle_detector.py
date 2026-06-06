#!/usr/bin/env python3
"""
障碍物检测模块 — 从 LiDAR 距离突变中检测安全区外的障碍物。

独立于 Flask 服务端，可单独测试：
    python3 obstacle_detector.py
"""

import rospy
import numpy as np
import math
from sensor_msgs.msg import LaserScan


# ============================================================
#  可调参数
# ============================================================

# LiDAR
LIDAR_TIMEOUT = 5.0            # wait_for_message 超时 (秒)

# 突变检测
JUMP_THRESHOLD = 0.5           # 相邻光束距离差超过此值视为突变 (米)

# 种子扩展
EXPAND_JUMP_RATIO = 2.0        # 扩展连续性: 相邻光束差 > JUMP_THRESHOLD × EXPAND_JUMP_RATIO 时停止

# 聚类
CLUSTER_MIN_BEAMS = 5          # 最少连续光束数，少于此数视为噪点丢弃
CLUSTER_MAX_GAP = 10            # 同一簇内允许的最大光束索引间隔
CROSS_ZERO_MERGE_MARGIN = 5    # 簇触及 0° 两侧 margin 光束内时尝试跨 0° 合并


# ============================================================
#  工具函数
# ============================================================

def point_in_rect(px, py, xmin, xmax, ymin, ymax):
    """点 (px, py) 是否在矩形内"""
    return xmin <= px <= xmax and ymin <= py <= ymax


def _compute_diag(n_beams, valid_count, inside_count, jump_count, all_dists):
    """组装诊断字典"""
    n = len(all_dists)
    return {
        "total_beams": n_beams,
        "valid_beams": valid_count,
        "inside_zone_beams": inside_count,
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

def _analyze_beams(data, car_x, car_y, safe_zone):
    """
    遍历所有激光光束，标记有效性、命中点坐标、是否在安全区内。

    返回:
        hit_dist, hit_angles, valid_mask, hit_inside, n_beams, valid_count, inside_count
    """
    sz_xmin, sz_xmax = safe_zone["xmin"], safe_zone["xmax"]
    sz_ymin, sz_ymax = safe_zone["ymin"], safe_zone["ymax"]

    n_beams = len(data.ranges)
    angle_min = data.angle_min
    angle_inc = data.angle_increment

    hit_dist, hit_angles, valid_mask, hit_inside = [], [], [], []

    for i in range(n_beams):
        d = data.ranges[i]
        if d == np.inf or d < data.range_min or d > data.range_max:
            valid_mask.append(False)
            hit_dist.append(None)
            hit_angles.append(None)
            hit_inside.append(None)
            continue
        valid_mask.append(True)
        hit_dist.append(d)
        angle = angle_min + i * angle_inc
        hit_angles.append(angle)
        hx = car_x - d * math.cos(angle)   # 前方 = 靠近前墙 = X 减小
        hy = car_y + d * math.sin(angle)   # 左侧 = 远离右墙 = Y 增大
        hit_inside.append(point_in_rect(hx, hy, sz_xmin, sz_xmax, sz_ymin, sz_ymax))

    valid_count = sum(valid_mask)
    inside_count = sum(1 for i in range(n_beams) if valid_mask[i] and hit_inside[i])

    return hit_dist, hit_angles, valid_mask, hit_inside, n_beams, valid_count, inside_count


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
#  主入口
# ============================================================

def detect_obstacles(safe_zone_p1, safe_zone_p2,
                     car_x, car_y,
                     jump_threshold=JUMP_THRESHOLD,
                     cluster_min_beams=CLUSTER_MIN_BEAMS):
    """
    通过 LiDAR 距离突变检测安全区外的障碍物。

    参数:
        safe_zone_p1, safe_zone_p2: 安全区对角线两点 (x, y)
        car_x, car_y:              小车当前位置（已减偏移量）
        jump_threshold:            突变阈值（米）
        cluster_min_beams:         最少连续光束数

    返回:
        (obstacles, diagnostics)
    """
    data = rospy.wait_for_message("scan", LaserScan, timeout=LIDAR_TIMEOUT)

    safe_zone = {
        "xmin": min(safe_zone_p1[0], safe_zone_p2[0]),
        "xmax": max(safe_zone_p1[0], safe_zone_p2[0]),
        "ymin": min(safe_zone_p1[1], safe_zone_p2[1]),
        "ymax": max(safe_zone_p1[1], safe_zone_p2[1]),
    }

    # 第一步：逐束分析
    hit_dist, hit_angles, valid_mask, hit_inside, \
        n_beams, valid_count, inside_count = \
        _analyze_beams(data, car_x, car_y, safe_zone)

    if valid_count == 0:
        return [], {"total_beams": n_beams, "valid_beams": 0}

    # 第二步：突变检测
    jump_indices = _detect_jumps(hit_dist, valid_mask, n_beams, jump_threshold)

    all_dists = sorted([hit_dist[i] for i in range(n_beams) if valid_mask[i]])
    diag = _compute_diag(n_beams, valid_count, inside_count, len(jump_indices), all_dists)

    if len(jump_indices) < 2:
        return [], diag

    # 第三步：种子扩展
    obstacle_beam = _expand_obstacle_beams(
        hit_dist, valid_mask, n_beams, jump_indices, jump_threshold)

    # 第四步：聚类
    clusters = _cluster_beams(obstacle_beam, n_beams, cluster_min_beams)
    if not clusters:
        return [], diag

    # 第五步：计算坐标
    obstacles = _compute_obstacles(clusters, hit_dist, hit_angles, car_x, car_y, n_beams)
    return obstacles, diag


# ── 独立测试入口 ──
if __name__ == "__main__":
    import sys
    rospy.init_node("obs_test", anonymous=True)
    rospy.wait_for_message("scan", LaserScan, timeout=10.0)
    print("障碍物检测模块就绪。请通过 /DetectObstacles 端点调用。")
