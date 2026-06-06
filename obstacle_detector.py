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


def point_in_rect(px, py, xmin, xmax, ymin, ymax):
    return xmin <= px <= xmax and ymin <= py <= ymax


def detect_obstacles(safe_zone_p1, safe_zone_p2,
                     car_x, car_y,
                     jump_threshold=0.1, cluster_min_beams=5):
    """
    通过 LiDAR 距离突变检测安全区外的障碍物。

    参数:
        safe_zone_p1, safe_zone_p2: 安全区对角线两点
        car_x, car_y:              小车当前位置（已减偏移量）
        jump_threshold:            突变阈值（米），相邻有效光束差超过此值视为突变
        cluster_min_beams:         最少连续光束数

    返回:
        (obstacles, diagnostics)
    """
    data = rospy.wait_for_message("scan", LaserScan, timeout=5.0)
    n_beams = len(data.ranges)
    angle_min = data.angle_min
    angle_inc = data.angle_increment

    sz_xmin = min(safe_zone_p1[0], safe_zone_p2[0])
    sz_xmax = max(safe_zone_p1[0], safe_zone_p2[0])
    sz_ymin = min(safe_zone_p1[1], safe_zone_p2[1])
    sz_ymax = max(safe_zone_p1[1], safe_zone_p2[1])

    # ── 第一步：逐束分析 ──
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
        hx = car_x + d * math.cos(angle)
        hy = car_y + d * math.sin(angle)
        hit_inside.append(point_in_rect(hx, hy, sz_xmin, sz_xmax, sz_ymin, sz_ymax))

    valid_count = sum(valid_mask)
    inside_count = sum(1 for i in range(n_beams) if valid_mask[i] and hit_inside[i])
    if valid_count == 0:
        return [], {"total_beams": n_beams, "valid_beams": 0}

    # ── 第二步：自适应突变检测（跳过 inf，比较相邻有效光束）──
    diffs = []
    prev_valid = None
    for i in range(n_beams):
        if not valid_mask[i]:
            continue
        if prev_valid is not None:
            diffs.append(abs(hit_dist[i] - hit_dist[prev_valid]))
        prev_valid = i
    first_valid = next((i for i in range(n_beams) if valid_mask[i]), None)
    if prev_valid is not None and first_valid is not None and prev_valid != first_valid:
        diffs.append(abs(hit_dist[first_valid] - hit_dist[prev_valid]))

    if not diffs:
        return [], _make_diag()

    mean_diff = sum(diffs) / len(diffs)
    var_diff = sum((d - mean_diff) ** 2 for d in diffs) / len(diffs)
    std_diff = math.sqrt(var_diff)
    effective = max(mean_diff + 3.0 * std_diff, jump_threshold)

    # 标记突变点
    jump_marks = [False] * n_beams
    prev_valid = None
    for i in range(n_beams):
        if not valid_mask[i]:
            continue
        if prev_valid is not None:
            if abs(hit_dist[i] - hit_dist[prev_valid]) > effective:
                jump_marks[prev_valid] = True
        prev_valid = i
    if prev_valid is not None and first_valid is not None and prev_valid != first_valid:
        if abs(hit_dist[first_valid] - hit_dist[prev_valid]) > effective:
            jump_marks[prev_valid] = True

    jump_indices = [i for i, m in enumerate(jump_marks) if m]

    # 诊断数据
    all_dists = sorted([hit_dist[i] for i in range(n_beams) if valid_mask[i]])
    n = len(all_dists)
    def _make_diag():
        return {
            "total_beams": n_beams, "valid_beams": valid_count,
            "inside_zone_beams": inside_count, "jump_count": len(jump_indices),
            "mean_diff": round(mean_diff, 4), "std_diff": round(std_diff, 4),
            "effective_threshold": round(effective, 3),
            "dist_min": round(all_dists[0], 3) if n else 0,
            "dist_p10": round(all_dists[n // 10], 3) if n > 10 else 0,
            "dist_p50": round(all_dists[n // 2], 3) if n else 0,
            "dist_p90": round(all_dists[n * 9 // 10], 3) if n > 10 else 0,
            "dist_max": round(all_dists[-1], 3) if n else 0,
        }

    if len(jump_indices) < 2:
        return [], _make_diag()

    # ── 第三步：种子扩展标记障碍物光束 ──
    background = all_dists[n // 2]
    obstacle_beam = [False] * n_beams

    for jump_idx in jump_indices:
        prev_v = next((jump_idx - d) % n_beams for d in range(1, n_beams)
                      if valid_mask[(jump_idx - d) % n_beams])
        next_v = (jump_idx + 1) % n_beams
        while not valid_mask[next_v]:
            next_v = (next_v + 1) % n_beams

        d_prev, d_next = hit_dist[prev_v], hit_dist[next_v]
        if d_next < d_prev and d_next < background - jump_threshold:
            seed = next_v
        elif d_prev < d_next and d_prev < background - jump_threshold:
            seed = prev_v
        else:
            continue

        # 向后扩展
        cur = seed
        while True:
            obstacle_beam[cur] = True
            pc = (cur - 1) % n_beams
            if not valid_mask[pc] or hit_dist[pc] > background - jump_threshold * 0.5:
                break
            if abs(hit_dist[pc] - hit_dist[cur]) > effective * 2:
                break
            cur = pc

        # 向前扩展
        cur = seed
        while True:
            obstacle_beam[cur] = True
            nc = (cur + 1) % n_beams
            if not valid_mask[nc] or hit_dist[nc] > background - jump_threshold * 0.5:
                break
            if abs(hit_dist[nc] - hit_dist[cur]) > effective * 2:
                break
            cur = nc

    # ── 第四步：聚类 ──
    obs_indices = [i for i, is_obs in enumerate(obstacle_beam) if is_obs]
    if not obs_indices:
        return [], _make_diag()

    clusters = []
    current = [obs_indices[0]]
    for i in range(1, len(obs_indices)):
        if obs_indices[i] - obs_indices[i - 1] <= 3:
            current.append(obs_indices[i])
        else:
            if len(current) >= cluster_min_beams:
                clusters.append(current)
            current = [obs_indices[i]]
    if len(current) >= cluster_min_beams:
        clusters.append(current)

    # 处理跨 0° 合并
    if len(clusters) >= 2:
        first, last = clusters[0], clusters[-1]
        if first[0] < 5 and last[-1] > n_beams - 5:
            merged = last + [idx + n_beams for idx in first]
            clusters = clusters[1:-1]
            if len(merged) >= cluster_min_beams:
                clusters.append(merged)

    # ── 第五步：计算障碍物坐标 ──
    obstacles = []
    for cluster in clusters:
        mid = cluster[len(cluster) // 2] % n_beams
        d, angle = hit_dist[mid], hit_angles[mid]
        obstacles.append({
            "x": round(car_x + d * math.cos(angle), 3),
            "y": round(car_y + d * math.sin(angle), 3),
            "distance": round(d, 3),
            "angle_deg": round(math.degrees(angle), 1),
            "beam_count": len(cluster),
        })

    return obstacles, _make_diag()


# ── 独立测试入口 ──
if __name__ == "__main__":
    import sys
    rospy.init_node("obs_test", anonymous=True)
    # 等待 LiDAR 数据就绪
    rospy.wait_for_message("scan", LaserScan, timeout=10.0)
    # 用 getx/gety 模拟获取小车位置（需此文件与主服务同目录运行时有 CarService 上下文，这里仅演示）
    # 实际调用由 Flask 端点传入 car_x, car_y
    print("障碍物检测模块就绪。请通过 /DetectObstacles 端点调用。")
