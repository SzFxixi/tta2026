#!/usr/bin/env python3
"""
墙壁直线建模定位 — 独立测试脚本。

思路：
  1. 从 LiDAR 全扫描中均匀下采样 ~SAMPLE_COUNT 个点
  2. 转笛卡尔坐标（inf 保留为 NaN）
  3. 墙角检测：相邻有效点笛卡尔间距最大处 + inf 返回处
  4. 按墙角分割点集 → 四面墙各一组
  5. 每面墙 RANSAC 拟合直线
  6. 计算原点到直线的垂直距离 → 小车到各墙的距离

用法（需 ROS + LiDAR）:
    python3 wall_positioning_test.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rospy
import numpy as np
import math
import time
from sensor_msgs.msg import LaserScan
from entities.obstacle_detector import get_obstacle_beam_mask

# ============================================================
#  可调参数（后续提取到 config.yaml）
# ============================================================

SAMPLE_COUNT = 1300              # 均匀采样点数
CORNER_GAP_FACTOR = 3.0         # 间隙 > 中位数 × 该因子 → 墙角
INF_GAP_BONUS = 3.0             # inf 点等价于多大间隙 (m)，用于综合评分
RANSAC_ITERATIONS = 80          # RANSAC 迭代次数
RANSAC_INLIER_THRESHOLD = 0.05  # 点到直线距离 < 该值 → 内点 (m)
RANSAC_MIN_INLIERS = 5          # 最少内点数，不足则拟合失败
WALL_GROUP_MIN_POINTS = 8       # 每面墙最少采样点数

# 房间参数（从 config.yaml 读取，这里先硬编码用于测试）
ROOM_X_MIN = 0.4
ROOM_X_MAX = 4.2
ROOM_Y_MIN = 0.4
ROOM_Y_MAX = 8.6

VISUALIZE = True                # 是否显示可视化


# ============================================================
#  工具函数
# ============================================================

def ransac_fit_line(points, max_iters=RANSAC_ITERATIONS,
                    inlier_thresh=RANSAC_INLIER_THRESHOLD,
                    min_inliers=RANSAC_MIN_INLIERS):
    """
    RANSAC 直线拟合。纯 numpy 实现，不依赖 sklearn。

    参数:
        points: N×2 数组，笛卡尔坐标
    返回:
        (a, b, c, inlier_count)  直线方程 a*x + b*y + c = 0，a²+b²=1
        None 如果内点不足
    """
    n = len(points)
    if n < 2:
        return None

    best_inliers = []
    best_line = None

    for _ in range(max_iters):
        # 随机选 2 个点
        i, j = np.random.choice(n, 2, replace=False)
        p1, p2 = points[i], points[j]

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if dx * dx + dy * dy < 1e-12:
            continue  # 重合点

        # 法向量 (a, b) = (dy, -dx) 归一化
        norm = math.hypot(dx, dy)
        a = dy / norm
        b = -dx / norm
        c = -(a * p1[0] + b * p1[1])

        # 确保法向量一致性（指向远离原点方向，便于后续判定墙壁方位）
        # 如果原点在法向量指向的反侧（即 a*0+b*0+c < 0），翻转
        if c > 0:
            a, b, c = -a, -b, -c

        # 统计内点
        dists = np.abs(a * points[:, 0] + b * points[:, 1] + c)
        inlier_mask = dists < inlier_thresh
        inlier_count = np.sum(inlier_mask)

        if inlier_count > len(best_inliers):
            best_inliers = np.where(inlier_mask)[0]
            best_line = (a, b, c)

    if best_line is None or len(best_inliers) < min_inliers:
        return None

    # 用所有内点重新拟合（总体最小二乘 = SVD）
    inlier_pts = points[best_inliers]
    centroid = inlier_pts.mean(axis=0)
    centered = inlier_pts - centroid
    _, _, vh = np.linalg.svd(centered)
    # 最小奇异值对应的右奇异向量 = 法向量
    a, b = vh[-1]
    c = -(a * centroid[0] + b * centroid[1])

    # 保持法向量一致性
    if c > 0:
        a, b, c = -a, -b, -c

    norm = math.hypot(a, b)
    return (a / norm, b / norm, c / norm, len(best_inliers))


def point_to_line_distance(x, y, a, b, c):
    """点到直线的垂直距离。a²+b²=1 时简化为 |a*x + b*y + c|"""
    return abs(a * x + b * y + c)


# ============================================================
#  主流程
# ============================================================

def main():
    rospy.init_node("wall_positioning_test", anonymous=True)
    scan = rospy.wait_for_message("scan", LaserScan, timeout=10.0)

    # ── 1. 均匀下采样 ──
    n_beams = len(scan.ranges)
    stride = max(1, n_beams // SAMPLE_COUNT)
    indices = list(range(0, n_beams, stride))[:SAMPLE_COUNT]
    angles = scan.angle_min + np.array(indices) * scan.angle_increment
    ranges = np.array([scan.ranges[i] for i in indices], dtype=np.float64)
    is_valid = (ranges > scan.range_min) & (ranges < scan.range_max) & np.isfinite(ranges)

    # ── 2. 障碍物检测 ──
    obs_mask_full = get_obstacle_beam_mask(0, 0)
    obs_mask = np.array([obs_mask_full[i] for i in indices], dtype=bool)
    is_valid = is_valid & (~obs_mask)
    if is_valid.sum() < 10:
        return

    # ── 3. 笛卡尔坐标 ──
    xs = np.where(is_valid, ranges * np.cos(angles), np.nan)
    ys = np.where(is_valid, ranges * np.sin(angles), np.nan)

    # ── 4. 墙角检测 + 分割（inf墙角映射到下一个有效点） ──
    corner_indices = _find_corners_by_gap_and_inf(xs, ys, is_valid, min_sep=len(ranges) // 4)
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
        wall_groups = _split_by_corners(valid_points, sorted(valid_corners))
    else:
        wall_groups = _split_by_angle_sectors(angles[is_valid], valid_points)
    wall_groups = [g for g in wall_groups if len(g) >= WALL_GROUP_MIN_POINTS]

    # ── 5. RANSAC 拟合 ──
    wall_lines = []
    for g in wall_groups:
        result = ransac_fit_line(g)
        if result is not None:
            a, b, c, inliers = result
            dist = point_to_line_distance(0, 0, a, b, c)
            cx, cy = g[:, 0].mean(), g[:, 1].mean()
            label = _wall_label(cx, cy)
            wall_lines.append((a, b, c, dist, cx, cy, label, inliers))

    # ── 6. 收集距离 ──
    front_dist = rear_dist = right_dist = left_dist = None
    for w in wall_lines:
        _, _, _, dist, _, _, label, _ = w
        if label == "前墙":   front_dist = dist
        elif label == "后墙": rear_dist = dist
        elif label == "右墙": right_dist = dist
        elif label == "左墙": left_dist = dist

    # ── 7. 对比表 ──
    raw = scan.ranges
    nf = len(raw)
    trad_x, trad_y = raw[nf // 2], raw[nf // 4]
    trad_r, trad_l = raw[nf - 2], raw[nf * 3 // 4]

    row = "  {:<12} {:>10} {:>15} {:>10}"
    print(f"\n  对比 (传统单光束 vs 墙壁建模):")
    print(f"  {'─' * 52}")
    print(row.format("方向", "传统(m)", "墙壁建模(m)", "差值(m)"))
    print(f"  {'─' * 52}")
    for name, trad, wall in [("前 (getx)", trad_x, front_dist),
                              ("右 (gety)", trad_y, right_dist),
                              ("后", trad_r, rear_dist),
                              ("左", trad_l, left_dist)]:
        if wall is not None and np.isfinite(trad):
            print(row.format(name, f"{trad:.3f}", f"{wall:.3f}", f"{wall-trad:+.3f}"))
        elif wall is not None:
            print(row.format(name, "inf", f"{wall:.3f}", "-"))
        else:
            print(row.format(name, f"{trad:.3f}" if np.isfinite(trad) else "inf", "未检测到", "-"))
    print(f"  {'─' * 52}")

    yaw = None
    front_wall = next((w for w in wall_lines if w[6] == "前墙"), None)
    if front_wall is not None:
        yaw = math.degrees(math.atan2(front_wall[1], front_wall[0]))
    if yaw is not None:
        print(f"  偏航角: {yaw:.1f}°")

    # ── 8. 可视化 ──
    if VISUALIZE:
        valid_angles = angles[is_valid]
        _visualize(valid_points, valid_angles, sorted(valid_corners), wall_lines, wall_groups)

