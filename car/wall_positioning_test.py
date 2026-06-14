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

SAMPLE_COUNT = 130              # 均匀采样点数
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

    print("等待 /scan 话题...")
    scan = rospy.wait_for_message("scan", LaserScan, timeout=10.0)
    print(f"收到扫描: {len(scan.ranges)} 个光束")

    # ── 1. 均匀下采样（保留 inf，后面墙角检测会用到） ──
    n_beams = len(scan.ranges)
    stride = max(1, n_beams // SAMPLE_COUNT)
    indices = list(range(0, n_beams, stride))[:SAMPLE_COUNT]

    angles = scan.angle_min + np.array(indices) * scan.angle_increment
    ranges = np.array([scan.ranges[i] for i in indices], dtype=np.float64)

    is_valid = (ranges > scan.range_min) & (ranges < scan.range_max) & np.isfinite(ranges)
    valid_count_before = is_valid.sum()

    # ── 障碍物检测，标记障碍物光束 ──
    print("障碍物检测中...")
    obs_mask_full = get_obstacle_beam_mask(0, 0)  # car_x,car_y 不重要，只需光束掩码
    # 下采样到与采样点相同的分辨率
    obs_mask = np.array([obs_mask_full[i] for i in indices], dtype=bool)
    # 障碍物光束也视为无效（和 inf 一样，不参与墙壁拟合）
    is_valid = is_valid & (~obs_mask)
    obs_count = valid_count_before - is_valid.sum()
    valid_count = is_valid.sum()
    print(f"采样点: {len(ranges)}  有效={valid_count}  "
          f"inf/无效={len(ranges)-valid_count_before}  障碍物={obs_count}")

    if valid_count < SAMPLE_COUNT * 0.5:
        print(f"⚠ 有效光束太少, 退出")
        return

    # ── 2. 极坐标 → 笛卡尔坐标（无效点填入 NaN） ──
    xs = np.where(is_valid, ranges * np.cos(angles), np.nan)
    ys = np.where(is_valid, ranges * np.sin(angles), np.nan)
    points = np.column_stack((xs, ys))

    # ── 3. 找墙角：间隙最大处 + inf 返回处 ──
    n = len(ranges)
    corner_indices = _find_corners_by_gap_and_inf(xs, ys, is_valid, min_sep=n // 4)
    print(f"检测到墙角: {len(corner_indices)} 个")
    for ci in corner_indices:
        ang = math.degrees(angles[ci])
        info = f"inf" if not is_valid[ci] else f"距离={ranges[ci]:.3f}m"
        print(f"  墙角 @索引{ci}  角度={ang:.1f}°  {info}")

    # ── 4. 按墙角分割点集（只取有效点，映射到有效点索引） ──
    valid_points = points[is_valid]
    valid_indices_map = np.cumsum(is_valid) - 1  # 原始索引 → 有效点索引
    valid_corner_indices = [valid_indices_map[ci] for ci in corner_indices if is_valid[ci]]

    if len(valid_corner_indices) < 4:
        print(f"⚠ 有效墙角不足 4 个 ({len(valid_corner_indices)})，降级为按角度扇区分割")
        valid_angles = angles[is_valid]
        wall_groups = _split_by_angle_sectors(valid_angles, valid_points)
    else:
        wall_groups = _split_by_corners(valid_points, valid_corner_indices)
        wall_groups = [g for g in wall_groups if len(g) >= WALL_GROUP_MIN_POINTS]

    print(f"\n墙壁分组: {len(wall_groups)} 组")
    for i, g in enumerate(wall_groups):
        cx, cy = g[:, 0].mean(), g[:, 1].mean()
        angle = math.degrees(math.atan2(cy, cx))
        print(f"  第{i}组: {len(g)}个点  质心角={angle:.1f}°  ({_wall_label(cx, cy)})")

    # ── 5. RANSAC 拟合每面墙 ──
    wall_lines = []
    for i, g in enumerate(wall_groups):
        result = ransac_fit_line(g)
        if result is None:
            print(f"  第{i}组: 拟合失败")
            wall_lines.append(None)
        else:
            a, b, c, inliers = result
            dist = point_to_line_distance(0, 0, a, b, c)
            cx, cy = g[:, 0].mean(), g[:, 1].mean()
            label = _wall_label(cx, cy)
            print(f"  第{i}组 → {label}: 距离={dist:.3f}m  内点={inliers}/{len(g)}  "
                  f"直线: {a:.4f}x + {b:.4f}y + {c:.4f} = 0")
            wall_lines.append((a, b, c, dist, cx, cy, label, inliers))

    # ── 6. 汇总 ──
    wall_lines = [w for w in wall_lines if w is not None]
    print(f"\n{'=' * 55}")
    print(f"  墙壁建模定位结果")
    print(f"{'=' * 55}")

    car_x = car_y = None
    front_dist = rear_dist = right_dist = left_dist = None

    for w in wall_lines:
        a, b, c, dist, cx, cy, label, inliers = w
        print(f"  {label:<6}: 直线距离={dist:.3f}m  内点数={inliers}")
        if label == "前墙":
            front_dist = dist
            car_x = dist
        elif label == "后墙":
            rear_dist = dist
            car_x = ROOM_X_MAX - dist  # 后墙距离 → 房间 X 坐标
        elif label == "右墙":
            right_dist = dist
            car_y = dist
        elif label == "左墙":
            left_dist = dist
            car_y = ROOM_Y_MAX - dist  # 左墙距离 → 房间 Y 坐标

    # 冗余校验：同一轴有两面墙时取加权平均
    if front_dist is not None and rear_dist is not None:
        # 加权：内点数多的权重高
        fw = next((w[7] for w in wall_lines if w[6] == "前墙"), 0)
        rw = next((w[7] for w in wall_lines if w[6] == "后墙"), 0)
        total_w = fw + rw
        if total_w > 0:
            car_x = (front_dist * fw + (ROOM_X_MAX - rear_dist) * rw) / total_w
        print(f"  X轴冗余: 前{front_dist:.3f}m + 后{rear_dist:.3f}m "
              f"≈ {front_dist + rear_dist:.3f}m  (房间宽={ROOM_X_MAX - ROOM_X_MIN:.1f}m)")

    if right_dist is not None and left_dist is not None:
        rw = next((w[7] for w in wall_lines if w[6] == "右墙"), 0)
        lw = next((w[7] for w in wall_lines if w[6] == "左墙"), 0)
        total_w = rw + lw
        if total_w > 0:
            car_y = (right_dist * rw + (ROOM_Y_MAX - left_dist) * lw) / total_w
        print(f"  Y轴冗余: 右{right_dist:.3f}m + 左{left_dist:.3f}m "
              f"≈ {right_dist + left_dist:.3f}m  (房间高={ROOM_Y_MAX - ROOM_Y_MIN:.1f}m)")

    print(f"\n  ─────────────────────────────")
    if car_x is not None:
        print(f"  小车 X 坐标: {car_x:.3f}m  (距前墙 {front_dist:.3f}m)" if front_dist else f"  小车 X 坐标: {car_x:.3f}m")
    if car_y is not None:
        print(f"  小车 Y 坐标: {car_y:.3f}m  (距右墙 {right_dist:.3f}m)" if right_dist else f"  小车 Y 坐标: {car_y:.3f}m")

    # 偏航角：前墙法向量与小车正前方向的夹角
    front_wall = next((w for w in wall_lines if w[6] == "前墙"), None)
    if front_wall is not None:
        a, b, c, dist, cx, cy, label, inliers = front_wall
        # 法向量 (a, b)，c<0 表明原点在法向量指向侧
        # 前墙的法向量指向小车（近似 +x 方向，即 car forward）
        yaw = math.degrees(math.atan2(b, a))
        print(f"  偏航角估计: {yaw:.1f}°  (前墙法向量方向)")

    print(f"{'=' * 55}")

    # ── 7. 与传统 getx/gety 对比 ──
    raw_ranges = np.array(scan.ranges)
    n_full = len(raw_ranges)
    trad_x = raw_ranges[n_full // 2]    # 前光束 → getx()
    trad_y = raw_ranges[n_full // 4]    # 右光束 → gety()
    trad_r = raw_ranges[n_full - 2]     # 后光束
    trad_l = raw_ranges[n_full * 3 // 4]  # 左光束

    print(f"\n  对比 (传统单光束 getx/gety):")
    print(f"  {'─' * 50}")
    row_fmt = "  {:<12} {:>10} {:>15} {:>12}"
    print(row_fmt.format("方向", "传统(m)", "墙壁建模(m)", "差值(m)"))
    print(f"  {'─' * 50}")
    comparisons = [
        ("前 (getx)",   trad_x, front_dist),
        ("右 (gety)",   trad_y, right_dist),
        ("后",          trad_r, rear_dist),
        ("左",          trad_l, left_dist),
    ]
    for name, trad, wall in comparisons:
        if wall is not None and not (np.isnan(trad) or np.isinf(trad)):
            diff = wall - trad
            print(row_fmt.format(name, f"{trad:.3f}", f"{wall:.3f}", f"{diff:+.3f}"))
        elif wall is not None:
            print(row_fmt.format(name, "inf/nan", f"{wall:.3f}", "-"))
        else:
            print(row_fmt.format(name, f"{trad:.3f}" if np.isfinite(trad) else "inf/nan", "未检测到", "-"))
    print(f"  {'─' * 50}")

    # ── 8. 可视化（可选） ──
    if VISUALIZE:
        valid_angles = angles[is_valid]
        _visualize(valid_points, valid_angles, valid_corner_indices, wall_lines, wall_groups)


# ============================================================
#  辅助函数
# ============================================================

def _find_corners_by_gap_and_inf(xs, ys, is_valid, min_sep=30):
    """
    综合两种信号找墙角：
      1. 相邻有效点之间笛卡尔间距最大处（间隙大 = 跨越墙角）
      2. inf / 无效返回处（光束打到了墙角的空处）

    两种信号合并评分，取 4 个互不干扰的候选。

    参数:
        xs, ys:    采样点笛卡尔坐标（无效点为 NaN）
        is_valid:  每个采样点是否有效
        min_sep:   两墙角最少间隔多少采样点
    返回:
        原始采样点索引列表（可能包含 inf 点）
    """
    n = len(xs)
    if n < 4:
        return []

    # 找出所有有效点的索引
    valid_indices = np.where(is_valid)[0]
    n_valid = len(valid_indices)

    if n_valid < 4:
        return []

    # 计算相邻有效点之间的笛卡尔间距（环形），同时记录夹了多少 inf 点
    gaps = []  # [(from_valid_idx, to_valid_idx, gap_size, inf_count, sample_idx_between)]

    for vi in range(n_valid):
        vi_next = (vi + 1) % n_valid
        fi = valid_indices[vi]       # 原始索引
        ti = valid_indices[vi_next]  # 原始索引

        # 环形距离：两有效点之间的采样点数
        if ti > fi:
            span = ti - fi
        else:
            span = n - fi + ti

        inf_count = span - 1  # 中间夹的无效点数（含 inf）

        # 笛卡尔间距
        dx = xs[ti] - xs[fi]
        dy = ys[ti] - ys[fi]
        gap = math.hypot(dx, dy)

        # 间隙位置取两有效点中间
        mid_sample = (fi + span // 2) % n

        gaps.append((fi, ti, gap, inf_count, mid_sample))

    if not gaps:
        return []

    # 综合评分 = 间隙大小 + inf 点等价间隙
    # inf 点多表示光束穿过了大空区域，应该是墙角
    scores = []
    for fi, ti, gap, inf_count, mid_sample in gaps:
        score = gap + inf_count * INF_GAP_BONUS
        scores.append((mid_sample, score, gap, inf_count))

    # 按评分从大到小排列
    scores.sort(key=lambda x: -x[1])

    # 打印信息
    median_gap = np.median([g[1] for g in gaps])
    print(f"有效点={n_valid}  间隙中位数: {median_gap:.3f}m")
    for mid_sample, score, gap, inf_count in scores[:8]:
        tag = "inf" if inf_count > 0 else ""
        print(f"  候选 @{mid_sample}: 间隙={gap:.3f}m  inf={inf_count}  "
              f"评分={score:.3f} {tag}")

    # 选 4 个互不干扰的（彼此间距 >= min_sep）
    selected = []
    for mid_sample, score, gap, inf_count in scores:
        too_close = False
        for s in selected:
            d = min(abs(mid_sample - s), n - abs(mid_sample - s))
            if d < min_sep:
                too_close = True
                break
        if not too_close:
            selected.append(mid_sample)
        if len(selected) >= 4:
            break

    return sorted(selected)


def _split_by_corners(points, corner_indices):
    """按墙角索引分割点集。corner_indices 已排序。"""
    n = len(points)
    groups = []
    prev = corner_indices[-1]  # 从最后一个墙角开始（环形）
    for ci in corner_indices:
        if ci > prev:
            group = points[prev + 1:ci + 1]
        else:
            # 跨越环形边界
            group = np.vstack((points[prev + 1:], points[:ci + 1]))
        prev = ci
        if len(group) >= WALL_GROUP_MIN_POINTS:
            groups.append(group)
    return groups


def _split_by_angle_sectors(angles, points):
    """降级方案：按角度扇区分割（前/右/后/左各 ±45°）。

    ROS LiDAR 角度约定（angle_min=-π，逆时针）：
      - 0°     → 前方 (+x)
      - 90°    → 左侧 (+y)
      - ±180°  → 后方 (-x)
      - -90°   → 右侧 (-y)
    """
    front_mask = (angles >= -math.pi / 4) & (angles <= math.pi / 4)
    right_mask = (angles >= -3 * math.pi / 4) & (angles <= -math.pi / 4)
    left_mask  = (angles >= math.pi / 4) & (angles <= 3 * math.pi / 4)
    rear_mask  = (angles >= 3 * math.pi / 4) | (angles <= -3 * math.pi / 4)

    groups = []
    for mask in [front_mask, right_mask, rear_mask, left_mask]:
        if mask.sum() >= WALL_GROUP_MIN_POINTS:
            groups.append(points[mask])
    return groups


def _wall_label(cx, cy):
    """根据质心位置判断墙壁方位（小车局部坐标系: +x前, +y左）"""
    angle = math.degrees(math.atan2(cy, cx))
    if -45 <= angle <= 45:
        return "前墙"
    elif 45 < angle <= 135:
        return "左墙"
    elif angle > 135 or angle <= -135:
        return "后墙"
    else:  # -135 < angle < -45
        return "右墙"


def _visualize(points, angles, corner_indices, wall_lines, wall_groups):
    """matplotlib 可视化：左=距离曲线+峰值，右=墙壁分组+拟合直线"""
    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n⚠ matplotlib 不可用，跳过可视化")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # ── 左图：距离 vs 角度曲线 + 墙角峰值 ──
    distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    angles_deg = np.degrees(angles)   # LiDAR 扫描角度
    ax1.plot(angles_deg, distances, "b-", linewidth=1, alpha=0.7, label="距离")
    ax1.scatter(angles_deg, distances, c="blue", s=8, alpha=0.4)

    if corner_indices:
        ax1.scatter(angles_deg[corner_indices], distances[corner_indices],
                   c="orange", s=100, marker="s", edgecolors="black",
                   linewidth=2, zorder=9, label="墙角(峰值)")

    ax1.set_xlabel("角度 (°)")
    ax1.set_ylabel("距离 (m)")
    ax1.set_title("距离-角度曲线 + 墙角峰值检测")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # ── 右图：墙壁分组 + 拟合直线 + 垂直距离 ──
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    for i, g in enumerate(wall_groups):
        c = colors[i % 4]
        ax2.scatter(g[:, 0], g[:, 1], c=c, s=10, alpha=0.5, label=f"第{i}组")

    for w in wall_lines:
        a, b, c_val, dist, cx, cy, label, inliers = w
        # 画拟合直线
        x_range = np.array([points[:, 0].min() - 0.5, points[:, 0].max() + 0.5])
        if abs(b) > 1e-6:
            y_vals = -(a * x_range + c_val) / b
            ax2.plot(x_range, y_vals, linewidth=2, label=f"{label} {dist:.2f}m")
        else:
            x_val = -c_val / a
            y_range = np.array([points[:, 1].min() - 0.5, points[:, 1].max() + 0.5])
            ax2.plot([x_val, x_val], y_range, linewidth=2, label=f"{label} {dist:.2f}m")

        # 画从原点到直线的垂线
        foot_x = -c_val * a   # a²+b²=1
        foot_y = -c_val * b
        ax2.plot([0, foot_x], [0, foot_y], "--", linewidth=1, alpha=0.7)
        ax2.scatter(foot_x, foot_y, s=30, marker="x")

    ax2.scatter(0, 0, c="red", s=120, marker="*", zorder=10, label="小车")
    ax2.set_xlabel("x (小车前方)")
    ax2.set_ylabel("y (小车左方)")
    ax2.set_title("墙壁分组 + RANSAC 拟合 + 垂直距离")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
