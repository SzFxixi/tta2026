#!/usr/bin/env python3
"""
路径规划模块 — 基于 A* 思想的避障路径规划。

在检测到的障碍物周围生成候选路点，构建轴对齐图（相邻点仅 X 或 Y 变化），
用 A* 搜索最优路径，通过打分函数平衡路径长度与障碍物距离。

依赖:
    obstacle_detector   — detect_obstacles()

可独立测试（需 ROS + LiDAR）:
    python3 path_planner.py <target_x> <target_y> [num_waypoints]
"""

import math
import heapq
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from obstacle_detector import detect_obstacles


# ============================================================
#  可调参数
# ============================================================

# 障碍物回避
OBSTACLE_MARGIN = 0.3     # 路点/路径段距障碍物的最小安全距离 (米)

# 栅格生成
GRID_EXPAND = 0.228         # 障碍物周围额外生成的偏移路点间距 (米)
GRID_X_STEP = 0.4         # 在起终点 X 之间均匀插入路点的间距 (米)
GRID_Y_STEP = 0.4         # 在起终点 Y 之间均匀插入路点的间距 (米)

# A* 打分
COST_DIST_WEIGHT = 1.0    # 路径长度权重
COST_RISK_WEIGHT = 0   # 默认走最短路径？？？

# 校正点规划
CORRECTION_CORRIDOR_WIDTH = 0.3   # 校正点前后左右走廊宽度 (米)，此范围内无障碍物才可校正
CORRECTION_MIN_INTERVAL = 1.0     # 相邻校正点最小间距 (米)，避免过于密集

# 房间有效范围（路点坐标不能超出此范围）
ROOM_X_MIN = 0.1          # X 最小 (m)，前墙边界
ROOM_Y_MIN = 0.1          # Y 最小 (m)，右墙边界
ROOM_X_MAX = 4.5          # X 最大 (m)，后墙边界
ROOM_Y_MAX = 8.8          # Y 最大 (m)，左墙边界




# 小车定位（与 CarControlServiceFlask 保持一致）
X_OFFSET = 0.0
Y_OFFSET = 0.0


# ============================================================
#  小车定位
# ============================================================

def getx():
    data = rospy.wait_for_message("scan", LaserScan)
    d = data.ranges[len(data.ranges) // 2]
    while d == np.inf:
        data = rospy.wait_for_message("scan", LaserScan)
        d = data.ranges[len(data.ranges) // 2]
    return d


def gety():
    data = rospy.wait_for_message("scan", LaserScan)
    d = data.ranges[len(data.ranges) // 4]
    while d == np.inf:
        data = rospy.wait_for_message("scan", LaserScan)
        d = data.ranges[len(data.ranges) // 4]
    return d


def get_car_position():
    return getx() - X_OFFSET, gety() - Y_OFFSET


# ============================================================
#  几何工具
# ============================================================

# ============================================================
#  禁区管理
# ============================================================

def setup_forbidden_zones():
    """
    交互式设置禁区。移动小车到禁区的一个角点按 Enter，再移到对角点按 Enter。
    按 Q 退出。坐标以调用此函数时的小车位置为原点（相对坐标）。

    返回:
        [(xmin, xmax, ymin, ymax), ...]  每个禁区由 X/Y 范围定义（相对坐标）
    """
    origin_x, origin_y = get_car_position()
    zones = []
    print(f"\n禁区设置 (原点={origin_x:.3f},{origin_y:.3f})")
    print("移动小车到禁区角点，按 Enter 记录；按 Q 退出")

    while True:
        choice = input(">>> ").strip()
        if choice.upper() == 'Q':
            break

        cx1, cy1 = get_car_position()
        rx1, ry1 = cx1 - origin_x, cy1 - origin_y
        print(f"  角点1 绝对:({cx1:.3f},{cy1:.3f})  相对:({rx1:+.3f},{ry1:+.3f})")

        input("  移动到对角点后按 Enter...")
        cx2, cy2 = get_car_position()
        rx2, ry2 = cx2 - origin_x, cy2 - origin_y
        print(f"  角点2 绝对:({cx2:.3f},{cy2:.3f})  相对:({rx2:+.3f},{ry2:+.3f})")

        zone = (min(rx1, rx2), max(rx1, rx2), min(ry1, ry2), max(ry1, ry2))
        zones.append(zone)
        print(f"  ✓ 禁区 #{len(zones)}: X[{zone[0]:.3f}~{zone[1]:.3f}] Y[{zone[2]:.3f}~{zone[3]:.3f}]")

    print(f"共设置 {len(zones)} 个禁区\n")
    return zones


def _point_in_forbidden(px, py, forbidden_zones, origin_x=0.0, origin_y=0.0):
    """点 (px,py) 是否在某个禁区内。禁区坐标是相对值，需要加原点偏移。"""
    for xmin, xmax, ymin, ymax in forbidden_zones:
        ax, bx = xmin + origin_x, xmax + origin_x
        ay, by = ymin + origin_y, ymax + origin_y
        if ax <= px <= bx and ay <= py <= by:
            return True
    return False


def _segment_crosses_forbidden(ax, ay, bx, by, forbidden_zones,
                                origin_x=0.0, origin_y=0.0):
    """线段 AB 是否穿过某个禁区（用分段采样检测）。"""
    if not forbidden_zones:
        return False
    seg_len = math.hypot(bx - ax, by - ay)
    if seg_len < 0.001:
        return _point_in_forbidden(ax, ay, forbidden_zones, origin_x, origin_y)
    n_samples = max(2, int(seg_len / 0.05))
    for i in range(n_samples + 1):
        t = i / n_samples
        px = ax + (bx - ax) * t
        py = ay + (by - ay) * t
        if _point_in_forbidden(px, py, forbidden_zones, origin_x, origin_y):
            return True
    return False


def _point_to_segment_dist(px, py, ax, ay, bx, by):
    """点 (px,py) 到线段 AB 的最短距离"""
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _segment_safe(ax, ay, bx, by, obstacles, margin=OBSTACLE_MARGIN,
                  forbidden_zones=None, origin_x=0.0, origin_y=0.0):
    """线段 AB 到所有障碍物的距离是否都大于 margin，且不穿过禁区"""
    for obs in obstacles:
        if _point_to_segment_dist(obs['x'], obs['y'], ax, ay, bx, by) < margin:
            return False
    if forbidden_zones and _segment_crosses_forbidden(ax, ay, bx, by,
                                                       forbidden_zones, origin_x, origin_y):
        return False
    return True


def _point_safe(px, py, obstacles, margin=OBSTACLE_MARGIN,
                forbidden_zones=None, origin_x=0.0, origin_y=0.0):
    """点 (px,py) 到所有障碍物的距离是否都大于 margin，且不在禁区内"""
    for obs in obstacles:
        if math.hypot(px - obs['x'], py - obs['y']) < margin:
            return False
    if forbidden_zones and _point_in_forbidden(px, py, forbidden_zones, origin_x, origin_y):
        return False
    return True


def _path_min_dist_to_obstacles(waypoints, obstacles):
    """路径到所有障碍物的最短距离"""
    d = float('inf')
    for i in range(len(waypoints) - 1):
        for obs in obstacles:
            d = min(d, _point_to_segment_dist(
                obs['x'], obs['y'],
                waypoints[i][0], waypoints[i][1],
                waypoints[i+1][0], waypoints[i+1][1]))
    return d


# ============================================================
#  候选路点生成
# ============================================================

def _generate_nodes(sx, sy, ex, ey, obstacles,
                    forbidden_zones=None, origin_x=0.0, origin_y=0.0):
    """在起终点及障碍物周围生成候选路点集合。"""
    margin = OBSTACLE_MARGIN + GRID_EXPAND

    xs = {sx, ex}
    ys = {sy, ey}

    for obs in obstacles:
        ox, oy = obs['x'], obs['y']
        for dx in (-margin, 0, margin):
            xs.add(ox + dx)
        for dy in (-margin, 0, margin):
            ys.add(oy + dy)

    if abs(ex - sx) > GRID_X_STEP:
        n = int(abs(ex - sx) / GRID_X_STEP)
        for i in range(1, n):
            xs.add(sx + (ex - sx) * i / n)
    if abs(ey - sy) > GRID_Y_STEP:
        n = int(abs(ey - sy) / GRID_Y_STEP)
        for i in range(1, n):
            ys.add(sy + (ey - sy) * i / n)

    nodes = []
    for x in xs:
        if x < ROOM_X_MIN or x > ROOM_X_MAX:
            continue
        for y in ys:
            if y < ROOM_Y_MIN or y > ROOM_Y_MAX:
                continue
            if _point_safe(x, y, obstacles, forbidden_zones=forbidden_zones,
                           origin_x=origin_x, origin_y=origin_y):
                nodes.append((x, y))

    if (sx, sy) not in nodes:
        nodes.append((sx, sy))
    if (ex, ey) not in nodes:
        nodes.append((ex, ey))

    return nodes


# ============================================================
#  A* 搜索（轴对齐图）
# ============================================================

def _heuristic(ax, ay, bx, by):
    """曼哈顿距离启发式"""
    return abs(ax - bx) + abs(ay - by)


def _cost(ax, ay, bx, by, obstacles):
    """边 (A → B) 的代价 = 距离 + 风险惩罚"""
    seg_len = math.hypot(bx - ax, by - ay)
    risk = 0.0
    for obs in obstacles:
        d = _point_to_segment_dist(obs['x'], obs['y'], ax, ay, bx, by)
        if d < OBSTACLE_MARGIN * 3:
            risk += 1.0 / max(d, 0.01)
    return COST_DIST_WEIGHT * seg_len + COST_RISK_WEIGHT * risk


def _astar(start, end, nodes, obstacles,
           forbidden_zones=None, origin_x=0.0, origin_y=0.0):
    """A* 搜索。相邻条件：轴对齐 + 不穿过障碍物/禁区。"""
    node_set = set(nodes)
    sx, sy = start
    ex, ey = end

    open_set = [(0, 0, sx, sy, None)]
    closed = {}

    while open_set:
        f, g, cx, cy, parent = heapq.heappop(open_set)

        key = (cx, cy)
        if key in closed and closed[key][0] <= g:
            continue
        closed[key] = (g, parent)

        if abs(cx - ex) < 0.001 and abs(cy - ey) < 0.001:
            path = [(ex, ey)]
            cur = key
            while cur != (sx, sy):
                _, prev = closed[cur]
                path.append(prev)
                cur = prev
            path.reverse()
            return path, g

        for nx, ny in node_set:
            nkey = (nx, ny)
            if nkey == key:
                continue
            if abs(nx - cx) > 0.001 and abs(ny - cy) > 0.001:
                continue
            if not _segment_safe(cx, cy, nx, ny, obstacles,
                                 forbidden_zones=forbidden_zones,
                                 origin_x=origin_x, origin_y=origin_y):
                continue

            step_cost = _cost(cx, cy, nx, ny, obstacles)
            ng = g + step_cost
            nf = ng + _heuristic(nx, ny, ex, ey)

            if nkey in closed and closed[nkey][0] <= ng:
                continue

            heapq.heappush(open_set, (nf, ng, nx, ny, key))

    return None, float('inf')


# ============================================================
#  路径平滑：合并共线点
# ============================================================

def _simplify_waypoints(waypoints):
    """移除连续的共线段中间点，只保留拐角点"""
    if len(waypoints) <= 2:
        return waypoints

    result = [waypoints[0]]
    for i in range(1, len(waypoints) - 1):
        px, py = result[-1]
        cx, cy = waypoints[i]
        nx, ny = waypoints[i + 1]
        # 如果当前点是拐角（方向改变），保留；否则跳过
        prev_dx = abs(cx - px) > 0.001
        prev_dy = abs(cy - py) > 0.001
        next_dx = abs(nx - cx) > 0.001
        next_dy = abs(ny - cy) > 0.001
        if (prev_dx != next_dx) or (prev_dy != next_dy):
            result.append(waypoints[i])
    result.append(waypoints[-1])
    return result


# ============================================================
#  路径重采样（按个数）
# ============================================================

def _resample_waypoints(waypoints, target_count):
    """
    将路径均匀采样到 target_count 个点，保持相邻点仅单轴变化。
    """
    if target_count < 2 or len(waypoints) < 2:
        return waypoints

    # 计算各段长度
    segs = []
    total = 0.0
    for i in range(len(waypoints) - 1):
        x1, y1 = waypoints[i]
        x2, y2 = waypoints[i + 1]
        L = math.hypot(x2 - x1, y2 - y1)
        segs.append((x1, y1, x2, y2, L))
        total += L

    if total == 0:
        return waypoints

    # 按长度成比例分配点数
    gaps_total = target_count - 1
    result = [waypoints[0]]
    remaining_gaps = gaps_total
    for idx, (x1, y1, x2, y2, L) in enumerate(segs):
        if idx == len(segs) - 1:
            n = remaining_gaps
        else:
            n = max(1, int(round(gaps_total * L / total)))
            remaining_gaps -= n
        for j in range(1, n):
            t = j / n
            result.append((x1 + (x2 - x1) * t, y1 + (y2 - y1) * t))
        if idx < len(segs) - 1:
            result.append((x2, y2))
    if result[-1] != waypoints[-1]:
        result.append(waypoints[-1])
    return result


# ============================================================
#  校正点规划
# ============================================================

def _is_correction_safe(cx, cy, obstacles, width=CORRECTION_CORRIDOR_WIDTH):
    """
    检查点 (cx,cy) 是否适合做位姿校正。

    校正依赖前后左右四个方向的 LiDAR 光束打到墙上：
      - 障碍物跟车在同一 Y 行 → 前后光束被挡
      - 障碍物跟车在同一 X 列 → 左右光束被挡

    所以障碍物对应的整条 X 列、整条 Y 行都不能校正。
    """
    for obs in obstacles:
        ox, oy = obs['x'], obs['y']
        # 同一 Y 行 → 前后 LiDAR 被挡
        if abs(oy - cy) < width:
            return False
        # 同一 X 列 → 左右 LiDAR 被挡
        if abs(ox - cx) < width:
            return False
    return True


def plan_correction_points(waypoints, obstacles,
                            min_interval=CORRECTION_MIN_INTERVAL):
    """
    从路径点序列中选出适合做位姿校正的点。

    规则：
      1. 起点和终点始终标记（即使不完全安全，调用方自行判断）
      2. 中间点须满足四方向走廊无障碍物
      3. 相邻校正点间距不小于 min_interval

    返回:
        [
            {"index": 0, "x": x0, "y": y0, "type": "start", "safe": bool},
            {"index": 2, "x": x2, "y": y2, "type": "intermediate", "safe": bool},
            {"index": 4, "x": x4, "y": y4, "type": "end", "safe": bool},
        ]
    """
    if not waypoints or not obstacles:
        # 无障碍物 → 起点终点都可以校正
        result = []
        if waypoints:
            result.append({"index": 0, "x": waypoints[0][0], "y": waypoints[0][1],
                           "type": "start", "safe": True})
        if len(waypoints) > 1:
            result.append({"index": len(waypoints) - 1,
                           "x": waypoints[-1][0], "y": waypoints[-1][1],
                           "type": "end", "safe": True})
        return result

    points = []

    for i, (x, y) in enumerate(waypoints):
        safe = _is_correction_safe(x, y, obstacles)

        if i == 0:
            ptype = "start"
        elif i == len(waypoints) - 1:
            ptype = "end"
        else:
            ptype = "intermediate"

        # 起点终点始终加入，中间点需满足安全条件
        if ptype in ("start", "end") or safe:
            # 距离检查：与上一个校正点不能太近
            if points:
                prev = points[-1]
                dist = math.hypot(x - prev['x'], y - prev['y'])
                if dist < min_interval:
                    # 如果新点比旧点更安全，替换
                    if safe and not prev['safe']:
                        points[-1] = {"index": i, "x": x, "y": y,
                                       "type": prev['type'], "safe": safe}
                    continue

            points.append({"index": i, "x": x, "y": y,
                           "type": ptype, "safe": safe})

    return points


# ============================================================
#  打分函数
# ============================================================

def _score_path(waypoints, obstacles):
    """
    对路径打分（越低越好）：
      score = 路径总长 + 风险惩罚 × 距离倒数
    """
    total_len = 0.0
    min_dist = float('inf')
    for i in range(len(waypoints) - 1):
        x1, y1 = waypoints[i]
        x2, y2 = waypoints[i + 1]
        total_len += math.hypot(x2 - x1, y2 - y1)
        for obs in obstacles:
            d = _point_to_segment_dist(obs['x'], obs['y'], x1, y1, x2, y2)
            min_dist = min(min_dist, d)

    risk = 0.0 if min_dist == float('inf') else 1.0 / max(min_dist, 0.01)
    return COST_DIST_WEIGHT * total_len + COST_RISK_WEIGHT * risk


# ============================================================
#  主入口
# ============================================================

def plan_path(start_x, start_y, end_x, end_y,
              car_x, car_y,
              forbidden_zones=None,
              origin_x=0.0, origin_y=0.0,
              num_waypoints=0,
              jump_threshold=None,
              cluster_min_beams=None):
    """
    规划从起点到终点的避障路径（A* 搜索）。

    参数:
        start_x, start_y:   起点坐标
        end_x, end_y:       终点坐标
        car_x, car_y:       小车位置（给障碍物检测用）
        forbidden_zones:    禁区列表 [(xmin,xmax,ymin,ymax), ...] 相对坐标
        origin_x, origin_y: 禁区坐标原点（小车设定禁区时的位置）
        num_waypoints:      期望输出点数（0 = 自动，按曼哈顿距离每米1个点）
    """
    kwargs = {}
    if jump_threshold is not None:
        kwargs['jump_threshold'] = jump_threshold
    if cluster_min_beams is not None:
        kwargs['cluster_min_beams'] = cluster_min_beams

    obstacles, _ = detect_obstacles(car_x, car_y, **kwargs)

    sx, sy = start_x, start_y
    ex, ey = end_x, end_y

    if num_waypoints <= 0:
        manhattan = abs(ex - sx) + abs(ey - sy)
        num_waypoints = max(2, math.ceil(manhattan))

    if not obstacles and not forbidden_zones:
        wp = [(sx, sy), (ex, sy), (ex, ey)]
        wp = _simplify_waypoints(wp)
        wp = _resample_waypoints(wp, num_waypoints)
        return {
            "path_name": "直接路径(无障碍)",
            "obstacle_margin": float('inf'),
            "waypoints": wp,
            "obstacles": [],
            "safe": True,
            "score": 0.0,
        }

    nodes = _generate_nodes(sx, sy, ex, ey, obstacles,
                            forbidden_zones=forbidden_zones,
                            origin_x=origin_x, origin_y=origin_y)
    raw_path, total_cost = _astar((sx, sy), (ex, ey), nodes, obstacles,
                                   forbidden_zones=forbidden_zones,
                                   origin_x=origin_x, origin_y=origin_y)

    if raw_path is None:
        wp = [(sx, sy), (ex, sy), (ex, ey)]
        margin = _path_min_dist_to_obstacles(wp, obstacles)
        wp = _simplify_waypoints(wp)
        wp = _resample_waypoints(wp, num_waypoints)
        return {
            "path_name": "先X后Y(安全无解)",
            "obstacle_margin": round(margin, 3),
            "waypoints": wp,
            "obstacles": obstacles,
            "safe": False,
            "score": float('inf'),
        }

    wp = _simplify_waypoints(raw_path)
    wp = _resample_waypoints(wp, num_waypoints)

    margin = _path_min_dist_to_obstacles(wp, obstacles)
    score = _score_path(wp, obstacles)
    safe_ok = margin >= OBSTACLE_MARGIN
    correction_points = plan_correction_points(wp, obstacles)

    return {
        "path_name": "A*避障",
        "obstacle_margin": round(margin, 3),
        "waypoints": wp,
        "correction_points": correction_points,
        "obstacles": obstacles,
        "safe": safe_ok,
        "score": round(score, 2),
    }


# ============================================================
#  便捷入口
# ============================================================

def plan_path_to(target_x, target_y,
                 forbidden_zones=None,
                 origin_x=0.0, origin_y=0.0,
                 num_waypoints=0,
                 jump_threshold=None,
                 cluster_min_beams=None):
    """从当前小车位置规划到终点的避障路径"""
    car_x, car_y = get_car_position()
    return plan_path(
        start_x=car_x, start_y=car_y,
        end_x=target_x, end_y=target_y,
        car_x=car_x, car_y=car_y,
        forbidden_zones=forbidden_zones,
        origin_x=origin_x, origin_y=origin_y,
        num_waypoints=num_waypoints,
        jump_threshold=jump_threshold,
        cluster_min_beams=cluster_min_beams,
    )


# ── 命令行测试入口 ──
if __name__ == "__main__":
    import sys

    rospy.init_node("path_planner_test", anonymous=True)
    rospy.wait_for_message("scan", LaserScan, timeout=5.0)

    car_x, car_y = get_car_position()

    if len(sys.argv) >= 3:
        ex, ey = float(sys.argv[1]), float(sys.argv[2])
        num = int(sys.argv[3]) if len(sys.argv) >= 4 else 0
    else:
        print(f"当前小车位置: X={car_x:.3f}, Y={car_y:.3f}")
        print("用法: python3 path_planner.py <target_x> <target_y> [num_waypoints]")
        sys.exit(1)

    print(f"小车位置: X={car_x:.3f}  Y={car_y:.3f}")
    print(f"终点: ({ex:.1f}, {ey:.1f})")

    result = plan_path_to(ex, ey, num_waypoints=num)

    print(f"\n路径: {result['path_name']}")
    print(f"得分: {result.get('score', '-')}")
    print(f"距障碍物: {result['obstacle_margin']}m  {'✓' if result['safe'] else '⚠ 危险!'}")
    print(f"障碍物 ({len(result['obstacles'])}个):")
    for obs in result['obstacles']:
        print(f"  ({obs['x']:.3f}, {obs['y']:.3f}) 距离={obs['distance']:.3f}m")
    print(f"\n路径点 ({len(result['waypoints'])}个):")
    for i, (x, y) in enumerate(result['waypoints']):
        marker = " ← 起点" if i == 0 else (" ← 终点" if i == len(result['waypoints']) - 1 else "")
        print(f"  {i}: ({x:.3f}, {y:.3f}){marker}")
