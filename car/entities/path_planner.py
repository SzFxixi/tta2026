#!/usr/bin/env python3
# 基于 A* 的避障路径规划，依赖 obstacle_detector
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import time
import heapq
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from entities.obstacle_detector import detect_obstacles
from entities.forbidden_zones import point_in_forbidden, segment_crosses_forbidden


# ============================================================
#  参数 — 从 config.yaml 读取
# ============================================================

from utils.config_loader import cfg

OBSTACLE_MARGIN = cfg.path_planning.obstacle_margin
WALL_MARGIN = cfg.path_planning.wall_margin
GRID_EXPAND = cfg.path_planning.grid_expand
GRID_X_STEP = cfg.path_planning.grid_x_step
GRID_Y_STEP = cfg.path_planning.grid_y_step
COST_DIST_WEIGHT = cfg.path_planning.cost_dist_weight
COST_RISK_WEIGHT = cfg.path_planning.cost_risk_weight
CORRECTION_CORRIDOR_WIDTH = cfg.path_planning.correction_corridor_width
CORRECTION_MIN_INTERVAL = cfg.path_planning.correction_min_interval
ROOM_X_MIN = cfg.room.x_min
ROOM_Y_MIN = cfg.room.y_min
ROOM_X_MAX = cfg.room.x_max
ROOM_Y_MAX = cfg.room.y_max
X_OFFSET = cfg.room.x_offset
Y_OFFSET = cfg.room.y_offset


# ============================================================
#  小车定位（墙壁直线建模）
# ============================================================

from utils.wall_positioning import fit_walls

_pos_cache = None
_pos_cache_time = 0
_POS_CACHE_TTL = 0.05

def _read_position():
    """读一次 LiDAR，拟合墙壁，50ms 内重复调用走缓存。返回 (x, y)。"""
    global _pos_cache, _pos_cache_time
    now = time.time()
    if _pos_cache is not None and (now - _pos_cache_time) < _POS_CACHE_TTL:
        return _pos_cache
    data = rospy.wait_for_message("scan", LaserScan)
    walls = fit_walls(data)
    x = walls.get("前墙")
    y = walls.get("右墙")
    if x is not None:
        x -= X_OFFSET
    if y is not None:
        y -= Y_OFFSET
    _pos_cache = (x, y)
    _pos_cache_time = now
    return x, y

def getx():
    """前墙距离 (m)。"""
    x, _ = _read_position()
    return x


def gety():
    """右墙距离 (m)。"""
    _, y = _read_position()
    return y


def get_x():
    """后墙距离 (m)。"""
    data = rospy.wait_for_message("scan", LaserScan)
    walls = fit_walls(data)
    return walls.get("后墙")


def get_y():
    """左墙距离 (m)。"""
    data = rospy.wait_for_message("scan", LaserScan)
    walls = fit_walls(data)
    return walls.get("左墙")


def get_car_position():
    """返回 (x, y)，兼容旧接口。"""
    return _read_position()


# ============================================================
#  几何工具
# ============================================================

def _point_to_segment_dist(px, py, ax, ay, bx, by):
    """点 (px,py) 到线段 AB 的最短距离"""
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


_cached_zones = None
_loaded_zones = False


def _get_forbidden_zones():
    """加载禁区（缓存），失败返回 None"""
    global _cached_zones, _loaded_zones
    if not _loaded_zones:
        _loaded_zones = True
        try:
            from entities.forbidden_zones import load_forbidden_zones
            _cached_zones = load_forbidden_zones()
        except Exception:
            _cached_zones = None
    return _cached_zones


def _segment_safe(ax, ay, bx, by, obstacles, margin=OBSTACLE_MARGIN,
                  forbidden_zones=None, start=None, end=None):
    """线段 AB 到所有障碍物的距离是否都大于 margin，且不穿过禁区。
       起终点(start/end)豁免禁区检测。"""
    for obs in obstacles:
        if _point_to_segment_dist(obs['x'], obs['y'], ax, ay, bx, by) < margin:
            return False
    # 任一端点是起/终点 → 跳过禁区检测
    is_a_se = (start and abs(ax - start[0]) < 0.001 and abs(ay - start[1]) < 0.001) or \
              (end   and abs(ax - end[0])   < 0.001 and abs(ay - end[1])   < 0.001)
    is_b_se = (start and abs(bx - start[0]) < 0.001 and abs(by - start[1]) < 0.001) or \
              (end   and abs(bx - end[0])   < 0.001 and abs(by - end[1])   < 0.001)
    if is_a_se or is_b_se:
        return True
    zones = forbidden_zones or _get_forbidden_zones()
    if zones and segment_crosses_forbidden(ax, ay, bx, by, zones):
        return False
    return True


def _point_safe(px, py, obstacles, margin=OBSTACLE_MARGIN,
                forbidden_zones=None, start=None, end=None):
    """点 (px,py) 到所有障碍物的距离是否都大于 margin，且不在禁区内。
       起终点(start/end)豁免禁区检测。"""
    for obs in obstacles:
        if math.hypot(px - obs['x'], py - obs['y']) < margin:
            return False
    # 起/终点不检测禁区
    if start and abs(px - start[0]) < 0.001 and abs(py - start[1]) < 0.001:
        return True
    if end and abs(px - end[0]) < 0.001 and abs(py - end[1]) < 0.001:
        return True
    zones = forbidden_zones or _get_forbidden_zones()
    if zones and point_in_forbidden(px, py, zones):
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

def _generate_nodes(sx, sy, ex, ey, obstacles, forbidden_zones=None,
                     obstacle_margin=None, grid_expand=None,
                     start=None, end=None):
    """在起终点及障碍物周围生成候选路点集合。"""
    om = OBSTACLE_MARGIN if obstacle_margin is None else obstacle_margin
    ge = GRID_EXPAND if grid_expand is None else grid_expand
    margin = om + ge

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

    # 禁区边界也撒候选节点
    zones = forbidden_zones or _get_forbidden_zones()
    if zones:
        for xmin, xmax, ymin, ymax in zones:
            for dx in (-OBSTACLE_MARGIN, 0, OBSTACLE_MARGIN):
                xs.add(xmin + dx)
                xs.add(xmax + dx)
            for dy in (-OBSTACLE_MARGIN, 0, OBSTACLE_MARGIN):
                ys.add(ymin + dy)
                ys.add(ymax + dy)

    nodes = []
    for x in xs:
        if x < ROOM_X_MIN + WALL_MARGIN or x > ROOM_X_MAX - WALL_MARGIN:
            continue
        for y in ys:
            if y < ROOM_Y_MIN + WALL_MARGIN or y > ROOM_Y_MAX - WALL_MARGIN:
                continue
            if _point_safe(x, y, obstacles, forbidden_zones=forbidden_zones,
                           start=start, end=end):
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


def _cost(ax, ay, bx, by, obstacles, obstacle_margin=None,
         cost_risk_weight=None):
    """边 (A → B) 的代价 = 距离 + 风险惩罚"""
    om = OBSTACLE_MARGIN if obstacle_margin is None else obstacle_margin
    rw = COST_RISK_WEIGHT if cost_risk_weight is None else cost_risk_weight
    seg_len = math.hypot(bx - ax, by - ay)
    risk = 0.0
    for obs in obstacles:
        d = _point_to_segment_dist(obs['x'], obs['y'], ax, ay, bx, by)
        if d < om * 3:
            risk += 1.0 / max(d, 0.01)
    return COST_DIST_WEIGHT * seg_len + rw * risk


def _build_neighbors(nodes, obstacles, om, forbidden_zones, allow_diagonal,
                     se_start, se_end):
    """预建邻接表：对每个节点找出所有可通过 _segment_safe 的邻居。"""
    neighbors = {}
    node_list = list(nodes)
    for i, (cx, cy) in enumerate(node_list):
        nbs = []
        for j, (nx, ny) in enumerate(node_list):
            if i == j:
                continue
            if not allow_diagonal:
                if abs(nx - cx) > 0.001 and abs(ny - cy) > 0.001:
                    continue
            if _segment_safe(cx, cy, nx, ny, obstacles, margin=om,
                             forbidden_zones=forbidden_zones,
                             start=se_start, end=se_end):
                nbs.append((nx, ny))
        neighbors[(cx, cy)] = nbs
    return neighbors


def _astar(start, end, nodes, obstacles, forbidden_zones=None,
           obstacle_margin=None, cost_risk_weight=None,
           allow_diagonal=False, se_start=None, se_end=None):
    """A* 搜索。相邻条件：轴对齐(默认) + 不穿过障碍物/禁区。
       allow_diagonal=True 时允许对角线移动。"""
    om = OBSTACLE_MARGIN if obstacle_margin is None else obstacle_margin
    sx, sy = start
    ex, ey = end

    # 预建邻接表（O(N²) 只算一次，而非每个节点扩展时都遍历全部）
    neighbors = _build_neighbors(nodes, obstacles, om, forbidden_zones,
                                 allow_diagonal, se_start, se_end)

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

        for nx, ny in neighbors.get(key, []):
            nkey = (nx, ny)

            step_cost = _cost(cx, cy, nx, ny, obstacles, obstacle_margin=om,
                               cost_risk_weight=cost_risk_weight)
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
    """将路径均匀采样到 target_count 个点"""
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
    """检查点是否适合做位姿校正（同 X/Y 列无障碍物阻挡 LiDAR 光束）"""
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
    """从路径点中选出适合位姿校正的点（所有点都须通过安全检查）"""
    if not waypoints:
        return []

    points = []

    for i, (x, y) in enumerate(waypoints):
        if not obstacles:
            safe = True
        else:
            safe = _is_correction_safe(x, y, obstacles)

        if not safe:
            continue

        if i == 0:
            ptype = "start"
        elif i == len(waypoints) - 1:
            ptype = "end"
        else:
            ptype = "intermediate"

        # 距离检查：与上一个校正点不能太近
        if points:
            prev = points[-1]
            dist = math.hypot(x - prev['x'], y - prev['y'])
            if dist < min_interval:
                continue

        points.append({"index": i, "x": x, "y": y,
                       "type": ptype, "safe": safe})

    return points



# ============================================================
#  打分函数
# ============================================================

def _score_path(waypoints, obstacles):
    """路径打分：总长 + 风险惩罚（越低越好）"""
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
              obstacles=None,
              num_waypoints=0,
              jump_threshold=None,
              cluster_min_beams=None,
              obstacle_margin=None,
              grid_expand=None,
              cost_risk_weight=None):
    """规划避障路径（A*）。obstacle_margin/grid_expand/cost_risk_weight 覆盖 config 值。"""
    om = OBSTACLE_MARGIN if obstacle_margin is None else obstacle_margin
    if obstacles is None:
        kwargs = {}
        if jump_threshold is not None:
            kwargs['jump_threshold'] = jump_threshold
        if cluster_min_beams is not None:
            kwargs['cluster_min_beams'] = cluster_min_beams
        obstacles, _ = detect_obstacles(car_x, car_y,
                                         forbidden_zones=forbidden_zones, **kwargs)

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

    se_start = (sx, sy)
    se_end = (ex, ey)
    nodes = _generate_nodes(sx, sy, ex, ey, obstacles,
                            forbidden_zones=forbidden_zones,
                            obstacle_margin=obstacle_margin,
                            grid_expand=grid_expand,
                            start=se_start, end=se_end)
    path_name = "A*避障"
    raw_path, total_cost = _astar((sx, sy), (ex, ey), nodes, obstacles,
                                   forbidden_zones=forbidden_zones,
                                   obstacle_margin=obstacle_margin,
                                   cost_risk_weight=cost_risk_weight,
                                   se_start=se_start, se_end=se_end)

    if raw_path is None:
        # 轴对齐A*无解，尝试对角线A*
        raw_path, total_cost = _astar((sx, sy), (ex, ey), nodes, obstacles,
                                       forbidden_zones=forbidden_zones,
                                       obstacle_margin=obstacle_margin,
                                       cost_risk_weight=cost_risk_weight,
                                       allow_diagonal=True,
                                       se_start=se_start, se_end=se_end)
        if raw_path is not None:
            path_name = "A*对角线"
    if raw_path is None:
        # 尝试两种直角路径，优先不穿禁区的
        paths = [
            [(sx, sy), (ex, sy), (ex, ey)],
            [(sx, sy), (sx, ey), (ex, ey)],
        ]
        zones = forbidden_zones or _get_forbidden_zones()
        best = None
        for wp in paths:
            safe = True
            if zones:
                for i in range(len(wp) - 1):
                    if segment_crosses_forbidden(wp[i][0], wp[i][1], wp[i+1][0], wp[i+1][1], zones):
                        safe = False
                        break
            if safe:
                best = wp
                break
            if best is None:
                best = wp
        wp = best
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
    safe_ok = margin >= om
    correction_points = plan_correction_points(wp, obstacles)

    return {
        "path_name": path_name,
        "obstacle_margin": round(margin, 3),
        "waypoints": wp,
        "correction_points": correction_points,
        "obstacles": obstacles,
        "safe": safe_ok,
        "score": round(score, 2),
    }



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

    result = plan_path(
        start_x=car_x, start_y=car_y,
        end_x=ex, end_y=ey,
        car_x=car_x, car_y=car_y,
        num_waypoints=num)

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
