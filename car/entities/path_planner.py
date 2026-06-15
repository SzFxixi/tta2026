#!/usr/bin/env python3
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

from utils.config_loader import cfg

OBSTACLE_MARGIN = cfg.path_planning.obstacle_margin
WALL_MARGIN = cfg.path_planning.wall_margin
GRID_EXPAND = cfg.path_planning.grid_expand
GRID_X_STEP = cfg.path_planning.grid_x_step
GRID_Y_STEP = cfg.path_planning.grid_y_step
EXPANSION_RADIUS_2 = cfg.path_planning.expansion_radius_2
EXPANSION_RADIUS_3 = cfg.path_planning.expansion_radius_3
ROOM_X_MIN = cfg.room.x_min
ROOM_Y_MIN = cfg.room.y_min
ROOM_X_MAX = cfg.room.x_max
ROOM_Y_MAX = cfg.room.y_max
X_OFFSET = cfg.room.x_offset
Y_OFFSET = cfg.room.y_offset

from utils.wall_positioning import fit_walls

_pos_cache = None
_pos_cache_time = 0
_POS_CACHE_TTL = 0.05

def _read_position():
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

def get_car_position():
    return _read_position()

def _point_to_segment_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))

_cached_zones = None
_loaded_zones = False

def _get_forbidden_zones():
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
    for obs in obstacles:
        if _point_to_segment_dist(obs['x'], obs['y'], ax, ay, bx, by) < margin:
            return False
    zones = forbidden_zones or _get_forbidden_zones()
    if not zones:
        return True
    # 只在端点本身落在禁区内时才豁免（让车能走出来）
    a_in = point_in_forbidden(ax, ay, zones)
    b_in = point_in_forbidden(bx, by, zones)
    if start and abs(ax - start[0]) < 0.001 and abs(ay - start[1]) < 0.001 and a_in:
        return True
    if start and abs(bx - start[0]) < 0.001 and abs(by - start[1]) < 0.001 and b_in:
        return True
    if end and abs(ax - end[0]) < 0.001 and abs(ay - end[1]) < 0.001 and a_in:
        return True
    if end and abs(bx - end[0]) < 0.001 and abs(by - end[1]) < 0.001 and b_in:
        return True
    if segment_crosses_forbidden(ax, ay, bx, by, zones):
        return False
    return True

def _point_safe(px, py, obstacles, margin=OBSTACLE_MARGIN,
                forbidden_zones=None, start=None, end=None):
    for obs in obstacles:
        if math.hypot(px - obs['x'], py - obs['y']) < margin:
            return False
    zones = forbidden_zones or _get_forbidden_zones()
    if not zones:
        return True
    # 起/终点本身在禁区内 → 放行（车就在这）；否则正常检查
    in_zone = point_in_forbidden(px, py, zones)
    if start and abs(px - start[0]) < 0.001 and abs(py - start[1]) < 0.001:
        return True  # 起点永远允许
    if end and abs(px - end[0]) < 0.001 and abs(py - end[1]) < 0.001:
        return True  # 终点永远允许
    if in_zone:
        return False
    return True

def _path_min_dist_to_obstacles(waypoints, obstacles):
    d = float('inf')
    for i in range(len(waypoints) - 1):
        for obs in obstacles:
            d = min(d, _point_to_segment_dist(
                obs['x'], obs['y'],
                waypoints[i][0], waypoints[i][1],
                waypoints[i+1][0], waypoints[i+1][1]))
    return d

def _generate_nodes(sx, sy, ex, ey, obstacles, forbidden_zones=None,
                     obstacle_margin=None, grid_expand=None,
                     start=None, end=None):
    x_min = ROOM_X_MIN + WALL_MARGIN
    x_max = ROOM_X_MAX - WALL_MARGIN
    y_min = ROOM_Y_MIN + WALL_MARGIN
    y_max = ROOM_Y_MAX - WALL_MARGIN

    nodes = set()
    x = x_min
    while x <= x_max + 0.0001:
        y = y_min
        while y <= y_max + 0.0001:
            if _point_safe(x, y, obstacles, forbidden_zones=forbidden_zones,
                           start=start, end=end):
                nodes.add((round(x, 4), round(y, 4)))
            y += GRID_Y_STEP
        x += GRID_X_STEP

    for bridge_x in (sx, ex):
        y = y_min
        while y <= y_max + 0.0001:
            key = (round(bridge_x, 4), round(y, 4))
            if key not in nodes and _point_safe(bridge_x, y, obstacles,
                                                 forbidden_zones=forbidden_zones,
                                                 start=start, end=end):
                nodes.add(key)
            y += GRID_Y_STEP
    for bridge_y in (sy, ey):
        x = x_min
        while x <= x_max + 0.0001:
            key = (round(x, 4), round(bridge_y, 4))
            if key not in nodes and _point_safe(x, bridge_y, obstacles,
                                                 forbidden_zones=forbidden_zones,
                                                 start=start, end=end):
                nodes.add(key)
            x += GRID_X_STEP

    nodes = list(nodes)
    if (sx, sy) not in nodes:
        nodes.append((sx, sy))
    if (ex, ey) not in nodes:
        nodes.append((ex, ey))
    return nodes

def _heuristic(ax, ay, bx, by):
    return abs(ax - bx) + abs(ay - by)

def _build_neighbors(nodes, obstacles, om, forbidden_zones, se_start, se_end):
    neighbors = {}
    node_list = list(nodes)
    for i, (cx, cy) in enumerate(node_list):
        nbs = []
        for j, (nx, ny) in enumerate(node_list):
            if i == j:
                continue
            if abs(nx - cx) > 0.001 and abs(ny - cy) > 0.001:
                continue
            if _segment_safe(cx, cy, nx, ny, obstacles, margin=om,
                             forbidden_zones=forbidden_zones,
                             start=se_start, end=se_end):
                nbs.append((nx, ny))
        neighbors[(cx, cy)] = nbs
    return neighbors

def _astar(start, end, nodes, obstacles, forbidden_zones=None,
           obstacle_margin=None, se_start=None, se_end=None):
    om = OBSTACLE_MARGIN if obstacle_margin is None else obstacle_margin
    sx, sy = start
    ex, ey = end

    neighbors = _build_neighbors(nodes, obstacles, om, forbidden_zones,
                                 se_start, se_end)

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
            ng = g + 1
            nf = ng + _heuristic(nx, ny, ex, ey)

            if nkey in closed and closed[nkey][0] <= ng:
                continue

            heapq.heappush(open_set, (nf, ng, nx, ny, key))

    return None, float('inf')

def _simplify_waypoints(waypoints):
    if len(waypoints) <= 2:
        return waypoints
    result = [waypoints[0]]
    for i in range(1, len(waypoints) - 1):
        px, py = result[-1]
        cx, cy = waypoints[i]
        nx, ny = waypoints[i + 1]
        prev_dx = abs(cx - px) > 0.001
        prev_dy = abs(cy - py) > 0.001
        next_dx = abs(nx - cx) > 0.001
        next_dy = abs(ny - cy) > 0.001
        if (prev_dx != next_dx) or (prev_dy != next_dy):
            result.append(waypoints[i])
    result.append(waypoints[-1])
    return result

def _check_forbidden(wp, zones, start, end):
    if not zones:
        return True
    for i in range(len(wp) - 1):
        x1, y1 = wp[i]
        x2, y2 = wp[i + 1]
        # 终点段豁免：允许进入目标点所在的禁区
        if end and abs(x2 - end[0]) < 0.001 and abs(y2 - end[1]) < 0.001:
            if point_in_forbidden(x2, y2, zones):
                continue
        if segment_crosses_forbidden(x1, y1, x2, y2, zones):
            return False
    return True

def plan_path(start_x, start_y, end_x, end_y,
              car_x, car_y,
              forbidden_zones=None,
              obstacles=None,
              jump_threshold=None,
              cluster_min_beams=None,
              obstacle_margin=None,
              grid_expand=None,
              _skip_escape=False):
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
    se_start = (sx, sy)
    se_end = (ex, ey)
    zones = forbidden_zones or _get_forbidden_zones()

    if not obstacles and not zones:
        wp = _simplify_waypoints([(sx, sy), (ex, sy), (ex, ey)])
        return {
            "path_name": "直接路径(无障碍)",
            "obstacle_margin": float('inf'),
            "waypoints": wp,
            "obstacles": [],
            "safe": True,
        }

    shortest_raw = [
        [(sx, sy), (ex, sy), (ex, ey)],
        [(sx, sy), (sx, ey), (ex, ey)],
    ]

    nodes = _generate_nodes(sx, sy, ex, ey, obstacles,
                            forbidden_zones=forbidden_zones,
                            obstacle_margin=obstacle_margin,
                            grid_expand=grid_expand,
                            start=se_start, end=se_end)
    astar_path, _ = _astar((sx, sy), (ex, ey), nodes, obstacles,
                            forbidden_zones=forbidden_zones,
                            obstacle_margin=obstacle_margin,
                            se_start=se_start, se_end=se_end)
    manhattan = abs(ex - sx) + abs(ey - sy)
    if astar_path is not None:
        astar_len = sum(math.hypot(astar_path[i+1][0] - astar_path[i][0],
                                    astar_path[i+1][1] - astar_path[i][1])
                        for i in range(len(astar_path) - 1))
        if abs(astar_len - manhattan) < 0.01:
            shortest_raw.append(astar_path)

    # (简化版, 原始版) — 禁区检查用原始版（细粒度），距离计算用简化版
    shortest_candidates = []  # [(simplified, raw)]
    seen = set()
    for wp in shortest_raw:
        swp = _simplify_waypoints(wp)
        key = tuple((round(x, 3), round(y, 3)) for x, y in swp)
        if key not in seen:
            seen.add(key)
            shortest_candidates.append((swp, wp))

    def _path_ok(simplified, raw=None):
        check = raw if raw else simplified
        if not _check_forbidden(check, zones, se_start, se_end):
            return False, float('inf')
        d = _path_min_dist_to_obstacles(simplified, obstacles)
        return True, d

    use_shortest = False
    for swp, raw in shortest_candidates:
        ok, d = _path_ok(swp, raw)
        if ok and d >= EXPANSION_RADIUS_2:
            use_shortest = True
            break

    # ── 最短路径 + 一级降级 ──
    for r2, r3, tag in [(EXPANSION_RADIUS_2, EXPANSION_RADIUS_3, ""),
                          (EXPANSION_RADIUS_2 * 0.8, EXPANSION_RADIUS_3 * 0.8, "(降级)")]:
        survivors = []
        for swp, raw in shortest_candidates:
            ok, min_d = _path_ok(swp, raw)
            if not ok:
                continue
            if min_d < r3:
                continue
            corners = len(swp) - 2
            survivors.append((corners, swp, min_d))
        if survivors:
            survivors.sort(key=lambda x: x[0])
            corners, wp, margin = survivors[0]
            safe = margin >= r2
            return {
                "path_name": f"最短路径{'（安全不足）' if not safe else ''}{tag}",
                "obstacle_margin": round(margin, 3),
                "waypoints": wp,
                "obstacles": obstacles,
                "safe": safe,
            }

    # ── A* 非最短路径 + om 降级 ──
    om = OBSTACLE_MARGIN if obstacle_margin is None else obstacle_margin
    ge = GRID_EXPAND if grid_expand is None else grid_expand
    for level in range(5):
        nodes = _generate_nodes(sx, sy, ex, ey, obstacles,
                                forbidden_zones=forbidden_zones,
                                obstacle_margin=om, grid_expand=ge,
                                start=se_start, end=se_end)
        apath, _ = _astar((sx, sy), (ex, ey), nodes, obstacles,
                           forbidden_zones=forbidden_zones,
                           obstacle_margin=om,
                           se_start=se_start, se_end=se_end)
        candidates = []
        if apath is not None:
            candidates.append((_simplify_waypoints(apath), apath))

        survivors = []
        for swp, raw in candidates:
            ok, min_d = _path_ok(swp, raw)
            if not ok:
                continue
            if min_d < EXPANSION_RADIUS_3:
                continue
            corners = len(swp) - 2
            survivors.append((corners, swp, min_d))

        if survivors:
            survivors.sort(key=lambda x: x[0])
            corners, wp, margin = survivors[0]
            level_tag = f"(降级{level})" if level > 0 else ""
            return {
                "path_name": "A*避障" + level_tag,
                "obstacle_margin": round(margin, 3),
                "waypoints": wp,
                "obstacles": obstacles,
                "safe": True,
            }

        if om <= 0.1:
            break
        om *= 0.8
        ge = max(0, ge * 0.5)

    # ── 紧急避险：迭代探路，逐步远离危险，直到找到安全路径 ──
    if not _skip_escape:
        def _danger_score(px, py):
            s = 0.0
            for o in obstacles:
                s += 1.0 / max(math.hypot(px - o['x'], py - o['y']), 0.01)
            if zones:
                for zxmin, zxmax, zymin, zymax in zones:
                    dx = max(zxmin - px, 0, px - zxmax)
                    dy = max(zymin - py, 0, py - zymax)
                    s += 2.0 / max(math.hypot(dx, dy), 0.01)
            return s

        print(f"  [避险] 起点危险，探路中...")
        best_path = {
            "path_name": "先X后Y(安全无解)",
            "obstacle_margin": round(_path_min_dist_to_obstacles(
                _simplify_waypoints([(sx, sy), (ex, sy), (ex, ey)]), obstacles), 3),
            "waypoints": _simplify_waypoints([(sx, sy), (ex, sy), (ex, ey)]),
            "obstacles": obstacles, "safe": False,
        }
        cx, cy = sx, sy
        cur_score = _danger_score(cx, cy)

        for it in range(20):
            r = plan_path(cx, cy, ex, ey, car_x, car_y,
                          forbidden_zones=forbidden_zones, obstacles=obstacles,
                          obstacle_margin=obstacle_margin, grid_expand=grid_expand,
                          _skip_escape=True)
            if r and r["safe"]:
                r["waypoints"] = [(sx, sy)] + r["waypoints"]
                print(f"  [避险] 迭代{it} 找到安全路径")
                return r
            if r and r.get("obstacle_margin", 0) > best_path.get("obstacle_margin", 0):
                best_path = r

            # 四方向打分，选最安全的
            best_nx, best_ny, best_ns = cx, cy, cur_score
            for dx, dy in [(0.1, 0), (-0.1, 0), (0, 0.1), (0, -0.1)]:
                nx, ny = cx + dx, cy + dy
                if not (ROOM_X_MIN <= nx <= ROOM_X_MAX and ROOM_Y_MIN <= ny <= ROOM_Y_MAX):
                    continue
                s = _danger_score(nx, ny)
                if s < best_ns:  # 分数越低越安全
                    best_nx, best_ny, best_ns = nx, ny, s
            if best_ns >= cur_score:
                break  # 无法更安全了
            cx, cy = best_nx, best_ny
            cur_score = best_ns

        if best_path.get("waypoints") and best_path["waypoints"][0] != (sx, sy):
            best_path["waypoints"] = [(sx, sy)] + best_path["waypoints"]
        print(f"  [避险] 未找到安全路径，选最佳备选")
        return best_path

    # 兜底
    wp = _simplify_waypoints([(sx, sy), (ex, sy), (ex, ey)])
    margin = _path_min_dist_to_obstacles(wp, obstacles)
    return {
        "path_name": "先X后Y(安全无解)",
        "obstacle_margin": round(margin, 3),
        "waypoints": wp,
        "obstacles": obstacles,
        "safe": False,
    }
