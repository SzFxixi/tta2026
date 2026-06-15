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
    for obs in obstacles:
        if math.hypot(px - obs['x'], py - obs['y']) < margin:
            return False
    if start and abs(px - start[0]) < 0.001 and abs(py - start[1]) < 0.001:
        return True
    if end and abs(px - end[0]) < 0.001 and abs(py - end[1]) < 0.001:
        return True
    zones = forbidden_zones or _get_forbidden_zones()
    if zones and point_in_forbidden(px, py, zones):
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

    nodes = []
    x = x_min
    while x <= x_max + 0.0001:
        y = y_min
        while y <= y_max + 0.0001:
            if _point_safe(x, y, obstacles, forbidden_zones=forbidden_zones,
                           start=start, end=end):
                nodes.append((x, y))
            y += GRID_Y_STEP
        x += GRID_X_STEP

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
        if start and abs(x1 - start[0]) < 0.001 and abs(y1 - start[1]) < 0.001:
            continue
        if start and abs(x2 - start[0]) < 0.001 and abs(y2 - start[1]) < 0.001:
            continue
        if end and abs(x1 - end[0]) < 0.001 and abs(y1 - end[1]) < 0.001:
            continue
        if end and abs(x2 - end[0]) < 0.001 and abs(y2 - end[1]) < 0.001:
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
              grid_expand=None):
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

    shortest_candidates = []
    seen = set()
    for wp in shortest_raw:
        key = tuple((round(x, 3), round(y, 3)) for x, y in _simplify_waypoints(wp))
        if key not in seen:
            seen.add(key)
            shortest_candidates.append(_simplify_waypoints(wp))

    def _path_ok(wp):
        if not _check_forbidden(wp, zones, se_start, se_end):
            return False, float('inf')
        d = _path_min_dist_to_obstacles(wp, obstacles)
        return True, d

    use_shortest = False
    for wp in shortest_candidates:
        ok, d = _path_ok(wp)
        if ok and d >= EXPANSION_RADIUS_2:
            use_shortest = True
            break

    if use_shortest:
        candidates = shortest_candidates
        path_name = "最短路径"
    else:
        candidates = []
        if astar_path is not None:
            candidates.append(_simplify_waypoints(astar_path))
        path_name = "A*避障"

    survivors = []
    for wp in candidates:
        ok, min_d = _path_ok(wp)
        if not ok:
            continue
        if min_d < EXPANSION_RADIUS_3:
            continue
        corners = len(_simplify_waypoints(wp)) - 2
        survivors.append((corners, wp, min_d))

    if survivors:
        survivors.sort(key=lambda x: x[0])
        corners, wp, margin = survivors[0]
        return {
            "path_name": path_name,
            "obstacle_margin": round(margin, 3),
            "waypoints": wp,
            "obstacles": obstacles,
            "safe": True,
        }

    wp = _simplify_waypoints([(sx, sy), (ex, sy), (ex, ey)])
    margin = _path_min_dist_to_obstacles(wp, obstacles)
    return {
        "path_name": "先X后Y(安全无解)",
        "obstacle_margin": round(margin, 3),
        "waypoints": wp,
        "obstacles": obstacles,
        "safe": False,
    }
