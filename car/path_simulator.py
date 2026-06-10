#!/usr/bin/env python3
"""
Path Planning Simulator — click to place start/goal/obstacles/no-go zones,
see A* obstacle-avoidance path + correction points in real time.

Controls:
  Left click  — place object in current mode
  Right click — delete nearby object
  Keyboard:
    b — Begin (start)   e — End (goal)
    t — obsTacle        x — no-go zone (click two corners)
    d — Delete          r — Reset all
    Up/Down / wheel — adjust obstacle radius

Dependencies: pip install matplotlib numpy
"""

import math
import heapq
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.widgets import Slider

# ============================================================
#  Parameters (matching path_planner.py)
# ============================================================

OBSTACLE_MARGIN    = 0.4     # min distance from waypoint/segment to obstacle
GRID_EXPAND        = 0.228   # extra offset spacing around obstacles
GRID_X_STEP        = 0.4     # X interpolation step between start and end
GRID_Y_STEP        = 0.4     # Y interpolation step between start and end
COST_DIST_WEIGHT   = 1.0     # path length weight
COST_RISK_WEIGHT   = 0.0     # risk penalty weight (0 = shortest path)
CORRECTION_CORRIDOR_WIDTH = 0.3   # max |dx| or |dy| to obstacle for correction
CORRECTION_MIN_INTERVAL    = 1.0   # min interval between correction points
ROOM_X_MIN, ROOM_X_MAX = 0.1, 4.5
ROOM_Y_MIN, ROOM_Y_MAX = 0.1, 8.8

# ============================================================
#  Geometry helpers
# ============================================================

def _point_to_segment_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _rect_has_point(rx1, ry1, rx2, ry2, px, py):
    x1, x2 = min(rx1, rx2), max(rx1, rx2)
    y1, y2 = min(ry1, ry2), max(ry1, ry2)
    return x1 <= px <= x2 and y1 <= py <= y2


def _seg_intersects_seg(ax, ay, bx, by, cx, cy, dx, dy):
    def _cross(px, py, qx, qy, rx, ry):
        return (qx - px) * (ry - py) - (qy - py) * (rx - px)
    d1 = _cross(cx, cy, dx, dy, ax, ay)
    d2 = _cross(cx, cy, dx, dy, bx, by)
    d3 = _cross(ax, ay, bx, by, cx, cy)
    d4 = _cross(ax, ay, bx, by, dx, dy)
    if (d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0):
        if (d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0):
            return True
    if d1 == 0 and _rect_has_point(cx, cy, dx, dy, ax, ay): return True
    if d2 == 0 and _rect_has_point(cx, cy, dx, dy, bx, by): return True
    if d3 == 0 and _rect_has_point(ax, ay, bx, by, cx, cy): return True
    if d4 == 0 and _rect_has_point(ax, ay, bx, by, dx, dy): return True
    return False


def _seg_hits_rect(ax, ay, bx, by, rx1, ry1, rx2, ry2):
    if _rect_has_point(rx1, ry1, rx2, ry2, ax, ay): return True
    if _rect_has_point(rx1, ry1, rx2, ry2, bx, by): return True
    x1, x2 = min(rx1, rx2), max(rx1, rx2)
    y1, y2 = min(ry1, ry2), max(ry1, ry2)
    for (cx, cy, dx, dy) in [(x1,y1,x2,y1),(x2,y1,x2,y2),(x2,y2,x1,y2),(x1,y2,x1,y1)]:
        if _seg_intersects_seg(ax, ay, bx, by, cx, cy, dx, dy):
            return True
    return False


# ============================================================
#  Path planning (ported from path_planner.py)
# ============================================================

def _point_safe(px, py, obstacles, no_go_zones, margin):
    for obs in obstacles:
        if math.hypot(px - obs['x'], py - obs['y']) < margin:
            return False
    for z in no_go_zones:
        if _rect_has_point(z['x1'], z['y1'], z['x2'], z['y2'], px, py):
            return False
    return True


def _segment_safe(ax, ay, bx, by, obstacles, no_go_zones, margin):
    dx, dy = bx - ax, by - ay
    seg_len2 = dx*dx + dy*dy
    for obs in obstacles:
        if _point_to_segment_dist(obs['x'], obs['y'], ax, ay, bx, by) < margin:
            return False
        # Direction check: don't go toward obstacles
        if seg_len2 > 0:
            t = ((obs['x'] - ax) * dx + (obs['y'] - ay) * dy) / seg_len2
            if t > 1.0:
                dA = math.hypot(ax - obs['x'], ay - obs['y'])
                dB = math.hypot(bx - obs['x'], by - obs['y'])
                if dB < dA * 0.85:
                    return False
    for z in no_go_zones:
        if _seg_hits_rect(ax, ay, bx, by, z['x1'], z['y1'], z['x2'], z['y2']):
            return False
    return True


def _generate_nodes(sx, sy, ex, ey, obstacles, no_go_zones, margin, expand):
    m = margin + expand
    xs, ys = {sx, ex}, {sy, ey}
    for obs in obstacles:
        ox, oy = obs['x'], obs['y']
        for dx in (-m, 0, m): xs.add(ox + dx)
        for dy in (-m, 0, m): ys.add(oy + dy)
    for z in no_go_zones:
        x1, x2 = min(z['x1'], z['x2']), max(z['x1'], z['x2'])
        y1, y2 = min(z['y1'], z['y2']), max(z['y1'], z['y2'])
        xs.update([x1 - margin, x2 + margin])
        ys.update([y1 - margin, y2 + margin])
    if abs(ex - sx) > GRID_X_STEP:
        n = int(abs(ex - sx) / GRID_X_STEP)
        for i in range(1, n): xs.add(sx + (ex - sx) * i / n)
    if abs(ey - sy) > GRID_Y_STEP:
        n = int(abs(ey - sy) / GRID_Y_STEP)
        for i in range(1, n): ys.add(sy + (ey - sy) * i / n)
    nodes = []
    for x in xs:
        if x < ROOM_X_MIN or x > ROOM_X_MAX: continue
        for y in ys:
            if y < ROOM_Y_MIN or y > ROOM_Y_MAX: continue
            if _point_safe(x, y, obstacles, no_go_zones, margin):
                nodes.append((x, y))
    if (sx, sy) not in nodes: nodes.append((sx, sy))
    if (ex, ey) not in nodes: nodes.append((ex, ey))
    return nodes


def _astar(start, end, nodes, obstacles, no_go_zones, margin, risk_weight):
    node_set = set(nodes)
    sx, sy = start; ex, ey = end
    open_set = [(0, 0, sx, sy, None)]
    closed = {}
    while open_set:
        f, g, cx, cy, parent = heapq.heappop(open_set)
        key = (cx, cy)
        if key in closed and closed[key][0] <= g: continue
        closed[key] = (g, parent)
        if abs(cx - ex) < 0.001 and abs(cy - ey) < 0.001:
            path = [(ex, ey)]; cur = key
            while cur != (sx, sy):
                _, prev = closed[cur]; path.append(prev); cur = prev
            path.reverse(); return path, g
        for nx, ny in node_set:
            nkey = (nx, ny)
            if nkey == key: continue
            if abs(nx - cx) > 0.001 and abs(ny - cy) > 0.001: continue
            if not _segment_safe(cx, cy, nx, ny, obstacles, no_go_zones, margin): continue
            seg_len = math.hypot(nx - cx, ny - cy)
            risk = 0.0
            for obs in obstacles:
                d = _point_to_segment_dist(obs['x'], obs['y'], cx, cy, nx, ny)
                if d < margin * 3: risk += 1.0 / max(d, 0.01)
            ng = g + COST_DIST_WEIGHT * seg_len + risk_weight * risk
            nf = ng + abs(nx - ex) + abs(ny - ey)
            if nkey in closed and closed[nkey][0] <= ng: continue
            heapq.heappush(open_set, (nf, ng, nx, ny, key))
    return None, float('inf')


def _simplify_waypoints(waypoints):
    if len(waypoints) <= 2: return waypoints
    result = [waypoints[0]]
    for i in range(1, len(waypoints) - 1):
        px, py = result[-1]; cx, cy = waypoints[i]; nx, ny = waypoints[i + 1]
        prev_dx = abs(cx - px) > 0.001; prev_dy = abs(cy - py) > 0.001
        next_dx = abs(nx - cx) > 0.001; next_dy = abs(ny - cy) > 0.001
        if (prev_dx != next_dx) or (prev_dy != next_dy): result.append(waypoints[i])
    result.append(waypoints[-1]); return result


def _is_correction_safe(cx, cy, obstacles, width):
    """Correction point is safe only if no obstacle shares its X-column or Y-row."""
    for obs in obstacles:
        if abs(obs['y'] - cy) < width: return False   # front/rear beam blocked
        if abs(obs['x'] - cx) < width: return False   # left/right beam blocked
    return True


def _plan_correction_points(waypoints, obstacles, width, min_interval):
    if not waypoints: return []
    if not obstacles:
        pts = [{'idx': 0, 'x': waypoints[0][0], 'y': waypoints[0][1], 'type': 'start', 'safe': True}]
        if len(waypoints) > 1:
            pts.append({'idx': len(waypoints)-1, 'x': waypoints[-1][0], 'y': waypoints[-1][1], 'type': 'end', 'safe': True})
        return pts
    points = []
    for i, (x, y) in enumerate(waypoints):
        safe = _is_correction_safe(x, y, obstacles, width)
        ptype = 'start' if i == 0 else ('end' if i == len(waypoints)-1 else 'intermediate')
        if ptype in ('start', 'end') or safe:
            if points:
                prev = points[-1]
                if math.hypot(x - prev['x'], y - prev['y']) < min_interval:
                    if safe and not prev['safe']:
                        points[-1] = {'idx': i, 'x': x, 'y': y, 'type': prev['type'], 'safe': safe}
                    continue
            points.append({'idx': i, 'x': x, 'y': y, 'type': ptype, 'safe': safe})
    return points


def _resample_waypoints(waypoints, target_count):
    if target_count < 2 or len(waypoints) < 2: return waypoints
    if len(waypoints) <= target_count:
        segs, total = [], 0.0
        for i in range(len(waypoints)-1):
            x1,y1=waypoints[i]; x2,y2=waypoints[i+1]; L=math.hypot(x2-x1,y2-y1)
            segs.append((x1,y1,x2,y2,L)); total+=L
        if total==0: return waypoints
        result=[waypoints[0]]; remaining=target_count-1
        for idx,(x1,y1,x2,y2,L) in enumerate(segs):
            n=remaining if idx==len(segs)-1 else max(1,int(round((target_count-1)*L/total)))
            remaining-=n
            for j in range(1,n): t=j/n; result.append((x1+(x2-x1)*t, y1+(y2-y1)*t))
            if idx<len(segs)-1: result.append((x2,y2))
        if result[-1]!=waypoints[-1]: result.append(waypoints[-1])
        return result
    else:
        result=[waypoints[0]]; step=(len(waypoints)-1)/(target_count-1)
        for i in range(1,target_count-1): result.append(waypoints[int(i*step)])
        result.append(waypoints[-1]); return result


def plan_path(start, end, obstacles, no_go_zones,
              margin=OBSTACLE_MARGIN, expand=GRID_EXPAND,
              risk_weight=COST_RISK_WEIGHT, num_waypoints=0):
    if not obstacles and not no_go_zones:
        wp = _simplify_waypoints([start, (end[0], start[1]), end])
        if num_waypoints > 0: wp = _resample_waypoints(wp, num_waypoints)
        return wp, [], True, float('inf'), []

    nodes = _generate_nodes(start[0], start[1], end[0], end[1],
                            obstacles, no_go_zones, margin, expand)
    raw, cost = _astar(start, end, nodes, obstacles, no_go_zones, margin, risk_weight)
    if raw is None:
        wp = _simplify_waypoints([start, (end[0], start[1]), end])
        if num_waypoints > 0: wp = _resample_waypoints(wp, num_waypoints)
        return wp, nodes, False, 0.0, []

    wp = _simplify_waypoints(raw)
    if num_waypoints > 0: wp = _resample_waypoints(wp, num_waypoints)
    correction = _plan_correction_points(wp, obstacles, CORRECTION_CORRIDOR_WIDTH, CORRECTION_MIN_INTERVAL)
    min_d = float('inf')
    for i in range(len(wp)-1):
        for obs in obstacles:
            d = _point_to_segment_dist(obs['x'], obs['y'], wp[i][0], wp[i][1], wp[i+1][0], wp[i+1][1])
            min_d = min(min_d, d)
    return wp, nodes, min_d >= margin, min_d, correction


# ============================================================
#  Interactive Simulator
# ============================================================

class Simulator:
    def __init__(self):
        self.start = None
        self.end = None
        self.obstacles = []       # [{'x':,'y':,'r':}, ...]
        self.no_go_zones = []     # [{'x1':,'y1':,'x2':,'y2':}, ...]

        self.waypoints = []
        self.nodes = []
        self.correction_points = []
        self.path_safe = True
        self.min_dist = float('inf')

        self.mode = 'start'       # start | end | obstacle | zone | delete
        self.obs_radius = 0.15
        self._zone_start = None

        self._margin  = OBSTACLE_MARGIN
        self._expand  = GRID_EXPAND
        self._risk    = COST_RISK_WEIGHT
        self._n_wp    = 0

        self._build()

    # ── UI ──
    def _build(self):
        self.fig = plt.figure(figsize=(13, 8.5))
        self.fig.canvas.manager.set_window_title('Path Planning Simulator')

        self.ax = self.fig.add_axes([0.05, 0.22, 0.68, 0.74])

        # Sliders (matching path_planner parameters)
        sliders = [
            ('obstacle_margin', 'OBSTACLE_MARGIN (m)', 0.1, 2.0, OBSTACLE_MARGIN),
            ('grid_expand',      'GRID_EXPAND (m)',      0.05, 1.0, GRID_EXPAND),
            ('risk_weight',      'COST_RISK_WEIGHT',     0.0, 50.0, COST_RISK_WEIGHT),
            ('num_waypoints',    'num_waypoints (0=auto)', 0, 20, 0),
        ]
        self._sliders = {}
        for i, (key, label, vmin, vmax, vinit) in enumerate(sliders):
            y = 0.65 - i * 0.11
            ax_s = self.fig.add_axes([0.76, y, 0.21, 0.035])
            s = Slider(ax_s, label, vmin, vmax, valinit=vinit,
                       valfmt='%.2f' if vmax > 10 else '%d')
            s.on_changed(self._on_slider)
            self._sliders[key] = s

        # Events
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)

        self._preset()
        self._replan()
        self._draw()

    def _preset(self):
        self.start = (0.5, 0.5)
        self.end = (3.5, 7.5)
        self.obstacles = [
            {'x': 1.2, 'y': 2.5, 'r': 0.15},
            {'x': 2.8, 'y': 3.5, 'r': 0.15},
            {'x': 1.5, 'y': 5.5, 'r': 0.15},
            {'x': 3.0, 'y': 6.2, 'r': 0.15},
        ]

    # ── Planning ──
    def _replan(self):
        if self.start is None or self.end is None:
            self.waypoints, self.nodes, self.correction_points = [], [], []
            return
        planner_obs = [{'x': o['x'], 'y': o['y']} for o in self.obstacles]
        wp, nodes, safe, min_d, cp = plan_path(
            self.start, self.end, planner_obs, self.no_go_zones,
            margin=self._margin, expand=self._expand,
            risk_weight=self._risk, num_waypoints=self._n_wp)
        self.waypoints = wp
        self.nodes = nodes
        self.path_safe = safe
        self.min_dist = min_d
        self.correction_points = cp

    # ── Drawing ──
    def _draw(self):
        ax = self.ax; ax.clear()
        ax.set_xlim(-0.3, 5.0); ax.set_ylim(-0.3, 9.2)
        ax.set_aspect('equal'); ax.set_facecolor('#fafaf5')

        # Grid
        for v in np.arange(0, 5.5, 0.5):
            ax.axvline(v, color='#e0e0e0', lw=0.4, ls=':')
        for v in np.arange(0, 9.5, 0.5):
            ax.axhline(v, color='#e0e0e0', lw=0.4, ls=':')

        # Room
        rw = ROOM_X_MAX - ROOM_X_MIN; rh = ROOM_Y_MAX - ROOM_Y_MIN
        ax.add_patch(Rectangle((ROOM_X_MIN, ROOM_Y_MIN), rw, rh,
                                fill=False, edgecolor='#333', lw=2))
        ax.text(2.3, 0.0, 'Front wall X=0.1', ha='center', fontsize=7, color='#888')
        ax.text(2.3, 8.95, 'Back wall X=4.5', ha='center', fontsize=7, color='#888')
        ax.text(4.65, 4.5, 'Left wall Y=8.8', ha='center', fontsize=7, color='#888')
        ax.text(-0.05, 4.5, 'Right wall Y=0.1', ha='center', fontsize=7, color='#888')

        # No-go zones
        for i, z in enumerate(self.no_go_zones):
            x1, x2 = min(z['x1'], z['x2']), max(z['x1'], z['x2'])
            y1, y2 = min(z['y1'], z['y2']), max(z['y1'], z['y2'])
            ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1,
                                    fill=True, facecolor='#ff9800', alpha=0.25,
                                    edgecolor='#e65100', lw=2, ls='--', zorder=2))
            ax.text((x1+x2)/2, (y1+y2)/2, f'NG#{i+1}', ha='center', va='center',
                    fontsize=7, color='#e65100', fontweight='bold')

        # Zone preview
        if self._zone_start and self.mode == 'zone':
            zx, zy = self._zone_start
            if hasattr(self, '_hover_pos') and self._hover_pos:
                hx, hy = self._hover_pos
                x1, x2 = min(zx, hx), max(zx, hx)
                y1, y2 = min(zy, hy), max(zy, hy)
                ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1,
                                        fill=True, facecolor='#ff9800', alpha=0.1,
                                        edgecolor='#e65100', lw=1, ls=':'))

        # Obstacle X/Y lines (show "forbidden correction zone")
        for obs in self.obstacles:
            ox, oy = obs['x'], obs['y']
            # faint cross lines showing blocked correction zone
            ax.axhline(oy, xmin=0, xmax=1, color='#ef9a9a', lw=0.5, ls='--', alpha=0.5)
            ax.axvline(ox, ymin=0, ymax=1, color='#ef9a9a', lw=0.5, ls='--', alpha=0.5)

        # Obstacles
        for i, obs in enumerate(self.obstacles):
            ox, oy, r = obs['x'], obs['y'], obs['r']
            # margin circle
            ax.add_patch(Circle((ox, oy), self._margin,
                                fill=True, facecolor='#ffcdd2', alpha=0.25,
                                edgecolor='none', zorder=2))
            # obstacle body
            ax.add_patch(Circle((ox, oy), r,
                                fill=True, facecolor='#e53935', alpha=0.7,
                                edgecolor='#b71c1c', lw=1.5, zorder=3))
            ax.text(ox, oy + r + 0.12, f'#{i+1}', ha='center',
                    fontsize=7, color='#c62828', fontweight='bold')

        # Candidate nodes
        if self.nodes:
            nx, ny = zip(*self.nodes)
            ax.scatter(nx, ny, c='#bdbdbd', s=8, alpha=0.5, zorder=4)

        # Path
        if len(self.waypoints) > 1:
            wx, wy = zip(*self.waypoints)
            color = '#2e7d32' if self.path_safe else '#c62828'
            ax.plot(wx, wy, color=color, lw=2.5, marker='o', ms=6,
                    mfc='white', mec=color, mew=2, zorder=5)
            # arrows
            for i in range(len(self.waypoints)-1):
                x1, y1 = self.waypoints[i]; x2, y2 = self.waypoints[i+1]
                s = math.hypot(x2-x1, y2-y1)
                if s > 0.05:
                    mx, my = (x1+x2)/2, (y1+y2)/2
                    dx, dy = (x2-x1)/s*0.1, (y2-y1)/s*0.1
                    ax.arrow(mx-dx, my-dy, dx*2, dy*2,
                             head_width=0.06, head_length=0.08,
                             fc=color, ec=color, alpha=0.5, zorder=6)

        # Correction points
        cp_indices = set()
        for cp in self.correction_points:
            idx, x, y, safe, ptype = cp['idx'], cp['x'], cp['y'], cp['safe'], cp['type']
            cp_indices.add(idx)
            if safe:
                marker, size, col = {'start': 's', 'end': '^', 'intermediate': 'D'}[ptype], \
                                    {'start': 80, 'end': 80, 'intermediate': 55}[ptype], \
                                    '#ff8f00'
            else:
                marker, size, col = 'D', 50, '#ffcc80'
            ax.scatter(x, y, c=col, s=size, marker=marker,
                       edgecolors='white', lw=1, zorder=9)

        # Start / End
        if self.start:
            ax.scatter(*self.start, c='#2e7d32', s=180, marker='o',
                       edgecolors='white', lw=2, zorder=10)
            ax.text(self.start[0]-0.12, self.start[1]-0.15, 'Start',
                    fontsize=9, color='#2e7d32', fontweight='bold', ha='right')
        if self.end:
            ax.scatter(*self.end, c='#c62828', s=180, marker='X',
                       edgecolors='white', lw=2, zorder=10)
            ax.text(self.end[0]+0.12, self.end[1]-0.15, 'Goal',
                    fontsize=9, color='#c62828', fontweight='bold', ha='left')

        # Title
        mode_names = {'start': 'Start', 'end': 'Goal', 'obstacle': 'Obstacle',
                      'zone': 'No-Go', 'delete': 'Delete'}
        info = f"Mode: [{mode_names[self.mode][0]}]{mode_names[self.mode]}"
        if self.mode == 'zone' and self._zone_start:
            info += f" (corner1 {self._zone_start[0]:.2f},{self._zone_start[1]:.2f})"
        info += f" | Obs: {len(self.obstacles)}  NG: {len(self.no_go_zones)}"
        if self.waypoints:
            safe_str = 'SAFE' if self.path_safe else 'DANGER!'
            info += f" | Path: {len(self.waypoints)} pts {safe_str}"
            info += f" | Correct: {len(self.correction_points)} pts"
            if self.min_dist != float('inf'):
                info += f" | minDist={self.min_dist:.2f}m"
        ax.set_title(info, fontsize=9, fontweight='bold', pad=6,
                     color='#c62828' if (self.waypoints and not self.path_safe) else '#333')

        # Help
        help_text = ("[b]Start [e]Goal [t]Obstacle [x]NoGo [d]Delete [r]Reset | "
                     "sliders: margin expand risk waypoints")
        ax.text(0.02, 0.01, help_text, transform=ax.transAxes,
                fontsize=7, color='#aaa', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        self.fig.canvas.draw_idle()

    # ── Events ──
    def _on_click(self, event):
        if event.inaxes != self.ax: return
        x, y = event.xdata, event.ydata
        in_room = ROOM_X_MIN-0.05 <= x <= ROOM_X_MAX+0.05 and ROOM_Y_MIN-0.05 <= y <= ROOM_Y_MAX+0.05

        if event.button == 1:
            if self.mode == 'start' and in_room:
                self.start = (round(x, 3), round(y, 3)); self._replan()
            elif self.mode == 'end' and in_room:
                self.end = (round(x, 3), round(y, 3)); self._replan()
            elif self.mode == 'obstacle' and in_room:
                ok = all(math.hypot(x-o['x'], y-o['y']) > self.obs_radius*2 for o in self.obstacles)
                if ok:
                    self.obstacles.append({'x': round(x,3), 'y': round(y,3), 'r': self.obs_radius})
                    self._replan()
            elif self.mode == 'zone' and in_room:
                if self._zone_start is None:
                    self._zone_start = (round(x,3), round(y,3))
                else:
                    x1, y1 = self._zone_start; x2, y2 = round(x,3), round(y,3)
                    if abs(x2-x1) > 0.05 and abs(y2-y1) > 0.05:
                        self.no_go_zones.append({'x1':x1,'y1':y1,'x2':x2,'y2':y2})
                        self._zone_start = None; self._replan()
                    else:
                        self._zone_start = None
            elif self.mode == 'delete':
                self._delete_near(x, y)
        elif event.button == 3:
            if self.mode == 'zone' and self._zone_start:
                self._zone_start = None
            else:
                self._delete_near(x, y)
        self._draw()

    def _on_motion(self, event):
        if event.inaxes == self.ax:
            self._hover_pos = (event.xdata, event.ydata)
            if self.mode == 'zone' and self._zone_start: self._draw()

    def _on_key(self, event):
        k = event.key.lower() if event.key else ''
        if   k == 'b': self.mode = 'start'; self._zone_start = None
        elif k == 'e': self.mode = 'end'; self._zone_start = None
        elif k == 't': self.mode = 'obstacle'; self._zone_start = None
        elif k == 'x': self.mode = 'zone'; self._zone_start = None
        elif k == 'd': self.mode = 'delete'; self._zone_start = None
        elif k == 'r':
            self.start = self.end = None; self.obstacles = []; self.no_go_zones = []
            self._zone_start = None; self._replan()
        elif k == 'up':    self.obs_radius = min(0.5, self.obs_radius + 0.05)
        elif k == 'down':  self.obs_radius = max(0.05, self.obs_radius - 0.05)
        elif k == 'escape': self._zone_start = None
        self._draw()

    def _on_scroll(self, event):
        d = 0.03 if event.button == 'up' else -0.03
        self.obs_radius = max(0.05, min(0.5, self.obs_radius + d)); self._draw()

    def _on_slider(self, val):
        self._margin = self._sliders['obstacle_margin'].val
        self._expand = self._sliders['grid_expand'].val
        self._risk   = self._sliders['risk_weight'].val
        self._n_wp   = int(self._sliders['num_waypoints'].val)
        self._replan(); self._draw()

    def _delete_near(self, x, y):
        th = 0.3
        if self.start and math.hypot(x-self.start[0], y-self.start[1]) < th:
            self.start = None; self._replan(); return
        if self.end and math.hypot(x-self.end[0], y-self.end[1]) < th:
            self.end = None; self._replan(); return
        for i in range(len(self.obstacles)-1, -1, -1):
            if math.hypot(x-self.obstacles[i]['x'], y-self.obstacles[i]['y']) < th:
                self.obstacles.pop(i); self._replan(); return
        for i in range(len(self.no_go_zones)-1, -1, -1):
            z = self.no_go_zones[i]
            if _rect_has_point(z['x1'], z['y1'], z['x2'], z['y2'], x, y):
                self.no_go_zones.pop(i); self._replan(); return


if __name__ == "__main__":
    import os as _os
    if _os.path.exists('/mnt/wslg/runtime-dir'):
        d = _os.environ.get('DISPLAY', '')
        if not d or d.startswith('172.'):
            _os.environ['DISPLAY'] = ':0'
            _os.environ['WAYLAND_DISPLAY'] = 'wayland-0'
    sim = Simulator()
    plt.show()
