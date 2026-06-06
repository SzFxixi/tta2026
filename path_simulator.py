#!/usr/bin/env python3
"""
路径规划微型仿真 — 点击画布放置起点/终点/障碍物/禁行区，实时查看 A* 避障路径。

操作:
  鼠标左键  — 根据当前模式放置物体
  鼠标右键  — 删除点击处的物体
  键盘:
    s — 起点模式      e — 终点模式
    o — 障碍物模式    z — 禁行区模式（连点两个角画矩形）
    d — 删除模式      r — 全部重置
    ↑↓ / 滚轮 — 调整障碍物半径
  右侧滑块 — 安全边距、栅格扩展、风险权重

依赖: pip install matplotlib numpy
"""

import math
import heapq
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.widgets import Slider

matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

ROOM_X, ROOM_Y = (0.0, 4.6), (0.0, 9.0)  # 场地 4.6m × 9m


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


def _rect_has_point(rx1, ry1, rx2, ry2, px, py):
    """点 (px,py) 是否在矩形内"""
    x1, x2 = min(rx1, rx2), max(rx1, rx2)
    y1, y2 = min(ry1, ry2), max(ry1, ry2)
    return x1 <= px <= x2 and y1 <= py <= y2


def _seg_intersects_seg(ax, ay, bx, by, cx, cy, dx, dy):
    """两线段 AB, CD 是否相交（含端点）"""
    def _cross(px, py, qx, qy, rx, ry):
        return (qx - px) * (ry - py) - (qy - py) * (rx - px)
    d1 = _cross(cx, cy, dx, dy, ax, ay)
    d2 = _cross(cx, cy, dx, dy, bx, by)
    d3 = _cross(ax, ay, bx, by, cx, cy)
    d4 = _cross(ax, ay, bx, by, dx, dy)
    if (d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0):
        if (d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0):
            return True
    if d1 == 0 and _rect_has_point(cx, cy, dx, dy, ax, ay):
        return True
    if d2 == 0 and _rect_has_point(cx, cy, dx, dy, bx, by):
        return True
    if d3 == 0 and _rect_has_point(ax, ay, bx, by, cx, cy):
        return True
    if d4 == 0 and _rect_has_point(ax, ay, bx, by, dx, dy):
        return True
    return False


def _seg_hits_rect(ax, ay, bx, by, rx1, ry1, rx2, ry2):
    """线段 AB 是否穿过矩形（含端点在内、与四边相交）"""
    if _rect_has_point(rx1, ry1, rx2, ry2, ax, ay):
        return True
    if _rect_has_point(rx1, ry1, rx2, ry2, bx, by):
        return True
    x1, x2 = min(rx1, rx2), max(rx1, rx2)
    y1, y2 = min(ry1, ry2), max(ry1, ry2)
    # 与矩形四条边逐一检测
    for (cx, cy, dx, dy) in [
        (x1, y1, x2, y1),  # 下边
        (x2, y1, x2, y2),  # 右边
        (x2, y2, x1, y2),  # 上边
        (x1, y2, x1, y1),  # 左边
    ]:
        if _seg_intersects_seg(ax, ay, bx, by, cx, cy, dx, dy):
            return True
    return False


# ============================================================
#  路径规划算法（从 path_planner.py 移植）
# ============================================================

def _point_safe(px, py, obstacles, no_go_zones, margin):
    """点是否安全（避开所有障碍物和禁行区）"""
    for obs in obstacles:
        if math.hypot(px - obs['x'], py - obs['y']) < margin:
            return False
    for z in no_go_zones:
        if _rect_has_point(z['x1'], z['y1'], z['x2'], z['y2'], px, py):
            return False
    return True


def _segment_safe(ax, ay, bx, by, obstacles, no_go_zones, margin):
    """线段是否安全（不穿过障碍物安全圆、不穿过禁行区）"""
    for obs in obstacles:
        if _point_to_segment_dist(obs['x'], obs['y'], ax, ay, bx, by) < margin:
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
        for dx in (-m, 0, m):
            xs.add(ox + dx)
        for dy in (-m, 0, m):
            ys.add(oy + dy)
    # 禁行区边界也生成候选路点
    for z in no_go_zones:
        x1, x2 = min(z['x1'], z['x2']), max(z['x1'], z['x2'])
        y1, y2 = min(z['y1'], z['y2']), max(z['y1'], z['y2'])
        for ox in (x1 - margin, x2 + margin):
            xs.add(ox)
        for oy in (y1 - margin, y2 + margin):
            ys.add(oy)
    # 起终点之间均匀插入
    if abs(ex - sx) > 0.5:
        n = int(abs(ex - sx) / 0.5)
        for i in range(1, n):
            xs.add(sx + (ex - sx) * i / n)
    if abs(ey - sy) > 0.5:
        n = int(abs(ey - sy) / 0.5)
        for i in range(1, n):
            ys.add(sy + (ey - sy) * i / n)
    nodes = []
    for x in xs:
        if x < ROOM_X[0] or x > ROOM_X[1]:
            continue
        for y in ys:
            if y < ROOM_Y[0] or y > ROOM_Y[1]:
                continue
            if _point_safe(x, y, obstacles, no_go_zones, margin):
                nodes.append((x, y))
    if (sx, sy) not in nodes:
        nodes.append((sx, sy))
    if (ex, ey) not in nodes:
        nodes.append((ex, ey))
    return nodes


def _astar(start, end, nodes, obstacles, no_go_zones, margin, risk_weight):
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
            if not _segment_safe(cx, cy, nx, ny, obstacles, no_go_zones, margin):
                continue
            seg_len = math.hypot(nx - cx, ny - cy)
            risk = 0.0
            for obs in obstacles:
                d = _point_to_segment_dist(obs['x'], obs['y'], cx, cy, nx, ny)
                if d < margin * 3:
                    risk += 1.0 / max(d, 0.01)
            ng = g + seg_len + risk_weight * risk
            nf = ng + abs(nx - ex) + abs(ny - ey)
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


def plan_path(start, end, obstacles, no_go_zones,
              margin=0.5, expand=0.2, risk_weight=10.0):
    """核心规划：返回 (waypoints, nodes, safe, min_dist)"""
    if not obstacles and not no_go_zones:
        wp = _simplify_waypoints([start, (end[0], start[1]), end])
        return wp, [], True, float('inf')

    nodes = _generate_nodes(start[0], start[1], end[0], end[1],
                            obstacles, no_go_zones, margin, expand)
    raw, cost = _astar(start, end, nodes, obstacles,
                        no_go_zones, margin, risk_weight)

    if raw is None:
        wp = _simplify_waypoints([start, (end[0], start[1]), end])
        return wp, nodes, False, 0.0

    wp = _simplify_waypoints(raw)
    min_d = float('inf')
    for i in range(len(wp) - 1):
        for obs in obstacles:
            d = _point_to_segment_dist(obs['x'], obs['y'],
                                        wp[i][0], wp[i][1],
                                        wp[i+1][0], wp[i+1][1])
            min_d = min(min_d, d)
    return wp, nodes, min_d >= margin, min_d


# ============================================================
#  交互式仿真器
# ============================================================

class Simulator:
    def __init__(self):
        # 场景
        self.start = None
        self.end = None
        self.obstacles = []       # [{'x':, 'y':, 'r':}, ...]
        self.no_go_zones = []     # [{'x1':, 'y1':, 'x2':, 'y2':}, ...]

        # 规划结果
        self.waypoints = []
        self.nodes = []
        self.path_safe = True
        self.min_dist = float('inf')

        # 交互
        self.mode = 'start'       # start | end | obstacle | zone | delete
        self.obs_radius = 0.15
        self._zone_start = None   # 禁行区第一个角

        # 参数
        self.margin = 0.5
        self.expand = 0.2
        self.risk_weight = 10.0

        self._build()

    def _build(self):
        self.fig = plt.figure(figsize=(12, 8))
        self.fig.canvas.manager.set_window_title('路径规划仿真')

        self.ax = self.fig.add_axes([0.05, 0.18, 0.70, 0.78])

        self.slider_margin = Slider(
            self.fig.add_axes([0.80, 0.60, 0.16, 0.03]),
            '安全边距 (m)', 0.1, 2.0, valinit=0.5)
        self.slider_expand = Slider(
            self.fig.add_axes([0.80, 0.50, 0.16, 0.03]),
            '栅格扩展 (m)', 0.05, 1.0, valinit=0.2)
        self.slider_risk = Slider(
            self.fig.add_axes([0.80, 0.40, 0.16, 0.03]),
            '风险权重', 0.0, 50.0, valinit=10.0)

        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.slider_margin.on_changed(self._on_slider)
        self.slider_expand.on_changed(self._on_slider)
        self.slider_risk.on_changed(self._on_slider)

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

    # ── 规划 ──

    def _replan(self):
        if self.start is None or self.end is None:
            self.waypoints, self.nodes = [], []
            return
        planner_obs = [{'x': o['x'], 'y': o['y']} for o in self.obstacles]
        self.waypoints, self.nodes, self.path_safe, self.min_dist = \
            plan_path(self.start, self.end, planner_obs, self.no_go_zones,
                      self.margin, self.expand, self.risk_weight)

    # ── 绘制 ──

    def _draw(self):
        ax = self.ax
        ax.clear()
        ax.set_xlim(-0.3, 5.0)
        ax.set_ylim(-0.3, 9.5)
        ax.set_aspect('equal')
        ax.set_facecolor('#fafaf5')

        # 网格
        for v in np.arange(0, 5.0, 0.5):
            ax.axvline(v, color='#e0e0e0', linewidth=0.5, linestyle=':')
        for v in np.arange(0, 9.5, 0.5):
            ax.axhline(v, color='#e0e0e0', linewidth=0.5, linestyle=':')

        # 房间
        ax.add_patch(Rectangle((0, 0), 4.6, 9.0, fill=False,
                                edgecolor='#333', linewidth=2))
        ax.text(2.3, -0.15, '前墙 (X=0)', ha='center', fontsize=8, color='#888')
        ax.text(2.3, 9.15, '后墙 (X=4.6)', ha='center', fontsize=8, color='#888')
        ax.text(4.75, 4.5, '左墙 Y=9 ↑', ha='center', fontsize=8, color='#888')
        ax.text(-0.15, 4.5, '右墙 Y=0 ↓', ha='center', fontsize=8, color='#888')

        # 禁行区
        for i, z in enumerate(self.no_go_zones):
            x1, x2 = min(z['x1'], z['x2']), max(z['x1'], z['x2'])
            y1, y2 = min(z['y1'], z['y2']), max(z['y1'], z['y2'])
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                    fill=True, facecolor='#ff9800', alpha=0.25,
                                    edgecolor='#e65100', linewidth=2,
                                    linestyle='--', zorder=2))
            ax.text((x1 + x2) / 2, (y1 + y2) / 2,
                    f'禁行#{i+1}', ha='center', va='center',
                    fontsize=8, color='#e65100', fontweight='bold')

        # 正在绘制的禁行区（半透明预览）
        if self._zone_start and self.mode == 'zone':
            zx, zy = self._zone_start
            if hasattr(self, '_hover_pos') and self._hover_pos:
                hx, hy = self._hover_pos
                x1, x2 = min(zx, hx), max(zx, hx)
                y1, y2 = min(zy, hy), max(zy, hy)
                ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                        fill=True, facecolor='#ff9800',
                                        alpha=0.15, edgecolor='#e65100',
                                        linewidth=1.5, linestyle=':'))

        # 障碍物
        for i, obs in enumerate(self.obstacles):
            ax.add_patch(Circle((obs['x'], obs['y']), self.margin,
                                fill=True, facecolor='#ffcdd2', alpha=0.25,
                                edgecolor='none', zorder=2))
            ax.add_patch(Circle((obs['x'], obs['y']), obs['r'],
                                fill=True, facecolor='#e53935', alpha=0.7,
                                edgecolor='#b71c1c', linewidth=1.5, zorder=3))
            ax.text(obs['x'], obs['y'] + obs['r'] + 0.12, f'#{i+1}',
                    ha='center', fontsize=7, color='#c62828', fontweight='bold')

        # 候选路点
        if self.nodes:
            nx, ny = zip(*self.nodes)
            ax.scatter(nx, ny, c='#bdbdbd', s=10, alpha=0.5, zorder=4)

        # 路径
        if len(self.waypoints) > 1:
            wx, wy = zip(*self.waypoints)
            color = '#2e7d32' if self.path_safe else '#c62828'
            ax.plot(wx, wy, color=color, linewidth=2.5, marker='o',
                    markersize=6, markerfacecolor='white',
                    markeredgecolor=color, markeredgewidth=2, zorder=5)
            for i in range(len(self.waypoints) - 1):
                x1, y1 = self.waypoints[i]
                x2, y2 = self.waypoints[i + 1]
                seg = math.hypot(x2 - x1, y2 - y1)
                if seg > 0.05:
                    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                    dx, dy = (x2 - x1) / seg * 0.1, (y2 - y1) / seg * 0.1
                    ax.arrow(mx - dx, my - dy, dx * 2, dy * 2,
                             head_width=0.06, head_length=0.08,
                             fc=color, ec=color, alpha=0.5, zorder=6)

        # 起点 / 终点
        if self.start:
            ax.scatter(*self.start, c='#2e7d32', s=180, marker='o',
                       edgecolors='white', linewidth=2, zorder=10)
            ax.text(self.start[0] - 0.12, self.start[1] - 0.15, '起点',
                    fontsize=9, color='#2e7d32', fontweight='bold', ha='right')
        if self.end:
            ax.scatter(*self.end, c='#c62828', s=180, marker='X',
                       edgecolors='white', linewidth=2, zorder=10)
            ax.text(self.end[0] + 0.12, self.end[1] - 0.15, '终点',
                    fontsize=9, color='#c62828', fontweight='bold', ha='left')

        # 标题
        mode_names = {'start': '起点', 'end': '终点',
                      'obstacle': '障碍物', 'zone': '禁行区', 'delete': '删除'}
        info = f"模式: [{self.mode[0].upper()}]{mode_names[self.mode]}"
        if self.mode == 'zone' and self._zone_start:
            info += f" (已选第一角 {self._zone_start[0]:.2f},{self._zone_start[1]:.2f})"
        info += f" | 障碍物: {len(self.obstacles)} 个"
        info += f" | 禁行区: {len(self.no_go_zones)} 个"
        if self.waypoints:
            safe_str = '✓ 安全' if self.path_safe else '✗ 危险!'
            info += f" | 路径: {len(self.waypoints)} 点 {safe_str}"
            if self.min_dist != float('inf'):
                info += f" 距障碍物 {self.min_dist:.2f}m"
        ax.set_title(info, fontsize=9, fontweight='bold', pad=6,
                     color='#333' if self.path_safe else '#c62828')

        help_text = "[s]起点 [e]终点 [o]障碍物 [z]禁行区 [d]删除 [r]重置 滚轮调半径"
        ax.text(0.02, 0.01, help_text, transform=ax.transAxes,
                fontsize=7, color='#aaa', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        self.fig.canvas.draw_idle()

    # ── 事件 ──

    def _on_click(self, event):
        if event.inaxes != self.ax:
            return
        x, y = event.xdata, event.ydata
        in_room = -0.05 <= x <= 4.65 and -0.05 <= y <= 9.05

        if event.button == 1:  # 左键
            if self.mode == 'start' and in_room:
                self.start = (round(x, 3), round(y, 3))
                self._replan()
            elif self.mode == 'end' and in_room:
                self.end = (round(x, 3), round(y, 3))
                self._replan()
            elif self.mode == 'obstacle' and in_room:
                ok = all(math.hypot(x - o['x'], y - o['y']) > self.obs_radius * 2
                         for o in self.obstacles)
                if ok:
                    self.obstacles.append({'x': round(x, 3), 'y': round(y, 3),
                                           'r': self.obs_radius})
                    self._replan()
            elif self.mode == 'zone' and in_room:
                if self._zone_start is None:
                    self._zone_start = (round(x, 3), round(y, 3))
                else:
                    x1, y1 = self._zone_start
                    x2, y2 = round(x, 3), round(y, 3)
                    if abs(x2 - x1) > 0.05 and abs(y2 - y1) > 0.05:
                        self.no_go_zones.append(
                            {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
                        self._zone_start = None
                        self._replan()
                    else:
                        self._zone_start = None  # 太小，取消
            elif self.mode == 'delete':
                self._delete_near(x, y)
        elif event.button == 3:  # 右键
            if self.mode == 'zone' and self._zone_start:
                self._zone_start = None  # 取消绘制
            else:
                self._delete_near(x, y)

        self._draw()

    def _on_motion(self, event):
        if event.inaxes == self.ax:
            self._hover_pos = (event.xdata, event.ydata)
            if self.mode == 'zone' and self._zone_start:
                self._draw()

    def _on_key(self, event):
        key = event.key.lower() if event.key else ''
        if key == 's':
            self.mode = 'start'
            self._zone_start = None
        elif key == 'e':
            self.mode = 'end'
            self._zone_start = None
        elif key == 'o':
            self.mode = 'obstacle'
            self._zone_start = None
        elif key == 'z':
            self.mode = 'zone'
            self._zone_start = None
        elif key == 'd':
            self.mode = 'delete'
            self._zone_start = None
        elif key == 'r':
            self.start = self.end = None
            self.obstacles = []
            self.no_go_zones = []
            self._zone_start = None
            self._replan()
        elif key == 'up':
            self.obs_radius = min(0.5, self.obs_radius + 0.05)
        elif key == 'down':
            self.obs_radius = max(0.05, self.obs_radius - 0.05)
        elif key == 'escape':
            self._zone_start = None
        self._draw()

    def _on_scroll(self, event):
        d = 0.03 if event.button == 'up' else -0.03
        self.obs_radius = max(0.05, min(0.5, self.obs_radius + d))
        self._draw()

    def _on_slider(self, val):
        self.margin = self.slider_margin.val
        self.expand = self.slider_expand.val
        self.risk_weight = self.slider_risk.val
        self._replan()
        self._draw()

    def _delete_near(self, x, y):
        th = 0.3
        if self.start and math.hypot(x - self.start[0], y - self.start[1]) < th:
            self.start = None
            self._replan()
            return
        if self.end and math.hypot(x - self.end[0], y - self.end[1]) < th:
            self.end = None
            self._replan()
            return
        for i in range(len(self.obstacles) - 1, -1, -1):
            if math.hypot(x - self.obstacles[i]['x'],
                          y - self.obstacles[i]['y']) < th:
                self.obstacles.pop(i)
                self._replan()
                return
        # 删除禁行区：点击在矩形内部即可
        for i in range(len(self.no_go_zones) - 1, -1, -1):
            z = self.no_go_zones[i]
            if _rect_has_point(z['x1'], z['y1'], z['x2'], z['y2'], x, y):
                self.no_go_zones.pop(i)
                self._replan()
                return


if __name__ == "__main__":
    sim = Simulator()
    plt.show()
