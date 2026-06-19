#!/usr/bin/env python3
"""
安全监控模块 — 障碍物距离日志、急停检测、探路逃离。

职责：
  1. 实时打印车到各障碍物的距离
  2. 离障碍物太近时触发急停 → 重新规划
  3. 规划前确保起点不在障碍物膨胀区内（否则先探路）

调用方（run.py）只需：
  from entities.safety_monitor import SafetyMonitor
  monitor = SafetyMonitor(cli, car_position_fn, obstacles, zones)
  cx, cy = monitor.ensure_safe(cx, cy)       # 规划前
  stop, reason = monitor.check(cx, cy)         # 监控中每帧
  summary = monitor.dist_summary(cx, cy)       # 日志
"""

import math
from utils.config_loader import cfg


class SafetyMonitor:
    def __init__(self, cli, car_position_fn, obstacles, zones=None):
        self.cli = cli
        self.car_position_fn = car_position_fn
        self.obstacles = obstacles or []
        self.zones = zones or []
        # 急停阈值：离障碍物 < emergency_stop_dist 就停
        self.stop_dist = getattr(cfg.path_planning, 'emergency_stop_dist',
                                 cfg.path_planning.expansion_radius_3)
        # 探路参数：min_safe_dist 也同步使用急停阈值
        self.max_probe_rounds = cfg.client.max_probe_rounds
        self.min_safe_dist = self.stop_dist

    # ── 距离工具 ──────────────────────────────────────────

    def min_obstacle_dist(self, cx, cy):
        """返回到最近障碍物的距离，无障碍物时返回一个大数。"""
        if not self.obstacles:
            return float('inf')
        return min(math.hypot(cx - o['x'], cy - o['y']) for o in self.obstacles)

    def dist_summary(self, cx, cy, top_n=3):
        """最近 top_n 个障碍物的距离字符串，用于日志输出。"""
        if not self.obstacles:
            return ""
        dists = [(i, math.hypot(cx - o['x'], cy - o['y'])) for i, o in enumerate(self.obstacles)]
        dists.sort(key=lambda x: x[1])
        parts = [f"#{i+1}:{d:.2f}m" for i, d in dists[:top_n]]
        return "  obs[" + " ".join(parts) + "]"

    # ── 急停检测 ──────────────────────────────────────────

    def check(self, cx, cy):
        """每帧调用。返回 (should_stop, reason)。
        should_stop=True 表示需要急停 + 重规划。"""
        if not self.obstacles:
            return False, None
        min_d = self.min_obstacle_dist(cx, cy)
        if min_d < self.stop_dist:
            return True, f"离障碍物仅 {min_d:.3f}m < {self.stop_dist}m，急停！"
        return False, None

    # ── 安全起点 ──────────────────────────────────────────

    def ensure_safe(self, cx, cy):
        """规划前调用。若当前位置离障碍物 < min_safe_dist，
        执行探路逃离，返回安全位置。否则原样返回。"""
        if not self.obstacles:
            return cx, cy

        from utils.emergency_escape import escape_danger_zone
        return escape_danger_zone(
            cx, cy, self.obstacles, self.zones, self.cli,
            max_rounds=self.max_probe_rounds,
            min_safe_dist=self.min_safe_dist,
            car_position_fn=self.car_position_fn,
        )
