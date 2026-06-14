#!/usr/bin/env python3
"""路径规划可视化工具 — 在房间地图上绘制路径、禁区、障碍物、校正点，输出 PNG。"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os, json

from utils.config_loader import cfg


def _load_zones():
    """加载禁区，失败返回 []"""
    try:
        import importlib
        from entities import forbidden_zones
        importlib.reload(forbidden_zones)
        zones = forbidden_zones.load_forbidden_zones()
        return zones if zones else []
    except Exception:
        return []


def visualize_plan(waypoints, correction_points=None, obstacles=None,
                   zones=None, start_label="S", end_label="E",
                   title="路径规划", save_path=None):
    """
    绘制路径规划结果。

    参数:
        waypoints:       [(x,y), ...]  路径点列表
        correction_points: [{"index":i, "x":x, "y":y, "type":t, "safe":bool}, ...]
        obstacles:       [{"x":x, "y":y, "distance":d}, ...]
        zones:           [[xmin,xmax,ymin,ymax], ...]  禁区列表，None=自动加载
        title:           图标题
        save_path:       输出 PNG 路径，None=自动命名到 ~/path_plan.png
    """
    if zones is None:
        zones = _load_zones()

    # 房间参数
    rx_min = cfg.room.x_min
    rx_max = cfg.room.x_max
    ry_min = cfg.room.y_min
    ry_max = cfg.room.y_max

    wall_margin = cfg.path_planning.wall_margin
    safe_x_min = rx_min + wall_margin
    safe_x_max = rx_max - wall_margin
    safe_y_min = ry_min + wall_margin
    safe_y_max = ry_max - wall_margin

    # -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 14))

    # 房间背景
    ax.add_patch(patches.Rectangle(
        (rx_min, ry_min), rx_max - rx_min, ry_max - ry_min,
        facecolor="#eeeeee", edgecolor="black", linewidth=2, zorder=1))

    # 墙边距区域（不可规划）
    # 四边各画一条
    margins = [
        (rx_min, safe_y_max, rx_max - rx_min, ry_max - safe_y_max),  # top
        (rx_min, ry_min, rx_max - rx_min, wall_margin),               # bottom
        (rx_min, ry_min, wall_margin, ry_max - ry_min),               # left
        (safe_x_max, ry_min, rx_max - safe_x_max, ry_max - ry_min),   # right
    ]
    for mx, my, mw, mh in margins:
        ax.add_patch(patches.Rectangle(
            (mx, my), mw, mh,
            facecolor="#ffcc80", alpha=0.35, hatch="///",
            edgecolor="none", zorder=2))

    # 安全规划区
    ax.add_patch(patches.Rectangle(
        (safe_x_min, safe_y_min), safe_x_max - safe_x_min, safe_y_max - safe_y_min,
        facecolor="#c8e6c9", alpha=0.35,
        edgecolor="#2e7d32", linewidth=1.5, linestyle="--", zorder=3))

    # 禁区
    for i, (xmin, xmax, ymin, ymax) in enumerate(zones):
        w = max(xmax - xmin, 0.03)
        h = max(ymax - ymin, 0.03)
        ax.add_patch(patches.Rectangle(
            (xmin, ymin), w, h,
            facecolor="#e53935", alpha=0.55,
            edgecolor="#b71c1c", linewidth=1.2, zorder=4))
        ax.text((xmin + xmax) / 2, (ymin + ymax) / 2, str(i + 1),
                ha="center", va="center", fontsize=7,
                fontweight="bold", color="white", zorder=5)

    # 障碍物
    if obstacles:
        oxs = [o["x"] for o in obstacles]
        oys = [o["y"] for o in obstacles]
        ax.scatter(oxs, oys, c="darkorange", s=60, marker="X",
                   edgecolors="black", linewidth=0.5, zorder=7, label="障碍物")

    # 校正点
    if correction_points:
        for cp in correction_points:
            color = "#2e7d32" if cp.get("safe", True) else "#ff6f00"
            marker = "*" if cp["type"] == "start" else ("o" if cp["type"] == "end" else "D")
            size = 140 if cp["type"] in ("start", "end") else 80
            ax.scatter(cp["x"], cp["y"], c=color, s=size, marker=marker,
                       edgecolors="black", linewidth=0.5, zorder=8)
            ax.annotate(cp["type"][0].upper(), (cp["x"], cp["y"]),
                        textcoords="offset points", xytext=(0, 8),
                        fontsize=6, fontweight="bold", ha="center", color=color)

    # 路径
    if waypoints and len(waypoints) >= 2:
        xs = [p[0] for p in waypoints]
        ys = [p[1] for p in waypoints]
        ax.plot(xs, ys, "b-", linewidth=2, alpha=0.7, zorder=6)
        ax.scatter(xs, ys, c="blue", s=25, zorder=7)
        # 起点/终点
        ax.scatter(xs[0], ys[0], c="#1b5e20", s=120, marker="s",
                   edgecolors="black", linewidth=1, zorder=8, label="起点")
        ax.scatter(xs[-1], ys[-1], c="#b71c1c", s=120, marker="s",
                   edgecolors="black", linewidth=1, zorder=8, label="终点")
        # 路点编号
        for i, (x, y) in enumerate(waypoints):
            offset = 10 if i % 2 == 0 else -15
            ax.annotate(str(i), (x, y), textcoords="offset points",
                        xytext=(0, offset), fontsize=6, ha="center",
                        color="#1565c0", fontweight="bold")

    # -------------------------------------------------------
    ax.set_xlim(rx_min - 0.3, rx_max + 0.5)
    ax.set_ylim(ry_min - 0.3, ry_max + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("X (LiDAR 前方)", fontsize=11)
    ax.set_ylabel("Y (LiDAR 右方)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.25, zorder=0)

    legend_elements = [
        patches.Patch(facecolor="#c8e6c9", alpha=0.35, edgecolor="#2e7d32",
                      linestyle="--", label="安全规划区"),
        patches.Patch(facecolor="#ffcc80", alpha=0.4, label="墙边距区域"),
        patches.Patch(facecolor="#e53935", alpha=0.55, edgecolor="#b71c1c",
                      label="禁区"),
    ]
    if obstacles:
        legend_elements.append(
            plt.Line2D([0], [0], marker="X", color="w", markerfacecolor="darkorange",
                       markersize=8, markeredgecolor="black", markeredgewidth=0.5,
                       label="障碍物"))
    legend_elements.append(
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#1b5e20",
                   markersize=8, markeredgecolor="black", label="起点"))
    legend_elements.append(
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#b71c1c",
                   markersize=8, markeredgecolor="black", label="终点"))
    if correction_points:
        legend_elements.append(
            plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#2e7d32",
                       markersize=7, markeredgecolor="black", label="校正点"))
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.expanduser("~/path_plan.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"路径图已保存: {save_path}")
    return save_path
