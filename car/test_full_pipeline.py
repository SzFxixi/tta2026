#!/usr/bin/env python3
"""
全流程测试脚本 — 检测 → 规划 → 执行 一体化验证。

用法（在小车上运行，需 ROS + LiDAR + Flask 服务端）:
    python3 test_full_pipeline.py <target_x> <target_y>

流程:
    阶段一：检测 — 输出 baseline + 障碍物坐标
    阶段二：规划 — 输出路径点 + 校正点表格
    阶段三：执行 — 逐段移动，检测偏角，打印执行状态
"""

import sys
import time
import math
import rospy
import requests
import numpy as np
from sensor_msgs.msg import LaserScan


# ============================================================
#  配置
# ============================================================

CAR_IP = "10.26.36.227"
CAR_PORT = 5000
BASE_URL = f"http://{CAR_IP}:{CAR_PORT}"
MOVE_TIMEOUT = 30


# ============================================================
#  服务器通信
# ============================================================

class _Client:
    def __init__(self):
        self.sess = requests.Session()
        self.tid = 1

    def _post(self, endpoint, payload=None, timeout=MOVE_TIMEOUT):
        if payload is None:
            payload = {}
        body = {**payload, "TaskId": self.tid}
        try:
            r = self.sess.post(f"{BASE_URL}/{endpoint}", json=body, timeout=timeout)
            resp = r.json()
            if resp.get("isSuccess"):
                self.tid += 1
                return True, resp
            if "expectedTaskId" in resp:
                self.tid = resp["expectedTaskId"]
                return self._post(endpoint, payload, timeout)
            return False, resp
        except Exception as e:
            return False, {"error": str(e)}

    def reset(self):
        ok, _ = self._post("Reset", timeout=5)
        self.tid = 1
        return ok

    def set_baseline(self):
        return self._post("SetBaseline")

    def sync_yaw(self):
        return self._post("SyncYaw", timeout=20)

    def move_relative(self, dx, dy):
        """相对位移（底盘坐标系：+x=前进, +y=右移）"""
        return self._post("MoveRelative", {"delta_x": dx, "delta_y": dy})

    def move_x(self, target_x, ref_y):
        """仅 X 轴移动（绝对坐标，服务端 LiDAR 反馈）"""
        return self._post("MoveOnlyX", {"location_x": target_x, "location_y": ref_y})

    def move_y(self, target_y, ref_x):
        """仅 Y 轴移动（绝对坐标，服务端 LiDAR 反馈）"""
        return self._post("MoveOnlyY", {"location_x": ref_x, "location_y": target_y})


# ============================================================
#  小车定位
# ============================================================

def _getx():
    data = rospy.wait_for_message("scan", LaserScan)
    d = data.ranges[len(data.ranges) // 2]
    while d == np.inf:
        data = rospy.wait_for_message("scan", LaserScan)
        d = data.ranges[len(data.ranges) // 2]
    return d


def _gety():
    data = rospy.wait_for_message("scan", LaserScan)
    d = data.ranges[len(data.ranges) // 4]
    while d == np.inf:
        data = rospy.wait_for_message("scan", LaserScan)
        d = data.ranges[len(data.ranges) // 4]
    return d


def car_position():
    return _getx(), _gety()


# ============================================================
#  主流程
# ============================================================

def main():
    rospy.init_node("full_pipeline_test", anonymous=True)
    rospy.wait_for_message("scan", LaserScan, timeout=5.0)

    cli = _Client()
    cli.reset()

    # 目标点
    if len(sys.argv) >= 3:
        tx, ty = float(sys.argv[1]), float(sys.argv[2])
    else:
        print("用法: python3 test_full_pipeline.py <target_x> <target_y>")
        sys.exit(1)

    # ────────────────────────────────────────────────
    #  阶段一：检测
    # ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  阶段一：环境检测")
    print("=" * 60)

    # baseline
    from obstacle_detector import detect_obstacles
    from lidar_utils import getsum

    # 获取小车位置
    cx, cy = car_position()
    print(f"\n[1] 小车位置: X={cx:.3f}, Y={cy:.3f}")

    print("\n[2] 设定校正基准 (SetBaseline)...")
    ok, _ = cli.set_baseline()
    baseline_sum = getsum()
    print(f"  {'✓' if ok else '✗'} 基准距离和: {baseline_sum:.3f} m")

    input("\n>>> 基准已设定，按 Enter 继续检测障碍物...")

    # 障碍物检测
    print(f"\n[3] 检测障碍物...")
    obstacles, diag = detect_obstacles(cx, cy)
    filtered = diag.get('filtered_out', 0)
    print(f"  雷达诊断: 有效 {diag['valid_beams']}/{diag['total_beams']} 光束, "
          f"突变 {diag['jump_count']} 点, "
          f"距离范围 {diag['dist_min']}~{diag['dist_max']}m"
          + (f", 过滤越界 {filtered} 个" if filtered else ""))

    if obstacles:
        print(f"\n  检测到 {len(obstacles)} 个障碍物:")
        for obs in obstacles:
            print(f"    · ({obs['x']:.3f}, {obs['y']:.3f})  "
                  f"距离={obs['distance']:.3f}m  角度={obs['angle_deg']:.1f}°  "
                  f"光束数={obs['beam_count']}")
    else:
        print(f"\n  未检测到障碍物 ✓")

    # ────────────────────────────────────────────────
    #  阶段二：路径规划
    # ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  阶段二：路径规划")
    print("=" * 60)

    from path_planner import plan_path

    print(f"\n[4] 规划路径: ({cx:.3f}, {cy:.3f}) → ({tx:.2f}, {ty:.2f})")
    result = plan_path(cx, cy, tx, ty, cx, cy)

    wp = result["waypoints"]
    cp = result.get("correction_points", [])
    cp_indices = {p["index"] for p in cp}

    print(f"  路径方案: {result['path_name']}")
    print(f"  得分: {result.get('score', '-')}  距障碍物: {result['obstacle_margin']}m  "
          f"{'✓ 安全' if result['safe'] else '⚠ 危险'}")

    # 表格
    print(f"\n  路径点与校正点:")
    print(f"  {'─' * 55}")
    print(f"  {'序号':<6} {'X (m)':<12} {'Y (m)':<12} {'类型':<12} {'备注'}")
    print(f"  {'─' * 55}")
    for i, (x, y) in enumerate(wp):
        if i == 0:
            kind = "起点"
        elif i == len(wp) - 1:
            kind = "终点"
        else:
            kind = "中间点"

        note = ""
        if i in cp_indices:
            cpi = next(p for p in cp if p["index"] == i)
            note = "★ 可校正" if cpi["safe"] else "★ 校正(有遮挡)"
        print(f"  {i:<6} {x:<12.3f} {y:<12.3f} {kind:<12} {note}")
    print(f"  {'─' * 55}")

    if cp:
        print(f"\n  校正点汇总 ({len(cp)} 个):")
        for p in cp:
            safe_str = "安全" if p["safe"] else "有遮挡"
            print(f"    #{p['index']} ({p['x']:.3f},{p['y']:.3f}) [{p['type']}] — {safe_str}")

    # ────────────────────────────────────────────────
    input("\n>>> 路径规划完成，按 Enter 开始执行移动...")

    # ────────────────────────────────────────────────
    #  阶段三：执行（MoveOnlyX/Y 绝对移动 + 强制终点校正）
    # ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  阶段三：执行移动")
    print("=" * 60)

    POS_THRESHOLD = 0.08
    MAX_RETRIES = 5

    for i in range(len(wp) - 1):
        x1, y1 = wp[i]
        x2, y2 = wp[i + 1]
        dx, dy = abs(x2 - x1), abs(y2 - y1)

        # ── 中间校正点 ──
        if i in cp_indices:
            cpi = next(p for p in cp if p["index"] == i)
            if cpi["safe"]:
                print(f"\n--- ★ 校正点 #{i} ({x1:.3f}, {y1:.3f}) [{cpi['type']}] ---")
                # 角度校正
                print(f"  角度校正 (SyncYaw)...")
                sum_before = getsum()
                ok, _ = cli.sync_yaw()
                sum_after = getsum()
                print(f"  LiDAR距离和: {sum_before:.3f}m → {sum_after:.3f}m  {'✓' if ok else '✗'}")
                # 坐标校正：LiDAR 偏差 → 底盘坐标系相对位移（底盘+x=LiDAR X↓，所以取反）
                for retry in range(1, MAX_RETRIES + 1):
                    cx, cy = car_position()
                    ex, ey = x1 - cx, y1 - cy           # LiDAR 偏差
                    cdx, cdy = cx - x1, cy - y1          # 底盘相对位移（取反）
                    err = math.hypot(ex, ey)
                    print(f"  坐标校正 #{retry}: 期望({x1:.3f},{y1:.3f}) 实际({cx:.3f},{cy:.3f}) 误差{err:.3f}m")
                    if err <= POS_THRESHOLD:
                        print(f"  ✓ 坐标已收敛")
                        break
                    print(f"    补底盘位移: ({cdx:+.3f}, {cdy:+.3f})")
                    cli.move_relative(cdx, cdy)
                    time.sleep(0.3)
                else:
                    print(f"  ⚠ 坐标校正未收敛，继续执行")
            else:
                print(f"\n--- 校正点 #{i} 有遮挡，跳过 ---")

        # ── 移动段（只用 MoveOnlyX 或 MoveOnlyY，服务端 LiDAR 闭环） ──
        if dx > 0.001 and dy < 0.001:
            direction = f"MoveOnlyX → X={x2:.3f} (ref Y={y1:.3f})"
            print(f"  移动: ({x1:.3f},{y1:.3f}) → ({x2:.3f},{y2:.3f})  [{direction}]")
            ok, resp = cli.move_x(x2, y1)
        elif dy > 0.001 and dx < 0.001:
            direction = f"MoveOnlyY → Y={y2:.3f} (ref X={x1:.3f})"
            print(f"  移动: ({x1:.3f},{y1:.3f}) → ({x2:.3f},{y2:.3f})  [{direction}]")
            ok, resp = cli.move_y(y2, x1)
        else:
            ok = True

        if ok:
            print(f"  ✓ 移动完成")
        else:
            print(f"  ✗ 移动失败: {resp.get('errorMessage', resp.get('error', '?'))}")
            break

        time.sleep(0.3)

    # ── 终点：强制坐标校正 ──
    final_x, final_y = car_position()
    print(f"\n--- 终点强制坐标校正 ---")
    print(f"  目标: ({tx:.2f}, {ty:.2f})")
    print(f"  到达: ({final_x:.3f}, {final_y:.3f})")
    print(f"  LiDAR距离和: {getsum():.3f}m")

    for retry in range(1, MAX_RETRIES + 1):
        cx, cy = car_position()
        ex, ey = tx - cx, ty - cy            # LiDAR 偏差
        cdx, cdy = cx - tx, cy - ty           # 底盘相对位移（取反）
        err = math.hypot(ex, ey)
        print(f"  校正 #{retry}: 期望({tx:.3f},{ty:.3f}) 实际({cx:.3f},{cy:.3f}) "
              f"误差{err:.3f}m (阈值{POS_THRESHOLD:.3f}m)")
        if err <= POS_THRESHOLD:
            print(f"  ✓ 终点坐标已收敛!")
            break
        print(f"  → 补底盘位移: ({cdx:+.3f}, {cdy:+.3f})")
        ok, _ = cli.move_relative(cdx, cdy)
        if not ok:
            print(f"  ✗ 纠正失败")
            break
        time.sleep(0.3)
        final_x, final_y = car_position()
    else:
        print(f"  ⚠ 达到最大重试 {MAX_RETRIES} 次")
        final_x, final_y = car_position()

    # ── 完成 ──
    print(f"\n{'=' * 60}")
    print(f"  全流程测试完成")
    print(f"  起点: ({cx:.3f}, {cy:.3f})")
    print(f"  目标: ({tx:.2f}, {ty:.2f})")
    print(f"  终点: ({final_x:.3f}, {final_y:.3f})")
    print(f"  终点误差: {math.hypot(tx-final_x, ty-final_y):.3f}m")
    print(f"{'=' * 60}")

    cli.sess.close()


if __name__ == "__main__":
    main()
