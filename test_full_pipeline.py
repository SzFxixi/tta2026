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

    def move(self, tx, ty):
        """使用 /Move 端点（X+Y while 循环，更稳健）"""
        return self._post("Move", {"location_x": tx, "location_y": ty})


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
    from pose_correction import getsum

    # 获取小车位置
    cx, cy = car_position()
    print(f"\n[1] 小车位置: X={cx:.3f}, Y={cy:.3f}")

    print("\n[2] 设定校正基准 (SetBaseline)...")
    ok, _ = cli.set_baseline()
    baseline_sum = getsum()
    print(f"  {'✓' if ok else '✗'} 基准距离和: {baseline_sum:.3f} m")

    input("\n>>> 基准已设定，按 Enter 继续检测障碍物...")

    # 障碍物检测
    safe_p1 = (0.0, 0.0)
    safe_p2 = (0.0, 0.0)   # 空安全区 = 取消

    print(f"\n[3] 检测障碍物...")
    print(f"  安全区: ({safe_p1[0]:.1f}, {safe_p1[1]:.1f}) → ({safe_p2[0]:.1f}, {safe_p2[1]:.1f})")

    obstacles, diag = detect_obstacles(safe_p1, safe_p2, cx, cy)
    print(f"  雷达诊断: 有效 {diag['valid_beams']}/{diag['total_beams']} 光束, "
          f"突变 {diag['jump_count']} 点, "
          f"距离范围 {diag['dist_min']}~{diag['dist_max']}m")

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

    #  阶段三：执行
    # ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  阶段三：执行移动")
    print("=" * 60)

    # 记录规划时的 LiDAR 基准值（纯诊断用，不连底盘）
    baseline_sum = getsum()

    for i in range(len(wp) - 1):
        x1, y1 = wp[i]
        x2, y2 = wp[i + 1]
        dx, dy = abs(x2 - x1), abs(y2 - y1)

        # ── 校正点处理 ──
        if i in cp_indices:
            cpi = next(p for p in cp if p["index"] == i)
            if cpi["safe"]:
                print(f"\n--- ★ 校正点 #{i} ({x1:.3f}, {y1:.3f}) [{cpi['type']}] ---")
                print(f"  现在执行位姿校正 (SyncYaw)...")
                sum_before = getsum()
                print(f"  校正前 LiDAR距离和: {sum_before:.3f}m")

                ok, _ = cli.sync_yaw()
                if ok:
                    sum_after = getsum()
                    print(f"  校正后 LiDAR距离和: {sum_after:.3f}m")
                    print(f"  ✓ 校正执行成功")
                else:
                    print(f"  ✗ 校正执行失败")
            else:
                print(f"\n--- 校正点 #{i} 有遮挡，跳过校正 ---")
        else:
            # 起点也检测一下
            label = "起点" if i == 0 else ""
            if label:
                print(f"\n--- {label} ({x1:.3f}, {y1:.3f}) ---")
                print(f"  LiDAR距离和: {getsum():.3f}m")

        # ── 移动段 ──
        if dx > 0.001 or dy > 0.001:
            direction = "MoveOnlyX" if dx > dy else "MoveOnlyY"
            print(f"  现在执行: Move → ({x2:.3f}, {y2:.3f}) [{direction}方向]")
            ok, _ = cli.move(x2, y2)
        else:
            print(f"  跳过（同一点）")
            ok = True

        if ok:
            print(f"  ✓ Move 执行成功: "
                  f"({x1:.3f}, {y1:.3f}) → ({x2:.3f}, {y2:.3f})")
        else:
            print(f"  ✗ Move 执行失败!")
            break

        time.sleep(0.3)

    # ── 终点偏角检测 ──
    final_x, final_y = car_position()
    print(f"\n--- 终点 ({final_x:.3f}, {final_y:.3f}) ---")
    # ── 终点诊断 ──
    final_sum = getsum()
    print(f"  终点 LiDAR距离和: {final_sum:.3f}m (基准: {baseline_sum:.3f}m)")
    if abs(final_sum - baseline_sum) > 0.3:
        print(f"  ⚠ 距离和偏差较大，执行终点校正...")
        _ = cli.sync_yaw()

    # ── 完成 ──
    print(f"\n{'=' * 60}")
    print(f"  全流程测试完成")
    print(f"  起点: ({cx:.3f}, {cy:.3f})")
    print(f"  终点: ({final_x:.3f}, {final_y:.3f})")
    print(f"  目标: ({tx:.2f}, {ty:.2f})")
    print(f"{'=' * 60}")

    cli.sess.close()


if __name__ == "__main__":
    main()
