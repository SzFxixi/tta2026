#!/usr/bin/env python3
"""
小车复合动作序列测试脚本（清晰封装版）
动作：移动、停留、旋转、仅X对齐、姿态纠正(X/Y)、设置基准(X/Y)
"""

import requests
import time
import json
import sys

# ================== 配置 ==================
CAR_IP   = "10.26.36.227"
CAR_PORT = 5000
BASE_URL = f"http://{CAR_IP}:{CAR_PORT}"

_sess   = requests.Session()
_tid    = 1

def _post(endpoint: str, payload: dict, timeout: int = 90):
    global _tid
    body = {**payload, "TaskId": _tid}
    print(f"\n  [{_tid:>3}] → POST /{endpoint}  {body}")
    try:
        r = _sess.post(f"{BASE_URL}/{endpoint}", json=body, timeout=timeout)
        res = r.json()
        print(f"        ← {json.dumps(res)}")
        if res.get("isSuccess"):
            _tid += 1
            return True, res
        if "expectedTaskId" in res:
            print(f"        [TaskId同步] {_tid} → {res['expectedTaskId']}")
            _tid = res["expectedTaskId"]
            return _post(endpoint, payload, timeout)
        return False, res
    except requests.Timeout:
        print(f"        [超时]")
        return False, None
    except Exception as e:
        print(f"        [错误] {e}")
        return False, None

def reset_task_id():
    global _tid
    _post("Reset", {}, timeout=5)
    _tid = 1
    print("[Reset] TaskId 已重置为 1")

# ================== 基础动作函数 ==================
def move_absolute(x: float, y: float, timeout: int = 20) -> bool:
    print(f"[Move] 目标坐标: ({x:.3f}, {y:.3f})")
    start = time.time()
    ok, resp = _post("Move", {"location_x": x, "location_y": y}, timeout=timeout)
    elapsed = time.time() - start
    print(f"[Move] 耗时: {elapsed:.2f}s")
    if resp and not ok:
        print(f"[Move] 失败: {resp.get('errorMessage', '未知')}")
    return ok

def rotate(degrees: float, timeout: int = 10) -> bool:
    print(f"[Rotate] {degrees}°")
    ok, _ = _post("Circle", {"rad_z": degrees}, timeout=timeout)
    if not ok:
        print("[Rotate] 失败")
    return ok

def move_only_x(x: float, y_ref: float, timeout: int = 20) -> bool:
    print(f"[MoveOnlyX] 对齐 X = {x:.3f} (参考 Y = {y_ref:.3f})")
    ok, _ = _post("MoveOnlyX", {"location_x": x, "location_y": y_ref}, timeout=timeout)
    if not ok:
        print("[MoveOnlyX] 失败")
    return ok

def wait(seconds: float):
    print(f"[Wait] 停留 {seconds} 秒...")
    time.sleep(seconds)

# ================== 新增：姿态纠正与基准设定 ==================
def set_baseline(timeout: int = 10) -> bool:
    print("[SetBaseline] 请求更新通用基准...")
    ok, _ = _post("SetBaseline", {}, timeout=timeout)
    if ok:
        print("[SetBaseline] 基准已更新")
    else:
        print("[SetBaseline] 失败")
    return ok

def sync_yaw(timeout: int = 15) -> bool:
    print("[SyncYaw] 开始姿态纠正...")
    ok, _ = _post("SyncYaw", {}, timeout=timeout)
    if ok:
        print("[SyncYaw] 完成")
    else:
        print("[SyncYaw] 失败")
    return ok

def set_x_baseline(timeout: int = 10) -> bool:
    print("[SetXBaseline] 请求更新 X 方向基准...")
    ok, _ = _post("SetXBaseline", {}, timeout=timeout)
    if ok:
        print("[SetXBaseline] 基准已更新")
    else:
        print("[SetXBaseline] 失败")
    return ok

def set_y_baseline(timeout: int = 10) -> bool:
    print("[SetYBaseline] 请求更新 Y 方向基准...")
    ok, _ = _post("SetYBaseline", {}, timeout=timeout)
    if ok:
        print("[SetYBaseline] 基准已更新")
    else:
        print("[SetYBaseline] 失败")
    return ok

def sync_x_yaw(timeout: int = 15) -> bool:
    print("[SyncXYaw] 开始 X 方向姿态纠正...")
    ok, _ = _post("SyncXYaw", {}, timeout=timeout)
    if ok:
        print("[SyncXYaw] 完成")
    else:
        print("[SyncXYaw] 失败")
    return ok

def sync_y_yaw(timeout: int = 15) -> bool:
    print("[SyncYYaw] 开始 Y 方向姿态纠正...")
    ok, _ = _post("SyncYYaw", {}, timeout=timeout)
    if ok:
        print("[SyncYYaw] 完成")
    else:
        print("[SyncYYaw] 失败")
    return ok

# ================== 序列执行器 ==================
def run_sequence(steps: list):
    fail_count = 0
    for i, step in enumerate(steps, 1):
        action = step[0]
        print(f"\n--- 步骤 {i}/{len(steps)}: {action} ---")

        if action == 'move':
            ok = move_absolute(step[1], step[2])
        elif action == 'rotate':
            ok = rotate(step[1])
        elif action == 'move_only_x':
            ok = move_only_x(step[1], step[2])
        elif action == 'wait':
            wait(step[1])
            ok = True
        elif action == 'set_baseline':
            ok = set_baseline()
        elif action == 'sync_yaw':
            ok = sync_yaw()
        elif action == 'set_x_baseline':
            ok = set_x_baseline()
        elif action == 'set_y_baseline':
            ok = set_y_baseline()
        elif action == 'sync_x_yaw':
            ok = sync_x_yaw()
        elif action == 'sync_y_yaw':
            ok = sync_y_yaw()
        else:
            print(f"未知动作: {action}")
            ok = False

        if ok:
            fail_count = 0
            print(f"    ✓ 完成")
        else:
            fail_count += 1
            print(f"    ✗ 失败 (连续{fail_count})")
            if fail_count >= 3:
                print("\n⚠ 连续失败，终止序列")
                break

# ================== 主流程 ==================
def main():
    print("=" * 60)
    print("  小车复合动作序列测试（含 X/Y 纠偏）")
    print(f"  服务器: {BASE_URL}")
    print("=" * 60)

    try:
        r = _sess.post(f"{BASE_URL}/Reset", json={"TaskId": 0}, timeout=5)
        print(f"连接正常: {r.json()}")
    except Exception as e:
        print(f"无法连接: {e}")
        sys.exit(1)

    reset_task_id()

    # ==================== 动作序列定义 ====================
    # 可用动作：'move','rotate','move_only_x','wait'
    #           'set_baseline','sync_yaw'
    #           'set_x_baseline','set_y_baseline','sync_x_yaw','sync_y_yaw'
    steps = [
        ('move', 1.5, 4.2),
        ('move_only_x', 0.8, 4.2),  
        ('wait', 10),
        ('move', 2.4, 4.2),
        ('rotate',   180),
        ('wait', 10),
        ('rotate', -180),
        ('move', 1.5, 0.9),   
        ('move_only_x', 0.7, 0.9),    
        ('rotate', 90),
        ('wait', 10),
        ('rotate', -90),
        ('move_only_x', 3.6, 0.9),
        ('rotate', 90),
        ('wait', 10),
        ('rotate', -90),
        ('move_only_x', 2.1, 0.9),
        ('move', 2.1, 4.3),
        ('move', 1.5, 7.6),
        ('move_only_x', 0.7, 7.6),
        ('rotate', -90),
        ('wait', 10),
        ('rotate', 90),
        ('move_only_x', 3.6, 7.6),
        ('rotate', -90),
        ('wait', 10),
        ('rotate', 90),
        ('move_only_x', 2.1, 7.6),
        ('move', 2.1, 4.3)
    ]

    print("\n[1] 开始执行动作序列...")
    run_sequence(steps)

    print("\n[2] 清理并退出")
    reset_task_id()
    print("测试结束。")

if __name__ == "__main__":
    main()