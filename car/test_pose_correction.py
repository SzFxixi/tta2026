#!/usr/bin/env python3
"""
位姿校正独立测试脚本

用法（在小车上运行）:
    python3 test_pose_correction.py

流程:
    1. 建立底盘连接
    2. 设定基准 (set_baseline)
    3. 等待用户按 Enter
    4. 检测偏角 + 校正
    5. 循环: 继续等待 Enter → 再次检测+校正, 直到按 q 退出
"""

import sys
from lidar_utils import PoseCorrector


def main():
    pc = PoseCorrector()
    pc.connect()

    # ── 设定基准 ──
    print("\n" + "=" * 50)
    print("  设定基准")
    print("=" * 50)
    pc.set_baseline()

    # ── 等待 Enter → 检测+校正 ──
    print("\n基准已设定。之后每次按 Enter 检测并校正，输入 q 退出。")

    while True:
        user = input("\n>>> 按 Enter 进行位姿检测与校正: ").strip()
        if user.lower() == 'q':
            print("退出。")
            break

        print("-" * 40)
        dev = pc.get_deviation()
        print(f"当前偏角: {dev:.1f}°")

        ok = pc.correct_if_needed()
        if ok:
            # 校正后重新检测确认
            dev_after = pc.get_deviation()
            print(f"校正后偏角: {dev_after:.1f}°")
        else:
            print("位姿正常，无需校正。")

    pc.disconnect()


if __name__ == "__main__":
    main()
