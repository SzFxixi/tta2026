#!/usr/bin/env python3
"""
点位动作配置 — 每个点位到达后按顺序依次执行的动作列表。

动作类型:
    {"type": "correct"}             — 校正坐标
    {"type": "rotate", "value": 90} — 旋转指定角度（度），正=逆时针
    {"type": "stay"}                — 等待操作者按 Enter 后继续

修改方式：直接编辑下方 ACTIONS 字典，每个点位是一个动作列表。
"""

ACTIONS = {
    1: [
        {"type": "correct"},
        {"type": "rotate", "value": 90},
        {"type": "stay"},
        {"type": "rotate", "value": -90},
    ],
    2: [
        {"type": "correct"},
        {"type": "rotate", "value": 90},
        {"type": "stay"},
        {"type": "rotate", "value": -90},
    ],
    3: [
        {"type": "correct"},
        {"type": "rotate", "value": -90},
        {"type": "stay"},
        {"type": "rotate", "value": 90},
    ],
    4: [
        {"type": "correct"},
        {"type": "rotate", "value": -90},
        {"type": "stay"},
        {"type": "rotate", "value": 90},
    ],
    5: [
        {"type": "stay"},
    ],
    6: [
        {"type": "correct"},
        {"type": "rotate", "value": 180},
        {"type": "stay"},
        {"type": "rotate", "value": -180},
    ],
}


def get_actions(pid):
    """获取点位 pid 的动作列表（按顺序执行）"""
    return ACTIONS.get(pid, [])
