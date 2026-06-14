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
