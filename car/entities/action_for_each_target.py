
# 这里x和y反了
ACTIONS = {
    1: [
        {"type": "correct"},
        {"type": "rotate", "value": 90},
        {"type": "stay"},
        {"type": "move_rel", "dx": 0, "dy": -0.5},
        {"type": "stay"},
        {"type": "rotate", "value": -90},
    ],
    2: [
        {"type": "correct"},
        {"type": "rotate", "value": 90},
        {"type": "stay"},
        {"type": "move_rel", "dx": 0, "dy": -0.2},
        {"type": "stay"},
        {"type": "rotate", "value": -90},
    ],
    3: [
        {"type": "correct"},
        {"type": "rotate", "value": -90},
        {"type": "stay"},
        {"type": "move_rel", "dx": 0, "dy": 0.2},
        {"type": "stay"},
        {"type": "rotate", "value": 90},
    ],
    4: [
        {"type": "correct"},
        {"type": "rotate", "value": -90},
        {"type": "stay"},
        {"type": "move_rel", "dx": 0, "dy": -0.2},
        {"type": "stay"},
        {"type": "rotate", "value": 90},
    ],
    5: [
        {"type": "correct"},
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
    return ACTIONS.get(pid, [])
