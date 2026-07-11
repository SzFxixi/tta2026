# 联调模式：每个点只做 correct（导航+精校）
# rotate / move_rel / stay / adjust_x 由 car_arm_integration.py 统一调度
#
# 原始动作序列（备份，如需恢复交互模式请取消注释）：
# ACTIONS = {
#     1: [
#         {"type": "correct"},
#         {"type": "rotate", "value": 90},
#         {"type": "stay"},
#         {"type": "move_rel", "dx": 0, "dy": -0.4},
#         {"type": "stay"},
#         {"type": "rotate", "value": -90},
#     ],
#     2: [
#         {"type": "correct"},
#         {"type": "rotate", "value": 90},
#         {"type": "stay"},
#         {"type": "move_rel", "dx": 0, "dy": -0.2},
#         {"type": "stay"},
#         {"type": "rotate", "value": -90},
#     ],
#     3: [
#         {"type": "correct"},
#         {"type": "rotate", "value": -90},
#         {"type": "stay"},
#         {"type": "move_rel", "dx": 0, "dy": 0.2},
#         {"type": "stay"},
#         {"type": "rotate", "value": 90},
#     ],
#     4: [
#         {"type": "correct"},
#         {"type": "rotate", "value": -90},
#         {"type": "stay"},
#         {"type": "move_rel", "dx": 0, "dy": -0.2},
#         {"type": "stay"},
#         {"type": "rotate", "value": 90},
#     ],
#     5: [
#         {"type": "correct"},
#         {"type": "adjust_x"},
#     ],
#     6: [
#         {"type": "correct"},
#         {"type": "rotate", "value": 180},
#         {"type": "stay"},
#         {"type": "rotate", "value": -180},
#     ],
# }
ACTIONS = {
    1: [{"type": "correct"}],
    2: [{"type": "correct"}],
    3: [{"type": "correct"}],
    4: [{"type": "correct"}],
    5: [{"type": "correct"}],
    6: [{"type": "correct"}],
}


def get_actions(pid):
    return ACTIONS.get(pid, [])
