#!/usr/bin/env python3
"""
点位动作绑定配置 — 定义每个点位到达后执行的动作。

动作类型:
    rotate: 旋转角度（度），正=逆时针，负=顺时针。None 表示不旋转。
    stay:   停留时间（秒）。None 表示不停留。

修改方式：直接编辑下方 ACTIONS 字典。
"""

# ============================================================
#  动作配置（按需修改）
# ============================================================

ACTIONS = {
    1:  {"rotate": None, "stay": None},
    2:  {"rotate": None, "stay": None},
    3:  {"rotate": None, "stay": None},
    4:  {"rotate": None, "stay": None},
    5:  {"rotate": None, "stay": None},
    6:  {"rotate": None, "stay": None},
}

# ============================================================
#  便捷查询
# ============================================================

def has_rotate(pid):
    """点位 pid 是否需要旋转"""
    a = ACTIONS.get(pid, {})
    return a.get("rotate") is not None


def has_stay(pid):
    """点位 pid 是否需要停留"""
    a = ACTIONS.get(pid, {})
    return a.get("stay") is not None


def get_action(pid):
    """获取点位 pid 的完整动作配置"""
    return ACTIONS.get(pid, {"rotate": None, "stay": None})
