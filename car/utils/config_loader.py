#!/usr/bin/env python3
"""
统一配置加载器 — 从 config.yaml 读取所有参数。

用法:
    from utils.config_loader import cfg
    print(cfg.room.x_min)            # → 0.1
    print(cfg.obstacle.jump_threshold)  # → 0.3
"""

import os
import re

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "config.yaml")


class _ConfigNode:
    """支持 . 访问的配置节点"""
    def __init__(self, data):
        for k, v in data.items():
            if isinstance(v, dict):
                setattr(self, k, _ConfigNode(v))
            else:
                setattr(self, k, v)

    def __repr__(self):
        return str({k: v for k, v in self.__dict__.items() if not k.startswith('_')})


def _parse_yaml(text):
    """极简 YAML 解析器（仅支持当前 config.yaml 的子集）"""
    result = {}
    stack = [(result, -1)]
    for line in text.split('\n'):
        stripped = line.rstrip()
        if not stripped or stripped.startswith('#'):
            continue
        indent = len(line) - len(line.lstrip())
        key, _, val = (s.strip() for s in stripped.partition(':'))
        val = val or None

        # 回退到正确的缩进层级
        while stack[-1][1] >= indent:
            stack.pop()
        parent, _ = stack[-1]

        if val is not None:
            val = val.strip('"\'')
            for t in (int, float):
                try:
                    val = t(val)
                    break
                except ValueError:
                    pass
            if isinstance(val, str) and val.lower() in ('true', 'false'):
                val = val.lower() == 'true'
            parent[key] = val
        else:
            parent[key] = {}
            stack.append((parent[key], indent))
    return result


def _load_config():
    with open(_CONFIG_PATH, 'r') as f:
        data = _parse_yaml(f.read())
    return _ConfigNode(data)


cfg = _load_config()
