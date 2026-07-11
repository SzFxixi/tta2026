#!/usr/bin/env python3
"""Keyboard-interactive launcher for run.py.

This entry point intentionally forces terminal interactive mode. It reuses the
navigation implementation from controllers/run.py but strips --external from
argv so it cannot accidentally start the HTTP-control mode.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controllers import run


def main():
    sys.argv = [arg for arg in sys.argv if arg != "--external"]
    print("[run1] 终端键盘交互模式")
    run.main()


if __name__ == "__main__":
    main()
