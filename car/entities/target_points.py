#!/usr/bin/env python3
"""Compatibility wrapper for target_zone_setup.py.

Navigation code still imports entities.target_points.load_targets. The actual
target recording and forbidden-zone generation tool now lives in
entities.target_zone_setup.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ROS_PYTHON_PATH = "/opt/ros/noetic/lib/python3/dist-packages"
if os.path.isdir(_ROS_PYTHON_PATH) and _ROS_PYTHON_PATH not in sys.path:
    sys.path.append(_ROS_PYTHON_PATH)

from entities.target_zone_setup import (
    generate_forbidden_zones,
    list_forbidden_zones,
    list_targets,
    load_targets,
    modify_target,
    regenerate_forbidden_zones,
    save_forbidden_zones,
    setup_targets,
)


if __name__ == "__main__":
    from entities.target_zone_setup import cli_main

    cli_main()
