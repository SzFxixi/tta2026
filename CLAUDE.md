# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System overview

Smart car control system: a Flask HTTP service running on a ROS-equipped car that receives movement commands and executes them via TCP chassis commands, while using LiDAR for positioning feedback, obstacle detection, path planning, and pose correction.

## Start / stop the car services

```bash
# Start (in order: roscore → lslidar → Flask)
bash start.sh

# Stop
pkill -f 'roscore|lslidar_net|CarControlServiceFlask'
```

Logs go to `~/car2_logs/`.

## Architecture

### How the car really works

- **Chassis control**: TCP socket to `192.168.42.2:40923`. Commands use `chassis move x <dx> y <dy> z <dz>;` format. The socket stays open for the session lifetime.
- **LiDAR**: ROS topic `/scan` (`sensor_msgs/LaserScan`). The car tracks its **absolute position** by reading the front beam (middle of range array → X-axis distance) and side beam (quarter of range array → Y-axis distance), then subtracting a **fixed sensor offset of 0.39m** from each.
- **Coordinate system**: X decreases toward the front wall, Y increases away from the right wall. The offset accounts for the LiDAR's physical mounting position relative to the car center.
- **Movement feedback loop**: Commands move the car, then the service polls `getx()`/`gety()` to measure actual position and issues corrective moves until the error is within tolerance (typically 0.08m, up to 5 retries). Long moves are split into 3 segments.

### Core modules

| Module | Role | Dependencies |
|--------|------|-------------|
| `CarControlServiceFlask.py` | Main Flask server (port 5000), ROS node, chassis TCP client | ROS, Flask, LiDAR |
| `CarController.py` | Remote HTTP client library with retry + TaskId sync | `requests`, `paramiko` |
| `obstacle_detector.py` | Detect obstacles outside a safe zone from LiDAR jump analysis | ROS, numpy |
| `path_planner.py` | Generate right-angle-turn paths avoiding detected obstacles | `obstacle_detector` |
| `pose_correction.py` | Angular drift detection + correction using LiDAR + chassis yaw | ROS, numpy, chassis TCP |
| `LidarTest.py` | Standalone wall-fitting via PCA + linear regression (debug tool) | sklearn, ROS |

### Task ID protocol

Every API call carries a `TaskId`. The server expects strictly `CurrentTaskID + 1`. If it receives a mismatched ID, it returns `expectedTaskId` in the error response. The client must re-sync and replay any missing tasks. `Reset` resets `CurrentTaskID` to 0. This ensures ordered execution across unreliable networks.

### Module dependency chain (new modules, work in progress)

```
obstacle_detector  (standalone — LiDAR → obstacle list)
       ↑
path_planner       (depends on obstacle_detector — LiDAR → obstacle list → safe path)
       ↑
pose_correction    (standalone — LiDAR + chassis yaw → angle correction)
```

These three modules are extracted from `CarControlServiceFlask.py` and are **not yet wired into the Flask endpoints**. They have their own test scripts and can run independently.

## Tests

All test scripts require ROS + LiDAR running on the car:

```bash
# Obstacle detection (safe zone: p1_x p1_y p2_x p2_y)
python3 test_obstacle_detector.py [x1 y1 x2 y2]

# Path planning (interactive: enter target coords and waypoint count)
python3 test_path_planner.py

# Pose correction (interactive: Enter to check/correct, q to quit)
python3 test_pose_correction.py

# Integration test (HTTP client, runs from remote machine)
python3 test_avoidance.py
```

`path_planner.py` can also be called directly for a quick test:
```bash
python3 path_planner.py <target_x> <target_y> [waypoint_count]
```

## Key conventions

- **Python 3** only. No virtualenv — system Python with ROS packages.
- **Sensor offset**: `X_OFFSET = 0.39`, `Y_OFFSET = 0.39` — must match `CarService` init in the Flask server. All positioning code subtracts these from raw LiDAR distances.
- **LiDAR inf handling**: When a beam returns `np.inf` (no echo), re-sample until valid data arrives. This is standard in the codebase.
- **Flask endpoints use `host='0.0.0.0', port=5000`**. The client (`CarController.py`) defaults to `10.203.94.227:5000`.
- **Chassis wheel stop detection**: Poll `chassis speed ?;` until all four wheel speeds drop below 10, with a 10s timeout.
