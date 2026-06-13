# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System overview

Smart car control system: a Flask HTTP service running on a ROS-equipped car that receives movement commands and executes them via TCP chassis commands, while using LiDAR for positioning feedback, obstacle detection, and path planning.

## Start / stop the car services

```bash
# Start (in order: roscore → lslidar → Flask)
bash start.sh

# Stop
pkill -f 'roscore|lslidar_net|CarControlServiceFlask'
```

Logs go to `~/car2_logs/`.

**Note**: `start.sh` references `~/catkin_ws/src/lsx10/scripts/CarControlServiceFlask.py` — update this path if the file lives elsewhere on the target machine.

## Architecture

### How the car really works

- **Chassis control**: TCP socket to `192.168.42.2:40923`. Init handshake: send `"command;"`, receive acknowledgement. Movement commands use `chassis move x <dx> y <dy> z <dz>;` format. Position query: `chassis position ?;` returns `x y yaw`. Speed query: `chassis speed ?;` returns four wheel speeds. The socket stays open for the session lifetime with a 3s recv timeout.
- **LiDAR**: ROS topic `/scan` (`sensor_msgs/LaserScan`). The car tracks its **absolute position** by reading the front beam (middle of range array → X-axis distance) and side beam (quarter of range array → Y-axis distance). The LiDAR sensor offset is currently set to **0** in both the Flask server and path planner (i.e., raw LiDAR distances are used directly as absolute coordinates).
- **Coordinate system**: X decreases toward the front wall, Y increases away from the right wall.
- **Movement feedback loop**: Commands move the car, then the service polls `getx()`/`gety()` to measure actual position and issues corrective moves until the error is within tolerance (typically 0.08m, up to 5 retries). Long moves are split into 3 segments.
- **Angle detection**: `getsum()` averages 5 samples of front-beam + rear-beam LiDAR distance. The car's rotation angle relative to the walls is `arccos(baseline / current_sum)`.

### Room bounds

- Room bounds: `ROOM_X_MIN=0.1`, `ROOM_X_MAX=4.5`, `ROOM_Y_MIN=0.1`, `ROOM_Y_MAX=8.8` (meters, in absolute coordinates). These are defined separately in `CarControlServiceFlask.py`, `obstacle_detector.py`, `path_planner.py`, and `path_simulator.py` — keep them in sync when changing.

### Core modules

| Module | Role | Dependencies |
|--------|------|-------------|
| `CarControlServiceFlask.py` | Main Flask server (port 5000), ROS node, chassis TCP client. Exposes 9 REST endpoints. | ROS, Flask, LiDAR |
| `CarController.py` | Remote HTTP client library with retry + TaskId sync | `requests` |
| `obstacle_detector.py` | Detect obstacles from LiDAR distance-jump analysis (5-stage pipeline: analyze → detect jumps → expand seeds → cluster → filter) | ROS, numpy |
| `path_planner.py` | A* path planning with axis-aligned edges, avoiding detected obstacles. Also plans correction points along the path. | `obstacle_detector`, ROS, numpy |
| `lidar_utils.py` | LiDAR utility: `getsum()` — averaged front+rear beam distance sum used for angle diagnostics in `test_full_pipeline.py`. | ROS, numpy |
| `forbidden_zones.py` | Forbidden zone (no-go area) management: interactive setup, save/load/edit via `forbidden_zones.json`. Uses absolute coordinates. Currently decoupled from path_planner — standalone tool. | ROS, LiDAR |
| `target_points.py` | Interactive tool to set 6 target point coordinates. Each point uses a different LiDAR beam combo (front/rear X, right/left Y) depending on car orientation at that position. Saved to `target_points.json`. | ROS, LiDAR |
| `action_for_each_target.py` | Per-point action config (rotate, stay, correct) for 6 points. Pure data module — no runtime dependencies. | None |
| `path_simulator.py` | Standalone matplotlib GUI simulator for A* path planning. Click to place start/goal/obstacles/no-go zones, see paths and correction points in real time. Runs without ROS on any machine. | matplotlib, numpy |
| `LidarTest.py` | Standalone wall-fitting via PCA + linear regression (debug/diagnostic tool, not used by any other module) | sklearn, ROS |

**Server IP addresses** vary by machine — the default (`10.203.94.227:5000`) is set in `CarController.py`, but `test_full_pipeline.py` uses `10.26.36.227`. Update these to match the target car before running.

### Flask REST endpoints

All endpoints (except `/ShutDown` and `/Reset`) require `TaskId = CurrentTaskID + 1`:

| Endpoint | Method | Purpose | Feedback |
|----------|--------|---------|----------|
| `/Move` | POST | Move to absolute (x, y) with LiDAR closed-loop correction | X then Y, 0.08m tolerance, 5 retries each |
| `/MoveRelative` | POST | Move by relative (dx, dy) in chassis frame. Supports `step` for splitting long moves. | No closed-loop feedback — one-shot chassis command |
| `/MoveOnlyX` | POST | Move to absolute X, then LiDAR-correct Y | X first (one-shot), then Y closed-loop |
| `/MoveOnlyY` | POST | Move to absolute Y, with LiDAR-correct Y feedback | X first (one-shot), then Y closed-loop |
| `/MoveLongDistance` | POST | Move to absolute (x, y), split X into 3 segments | 0.00001m tolerance (extremely tight) |
| `/Circle` | POST | Rotate in place by `rad_z` radians | One-shot chassis command |
| `/SyncYaw` | POST | Angle correction using LiDAR front/rear distance sum vs baseline | Rotates in 2.5°–10° steps |
| `/SetBaseline` | POST | Record current LiDAR distance sum + chassis yaw as correction baseline | No movement |
| `/ShutDown` | POST | Close chassis TCP connection, reset TaskId to 0 | — |
| `/Reset` | POST | Reset TaskId to 0 without disconnecting | — |

### Task ID protocol

Every API call carries a `TaskId`. The server expects strictly `CurrentTaskID + 1`. If it receives a mismatched ID, it returns `expectedTaskId` in the error response. The client must re-sync and replay any missing tasks. `/Reset` resets `CurrentTaskID` to 0. This ensures ordered execution across unreliable networks.

**Caveat**: `/MoveOnlyX` returns `expectedTaskId: CurrentTaskID` (not `CurrentTaskID + 1`), inconsistent with all other endpoints. Clients relying on this value for re-sync may get stuck.

### Module dependency chain

```
obstacle_detector  (standalone — LiDAR → obstacle list)
       ↑
path_planner       (depends on obstacle_detector → safe A* path + correction points)
       ↑
test_full_pipeline (entry point — calls path_planner → Flask endpoints to execute)
```

**Integration gap**: `test_full_pipeline.py` is the entry point that bridges these modules — it calls `path_planner` to generate a path, then calls the Flask server's movement endpoints to execute it. But the Flask server itself has no awareness of obstacles or planned paths.

## Tests

All test scripts (except `path_simulator.py`) require ROS + LiDAR running on the car:

```bash
# Obstacle detection — prints detected obstacles and LiDAR diagnostics
python3 test_obstacle_detector.py

# Path planning — interactive: enter target coords and waypoint count
python3 test_path_planner.py

# Full pipeline integration test — 3 phases: detect → plan → execute
python3 test_full_pipeline.py <target_x> <target_y>
```

`path_planner.py` can also be called directly:
```bash
python3 path_planner.py <target_x> <target_y> [waypoint_count]
```

`path_simulator.py` runs on any machine with matplotlib — no ROS required:
```bash
python3 path_simulator.py
```

## Key conventions

- **Python 3** only. No virtualenv — system Python with ROS packages.
- **Sensor offset**: `X_OFFSET = 0.0`, `Y_OFFSET = 0.0` in both `path_planner.py` and `CarControlServiceFlask.py` (`CarService` init at line 173). Raw LiDAR distances are used directly as absolute coordinates.
- **LiDAR inf handling**: When a beam returns `np.inf` (no echo), re-sample until valid data arrives. This pattern is duplicated across all files that read LiDAR — if you change it, update all copies.
- **Flask endpoints use `host='0.0.0.0', port=5000`**.
- **Chassis wheel stop detection**: Poll `chassis speed ?;` until all four wheel speeds drop below 10, with a 10s timeout.

## Known issues

### `/MoveOnlyX` returns wrong expectedTaskId

Line 362 returns `"expectedTaskId": CurrentTaskID` instead of `CurrentTaskID + 1`, inconsistent with every other endpoint. Clients that rely on this value for re-sync may get stuck.

### `MoveLongDistance` tolerance too tight

Uses tolerance `0.00001` (vs `0.08` for all other endpoints). This may cause infinite retry loops if LiDAR noise keeps the error above 10 microns.

### Obstacle detection not yet reliable

The path planner and obstacle detector run, but the full pipeline from detection → planning → movement is not producing correct obstacle avoidance behavior. Root cause is not yet diagnosed.

### Duplicated getx/gety/getsum

The LiDAR reading functions (`getx`, `gety`, `getsum`) are duplicated across `CarControlServiceFlask.py`, `path_planner.py`, `test_obstacle_detector.py`, and `test_full_pipeline.py`. Changes to sampling strategy or inf handling must be applied in all copies.
