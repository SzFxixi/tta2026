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

- **Chassis control**: TCP socket to `192.168.42.2:40923`. Init handshake: send `"command;"`, receive acknowledgement. Movement commands use `chassis move x <dx> y <dy> z <dz>;` format. Position query: `chassis position ?;` returns `x y yaw`. Speed query: `chassis speed ?;` returns four wheel speeds. The socket stays open for the session lifetime with a 3s recv timeout.
- **LiDAR**: ROS topic `/scan` (`sensor_msgs/LaserScan`). The car tracks its **absolute position** by reading the front beam (middle of range array → X-axis distance) and side beam (quarter of range array → Y-axis distance), then subtracting a **fixed sensor offset of 0.39m** from each.
- **Coordinate system**: X decreases toward the front wall, Y increases away from the right wall. The offset accounts for the LiDAR's physical mounting position relative to the car center.
- **Movement feedback loop**: Commands move the car, then the service polls `getx()`/`gety()` to measure actual position and issues corrective moves until the error is within tolerance (typically 0.08m, up to 5 retries). Long moves are split into 3 segments.
- **Angle detection**: `getsum()` averages 5 samples of front-beam + rear-beam LiDAR distance. The car's rotation angle relative to the walls is `arccos(baseline / current_sum)`.

### Room bounds and safe zone

- Room bounds: `ROOM_X_MIN=0.1`, `ROOM_X_MAX=4.5`, `ROOM_Y_MIN=0.1`, `ROOM_Y_MAX=8.8` (meters, in absolute coordinates).
- Safe zone (obstacle-free region): defaults to `(0.5, 0.5)` to `(3.6, 7.8)`. Obstacles outside this zone are used for path planning avoidance.

### Core modules

| Module | Role | Dependencies |
|--------|------|-------------|
| `CarControlServiceFlask.py` | Main Flask server (port 5000), ROS node, chassis TCP client | ROS, Flask, LiDAR |
| `CarController.py` | Remote HTTP client library with retry + TaskId sync | `requests` |
| `obstacle_detector.py` | Detect obstacles outside a safe zone from LiDAR jump analysis | ROS, numpy |
| `path_planner.py` | Generate right-angle-turn paths avoiding detected obstacles | `obstacle_detector` |
| `pose_correction.py` | Angular drift detection + correction using LiDAR + chassis yaw | ROS, numpy, chassis TCP |
| `mission_runner.py` | High-level mission executor: plan → move segment-by-segment → correct | `path_planner`, Flask endpoints |
| `LidarTest.py` | Standalone wall-fitting via PCA + linear regression (debug tool, not used by any other module) | sklearn, ROS |

**Server IP addresses** vary by machine — the default (`10.203.94.227:5000`) is set in `CarController.py`, but `mission_runner.py` uses a different IP (`10.26.36.227`). Update these to match the target car before running.

### Task ID protocol

Every API call carries a `TaskId`. The server expects strictly `CurrentTaskID + 1`. If it receives a mismatched ID, it returns `expectedTaskId` in the error response. The client must re-sync and replay any missing tasks. `Reset` resets `CurrentTaskID` to 0. This ensures ordered execution across unreliable networks.

**Caveat**: `/MoveOnlyX` returns `expectedTaskId: CurrentTaskID` (not `CurrentTaskID + 1`), which is inconsistent with all other endpoints.

### Module dependency chain (new modules, work in progress)

```
obstacle_detector  (standalone — LiDAR → obstacle list)
       ↑
path_planner       (depends on obstacle_detector — LiDAR → obstacle list → safe path)
       ↑
pose_correction    (standalone — LiDAR + chassis yaw → angle correction)
```

These three modules are extracted from `CarControlServiceFlask.py` and are **not yet wired into the Flask endpoints**. They have their own test scripts and can run independently. The Flask server has its own built-in `SyncYaw()` method in `CarService` which duplicates the concept of `pose_correction.py` but with different thresholds and rotation step sizes (2.5° vs 8°).

**Integration gap**: `mission_runner.py` and `test_full_pipeline.py` are the entry points that bridge these modules — they call `path_planner` to generate a path, then call the Flask server's movement endpoints to execute it. But the Flask server itself has no awareness of obstacles or planned paths.

## Competition points

`mission_runner.py` contains a `POINTS` dict with 6 waypoints, all set to `None`. These must be configured before a competition run. The `STEPS` dict defines 4 steps that select subsets of these points. Run `python3 mission_runner.py` and use the interactive menu to select which step to execute.

## Tests

All test scripts require ROS + LiDAR running on the car:

```bash
# Obstacle detection (safe zone: p1_x p1_y p2_x p2_y)
python3 test_obstacle_detector.py [x1 y1 x2 y2]

# Path planning (interactive: enter target coords and waypoint count)
python3 test_path_planner.py

# Pose correction (interactive: Enter to check/correct, q to quit)
python3 test_pose_correction.py

# Full pipeline integration test (3 phases: detect → plan → execute)
python3 test_full_pipeline.py

# Legacy integration test (HTTP client, runs from remote machine)
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
- **Flask endpoints use `host='0.0.0.0', port=5000`**.
- **Chassis wheel stop detection**: Poll `chassis speed ?;` until all four wheel speeds drop below 10, with a 10s timeout.

## Known issues

### Critical: offset mismatch between path_planner and Flask server

`path_planner.py` sets `X_OFFSET = 0.0, Y_OFFSET = 0.0` (line 56-57), so `get_car_position()` returns **raw LiDAR distances**. But `CarControlServiceFlask.py` subtracts `0.39` from both X and Y. This means coordinates computed by the path planner and coordinates expected by the Flask endpoints are in different frames — off by 0.39m in both axes. This likely contributes to the "still cannot avoid obstacles accurately" problem noted in the last commit.

### Dead code

- **`CarController.py`** imports `paramiko` and `scp` but never uses them.
- **`CarControlServiceFlask.py`** `adjust_angle()` function (line 67-72) uses a hardcoded `arccos(9.0/s)` and is never called by any endpoint.
- **`test_avoidance.py`** calls `SetXBaseline`, `SetYBaseline`, `SyncXYaw`, `SyncYYaw` endpoints that **do not exist** in the Flask server. These actions will fail.

### `/MoveOnlyX` returns wrong expectedTaskId

Line 387 returns `CurrentTaskID` instead of `CurrentTaskID + 1`, inconsistent with every other endpoint. Clients relying on this value for re-sync may get stuck.

### `MoveLongDistance` tolerance too tight

Uses tolerance `0.00001` (vs `0.08` for all other endpoints). This may cause infinite retry loops if LiDAR noise keeps the error above 10 microns.

### Obstacle detection not yet reliable

The most recent commit (`7a2a606`) states "still cannot avoid obstacles accurately." The path planner and obstacle detector run, but the full pipeline from detection → planning → movement is not producing correct obstacle avoidance behavior. Root cause is not yet diagnosed.
