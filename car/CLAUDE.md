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

**Note**: `start.sh` references `~/catkin_ws/src/lsx10/scripts/CarControlServiceFlask.py` — update if needed.

## Parameter configuration

All tunable parameters are in `config.yaml` — room bounds, chassis IP, obstacle detection thresholds, path planning margins, client settings, etc. Loaded by `config_loader.py`:

```python
from config_loader import cfg
margin = cfg.path_planning.obstacle_margin   # → 0.5
ip = cfg.client.car_ip                       # → "10.152.203.227"
```

Changing `config.yaml` does not require code edits. All Python modules import from `cfg` — no hardcoded magic numbers.

## Architecture

### How the car really works

- **Chassis control**: TCP socket to `192.168.42.2:40923` (configured in `config.yaml` → `chassis.ip`). Init handshake: send `"command;"`, receive acknowledgement. Movement: `chassis move x <dx> y <dy> z <dz>;`. Position query: `chassis position ?;` → `x y yaw`. Speed query: `chassis speed ?;` → four wheel speeds. Socket stays open for session lifetime with 3s recv timeout. **Rotation `z` uses degrees, not radians.**
- **LiDAR**: ROS topic `/scan` (`sensor_msgs/LaserScan`). Position by reading front beam (middle of range → X) and side beam (quarter of range → Y). Offset is 0 (raw distances = absolute coordinates).
- **Coordinate system**: X decreases toward the front wall, Y increases away from the right wall.
- **Movement feedback**: Commands move the car, then `/MoveOnlyX` and `/MoveOnlyY` poll `getx()`/`gety()` for LiDAR closed-loop correction (0.08m tolerance, up to 5 retries).
- **Angle detection**: `getsum()` = average of 5 front+rear beam samples. Rotation angle = `arccos(baseline / current_sum)`.

### Core modules

| Module | Role |
|--------|------|
| `config.yaml` + `config_loader.py` | All parameters in one place; `cfg.xxx.yyy` dot access |
| `CarControlServiceFlask.py` | Flask server (port 5000), ROS node, chassis TCP. 9 REST endpoints. |
| `CarController.py` | Remote HTTP client library with retry + TaskId sync (not used by pipeline) |
| `obstacle_detector.py` | 5-stage obstacle detection: analyze beams → detect jumps → expand seeds → cluster → filter (bounds + forbidden zones) |
| `path_planner.py` | A* path planning with axis-aligned edges. Avoids obstacles, forbidden zones, and walls. Plans correction points. Self-loads forbidden zones if not passed. |
| `forbidden_zones.py` | Forbidden zone management CLI. Saves absolute-coordinate rectangles to `forbidden_zones.json`. Used by both `obstacle_detector` (filter false obstacles) and `path_planner` (geometry avoidance). |
| `target_points.py` | Interactive 6-point coordinate recording. Each point stores dual coordinates: `plan` (front+right beams, for path planning) and `correct` (point-specific beams per table below, for fine correction). Saved to `target_points.json`. |
| `action_for_each_target.py` | Per-point sequential action list (correct / rotate / stay). Actions execute in order after reaching target. |
| `lidar_utils.py` | `getsum()` — averaged front+rear beam distance sum, used by `run.py`. |
| `path_simulator.py` | matplotlib GUI simulator for A* path planning. No ROS required. |

### Points and correction beams

Points 1~4 have two coordinate sets in `target_points.json`:

| Point | Plan X | Plan Y | Correct X (beam) | Correct Y (beam) | Correct axes |
|-------|--------|--------|------------------|------------------|-------------|
| 1 | front | right | front | right | xy |
| 2 | front | right | **rear** | right | xy |
| 3 | front | right | **rear** | **left** | xy |
| 4 | front | right | front | **left** | xy |
| 5 | front | right | **rear** | right | **x only** |
| 6 | front | right | **rear** | right | **x only** |

Correction values in JSON are **raw beam distances**, not inverted. Inversion (`4.5 - rear`, `8.8 - left`) happens at correction time.

### Flask REST endpoints

| Endpoint | Key detail |
|----------|-----------|
| `/Move` | X+Y LiDAR closed-loop, 0.08m tolerance |
| `/MoveRelative` | One-shot chassis command, no feedback |
| `/MoveOnlyX` | X one-shot, then Y closed-loop. TaskId mismatch was FIXED (now returns `CurrentTaskID + 1`) |
| `/MoveOnlyY` | X one-shot, then Y closed-loop |
| `/MoveLongDistance` | X split into 3 segments, 0.00001m tolerance (too tight) |
| `/Circle` | Takes **degrees** (not radians despite param name `rad_z`) |
| `/SyncYaw` | Angle correction: `arccos(baseline/current)` → rotate in 2.5°~10° steps. Threshold: 20° (configurable) |
| `/SetBaseline` | Records current front+rear sum + yaw as correction baseline |
| `/ShutDown` / `/Reset` | Cleanup / TaskId reset |

### Task ID protocol

`TaskId` must equal `CurrentTaskID + 1`. Mismatch → server returns `expectedTaskId`, client re-syncs and replays. `/Reset` zeroes the counter.

### Module dependency chain

```
obstacle_detector  ←  forbidden_zones (filter)
       ↑                        ↑
path_planner  ←  forbidden_zones (avoid)
       ↑
run  ←  target_points + action_for_each_target + Flask endpoints
```

**Integration gap**: The Flask server has no awareness of obstacles or planned paths — all intelligence lives in the pipeline client.

### Correction flow

**Waypoint correction** (during movement):
1. `SyncYaw` — angle correction via Flask endpoint
2. Coordinate correction — LiDAR vs expected position, with sanity check (front+rear ≈ baseline, right+left ≈ baseline, tolerance configurable in `config.yaml`)

**Target correction** (after arrival, if `correct` in actions):
- Points with correction coordinates → compare stored vs current raw beam distances, compute chassis displacement
- Points without → fall back to planning coordinate closed-loop

Correction points are chosen by `plan_correction_points()` — **all points (start, end, intermediate) must pass safety check**. No point gets automatic inclusion.

### Sanity check for LiDAR readings

Before each coordinate correction attempt, `_lidar_readings_sane()` verifies:
- `front + rear ≈ baseline_x_sum` (within `sanity_check_tolerance`)
- `right + left ≈ baseline_y_sum`

If either fails, the LiDAR is hitting the wrong walls (car is angled), correction is aborted.

## Key commands

```bash
# Main entry — interactive task loop
python3 run.py
# Input 1~6: go to target. 7: return to start. Q: quit.

# Set up forbidden zones (no-go areas)
python3 forbidden_zones.py
# s=setup, l=list, a=add, d=delete

# Record target point coordinates (6 points)
python3 target_points.py
# s=setup, l=list

# Standalone tests (require ROS + LiDAR)
python3 test_obstacle_detector.py
python3 test_path_planner.py
python3 path_planner.py <target_x> <target_y>

# Offline path simulator (no ROS required)
python3 path_simulator.py
```

## Key conventions

- **Python 3** only. System Python with ROS packages, no virtualenv.
- **Sensor offset is 0** — raw LiDAR distances used directly as absolute coordinates.
- **LiDAR inf → re-sample**. Pattern duplicated across files that read LiDAR.
- **Chassis rotation `z` is degrees**, not radians. `/Circle` param name is misleading (`rad_z`).
- **All parameters in `config.yaml`**. When adding new tunables, put them there, not as module constants.
- **Forbidden zones** use absolute coordinates. Must re-record with `forbidden_zones.py` if old JSON has relative format with `origin_x`/`origin_y`.

## Known issues

### `/MoveLongDistance` tolerance too tight

Uses 0.00001m tolerance (vs 0.08m for all others). LiDAR noise can cause infinite retry loops.

### Obstacle detection intermittency

Objects near detection thresholds (JUMP_THRESHOLD=0.3, CLUSTER_MIN_BEAMS=3) are sometimes missed. Sensitive to LiDAR noise and angle.

### Duplicated LiDAR reading functions

`getx`/`gety`/`getsum` are duplicated across `CarControlServiceFlask.py`, `path_planner.py`, `run.py`, and `test_obstacle_detector.py`. Changes must be applied to all copies.

### Correction may trigger when LiDAR is blocked

`plan_correction_points` uses column/row clearance check but doesn't account for the fan-shaped beam pattern. A correction point may be marked safe even if the LiDAR beam hits an obstacle at an angle.

### Flask server unaware of obstacles

All planning intelligence is client-side. Flask endpoints execute moves blindly — if the client sends a move into an obstacle, the server won't stop it.
