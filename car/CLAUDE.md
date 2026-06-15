# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System overview

Smart car control system: a Flask HTTP service running on a ROS-equipped car that receives movement commands and executes them via TCP chassis commands, while using LiDAR for wall-fitting-based positioning, obstacle detection, path planning, and pose correction.

## Start / stop the car services

```bash
# Start (in order: roscore → lslidar → Flask)
# start.sh is at the repo root level (../tta@10.152.203.227/start.sh), not in car/
bash start.sh

# Stop
pkill -f 'roscore|lslidar_net|CarControlServiceFlask'
```

Logs go to `~/car2_logs/`.

## Parameter configuration

All tunable parameters are in `config.yaml` — room bounds, chassis IP, obstacle detection thresholds, path planning margins, client settings, etc. Loaded by `config_loader.py`:

```python
from utils.config_loader import cfg
margin = cfg.path_planning.obstacle_margin   # → 0.5
ip = cfg.client.car_ip                       # → "10.152.203.227"
```

Changing `config.yaml` does not require code edits. All Python modules import from `cfg` — no hardcoded magic numbers.

**⚠️ Custom YAML parser**: `config_loader.py` uses a hand-rolled minimal YAML parser (not PyYAML). It supports only a strict subset: `key: value` pairs, nesting via indentation (spaces only), `#` comments, and string/int/float/bool values. It does **not** support: lists (`- item`), flow style (`{key: val}`), multi-line strings, anchors/aliases, or tabs for indentation.

## Architecture

### How the car really works

- **Chassis control**: TCP socket to `192.168.42.2:40923` (configured in `config.yaml` → `chassis.ip`). Init handshake: send `"command;"`, receive acknowledgement. Movement: `chassis move x <dx> y <dy> z <dz>;`. Position query: `chassis position ?;` → `x y yaw`. Speed query: `chassis speed ?;` → four wheel speeds. Socket stays open for session lifetime with 3s recv timeout. **Rotation `z` uses degrees, not radians.**
- **LiDAR positioning — wall fitting (current)**: ROS topic `/scan` (`sensor_msgs/LaserScan`). **Wall straight-line modeling via RANSAC**: `fit_walls()` in `utils/wall_positioning.py` downsamples the scan, detects corners via Cartesian gap analysis, splits points into 4 wall groups, fits each with RANSAC, and returns perpendicular distances to the front/right/rear/left walls plus yaw angle from the front wall's normal vector. Offset is 0 (raw distances = absolute coordinates). See "Wall positioning system" below.
- **LiDAR positioning — raw beam (legacy, for target correction only)**: Front beam = `ranges[n//2]` → X. Side beam = `ranges[n//4]` → Y. Rear beam = `ranges[n-2]`. Left beam = `ranges[n*3//4]`. Still used by `_get_beam()` in `run.py` for target-point correction beam readings and in `target_points.py` for recording correction coordinates.
- **Coordinate system**: X decreases toward the front wall, Y increases away from the right wall. The car's position is `(前墙距离, 右墙距离)` — distance from front wall and right wall respectively.
- **Angle detection**: `fit_walls()` returns `yaw` (degrees) from the front wall's fitted normal vector. Positive = car nose points left. `/SyncYaw` rotates to zero this angle. The old `getsum()`/`arccos` method is no longer used for yaw.
- **Movement feedback**: Commands move the car, then `/MoveOnlyX` and `/MoveOnlyY` poll wall-based `getx()`/`gety()` for LiDAR closed-loop correction (0.08m tolerance, up to 5 retries).

### Wall positioning system (墙壁直线建模)

The biggest architectural change from the original single-beam approach. `utils/wall_positioning.py` provides `fit_walls(scan, ...)`:

1. **Uniform downsampling** — ~200 samples from the full scan
2. **Obstacle filtering** — calls `get_obstacle_beam_mask()` to exclude obstacle-hit beams from wall fitting
3. **Cartesian conversion** — polar → (x, y), inf → NaN
4. **Corner detection** — finds 4 corners via largest Cartesian gaps between consecutive valid points + inf-gap scoring
5. **Wall segmentation** — splits points into 4 groups by corner indices (or by angle sector fallback if <4 corners found)
6. **RANSAC line fitting** — per-group, 80 iterations, 0.05m inlier threshold, SVD refinement on inliers. Returns `(a, b, c)` where `|c|` = perpendicular distance from origin to wall
7. **Wall labeling** — by centroid angle: front (±45°), left (45°–135°), rear (±135°–180°), right (-45°–-135°)

Returns: `{"前墙": dist_m, "右墙": dist_m, "后墙": dist_m, "左墙": dist_m, "yaw": deg}`. Failed walls return `None`.

**Critical detail — wall fitting uses obstacle-filtered points**: obstacles are excluded so they don't distort the wall line. This means obstacle detection must work for positioning to be accurate.

**50ms cache**: `_read_walls()` (in `run.py` and `CarControlServiceFlask.py`) and `_read_position()` (in `path_planner.py`) cache the last `fit_walls()` result for 50ms to avoid re-reading `/scan` when multiple functions call positioning in quick succession. This is intentionally duplicated across modules (same reason as the old LiDAR duplication — ROS topic contention).

### Directory structure

```
car/
├── config.yaml              # All tunable parameters
├── controllers/             # Control & entry points
│   ├── CarControlServiceFlask.py   # Flask server (port 5000), ROS node, chassis TCP
│   ├── CarController.py            # Remote HTTP client library (not used by pipeline)
│   └── run.py                      # Main entry point (interactive + --external modes)
├── entities/                # Business logic
│   ├── obstacle_detector.py        # 5-stage obstacle detection
│   ├── path_planner.py             # A* path planning
│   ├── forbidden_zones.py          # Forbidden zone CLI + geometry checks
│   ├── forbidden_zones.json        # Forbidden zone data
│   ├── target_points.py            # Target point recording (6 points)
│   ├── target_points.json          # Target point data
│   └── action_for_each_target.py   # Per-point action sequences
├── utils/                   # Utilities
│   ├── config_loader.py            # Custom YAML config loader
│   ├── lidar_utils.py              # getsum() — averaged front+rear wall distance (legacy API)
│   ├── wall_positioning.py         # RANSAC wall-fitting positioning (current)
│   └── visualize_path.py           # Path visualization → PNG
├── wall_positioning_test.py        # Wall positioning standalone test
├── actions.py                      # Old action definitions (unused, kept for reference)
└── path_simulator.py               # Offline path planning simulator (matplotlib GUI)
```

**Note**: The `tests/` directory referenced in older docs does not exist. The only standalone test is `wall_positioning_test.py`.

### Core modules

| Module | Role |
|--------|------|
| `config.yaml` + `utils/config_loader.py` | All parameters in one place; `cfg.xxx.yyy` dot access |
| `controllers/CarControlServiceFlask.py` | Flask server (port 5000), ROS node, chassis TCP. 9 REST endpoints. Uses wall fitting for all positioning. |
| `controllers/CarController.py` | Remote HTTP client library with retry + TaskId sync. **Not used by the main pipeline** — `run.py` has its own `_Client` class. |
| `controllers/run.py` | **Main entry point**. Interactive loop (1~6=target, 7=start, Q=quit) or `--external` mode (HTTP API on port 6000 for airplane/arm coordination). Uses wall fitting for positioning, raw beams for target correction. |
| `utils/wall_positioning.py` | **Current positioning system**. `fit_walls()` — RANSAC wall line fitting from LiDAR scan. Returns 4 wall distances + yaw. Used by server, run.py, and path_planner. |
| `entities/obstacle_detector.py` | 5-stage obstacle detection: analyze → jump detect → expand → cluster → filter. Also provides `get_obstacle_beam_mask()` for wall fitting. |
| `entities/path_planner.py` | A* path planning. Avoids obstacles, forbidden zones, walls. Self-loads zones. Uses wall fitting for car position. |
| `entities/forbidden_zones.py` | Forbidden zone CLI + geometry checks (`point_in_forbidden`, `segment_crosses_forbidden`). Saves to `forbidden_zones.json`. |
| `entities/target_points.py` | Interactive 6-point recording. Each point has `plan` (front+right beams) and optionally `correct` (point-specific beams). Saves to `target_points.json`. |
| `entities/action_for_each_target.py` | Per-point sequential action list (correct / rotate / stay / move_rel). |
| `utils/lidar_utils.py` | `getsum()` — averaged front+rear **wall** distance. Server's `getsum_y()` for left+right wall sanity checks. |
| `utils/visualize_path.py` | Matplotlib-based path visualization. Draws room, walls, forbidden zones, obstacles, path waypoints, correction points. Outputs PNG to `/tmp/path_plan.png`. |
| `wall_positioning_test.py` | Standalone wall positioning test. Compares wall-fitting distances vs traditional single-beam readings. Requires ROS + LiDAR. |
| `path_simulator.py` | Offline interactive simulator. Click to place start/goal/obstacles/zones, see A* path + correction points in real time. No ROS/LiDAR needed. |

### Dual positioning system

The codebase now has two positioning systems running side by side:

| System | Used by | Method | Purpose |
|--------|---------|--------|---------|
| **Wall fitting** | Server, run.py (navigation), path_planner | `fit_walls()` → RANSAC wall distances + yaw | Car position, movement feedback, SyncYaw, sanity checks |
| **Raw beams** | run.py (correction), target_points.py | `_get_beam(index)` → single beam distance | Target-point correction beam readings (stored in JSON) |

The wall-fitting system is the **primary** positioning method. Raw beams are kept only for backward compatibility with existing `target_points.json` correction data. The old `_CORRECT_BEAMS` table in `run.py` uses raw beam indices for each point's correction coordinates.

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
| 7 | front | right | **rear** | right | xy |

- **Plan coordinates**: Always front beam (`n//2`) for X, right beam (`n//4`) for Y. These are wall-based now for the main pipeline, but `target_points.py` still records raw beam values.
- **Correction coordinates**: Raw beam distances per the table above. Inversion (`4.5 - rear`, `8.8 - left`) happens at correction time in `run.py:_do_target_correction()`.

Point 7 ("return to start") correction beams use the start position's recorded beam distances (`_start_correct_beams` captured at startup), not stored JSON values.

### Flask REST endpoints

| Endpoint | Key detail |
|----------|-----------|
| `/Move` | X+Y LiDAR closed-loop using wall-based `getx()`/`gety()`, 0.08m tolerance |
| `/MoveRelative` | One-shot chassis command, no feedback. Supports optional `step` parameter for segmented moves. |
| `/MoveOnlyX` | X one-shot (wall-based `getx()`), then Y closed-loop (only when `correct=true` + `readings_sane()`). |
| `/MoveOnlyY` | X one-shot (wall-based `getx()`), then Y closed-loop (only when `correct=true` + `readings_sane()`). |
| `/MoveLongDistance` | X split into 3 segments, 0.00001m tolerance (too tight — known issue) |
| `/Circle` | Takes **degrees** (not radians despite param name `rad_z`) |
| `/SyncYaw` | **Wall-normal based**: reads front wall yaw from `fit_walls()`, rotates in configurable steps. Threshold: `sync_yaw_threshold_deg` (default 10°). Max iterations: `sync_yaw_max_iterations` (default 6). |
| `/SetBaseline` | **Deprecated** — no-op on server. Wall fitting doesn't need a baseline. Kept for backward compatibility with old clients. |
| `/ShutDown` / `/Reset` | Cleanup / TaskId reset |

### Task ID protocol

`TaskId` must equal `CurrentTaskID + 1`. Mismatch → server returns `expectedTaskId`, client re-syncs and replays. `/Reset` zeroes the counter.

### Module dependency chain

```
obstacle_detector  ←  forbidden_zones (filter obstacles in zones)
       ↑                        ↑
wall_positioning   ←  obstacle_detector (get_obstacle_beam_mask for filtering)
       ↑                        ↑
path_planner  ←  forbidden_zones (avoid) + wall_positioning (car position)
       ↑
run  ←  target_points + action_for_each_target + wall_positioning + Flask endpoints
       ↑
CarControlServiceFlask  ←  wall_positioning (all LiDAR reads)
```

**Integration gap**: The Flask server has no awareness of obstacles or planned paths — all intelligence lives in the pipeline client (`run.py`).

### Path planning degradation strategy

When A* fails with default obstacle margins, `run.py` retries with progressively smaller margins in 4 levels:

| Level | obstacle_margin | grid_expand | Description |
|-------|----------------|-------------|-------------|
| 默认   | base × 1.0 | base × 1.0 | Default |
| 降级1  | base × 0.6 | base × 0.5 | Reduced clearance |
| 降级2  | base × 0.3 | base × 0.2 | Tight clearance |
| 降级3  | base × 0.1 | 0.0 | Minimal — fallback to "先X后Y(安全无解)" if still unsolvable |

When start→end is within 15° of X or Y axis, `cost_risk_weight=1.0` (prefer paths farther from obstacles); otherwise `cost_risk_weight=0.0` (shortest path). A* also falls back from axis-aligned to diagonal search before giving up.

### LiDAR fallback (`_last_good_position`)

When LiDAR readings are unreliable (wall fitting returns `None`, or sanity check fails), the pipeline uses the last successfully reached target coordinates as the current position estimate, allowing movement to continue without LiDAR feedback.

### Coordinate system inversion

LiDAR and chassis coordinate systems are **opposite directions**:
- Moving toward front wall → LiDAR X **decreases**; chassis delta must be **negated** (`-dx`)
- Moving away from right wall → LiDAR Y **decreases**; chassis delta must be **negated** (`-dy`)

This inversion is handled in `_move_segment()` by passing `-dx`/`-dy` to `/MoveRelative`.

### Correction flow

**Waypoint correction** (during movement):
1. `SyncYaw` — wall-normal-based angle correction via Flask endpoint
2. Coordinate correction — wall-based LiDAR vs expected position, with sanity check (`front+rear ≈ room_width`, `right+left ≈ room_height`, tolerance configurable in `config.yaml`)

**Target correction** (after arrival, if `correct` in actions):
- Points with correction coordinates → compare stored raw beam distances vs current raw beam readings, convert to absolute coordinates, compute chassis displacement
- Points without → fall back to wall-based planning coordinate closed-loop

Correction points are chosen by `plan_correction_points()` — **all points (start, end, intermediate) must pass safety check** (no obstacle in same X/Y column within `correction_corridor_width`). No point gets automatic inclusion.

### Sanity check for LiDAR readings

Two layers of sanity check guard against bad LiDAR corrections, both now using wall distances:

1. **Client-side** (`run.py:_lidar_readings_sane()`): before `_do_correction()`, verifies `front+rear ≈ room_width` and `right+left ≈ room_height`. Aborts correction on failure.

2. **Server-side** (`CarService.readings_sane()`): before the closed-loop in `/MoveOnlyX` and `/MoveOnlyY`, verifies the same wall-distance sums. On failure, the LiDAR closed-loop is skipped but the initial one-shot chassis move still counts as success.

The client passes `correct=true` to `/MoveOnlyX`/`/MoveOnlyY` only at designated correction points; all other segments use `correct=false` to skip the server-side closed-loop entirely.

## Key commands

```bash
# Main entry — interactive task loop (uses config.yaml)
python3 controllers/run.py
# Input 1~6: go to target. 7: return to start. Q: quit.

# External mode — HTTP API for airplane/arm coordination
python3 controllers/run.py --external
# Listens on port 6000: POST /go (point), POST /continue (arm), GET /status

# Set up forbidden zones (no-go areas)
python3 entities/forbidden_zones.py
# s=setup, l=list, a=add, d=delete

# Record target point coordinates (6 points)
python3 entities/target_points.py
# s=setup, l=list

# A* path planning (standalone, requires ROS + LiDAR)
python3 entities/path_planner.py <target_x> <target_y> [num_waypoints]

# Wall positioning test (requires ROS + LiDAR)
python3 wall_positioning_test.py

# Offline path planning simulator (no ROS/LiDAR needed, matplotlib GUI)
python3 path_simulator.py
```

## Key conventions

- **Python 3** only. System Python with ROS packages, no virtualenv.
- **Sensor offset is 0** — raw LiDAR/wall distances used directly as absolute coordinates.
- **LiDAR inf → re-sample**. Pattern duplicated across files that read LiDAR.
- **Chassis rotation `z` is degrees**, not radians. `/Circle` param name is misleading (`rad_z`).
- **All parameters in `config.yaml`**. When adding new tunables, put them there, not as module constants.
- **Forbidden zones** use absolute coordinates. Must re-record with `forbidden_zones.py` if old JSON has relative format with `origin_x`/`origin_y`.
- **Chassis deltas are negated** from LiDAR deltas. LiDAR and chassis coordinate axes point in opposite directions — `_move_segment()` sends `-dx`/`-dy` to `/MoveRelative`.
- **Wall fitting is the primary positioning system**. `fit_walls()` is the canonical way to read car position. Raw beam `_get_beam()` is legacy, used only for target correction compatibility.
- **50ms LiDAR cache is intentionally duplicated** across `run.py`, `CarControlServiceFlask.py`, and `path_planner.py`. Same rationale as the old getx/gety duplication: sharing via import would cause ROS topic `/scan` contention.

## Known issues

### `/MoveLongDistance` tolerance too tight

Uses 0.00001m tolerance (vs 0.08m for all others). LiDAR noise can cause infinite retry loops.

### Obstacle detection intermittency

Objects near detection thresholds (JUMP_THRESHOLD=0.3, CLUSTER_MIN_BEAMS=3) are sometimes missed. Sensitive to LiDAR noise and angle.

### Duplicated LiDAR reading + cache (intentional)

Wall-fitting read functions and their 50ms caches are duplicated across `controllers/CarControlServiceFlask.py`, `entities/path_planner.py`, and `controllers/run.py`. **This is intentional** — sharing a single LiDAR reader via import would cause multiple modules to contend for the same ROS topic `/scan`, interfering with chassis communication. Changes to sampling strategy or cache behavior must be applied to all copies.

### Correction may trigger when LiDAR is blocked

`plan_correction_points` uses column/row clearance check but doesn't account for the fan-shaped beam pattern. A correction point may be marked safe even if the LiDAR beam hits an obstacle at an angle.

### Flask server unaware of obstacles

All planning intelligence is client-side. Flask endpoints execute moves blindly — if the client sends a move into an obstacle, the server won't stop it.

### Wall fitting can fail with insufficient valid points

If `<20` valid points remain after obstacle filtering, `fit_walls()` returns all `None` values. This triggers the `_last_good_position` fallback in the pipeline.

### Dual positioning systems may diverge

Wall-based positioning and raw-beam positioning can give different results when walls are not perfectly straight or perpendicular. This can cause discrepancies between navigation (wall-based) and target correction (raw-beam-based).
