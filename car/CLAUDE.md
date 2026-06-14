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

**Note**: `start.sh` now points to `~/smart_car/scripts/car/controllers/CarControlServiceFlask.py`.

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
- **LiDAR**: ROS topic `/scan` (`sensor_msgs/LaserScan`). Position by reading front beam (middle of range → X) and side beam (quarter of range → Y). Offset is 0 (raw distances = absolute coordinates).
- **Coordinate system**: X decreases toward the front wall, Y increases away from the right wall.
- **Movement feedback**: Commands move the car, then `/MoveOnlyX` and `/MoveOnlyY` poll `getx()`/`gety()` for LiDAR closed-loop correction (0.08m tolerance, up to 5 retries).
- **Angle detection**: `getsum()` = average of 5 front+rear beam samples. Rotation angle = `arccos(baseline / current_sum)`.

### Directory structure

```
car/
├── config.yaml              # 全局参数
├── start.sh                 # 一键启动
├── controllers/             # 控制与入口
│   ├── CarControlServiceFlask.py   # Flask 服务端
│   ├── CarController.py            # 远程客户端库
│   └── run.py                      # 主程序入口
├── entities/                # 业务逻辑
│   ├── obstacle_detector.py        # 障碍物检测
│   ├── path_planner.py             # A* 路径规划
│   ├── forbidden_zones.py          # 禁区管理
│   ├── forbidden_zones.json        # 禁区数据
│   ├── target_points.py            # 目标点管理
│   ├── target_points.json          # 目标点数据
│   └── action_for_each_target.py   # 点位动作配置
├── utils/                   # 工具
│   ├── config_loader.py            # 配置加载器
│   ├── lidar_utils.py              # LiDAR 工具函数
│   └── visualize_path.py           # 路径可视化（PNG 输出）
├── tests/                   # 测试
│   ├── test_obstacle_detector.py
│   └── test_path_planner.py
└── path_simulator.py        # 离线路径规划模拟器（matplotlib 交互）
```

**⚠️ Root-level duplicates removed** (Jun 2026). Canonical versions live in `controllers/`, `entities/`, `utils/`, `tests/`. Root only retains standalone/unclassified files: `actions.py` (old), `LidarTest.py` (prototype), `path_simulator.py`, `test_full_pipeline.py`, `wall_positioning_test.py`.

### Core modules

| Module | Role |
|--------|------|
| `config.yaml` + `utils/config_loader.py` | All parameters in one place; `cfg.xxx.yyy` dot access |
| `controllers/CarControlServiceFlask.py` | Flask server (port 5000), ROS node, chassis TCP. 9 REST endpoints. |
| `controllers/CarController.py` | Remote HTTP client library with retry + TaskId sync (not used by pipeline) |
| `controllers/run.py` | **Main entry point**. Interactive loop (1~6=target, 7=start, Q=quit) or `--external` mode (HTTP API on port 6000 for airplane/arm coordination). |
| `entities/obstacle_detector.py` | 5-stage obstacle detection: analyze → jump detect → expand → cluster → filter |
| `entities/path_planner.py` | A* path planning. Avoids obstacles, forbidden zones, walls. Self-loads zones. |
| `entities/forbidden_zones.py` | Forbidden zone CLI + geometry checks. Saves to `forbidden_zones.json`. |
| `entities/target_points.py` | Interactive 6-point recording. Dual coords per point. Saves to `target_points.json`. |
| `entities/action_for_each_target.py` | Per-point sequential action list (correct / rotate / stay / move_rel). |
| `utils/lidar_utils.py` | `getsum()` — averaged front+rear beam distance for angle diagnostics. Server also has `getsum_y()` for left+right sanity checks. |
| `utils/visualize_path.py` | Matplotlib-based path visualization. Draws room, walls, forbidden zones, obstacles, path waypoints, correction points. Outputs PNG. |
| `path_simulator.py` | Offline interactive simulator. Click to place start/goal/obstacles/zones, see A* path + correction points in real time. No ROS/LiDAR needed. |

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

Correction values in JSON are **raw beam distances**, not inverted. Inversion (`4.5 - rear`, `8.8 - left`) happens at correction time.

Point 7 ("return to start") correction beams are defined in `run.py`'s `_CORRECT_BEAMS` but use the start position's recorded beam distances (`_start_correct_beams`), not stored JSON values.

### Flask REST endpoints

| Endpoint | Key detail |
|----------|-----------|
| `/Move` | X+Y LiDAR closed-loop, 0.08m tolerance |
| `/MoveRelative` | One-shot chassis command, no feedback |
| `/MoveOnlyX` | X one-shot, then Y closed-loop (only when `correct=true` + LiDAR readings sane). TaskId mismatch was FIXED |
| `/MoveOnlyY` | X one-shot, then Y closed-loop (only when `correct=true` + LiDAR readings sane) |
| `/MoveLongDistance` | X split into 3 segments, 0.00001m tolerance (too tight) |
| `/Circle` | Takes **degrees** (not radians despite param name `rad_z`) |
| `/SyncYaw` | Angle correction: `arccos(baseline/current)` → rotate in 2.5°~10° steps. Threshold: 10° (configurable via `sync_yaw_threshold_deg`) |
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

When LiDAR readings are unreliable (inf, or sanity check fails), the pipeline uses the last successfully reached target coordinates as the current position estimate, allowing movement to continue without LiDAR feedback.

### Coordinate system inversion

LiDAR and chassis coordinate systems are **opposite directions**:
- Moving toward front wall → LiDAR X **decreases**; chassis delta must be **negated** (`-dx`)
- Moving away from right wall → LiDAR Y **decreases**; chassis delta must be **negated** (`-dy`)

This inversion is handled in `_move_segment()` by passing `-dx`/`-dy` to `/MoveRelative`.

### Correction flow

**Waypoint correction** (during movement):
1. `SyncYaw` — angle correction via Flask endpoint
2. Coordinate correction — LiDAR vs expected position, with sanity check (front+rear ≈ baseline, right+left ≈ baseline, tolerance configurable in `config.yaml`)

**Target correction** (after arrival, if `correct` in actions):
- Points with correction coordinates → compare stored vs current raw beam distances, compute chassis displacement
- Points without → fall back to planning coordinate closed-loop

Correction points are chosen by `plan_correction_points()` — **all points (start, end, intermediate) must pass safety check**. No point gets automatic inclusion.

### Sanity check for LiDAR readings

Two layers of sanity check guard against bad LiDAR corrections:

1. **Client-side** (`run.py:_lidar_readings_sane()`): before `_do_correction()`, verifies `front+rear≈baseline` and `right+left≈baseline`. Aborts correction on failure.

2. **Server-side** (`CarService.readings_sane()`): before the closed-loop in `/MoveOnlyX` and `/MoveOnlyY`, verifies the same sums against the server's stored baselines (`car.distance`, `car.distance_y`). On failure, the LiDAR closed-loop is skipped but the initial one-shot chassis move still counts as success.

The client passes `correct=true` to `/MoveOnlyX`/`/MoveOnlyY` only at designated correction points; all other segments use `correct=false` to skip the server-side closed-loop entirely.

## Key commands

```bash
# Main entry — interactive task loop (uses config.yaml)
python3 controllers/run.py
# Input 1~6: go to target. 7: return to start. Q: quit.

# External mode — HTTP API for airplane/arm coordination
python3 controllers/run.py --external
# Listens on port 6000: POST /go (point), POST /continue (arm), GET /status

# Legacy pipeline (hardcoded config, no obstacle-aware correction)
python3 test_full_pipeline.py

# Set up forbidden zones (no-go areas)
python3 entities/forbidden_zones.py
# s=setup, l=list, a=add, d=delete

# Record target point coordinates (6 points)
python3 entities/target_points.py
# s=setup, l=list

# Standalone tests (require ROS + LiDAR)
python3 tests/test_obstacle_detector.py
python3 tests/test_path_planner.py

# A* path planning (standalone, requires ROS + LiDAR)
python3 entities/path_planner.py <target_x> <target_y> [num_waypoints]

# Offline path planning simulator (no ROS/LiDAR needed, matplotlib GUI)
python3 path_simulator.py
```

## Key conventions

- **Python 3** only. System Python with ROS packages, no virtualenv.
- **Sensor offset is 0** — raw LiDAR distances used directly as absolute coordinates.
- **LiDAR inf → re-sample**. Pattern duplicated across files that read LiDAR.
- **Chassis rotation `z` is degrees**, not radians. `/Circle` param name is misleading (`rad_z`).
- **All parameters in `config.yaml`**. When adding new tunables, put them there, not as module constants.
- **Forbidden zones** use absolute coordinates. Must re-record with `forbidden_zones.py` if old JSON has relative format with `origin_x`/`origin_y`.
- **Chassis deltas are negated** from LiDAR deltas. LiDAR and chassis coordinate axes point in opposite directions — `_move_segment()` sends `-dx`/`-dy` to `/MoveRelative`.

## Known issues

### `/MoveLongDistance` tolerance too tight

Uses 0.00001m tolerance (vs 0.08m for all others). LiDAR noise can cause infinite retry loops.

### Obstacle detection intermittency

Objects near detection thresholds (JUMP_THRESHOLD=0.3, CLUSTER_MIN_BEAMS=3) are sometimes missed. Sensitive to LiDAR noise and angle.

### Duplicated LiDAR reading functions (intentional)

`getx`/`gety`/`getsum`/`getsum_y` are duplicated across `controllers/CarControlServiceFlask.py`, `entities/path_planner.py`, `controllers/run.py`, and `tests/test_obstacle_detector.py`. **This is intentional** — sharing a single LiDAR reader via import would cause multiple modules to contend for the same ROS topic `/scan`, interfering with chassis communication. Each module reads LiDAR independently. Changes to sampling strategy or inf handling must be applied to all copies.

### Correction may trigger when LiDAR is blocked

`plan_correction_points` uses column/row clearance check but doesn't account for the fan-shaped beam pattern. A correction point may be marked safe even if the LiDAR beam hits an obstacle at an angle.

### Flask server unaware of obstacles

All planning intelligence is client-side. Flask endpoints execute moves blindly — if the client sends a move into an obstacle, the server won't stop it.
