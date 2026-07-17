# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TTA2026 智慧救援 — autonomous drone competition system. Drone inspects 4 rescue points using YOLO vision, recognizes disaster grades (1/2/3) via H-marker visual alignment, then delivers supplies to the target point. Car (ground robot) coordinates with drone for loading/unloading.

## Commands

**Scan mission (drone only):**
```bash
python Main.py --config configs/rescue_config.json --mission scan
```

**Full delivery mission (drone + car):**
```bash
python Main.py --config configs/rescue_config.json --mission delivery
```

**Single waypoint test:**
```bash
python Main.py --config configs/rescue_config.json --waypoint "救援点1"
```

**H servo alignment test:**
```bash
python test_h_servo.py --config configs/rescue_config.json
```

**Stream test (no flight, camera only):**
```bash
python Main.py --config configs/rescue_config.json --stream
```

**Single image detection:**
```bash
python Main.py --config configs/rescue_config.json --image path/to/image.jpg
```

**Half-mission test (starts from loading area):**
```bash
python Main.py --config configs/rescue_config.json --half "救援点1"
```

**Full competition flow with MasterController (drone + car orchestration):**
```bash
python MasterController.py --config configs/rescue_config.json --car-url http://<car-ip>:6001
```

Config files: `configs/rescue_config.json` (main), `configs/config_scan.json` (scan only), `configs/config_drone.json`, `configs/rescue_config_pi.json` (onboard Pi).

## Architecture

### Layer stack (top-down)
```
Main.py ──► RescueController / MasterController  (mission orchestration)
               │
               ├──► DroneNavigator           (flight + YOLO detection + servo)
               │       ├──► DroneControlClient   (HTTP PUT → PlaneServer PSDK)
               │       ├──► CameraSource         (cv2/ffmpeg stream input)
               │       ├──► HAngleDetector       (edge/Hough angle measurement)
               │       └──► MathHelper           (coordinate transforms)
               │
               └──► CarController / CarHttpClient  (HTTP to car Flask service)
```

### Three YOLO models, loaded separately
- `yolo_weights` → primary detection model (`yolov8n.pt`)
- `yolo_weights_h` → H-marker detector (`yolov8n_h.pt`)
- `yolo_weights_grade` → disaster grade classifier (`yolov8n_grade_new.pt`)

`detect_all(frame)` runs both H and grade models on every frame. Grade label is mapped to level (1/2/3) via `grade_mapping` in config.

### Visual servo flow (H alignment)
1. `detect_all(frame)` → find H bbox and grade
2. `_servo_toward_h(h_box, frame_shape)` → pixel offset → meters-per-pixel (based on `h_marker_size`) → `drone.move_to(x+dx, y+dy, z)`
3. Repeat until centered (offset < `servo_tolerance_preview`) or max iterations
4. `correct_h_rotation(frame, h_box)` → Hough lines measure angle deviation → `drone.rotate_yaw(correction)`
5. If H not found, ascend by `h_search_step_height` up to `h_search_max_height` and retry

### Coordinate systems
- `waypoint_frame`: `"world"` or `"body"`. Body-frame waypoints are rotated by takeoff yaw via `MathHelper.rotate_axis()`.
- `control_frame`: `"world"` or `"body"`. World-frame sends raw dx/dy to PlaneServer; body-frame rotates by drone yaw.
- After H servo at home point, coordinate origin is asserted: `drone.state['x'] = 0.0; drone.state['y'] = 0.0`.

### CameraSource modes
- USB/webcam: integer source (`"0"`)
- Video file: path to file
- Image directory: path to folder (loops)
- RTSP/RTMP pull: URL string, uses `cv2.CAP_FFMPEG` with optional `camera_ffmpeg_opts`
- RTMP listen mode: `"listen": true` — ffmpeg subprocess listens for drone push stream, decodes JPEG frames via pipe

### PlaneServer communication (DroneControlClient)
- All commands: HTTP PUT to `http://<ip>:<port>/<Endpoint>` with JSON body
- `taskId` counter required by server; auto-syncs on taskId errors via `/GetTaskId` + `/Reset`
- Retry: exponential backoff (0.5s–5s), up to `max_retries` times
- `Translate` command: segmented for long distances (`max_translate_ms` chunks) to avoid HTTP timeout

### Key config parameters (in `configs/rescue_config.json`)
- `landing_offset` — forward shift before landing (compensates for H-to-platform offset)
- `back_offset` — reverse shift to undo landing forward move before next takeoff
- `home_point` — calibration point for coordinate origin assertion
- `h_angle_sign` / `h_angle_scale` — flip/scale H rotation correction direction
- `servo_tolerance_preview` — pixel offset threshold for "centered" (meters)
- `h_search_max_height` / `h_search_step_height` — vertical search bounds for H
- `target_grade` / `target_labels` — priority rules for selecting delivery target waypoint
