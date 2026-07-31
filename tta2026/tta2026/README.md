# 新代码/ 无人机四点位 YOLO 扫描

## 当前文件说明

- `Main.py`：入口脚本，执行四点位巡检或单张图像检测。
- `Controllers/DroneNavigator.py`：无人机巡检器，负责起飞、移动、拍照、YOLO 检测和结果保存。
- `Clients/DroneControlClient.py`：基础无人机运动 SDK，支持模拟模式和 HTTP 控制。
- `Utils/CameraSource.py`：摄像头、视频文件或图片目录输入源。
- `Utils/JsonHelper.py`：JSON 读取/写入助手。
- `Entities/Waypoint.py`：航点数据结构。
- `drone_config_template.json`：配置模板，包括 YOLO、相机、无人机控制和航点设置。

## 运行说明

### 使用图片目录（默认）
```bash
python3 Main.py --config config_scan.json
```

### 使用 RTSP/RTMP 流（车载盒子推流）
修改 `config_scan.json` 中的 `camera_source`：
```json
{
  "camera_source": "rtsp://192.168.1.100:554/stream",
  "camera_ffmpeg_opts": {
    "rtsp_transport": "tcp"
  }
}
```

### 只检测单张图片
```bash
python3 Main.py --config config_scan.json --image path/to/image.jpg
```

## 配置说明

- `yolo_weights`：YOLO 权重文件路径，例如 `yolov8n.pt`。
- `confidence`：检测置信度阈值。
- `output_folder`：保存检测结果的文件夹。
- `camera_source`：摄像头索引（如 `0`）、视频文件路径、图片目录，或 RTSP/RTMP 流 URL（如 `rtsp://192.168.1.100:554/stream`）。
- `camera_ffmpeg_opts`：FFMPEG 流参数（仅在使用流时生效），例如 `"rtsp_transport": "tcp"`。
- `drone.enabled`：是否启用真实控制；默认 `false` 为模拟模式。
- `drone.control_url`：真实控制服务器地址，例如 `http://127.0.0.1:5000/drone_command`。
- `waypoints`：四点位巡检坐标列表，每个点包含 `name`、`x`、`y`、`z`。
- `locator.enabled`：是否启用 ArUco 标记定位；启用后，代码将在搜索时等待检测到指定标记点。
- `locator.camera_intrinsics`：摄像头内参数据，支持直接写入 `camera_matrix` 与 `distortion_coefficients`，也支持文件路径。
- `locator.dictionary`：ArUco 字典名，比如 `DICT_4X4_50`。
- `locator.marker_size`：标记实际边长（单位：米）。
- `locator.target_id`：期望检测的 ArUco 标记 ID。
- `locator.coefficients`：PnP 翻译向量的系数校正，默认 `[1.0, 1.0, 1.0]`。

- `search_max_attempts`：在近似航点周围进行本地搜索的最大尝试轮数（默认 `8`）。
- `search_step`：每次本地微调移动的步长（单位与航点坐标一致，默认 `0.2`）。

注意：代码负责在近似点位周围以小步长移动并拍照；图像等级判断（你的 YOLO 逻辑）应通过替换或扩展 `DroneNavigator.detect_frame()` / 在上层处理检测结果来实现，当前框架只在检测到任意对象时视为“找到目标”。

## 说明

- 当前代码已经使用 YOLO 对四个巡检点拍摄图片进行检测，并将检测结果保存为标注图像。
- 如果你需要本地 HTTP 控制服务器，请继续使用 `drone_control_server.py` 并把 `drone.enabled` 设置为 `true`。
