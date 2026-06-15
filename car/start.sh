#!/bin/bash
# 一键启动智能小车服务
# 顺序：roscore → lslidar → Flask

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CAR_DIR="$SCRIPT_DIR/../car"
LOG_DIR="$HOME/car2_logs"
mkdir -p "$LOG_DIR"

echo "=== 启动 roscore ==="
roscore &
sleep 3

echo "=== 启动 lslidar 驱动 ==="
rosrun lslidar_driver lslidar_net &
sleep 2

echo "=== 启动 Flask 控制服务 ==="
cd "$CAR_DIR"
python3 controllers/CarControlServiceFlask.py &

echo ""
echo "全部服务已启动"
echo "  日志: $LOG_DIR"
echo "  停止: pkill -f 'roscore|lslidar_net|CarControlServiceFlask'"
wait
