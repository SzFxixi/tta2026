#!/bin/bash
# 启动顺序：roscore → lslidar → Flask
# 日志保存在 ~/car2_logs/

LOG_DIR=~/car2_logs
CAR_DIR=~/catkin_ws/src/lsx10/car
mkdir -p $LOG_DIR

source /opt/ros/noetic/setup.bash
if [ -f ~/catkin_ws/devel/setup.bash ]; then
    source ~/catkin_ws/devel/setup.bash
fi

echo "[1/3] 启动 roscore..."
nohup roscore > $LOG_DIR/roscore.log 2>&1 &
sleep 3

echo "[2/3] 启动 lslidar..."
nohup roslaunch lslidar_driver lslidar_net.launch > $LOG_DIR/lslidar.log 2>&1 &
sleep 5

echo "[3/3] 启动 CarControlServiceFlask..."
nohup python3 $CAR_DIR/controllers/CarControlServiceFlask.py > $LOG_DIR/flask.log 2>&1 &

echo ""
echo "全部启动完成！查看日志："
echo "  tail -f ~/car2_logs/roscore.log"
echo "  tail -f ~/car2_logs/lslidar.log"
echo "  tail -f ~/car2_logs/flask.log"
echo ""
echo "停止所有服务："
echo "  pkill -f 'roscore|lslidar_net|CarControlServiceFlask'"

if [ "$1" = "interactive" ]; then
    echo ""
    echo "[interactive] 启动键盘交互模式..."
    cd $CAR_DIR
    python3 controllers/run1.py
fi
