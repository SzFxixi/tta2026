# TTA2026 智慧救援 — 测试指南

## 一、环境准备

###  无人机端（机载盒子 T507-H）

SSH 登录：
```bash
ssh root@192.168.31.172
# 密码: koalastudio
```

启动 PlaneServer（飞控服务）：
```bash
# 前台运行（可看日志）
pkill PlaneServer && /home/forlinx/PSDK/build/bin/PlaneServer
```

若日志没出现，再运行一遍这个：
```
/home/forlinx/PSDK/build/bin/PlaneServer
```

## 测试命令


### 仅巡检（任务一）— 实机飞行

```bash
python Main.py --config configs/rescue_config.json --mission scan
# 或简写:
python Main.py --config configs/rescue_config.json
```

流程：起飞 → 依次巡检 4 个救援点 → 装货区 H 对齐 → 降落
输出：`output/rescue_levels.csv`

### 巡检 + 投送（完整任务）

```bash
python Main.py --config configs/rescue_config.json --mission delivery
```

流程：巡检 → 装货区降落 → 等小车装货 → 起飞回原点 → 飞目标点 → H 对齐降落 → 等小车取货 → 返航

### H 伺服对齐专项测试

```bash
python test_h_servo.py --config configs/rescue_config.json
```

流程：起飞 → 云台朝下 → 找 H → 迭代伺服居中 → 降落

---


## 小车测试（联调时）（AI生成的）

小车上启动 Flask 服务（需 ROS 环境）：
```bash
cd ~/car/
bash start.sh
```

验证小车服务：
```bash
curl -X POST http://<小车IP>:5000/Reset -H "Content-Type: application/json" -d "{\"TaskId\": 1}"
```

配置文件加：
```json
"car": { "enabled": true, "ip": "<小车IP>", "port": 5000 }
```
