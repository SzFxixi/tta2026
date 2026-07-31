import sys
import os
import json
import time

# 将工作目录加入模块搜索路径，确保能够导入 workspace 内的包
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Clients.DroneControlClient import DroneControlClient

if __name__ == '__main__':
    cfg = json.load(open('configs/rescue_config.json', 'r', encoding='utf-8'))
    # 使用配置中的 drone 部分，但强制模拟（不实际发送网络请求）
    drone_cfg = cfg.get('drone', {})
    drone_cfg['enabled'] = False
    drone = DroneControlClient(drone_cfg)

    waypoints = cfg.get('waypoints', [])
    print('模拟运行：将按配置中的航点顺序调用 move_to（drone.enabled=False，安全）。')
    for wp in waypoints:
        name = wp.get('name')
        x = float(wp.get('x'))
        y = float(wp.get('y'))
        z = float(wp.get('z'))
        print('-' * 60)
        print(f'前往 {name}: ({x},{y},{z})')
        ok = drone.move_to(x, y, z)
        print(f'返回: {ok}; 当前状态: {drone.state}')
        time.sleep(0.5)
    print('模拟运行结束')
