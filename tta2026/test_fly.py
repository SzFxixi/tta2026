from Clients.DroneControlClient import DroneControlClient
import time

c = DroneControlClient({
    'enabled': True, 'ip': '192.168.31.172', 'port': 18080,
    'speed': {'translate': 0.5, 'rotate': 20},
    'threshold': {'translate': 200, 'rotate': 300},
    'initial_position': {'x': 0, 'y': 0, 'z': 0, 'yaw': 0},
    'max_retries': 3,
})

print('=== 起飞 ===')
c.takeoff()
time.sleep(3)

print()
print('=== 测试平移 0.5m ===')
print(f'移动前 state: {c.state}')
c.move_to(0.5, 0.0, 1.2)
print(f'移动后 state: {c.state}')

time.sleep(2)

print()
print('=== 降�===')
c.land()