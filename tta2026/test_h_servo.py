"""测试 H 检测与伺服对齐。起飞 → 找 H → 对齐 → 降落。"""
import argparse, time
from Controllers.DroneNavigator import DroneNavigator
from Utils.JsonHelper import JsonHelper

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    config = JsonHelper.load_json(args.config)
    nav = DroneNavigator(config)

    print('=== 起飞 ===')
    nav.drone.reset()
    time.sleep(1)
    if not nav.takeoff():
        print('起飞失败')
        return
    time.sleep(8)

    print('=== 云台朝下，搜索 H ===')
    nav._rotate_gimbal_with_recovery(-90)

    for attempt in range(5):
        frame = nav._capture_fresh_frame(settle=3.0)
        if frame is None:
            continue
        all_det = nav.detect_all(frame)
        h = all_det["h_candidate"]
        print(f"  第{attempt}轮: H={h['label'] if h else 'none'}")
        if h is None:
            if attempt > 0:
                ox, oy = nav._next_spiral_offset(attempt)
                nav.move_to(nav.drone.state['x'] + ox, nav.drone.state['y'] + oy, nav.drone.state['z'])
            continue
        for servo_i in range(5):
            moved = nav._servo_toward_h(h['box'], frame.shape)
            if not moved:
                print(f"  H 已居中 (迭代{servo_i}次)")
                break
            time.sleep(2)
            frame = nav._capture_fresh_frame(settle=2.0)
            if frame is None:
                break
            all_det = nav.detect_all(frame)
            h = all_det["h_candidate"]
            if h is None:
                break
        break
    else:
        print('未找到 H')

    print('=== 降落 ===')
    nav._rotate_gimbal_with_recovery(0)
    time.sleep(1)
    nav.land()

if __name__ == '__main__':
    main()
