import argparse

from Controllers.CarController import CarController
from Controllers.DroneNavigator import DroneNavigator
from RescueController import RescueController
from Utils.JsonHelper import JsonHelper


def main() -> None:
    parser = argparse.ArgumentParser(description='智慧救援 — 无人机巡检+等级识别')
    parser.add_argument('--config', type=str, required=True, help='配置文件 JSON 路径')
    parser.add_argument('--image', type=str, default=None, help='单张图片检测（不飞行）')
    parser.add_argument('--stream', action='store_true', help='实时拉流测试（H + 等级检测画面，不飞行）')
    parser.add_argument('--mission', type=str, default='scan', choices=['scan', 'delivery'],
                        help='任务模式：scan=仅巡检，delivery=巡检+等待小车信号+送达目标+返回原点')
    args = parser.parse_args()

    config = JsonHelper.load_json(args.config)

    if args.stream:
        navigator = DroneNavigator(config)
        navigator.stream_test()
        return

    if args.image:
        navigator = DroneNavigator(config)
        result = navigator.detect_image_file(args.image)
        print('检测结果:', result)
        return

    controller = RescueController(config)
    controller.set_car_controller(CarController(config))

    if args.mission == 'delivery':
        controller.execute_delivery_mission()
    else:
        controller.execute_scan_mission()


if __name__ == '__main__':
    main()
