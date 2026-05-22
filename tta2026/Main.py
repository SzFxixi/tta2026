import argparse

from Controllers.DroneNavigator import DroneNavigator
from RescueController import RescueController
from Utils.JsonHelper import JsonHelper


def main() -> None:
    parser = argparse.ArgumentParser(description='智慧救援 — 无人机巡检+等级识别')
    parser.add_argument('--config', type=str, required=True, help='配置文件 JSON 路径')
    parser.add_argument('--image', type=str, default=None, help='单张图片检测（不飞行）')
    args = parser.parse_args()

    config = JsonHelper.load_json(args.config)

    if args.image:
        navigator = DroneNavigator(config)
        result = navigator.detect_image_file(args.image)
        print('检测结果:', result)
        return

    controller = RescueController(config)
    controller.execute_scan_mission()


if __name__ == '__main__':
    main()
