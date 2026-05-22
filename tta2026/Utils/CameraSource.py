import os
from typing import Any, Dict, List, Optional, Tuple
import cv2


class CameraSource:
    """摄像头、视频文件、图片目录或流（RTSP/RTMP）读取工具。"""

    def __init__(self, source: str, ffmpeg_opts: Optional[Dict[str, Any]] = None, loop: bool = True):
        self.source = source
        self.ffmpeg_opts = ffmpeg_opts or {}
        self.capture = None
        self.image_files: List[str] = []
        self.image_index = 0
        self.loop = loop  # 图片目录模式：True=读完后循环, False=读完后返回 None
        self._init_source()

    def _init_source(self) -> None:
        # 尝试作为摄像头索引
        if self.source.isdigit():
            self.capture = cv2.VideoCapture(int(self.source))
            print(f"[CameraSource] 打开摄像头: {self.source}")
            return

        # 尝试作为文件或流（URL）
        if os.path.isfile(self.source):
            self.capture = cv2.VideoCapture(self.source)
            print(f"[CameraSource] 打开视频文件: {self.source}")
            return

        # 尝试作为目录
        if os.path.isdir(self.source):
            self.image_files = sorted(
                [os.path.join(self.source, f) for f in os.listdir(self.source) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            )
            print(f"[CameraSource] 图片目录: {self.source}, 共 {len(self.image_files)} 张")
            return

        # 尝试作为 RTSP/RTMP 流
        if self.source.lower().startswith(('rtsp://', 'rtmp://')):
            # 通过环境变量设置 FFMPEG 选项（OpenCV 的正确方式）
            if self.ffmpeg_opts:
                opts_str = '|'.join(f'{k}={v}' for k, v in self.ffmpeg_opts.items())
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = opts_str
                print(f"[CameraSource] 设置 FFMPEG 选项: {opts_str}")

            self.capture = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            if self.capture.isOpened():
                print(f"[CameraSource] 打开流: {self.source}")
                return

            # 回退：尝试不指定 CAP_FFMPEG 让 OpenCV 自动选择后端
            print(f"[CameraSource] CAP_FFMPEG 打开失败，尝试自动后端...")
            self.capture = cv2.VideoCapture(self.source)
            if self.capture.isOpened():
                print(f"[CameraSource] 自动后端打开流成功: {self.source}")
                return

            raise ConnectionError(
                f"无法打开流: {self.source}\n"
                f"  请检查: (1) 无人机是否正在推流 (2) IP 和端口是否正确 (3) 防火墙是否放行\n"
                f"  提示: 先用 ffplay {self.source} 验证流是否可达"
            )

        raise ValueError(f"未知的 camera_source: {self.source}")

    def reconnect_stream(self) -> bool:
        """重新连接流，返回是否成功。"""
        self.release()
        self.capture = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        if not self.capture.isOpened():
            self.capture = cv2.VideoCapture(self.source)
        ok = self.capture is not None and self.capture.isOpened()
        print(f"[CameraSource] 流重连{'成功' if ok else '失败'}")
        return ok

    def read(self) -> Tuple[bool, Optional[Any]]:
        if self.capture is not None:
            success, frame = self.capture.read()
            return success, frame if success else None

        if self.image_files:
            if self.image_index >= len(self.image_files):
                if self.loop:
                    self.image_index = 0
                    print(f"[CameraSource] 图片目录已循环，重新从第 1 张开始")
                else:
                    return False, None
            path = self.image_files[self.image_index]
            self.image_index += 1
            frame = cv2.imread(path)
            return frame is not None, frame

        return False, None

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            print("[CameraSource] 已释放视频流")
