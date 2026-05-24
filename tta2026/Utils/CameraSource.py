import os
import subprocess
import sys
import threading
from queue import Queue, Empty
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


class CameraSource:
    """摄像头、视频文件、图片目录或流（RTSP/RTMP）读取工具。

    支持两种 RTMP 模式：
    - 拉流（默认）：cv2.VideoCapture 主动连接 RTMP 服务端
    - 监听（listen=True）：ffmpeg 子进程监听端口，接收无人机推流
      连接断开后自动重连。
    """

    def __init__(
        self,
        source: str,
        ffmpeg_opts: Optional[Dict[str, Any]] = None,
        loop: bool = True,
        listen: bool = False,
        listen_fps: int = 30,
    ):
        self.source = source
        self.ffmpeg_opts = ffmpeg_opts or {}
        self.capture = None
        self.image_files: List[str] = []
        self.image_index = 0
        self.loop = loop
        self.listen = listen
        self.listen_fps = listen_fps
        self._process = None
        self._frame_queue: Queue = Queue(maxsize=30)
        self._reader_thread = None
        self._running = False
        self._init_source()

    # ── ffmpeg 路径查找 ──────────────────────────────────────

    @staticmethod
    def _find_ffmpeg() -> str:
        candidates = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ffmpeg.exe'),
            os.path.join(os.getcwd(), 'ffmpeg.exe'),
            'ffmpeg.exe',
        ]
        for candidate in candidates:
            if candidate == 'ffmpeg.exe':
                try:
                    subprocess.run(['ffmpeg.exe', '-version'], capture_output=True, check=True)
                    return 'ffmpeg.exe'
                except Exception:
                    continue
            elif os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(
            "找不到 ffmpeg.exe，请将其放到项目根目录或加入系统 PATH"
        )

    # ── 推流监听模式 ──────────────────────────────────────────

    def _init_listen(self) -> None:
        """启动 ffmpeg 子进程监听 RTMP 推流，后台线程读取 JPEG 帧。"""
        ffmpeg = self._find_ffmpeg()

        # 不用 fps 滤镜，保持原始分辨率与帧率
        cmd = [
            ffmpeg,
            '-loglevel', 'error',
            '-listen', '1',
            '-rtmp_live', 'live',
            '-fflags', '+nobuffer+discardcorrupt+genpts',
            '-flags', 'low_delay',
            '-err_detect', 'ignore_err',
            '-i', self.source,
            '-f', 'image2pipe',
            '-vcodec', 'mjpeg',
            '-q:v', '3',
            '-vsync', 'drop',
            '-',
        ]
        print(f"[CameraSource] 监听中: {self.source}")
        print(f"[CameraSource] 等待无人机推流... (确保无人机推流地址为 rtmp://<你电脑IP>/live)")

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=None,
        )
        self._running = True
        self._reader_thread = threading.Thread(target=self._pipe_reader, daemon=True)
        self._reader_thread.start()

    def _pipe_reader(self) -> None:
        """后台线程：从 ffmpeg 管道读取 JPEG 帧，放入队列。"""
        buf = b''
        try:
            while self._running and self._process and self._process.stdout:
                chunk = self._process.stdout.read(4096)
                if not chunk:
                    break
                buf += chunk

                while True:
                    start = buf.find(b'\xff\xd8')
                    if start == -1:
                        buf = b''
                        break
                    end = buf.find(b'\xff\xd9', start + 2)
                    if end == -1:
                        break

                    jpeg_data = buf[start:end + 2]
                    buf = buf[end + 2:]

                    frame = cv2.imdecode(
                        np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    if frame is not None:
                        try:
                            self._frame_queue.put(frame, timeout=1)
                        except Exception:
                            pass
        except Exception:
            pass
        finally:
            print("[CameraSource] ffmpeg 管道断开，将自动重连...")

    def _process_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _read_from_pipe(self) -> Tuple[bool, Optional[Any]]:
        """从帧队列取一帧（非阻塞），进程死了则自动重连。"""
        # ffmpeg 挂了则自动重启
        if not self._process_alive():
            self._restart_listen()

        try:
            frame = self._frame_queue.get_nowait()
            return True, frame
        except Empty:
            return False, None

    def _restart_listen(self) -> None:
        """重启 ffmpeg 监听进程。"""
        self._running = False
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=3)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
        # 清空残留缓冲区
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except Empty:
                break
        self._init_listen()

    # ── 初始化入口 ───────────────────────────────────────────

    def _init_source(self) -> None:
        if self.listen and self.source.lower().startswith('rtmp://'):
            self._init_listen()
            return

        if self.source.isdigit():
            self.capture = cv2.VideoCapture(int(self.source))
            print(f"[CameraSource] 打开摄像头: {self.source}")
            return

        if os.path.isfile(self.source):
            self.capture = cv2.VideoCapture(self.source)
            print(f"[CameraSource] 打开视频文件: {self.source}")
            return

        if os.path.isdir(self.source):
            self.image_files = sorted(
                [os.path.join(self.source, f) for f in os.listdir(self.source)
                 if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            )
            print(f"[CameraSource] 图片目录: {self.source}, 共 {len(self.image_files)} 张")
            return

        if self.source.lower().startswith(('rtsp://', 'rtmp://')):
            if self.ffmpeg_opts:
                opts_str = '|'.join(f'{k}={v}' for k, v in self.ffmpeg_opts.items())
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = opts_str
                print(f"[CameraSource] 设置 FFMPEG 选项: {opts_str}")

            self.capture = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            if self.capture.isOpened():
                print(f"[CameraSource] 打开流: {self.source}")
                return

            print("[CameraSource] CAP_FFMPEG 打开失败，尝试自动后端...")
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

    # ── 帧读取 ────────────────────────────────────────────────

    def read(self) -> Tuple[bool, Optional[Any]]:
        if self._process is not None:
            return self._read_from_pipe()

        if self.capture is not None:
            success, frame = self.capture.read()
            return success, frame if success else None

        if self.image_files:
            if self.image_index >= len(self.image_files):
                if self.loop:
                    self.image_index = 0
                    print("[CameraSource] 图片目录已循环，重新从第 1 张开始")
                else:
                    return False, None
            path = self.image_files[self.image_index]
            self.image_index += 1
            frame = cv2.imread(path)
            return frame is not None, frame

        return False, None

    # ── 重连 ──────────────────────────────────────────────────

    def reconnect_stream(self) -> bool:
        self.release()
        if self.listen:
            try:
                self._init_listen()
                return True
            except Exception as e:
                print(f"[CameraSource] 监听重连失败: {e}")
                return False
        self.capture = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        if not self.capture.isOpened():
            self.capture = cv2.VideoCapture(self.source)
        ok = self.capture is not None and self.capture.isOpened()
        print(f"[CameraSource] 流重连{'成功' if ok else '失败'}")
        return ok

    # ── 释放 ──────────────────────────────────────────────────

    def release(self) -> None:
        self._running = False
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
            print("[CameraSource] 已终止 ffmpeg 监听进程")
        if self.capture is not None:
            self.capture.release()
            print("[CameraSource] 已释放视频流")
