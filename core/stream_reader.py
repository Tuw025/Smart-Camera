"""Bộ đọc luồng RTSP đa luồng: chỉ giữ khung hình mới nhất và tự kết nối lại."""

from __future__ import annotations

import os

# Phải đặt TRƯỚC khi FFmpeg backend mở luồng: ép RTSP qua TCP để tránh mất gói
# UDP gây vỡ hình. Dùng setdefault để người vận hành vẫn có thể ghi đè bằng biến môi trường.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import threading
import time
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from utils.config_loader import CameraConfig
from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_OPEN_TIMEOUT_MS: int = 3000
DEFAULT_READ_TIMEOUT_MS: int = 3000
DEFAULT_STOP_TIMEOUT_SECONDS: float = 1.5
FPS_WINDOW_SECONDS: float = 1.0


class ConnectionState(Enum):
    """Trạng thái kết nối của luồng RTSP."""

    IDLE = "IDLE"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"
    STOPPED = "STOPPED"


@dataclass(frozen=True)
class FramePacket:
    """Khung hình mới nhất kèm định danh tăng dần.

    ``image`` được chia sẻ giữa các luồng — bên đọc KHÔNG được vẽ trực tiếp lên nó,
    hãy ``image.copy()`` trước khi chỉnh sửa.
    """

    image: np.ndarray
    frame_id: int
    timestamp: float  # time.monotonic() lúc đọc được frame


class RTSPReader:
    """Đọc ``cv2.VideoCapture`` ở luồng nền, luôn ghi đè để chỉ giữ frame mới nhất.

    Khi mở luồng thất bại hoặc ``read()`` lỗi/timeout, reader giải phóng capture,
    chờ ``reconnect_interval_seconds`` rồi thử lại cho tới khi ``stop()`` được gọi.
    """

    def __init__(
        self,
        camera: CameraConfig,
        open_timeout_ms: int = DEFAULT_OPEN_TIMEOUT_MS,
        read_timeout_ms: int = DEFAULT_READ_TIMEOUT_MS,
    ) -> None:
        self._name = camera.name
        self._rtsp_url = camera.rtsp_url
        self._reconnect_interval = camera.reconnect_interval_seconds
        self._open_timeout_ms = open_timeout_ms
        self._read_timeout_ms = read_timeout_ms

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._latest: FramePacket | None = None
        self._frame_counter = 0
        self._state = ConnectionState.IDLE
        self._fps = 0.0

    # ------------------------------------------------------------------ API

    def start(self) -> None:
        """Khởi động luồng đọc nền (gọi lại khi đang chạy sẽ bị bỏ qua)."""
        if self._thread is not None and self._thread.is_alive():
            logger.debug("RTSPReader '%s' đã chạy, bỏ qua start()", self._name)
            return
        self._stop_event.clear()
        self._set_state(ConnectionState.CONNECTING)
        self._thread = threading.Thread(
            target=self._run, name=f"RTSPReader-{self._name}", daemon=True
        )
        self._thread.start()
        logger.info("Khởi động RTSPReader cho '%s' (%s)", self._name, self._rtsp_url)

    def get_latest_frame(self) -> FramePacket | None:
        """Trả về frame mới nhất (hoặc ``None`` nếu chưa từng đọc được frame nào)."""
        with self._lock:
            return self._latest

    def stop(self, timeout: float = DEFAULT_STOP_TIMEOUT_SECONDS) -> None:
        """Dừng luồng nền và chờ tối đa ``timeout`` giây.

        Capture được giải phóng ngay trong luồng nền (không release chéo luồng khi
        ``read()`` đang chạy). Nếu luồng còn kẹt trong lệnh mở/đọc của FFmpeg quá
        ``timeout``, nó là daemon thread nên không chặn tiến trình thoát.
        """
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning(
                    "RTSPReader '%s' chưa dừng sau %.1fs (đang chờ FFmpeg timeout) — bỏ qua",
                    self._name,
                    timeout,
                )
        self._set_state(ConnectionState.STOPPED)
        logger.info("Đã dừng RTSPReader '%s'", self._name)

    @property
    def name(self) -> str:
        """Tên camera."""
        return self._name

    @property
    def state(self) -> ConnectionState:
        """Trạng thái kết nối hiện tại."""
        with self._lock:
            return self._state

    @property
    def is_connected(self) -> bool:
        """``True`` khi đang nhận frame bình thường."""
        return self.state is ConnectionState.CONNECTED

    @property
    def fps(self) -> float:
        """FPS đọc luồng đo trong cửa sổ ~1 giây gần nhất (0 khi mất kết nối)."""
        with self._lock:
            return self._fps

    @property
    def is_running(self) -> bool:
        """``True`` khi luồng nền còn sống."""
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------ internals

    def _set_state(self, state: ConnectionState) -> None:
        with self._lock:
            self._state = state
            if state is not ConnectionState.CONNECTED:
                self._fps = 0.0

    def _open_capture(self) -> cv2.VideoCapture | None:
        capture = cv2.VideoCapture(
            self._rtsp_url,
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self._open_timeout_ms,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, self._read_timeout_ms,
            ],
        )
        if capture.isOpened():
            return capture
        capture.release()
        return None

    def _run(self) -> None:
        attempt = 0
        while not self._stop_event.is_set():
            attempt_started = time.monotonic()
            capture = self._open_capture()
            if capture is None:
                attempt += 1
                self._set_state(ConnectionState.RECONNECTING)
                logger.warning(
                    "Không mở được luồng '%s' (%s) — thử lại sau %ss (lần %d)",
                    self._name, self._rtsp_url, self._reconnect_interval, attempt,
                )
                # Trừ thời gian đã chờ open-timeout để nhịp thử lại đúng reconnect_interval.
                elapsed = time.monotonic() - attempt_started
                self._stop_event.wait(max(0.0, self._reconnect_interval - elapsed))
                continue

            attempt = 0
            self._set_state(ConnectionState.CONNECTED)
            logger.info("Đã kết nối luồng '%s' (%s)", self._name, self._rtsp_url)
            try:
                self._read_loop(capture)
            finally:
                capture.release()

            if not self._stop_event.is_set():
                self._set_state(ConnectionState.RECONNECTING)
                logger.warning(
                    "Mất kết nối luồng '%s' — thử kết nối lại sau %ss",
                    self._name, self._reconnect_interval,
                )
                self._stop_event.wait(self._reconnect_interval)

    def _read_loop(self, capture: cv2.VideoCapture) -> None:
        window_start = time.monotonic()
        window_frames = 0
        while not self._stop_event.is_set():
            ok, image = capture.read()
            if not ok or image is None:
                return
            now = time.monotonic()
            window_frames += 1
            elapsed = now - window_start
            with self._lock:
                self._frame_counter += 1
                self._latest = FramePacket(image=image, frame_id=self._frame_counter, timestamp=now)
                if elapsed >= FPS_WINDOW_SECONDS:
                    self._fps = window_frames / elapsed
            if elapsed >= FPS_WINDOW_SECONDS:
                window_start = now
                window_frames = 0
