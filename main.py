"""Điểm khởi chạy ứng dụng (bản tối giản Task 1.2): Config + RTSPReader + cửa sổ OpenCV."""

from __future__ import annotations

import logging
import sys
import time
import unicodedata

import cv2
import numpy as np

from core.stream_reader import ConnectionState, FramePacket, RTSPReader
from utils.config_loader import AppConfig, ConfigError, load_config
from utils.logger import get_logger, setup_logging

WINDOW_TITLE_PREFIX: str = "Smart Camera"
PLACEHOLDER_SIZE: tuple[int, int] = (640, 360)  # (width, height)
PLACEHOLDER_REFRESH_SECONDS: float = 0.5
WAIT_KEY_MS: int = 5
HEADLESS_SLEEP_SECONDS: float = 0.1
HEADLESS_STATUS_INTERVAL_SECONDS: float = 10.0
QUIT_KEYS: frozenset[int] = frozenset({ord("q"), ord("Q")})

HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
HUD_COLOR_OK: tuple[int, int, int] = (0, 255, 0)
HUD_COLOR_WARN: tuple[int, int, int] = (0, 200, 255)

logger = get_logger("main")


def to_ascii(text: str) -> str:
    """Bỏ dấu tiếng Việt để hiển thị bằng font Hershey của OpenCV (chỉ hỗ trợ ASCII)."""
    text = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return stripped.encode("ascii", "ignore").decode("ascii").strip()


def _draw_text(image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    # Viền đen dưới chữ để HUD đọc được trên mọi nền.
    cv2.putText(image, text, origin, HUD_FONT, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, origin, HUD_FONT, 0.7, color, 2, cv2.LINE_AA)


def render_view(reader: RTSPReader, packet: FramePacket | None, camera_label: str) -> np.ndarray:
    """Tạo khung hiển thị: frame thật kèm HUD, hoặc khung chờ khi chưa/mất kết nối."""
    state = reader.state
    if state is ConnectionState.CONNECTED and packet is not None:
        view = packet.image.copy()
        _draw_text(view, camera_label, (10, 30), HUD_COLOR_OK)
        _draw_text(view, f"Stream FPS: {reader.fps:.1f}", (10, 60), HUD_COLOR_OK)
        _draw_text(view, f"Status: {state.value}", (10, 90), HUD_COLOR_OK)
        return view

    width, height = PLACEHOLDER_SIZE
    view = np.zeros((height, width, 3), dtype=np.uint8)
    message = "Dang ket noi..." if state is ConnectionState.CONNECTING else "Mat ket noi - dang thu lai..."
    _draw_text(view, camera_label, (20, 40), HUD_COLOR_WARN)
    _draw_text(view, f"Status: {state.value}", (20, 80), HUD_COLOR_WARN)
    _draw_text(view, message, (20, height // 2), HUD_COLOR_WARN)
    return view


def run_window_loop(reader: RTSPReader, camera_label: str) -> None:
    """Hiển thị cửa sổ OpenCV cho tới khi nhấn ``q`` hoặc đóng cửa sổ bằng nút X."""
    title = f"{WINDOW_TITLE_PREFIX} - {camera_label}"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    last_frame_id = -1
    last_placeholder_at = 0.0

    while True:
        packet = reader.get_latest_frame()
        frame_id = packet.frame_id if packet is not None else -1
        now = time.monotonic()
        # Chỉ vẽ lại khi có frame mới; màn hình chờ được làm mới thưa để không ăn CPU.
        if reader.is_connected and frame_id != last_frame_id:
            cv2.imshow(title, render_view(reader, packet, camera_label))
            last_frame_id = frame_id
        elif not reader.is_connected and now - last_placeholder_at >= PLACEHOLDER_REFRESH_SECONDS:
            cv2.imshow(title, render_view(reader, packet, camera_label))
            last_placeholder_at = now

        key = cv2.waitKey(WAIT_KEY_MS) & 0xFF
        if key in QUIT_KEYS:
            logger.info("Nhận phím 'q' — đang thoát")
            return
        if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            logger.info("Cửa sổ OpenCV đã bị đóng — đang thoát")
            return


def run_headless_loop(reader: RTSPReader) -> None:
    """Vòng lặp nền không cửa sổ: log trạng thái định kỳ, dừng bằng Ctrl+C."""
    logger.info("show_window=false — chạy nền, nhấn Ctrl+C để thoát")
    next_status = time.monotonic()
    while True:
        now = time.monotonic()
        if now >= next_status:
            packet = reader.get_latest_frame()
            logger.info(
                "Trạng thái '%s': %s | FPS đọc luồng: %.1f | frame_id: %s",
                reader.name, reader.state.value, reader.fps,
                packet.frame_id if packet is not None else "-",
            )
            next_status = now + HEADLESS_STATUS_INTERVAL_SECONDS
        time.sleep(HEADLESS_SLEEP_SECONDS)


def run(config: AppConfig) -> None:
    """Khởi động reader và vòng lặp hiển thị; luôn dọn dẹp tài nguyên khi thoát."""
    reader = RTSPReader(config.camera)
    camera_label = to_ascii(config.camera.name) or "Camera"
    reader.start()
    try:
        if config.display.show_window:
            run_window_loop(reader, camera_label)
        else:
            run_headless_loop(reader)
    except KeyboardInterrupt:
        logger.info("Nhận Ctrl+C — đang thoát")
    finally:
        shutdown_started = time.monotonic()
        reader.stop()
        if config.display.show_window:
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # Cho HighGUI trên Windows xử lý sự kiện đóng cửa sổ.
        logger.info("Đã thoát sạch sau %.2fs", time.monotonic() - shutdown_started)


def main() -> int:
    """Nạp cấu hình, bật logging và chạy ứng dụng. Trả về exit code."""
    try:
        config = load_config()
    except ConfigError as exc:
        logging.basicConfig(level=logging.ERROR)
        logging.getLogger("main").error("Cấu hình không hợp lệ: %s", exc)
        return 1

    setup_logging(
        level=config.logging.level,
        log_dir=config.logging.log_dir,
        max_bytes=config.logging.max_bytes,
        backup_count=config.logging.backup_count,
    )
    logger.info(
        "Khởi động | camera='%s' | mode=%s | show_window=%s",
        config.camera.name, config.mode.active, config.display.show_window,
    )
    try:
        run(config)
    except KeyboardInterrupt:
        # Ctrl+C rơi vào đúng lúc đang dọn dẹp — vẫn coi là thoát bình thường.
        logger.info("Nhận Ctrl+C trong lúc dọn dẹp — thoát")
    return 0


if __name__ == "__main__":
    sys.exit(main())
