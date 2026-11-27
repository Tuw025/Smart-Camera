"""Thiết lập logging chuẩn cho ứng dụng: console + file xoay vòng UTF-8."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
LOG_FILE_NAME: str = "app.log"
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

_HANDLER_MARKER: str = "_camera_app_handler"


def setup_logging(
    level: str = "INFO",
    log_dir: str | Path = "logs",
    max_bytes: int = 5_242_880,
    backup_count: int = 3,
) -> Path:
    """Cấu hình root logger ghi đồng thời ra console và file xoay vòng.

    Args:
        level: Tên mức log chuẩn (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_dir: Thư mục log; đường dẫn tương đối được tính từ thư mục ``src``.
        max_bytes: Kích thước tối đa của một file log trước khi xoay vòng.
        backup_count: Số file log cũ được giữ lại.

    Returns:
        Đường dẫn tuyệt đối tới file log đang ghi.
    """
    log_dir_path = Path(log_dir)
    if not log_dir_path.is_absolute():
        log_dir_path = PROJECT_ROOT / log_dir_path
    log_dir_path.mkdir(parents=True, exist_ok=True)
    log_file = log_dir_path / LOG_FILE_NAME

    root = logging.getLogger()
    root.setLevel(level.upper())

    # Gọi lại setup_logging không được nhân bản handler (tránh log lặp dòng).
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            root.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # Khi stdout bị chuyển hướng (pipe/file), Windows mặc định dùng cp1252 và sẽ
    # lỗi với tiếng Việt/emoji; ép UTF-8 để console log không bị "Logging error".
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    for handler in (console_handler, file_handler):
        setattr(handler, _HANDLER_MARKER, True)
        root.addHandler(handler)

    return log_file


def get_logger(name: str) -> logging.Logger:
    """Trả về logger theo tên module (dùng ``get_logger(__name__)``)."""
    return logging.getLogger(name)
