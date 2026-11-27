"""Nạp & validate ``config.yaml`` (UTF-8) và biến môi trường ``.env``.

Secret Zalo/ImgBB được nạp theo cơ chế lazy: thiếu secret KHÔNG gây lỗi khi
khởi động (Spec Addendum v1.1); việc bắt buộc có secret thuộc về Notifier.
"""

from __future__ import annotations

import os
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config.yaml"
DEFAULT_ENV_PATH: Path = PROJECT_ROOT / ".env"

VALID_MODES: frozenset[str] = frozenset({"AWAY", "HOME"})
VALID_LOG_LEVELS: frozenset[str] = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
MESSAGE_PLACEHOLDERS: frozenset[str] = frozenset({"camera_name", "timestamp", "count"})
_DEVICE_PATTERN = re.compile(r"^(cpu|cuda(:\d+)?)$")

Point = tuple[float, float]


class ConfigError(ValueError):
    """Lỗi cấu hình không hợp lệ (sai schema, sai kiểu hoặc sai miền giá trị)."""


@dataclass(frozen=True)
class CameraConfig:
    """Thông tin camera đang giám sát."""

    name: str
    rtsp_url: str
    reconnect_interval_seconds: float


@dataclass(frozen=True)
class ModeConfig:
    """Chế độ vận hành hiện hành."""

    active: str


@dataclass(frozen=True)
class DetectorConfig:
    """Tham số mô hình YOLO."""

    model_path: str
    device: str
    imgsz: int
    confidence_threshold: float
    inference_fps: float


@dataclass(frozen=True)
class AwayBehaviorConfig:
    """Luật kích hoạt chế độ AWAY."""

    target_classes: tuple[str, ...]
    consecutive_frames: int
    roi_polygon: tuple[Point, ...] | None


@dataclass(frozen=True)
class BehaviorConfig:
    """Cấu hình tầng hành vi theo từng mode."""

    away: AwayBehaviorConfig


@dataclass(frozen=True)
class NotifierConfig:
    """Cấu hình cảnh báo."""

    enabled: bool
    cooldown_seconds: float
    image_expiration_seconds: int
    message_template: str


@dataclass(frozen=True)
class DisplayConfig:
    """Cấu hình cửa sổ debug OpenCV."""

    show_window: bool


@dataclass(frozen=True)
class LoggingConfig:
    """Cấu hình logging."""

    level: str
    log_dir: str
    max_bytes: int
    backup_count: int


@dataclass(frozen=True)
class Secrets:
    """Secret nạp từ ``.env``/biến môi trường; có thể rỗng ở Task 1.1–1.4."""

    zalo_bot_token: str | None
    zalo_chat_ids: tuple[str, ...]
    image_host_api_key: str | None

    def __repr__(self) -> str:
        # Không bao giờ in giá trị secret ra log.
        return (
            "Secrets(zalo_bot_token=%s, zalo_chat_ids=%d id, image_host_api_key=%s)"
            % (
                "<set>" if self.zalo_bot_token else "<missing>",
                len(self.zalo_chat_ids),
                "<set>" if self.image_host_api_key else "<missing>",
            )
        )


@dataclass(frozen=True)
class AppConfig:
    """Toàn bộ cấu hình ứng dụng đã được validate."""

    camera: CameraConfig
    mode: ModeConfig
    detector: DetectorConfig
    behavior: BehaviorConfig
    notifier: NotifierConfig
    display: DisplayConfig
    logging: LoggingConfig
    secrets: Secrets
    config_path: Path


# ---------------------------------------------------------------------------
# Helpers validate kiểu & miền giá trị
# ---------------------------------------------------------------------------


def _section(data: dict[str, Any], key: str, path: str, expected_keys: set[str]) -> dict[str, Any]:
    full = f"{path}.{key}" if path else key
    if key not in data:
        raise ConfigError(f"Thiếu mục bắt buộc '{full}'")
    value = data[key]
    if not isinstance(value, dict):
        raise ConfigError(f"'{full}' phải là một mapping (khối key: value)")
    _check_keys(value, full, expected_keys)
    return value


def _check_keys(data: dict[str, Any], path: str, expected_keys: set[str]) -> None:
    missing = expected_keys - data.keys()
    unknown = data.keys() - expected_keys
    if missing:
        raise ConfigError(f"'{path}' thiếu khóa: {', '.join(sorted(missing))}")
    if unknown:
        raise ConfigError(f"'{path}' có khóa không hợp lệ (sai chính tả?): {', '.join(sorted(unknown))}")


def _non_empty_str(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{path}' phải là chuỗi không rỗng")
    return value.strip()


def _bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"'{path}' phải là true/false")
    return value


def _number(value: Any, path: str) -> float:
    # bool là subclass của int trong Python nên phải loại riêng.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"'{path}' phải là số")
    return float(value)


def _positive_number(value: Any, path: str) -> float:
    number = _number(value, path)
    if number <= 0:
        raise ConfigError(f"'{path}' phải > 0 (nhận {value})")
    return number


def _positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"'{path}' phải là số nguyên")
    if value <= 0:
        raise ConfigError(f"'{path}' phải > 0 (nhận {value})")
    return value


def _parse_roi(value: Any, path: str) -> tuple[Point, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) < 3:
        raise ConfigError(f"'{path}' phải là null hoặc danh sách >= 3 điểm [x, y]")
    points: list[Point] = []
    for index, point in enumerate(value):
        point_path = f"{path}[{index}]"
        if not isinstance(point, list) or len(point) != 2:
            raise ConfigError(f"'{point_path}' phải có dạng [x, y]")
        x = _number(point[0], f"{point_path}.x")
        y = _number(point[1], f"{point_path}.y")
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ConfigError(f"'{point_path}' = [{x}, {y}] nằm ngoài đoạn [0.0, 1.0]")
        points.append((x, y))
    return tuple(points)


def _validate_template(value: Any, path: str) -> str:
    template = _non_empty_str(value, path)
    try:
        fields = {name for _, name, _, _ in string.Formatter().parse(template) if name is not None}
    except ValueError as exc:
        raise ConfigError(f"'{path}' sai cú pháp ngoặc nhọn: {exc}") from exc
    unknown = fields - MESSAGE_PLACEHOLDERS
    if unknown:
        raise ConfigError(
            f"'{path}' dùng biến không hỗ trợ: {', '.join(sorted(unknown))} "
            f"(chỉ hỗ trợ: {', '.join(sorted(MESSAGE_PLACEHOLDERS))})"
        )
    return template


# ---------------------------------------------------------------------------
# Parse từng mục
# ---------------------------------------------------------------------------


def _parse_camera(raw: dict[str, Any]) -> CameraConfig:
    sec = _section(raw, "camera", "", {"name", "rtsp_url", "reconnect_interval_seconds"})
    return CameraConfig(
        name=_non_empty_str(sec["name"], "camera.name"),
        rtsp_url=_non_empty_str(sec["rtsp_url"], "camera.rtsp_url"),
        reconnect_interval_seconds=_positive_number(
            sec["reconnect_interval_seconds"], "camera.reconnect_interval_seconds"
        ),
    )


def _parse_mode(raw: dict[str, Any]) -> ModeConfig:
    sec = _section(raw, "mode", "", {"active"})
    active = _non_empty_str(sec["active"], "mode.active")
    if active not in VALID_MODES:
        raise ConfigError(f"'mode.active' phải là một trong {sorted(VALID_MODES)} (nhận '{active}')")
    return ModeConfig(active=active)


def _parse_detector(raw: dict[str, Any]) -> DetectorConfig:
    sec = _section(
        raw,
        "detector",
        "",
        {"model_path", "device", "imgsz", "confidence_threshold", "inference_fps"},
    )
    device = _non_empty_str(sec["device"], "detector.device")
    if not _DEVICE_PATTERN.match(device):
        raise ConfigError(f"'detector.device' phải là 'cpu', 'cuda' hoặc 'cuda:N' (nhận '{device}')")
    confidence = _number(sec["confidence_threshold"], "detector.confidence_threshold")
    if not 0.0 < confidence <= 1.0:
        raise ConfigError(f"'detector.confidence_threshold' phải trong (0, 1] (nhận {confidence})")
    return DetectorConfig(
        model_path=_non_empty_str(sec["model_path"], "detector.model_path"),
        device=device,
        imgsz=_positive_int(sec["imgsz"], "detector.imgsz"),
        confidence_threshold=confidence,
        inference_fps=_positive_number(sec["inference_fps"], "detector.inference_fps"),
    )


def _parse_behavior(raw: dict[str, Any]) -> BehaviorConfig:
    sec = _section(raw, "behavior", "", {"away"})
    away = _section(sec, "away", "behavior", {"target_classes", "consecutive_frames", "roi_polygon"})
    classes = away["target_classes"]
    if not isinstance(classes, list) or not classes:
        raise ConfigError("'behavior.away.target_classes' phải là danh sách nhãn không rỗng")
    target_classes = tuple(
        _non_empty_str(name, f"behavior.away.target_classes[{i}]") for i, name in enumerate(classes)
    )
    return BehaviorConfig(
        away=AwayBehaviorConfig(
            target_classes=target_classes,
            consecutive_frames=_positive_int(away["consecutive_frames"], "behavior.away.consecutive_frames"),
            roi_polygon=_parse_roi(away["roi_polygon"], "behavior.away.roi_polygon"),
        )
    )


def _parse_notifier(raw: dict[str, Any]) -> NotifierConfig:
    sec = _section(
        raw,
        "notifier",
        "",
        {"enabled", "cooldown_seconds", "image_expiration_seconds", "message_template"},
    )
    return NotifierConfig(
        enabled=_bool(sec["enabled"], "notifier.enabled"),
        cooldown_seconds=_positive_number(sec["cooldown_seconds"], "notifier.cooldown_seconds"),
        image_expiration_seconds=_positive_int(
            sec["image_expiration_seconds"], "notifier.image_expiration_seconds"
        ),
        message_template=_validate_template(sec["message_template"], "notifier.message_template"),
    )


def _parse_display(raw: dict[str, Any]) -> DisplayConfig:
    sec = _section(raw, "display", "", {"show_window"})
    return DisplayConfig(show_window=_bool(sec["show_window"], "display.show_window"))


def _parse_logging(raw: dict[str, Any]) -> LoggingConfig:
    sec = _section(raw, "logging", "", {"level", "log_dir", "max_bytes", "backup_count"})
    level = _non_empty_str(sec["level"], "logging.level").upper()
    if level not in VALID_LOG_LEVELS:
        raise ConfigError(f"'logging.level' phải là một trong {sorted(VALID_LOG_LEVELS)} (nhận '{level}')")
    return LoggingConfig(
        level=level,
        log_dir=_non_empty_str(sec["log_dir"], "logging.log_dir"),
        max_bytes=_positive_int(sec["max_bytes"], "logging.max_bytes"),
        backup_count=_positive_int(sec["backup_count"], "logging.backup_count"),
    )


def _optional_env(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def parse_chat_ids(raw: str | None) -> tuple[str, ...]:
    """Tách ``ZALO_CHAT_IDS`` theo dấu phẩy, cắt khoảng trắng, bỏ phần tử rỗng."""
    if not raw:
        return ()
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def load_secrets(env_path: str | Path | None = None) -> Secrets:
    """Nạp secret từ ``.env`` (nếu tồn tại) và biến môi trường; không báo lỗi khi thiếu.

    Biến môi trường đã có sẵn trong hệ thống được ưu tiên hơn giá trị trong ``.env``.
    """
    path = Path(env_path) if env_path is not None else DEFAULT_ENV_PATH
    if path.is_file():
        load_dotenv(path, override=False, encoding="utf-8")
    return Secrets(
        zalo_bot_token=_optional_env("ZALO_BOT_TOKEN"),
        zalo_chat_ids=parse_chat_ids(os.getenv("ZALO_CHAT_IDS")),
        image_host_api_key=_optional_env("IMAGE_HOST_API_KEY"),
    )


def load_config(
    config_path: str | Path | None = None,
    env_path: str | Path | None = None,
) -> AppConfig:
    """Đọc ``config.yaml`` (UTF-8), validate toàn bộ schema và nạp secret.

    Args:
        config_path: Đường dẫn file cấu hình; mặc định ``src/config.yaml``.
        env_path: Đường dẫn file ``.env``; mặc định ``src/.env``.

    Returns:
        ``AppConfig`` bất biến đã được validate.

    Raises:
        ConfigError: File không tồn tại, sai cú pháp YAML hoặc sai schema/giá trị.
    """
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise ConfigError(f"Không tìm thấy file cấu hình: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Sai cú pháp YAML trong {path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ConfigError(f"File {path} phải được lưu bằng UTF-8: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"Nội dung {path} phải là một mapping ở cấp gốc")
    _check_keys(raw, "<root>", {"camera", "mode", "detector", "behavior", "notifier", "display", "logging"})

    return AppConfig(
        camera=_parse_camera(raw),
        mode=_parse_mode(raw),
        detector=_parse_detector(raw),
        behavior=_parse_behavior(raw),
        notifier=_parse_notifier(raw),
        display=_parse_display(raw),
        logging=_parse_logging(raw),
        secrets=load_secrets(env_path),
        config_path=path.resolve(),
    )


def _self_check() -> int:
    """Nạp cấu hình, bật logging và in tóm tắt (dùng cho bước nghiệm thu Task 1.1)."""
    import logging

    from utils.logger import setup_logging

    try:
        config = load_config()
    except ConfigError as exc:
        logging.basicConfig(level=logging.ERROR)
        logging.getLogger(__name__).error("Cấu hình không hợp lệ: %s", exc)
        return 1

    log_file = setup_logging(
        level=config.logging.level,
        log_dir=config.logging.log_dir,
        max_bytes=config.logging.max_bytes,
        backup_count=config.logging.backup_count,
    )
    logger = logging.getLogger("utils.config_loader")
    logger.info("Đã nạp cấu hình: %s", config.config_path)
    logger.info("Camera: %s | %s | reconnect %ss", config.camera.name, config.camera.rtsp_url,
                config.camera.reconnect_interval_seconds)
    logger.info("Mode: %s", config.mode.active)
    logger.info("Detector: %s", config.detector)
    logger.info("Behavior AWAY: %s", config.behavior.away)
    logger.info("Notifier: enabled=%s cooldown=%ss template='%s'", config.notifier.enabled,
                config.notifier.cooldown_seconds, config.notifier.message_template)
    logger.info("Display: %s | Logging: %s", config.display, config.logging)
    logger.info("Secrets: %r", config.secrets)
    if config.notifier.enabled and not (config.secrets.zalo_bot_token and config.secrets.zalo_chat_ids):
        logger.info("Chưa có ZALO_BOT_TOKEN/ZALO_CHAT_IDS — sẽ bắt buộc khi khởi tạo Notifier (Task 1.5–1.6)")
    logger.info("File log: %s", log_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_check())
