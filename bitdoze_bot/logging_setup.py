from __future__ import annotations

import logging
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Mapping

from bitdoze_bot.config import Config

DEFAULT_LEVEL_NAME = "INFO"
DEFAULT_FORMATS = {
    "detailed": "%(asctime)s %(levelname)s %(name)s: %(message)s",
    "simple": "%(levelname)s %(name)s: %(message)s",
}


@dataclass(frozen=True)
class FileLoggingSettings:
    enabled: bool = True
    path: Path = Path("logs/bitdoze-bot.log")
    max_bytes: int = 10 * 1024 * 1024
    backup_count: int = 5


@dataclass(frozen=True)
class LoggingSettings:
    configured_level: str | None = None
    level_name: str = DEFAULT_LEVEL_NAME
    level: int = logging.INFO
    level_fallback_used: bool = False
    fmt: str = DEFAULT_FORMATS["detailed"]
    file: FileLoggingSettings = FileLoggingSettings()


def _parse_level(level_value: Any) -> tuple[str, int, bool]:
    if isinstance(level_value, str):
        candidate = level_value.strip().upper()
        if candidate:
            parsed = logging.getLevelName(candidate)
            if isinstance(parsed, int):
                return candidate, parsed, False
            return DEFAULT_LEVEL_NAME, logging.INFO, True
    return DEFAULT_LEVEL_NAME, logging.INFO, False


def _parse_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _parse_format(format_value: Any) -> str:
    if isinstance(format_value, str):
        key = format_value.strip().lower()
        if key in DEFAULT_FORMATS:
            return DEFAULT_FORMATS[key]
        if format_value.strip():
            return format_value
    return DEFAULT_FORMATS["detailed"]


def _parse_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def build_logging_settings(config: Config) -> LoggingSettings:
    logging_cfg = _as_mapping(config.get("logging", default={}))
    raw_level = logging_cfg.get("level")
    configured_level = str(raw_level) if raw_level is not None else None
    level_name, level, level_fallback_used = _parse_level(raw_level)
    fmt = _parse_format(logging_cfg.get("format"))

    file_cfg = _as_mapping(logging_cfg.get("file", {}))
    file_settings = FileLoggingSettings(
        enabled=_parse_bool(file_cfg.get("enabled", True), True),
        path=Path(str(file_cfg.get("path", "logs/bitdoze-bot.log"))),
        max_bytes=_parse_positive_int(file_cfg.get("max_bytes"), 10 * 1024 * 1024),
        backup_count=_parse_positive_int(file_cfg.get("backup_count"), 5),
    )

    return LoggingSettings(
        configured_level=configured_level,
        level_name=level_name,
        level=level,
        level_fallback_used=level_fallback_used,
        fmt=fmt,
        file=file_settings,
    )


def configure_logging(settings: LoggingSettings) -> None:
    formatter = logging.Formatter(settings.fmt)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(settings.level)
    root.handlers.clear()
    root.addHandler(stream_handler)

    if not settings.file.enabled:
        return

    log_path = settings.file.path
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            filename=log_path,
            maxBytes=settings.file.max_bytes,
            backupCount=settings.file.backup_count,
            encoding="utf-8",
        )
    except OSError:
        root.warning(
            "Failed to initialize file logging at path=%s; continuing with stdout only",
            log_path,
        )
        return

    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def configure_logging_from_config(config: Config) -> LoggingSettings:
    settings = build_logging_settings(config)
    configure_logging(settings)
    logger = logging.getLogger(__name__)
    if settings.level_fallback_used:
        logger.warning(
            "Configured logging level '%s' is invalid; using INFO",
            settings.configured_level,
        )
    logger.info(
        "Logging configured level=%s file_enabled=%s file_path=%s",
        settings.level_name,
        settings.file.enabled,
        settings.file.path,
    )
    return settings
