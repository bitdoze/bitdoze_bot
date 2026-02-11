from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest

from bitdoze_bot.config import Config
from bitdoze_bot.logging_setup import (
    DEFAULT_FORMATS,
    build_logging_settings,
    configure_logging,
    configure_logging_from_config,
)


@pytest.fixture(autouse=True)
def _restore_root_logging():
    root = logging.getLogger()
    previous_handlers = list(root.handlers)
    previous_level = root.level
    try:
        yield
    finally:
        root.handlers.clear()
        for handler in previous_handlers:
            root.addHandler(handler)
        root.setLevel(previous_level)


def test_build_logging_settings_defaults_when_section_missing() -> None:
    config = Config(data={}, path=Path("config.yaml"))

    settings = build_logging_settings(config)

    assert settings.level == logging.INFO
    assert settings.level_name == "INFO"
    assert settings.level_fallback_used is False
    assert settings.fmt == DEFAULT_FORMATS["detailed"]
    assert settings.file.enabled is True
    assert settings.file.path == config.resolve_path(None, default="logs/bitdoze-bot.log")
    assert settings.file.max_bytes == 10 * 1024 * 1024
    assert settings.file.backup_count == 5


def test_build_logging_settings_invalid_level_falls_back_to_info() -> None:
    config = Config(
        data={"logging": {"level": "not-a-real-level"}},
        path=Path("config.yaml"),
    )

    settings = build_logging_settings(config)

    assert settings.level == logging.INFO
    assert settings.level_name == "INFO"
    assert settings.level_fallback_used is True
    assert settings.configured_level == "not-a-real-level"


def test_build_logging_settings_format_shortcuts_and_file_overrides(tmp_path: Path) -> None:
    log_path = tmp_path / "runtime" / "bot.log"
    config = Config(
        data={
            "logging": {
                "level": "DEBUG",
                "format": "simple",
                "file": {
                    "enabled": "false",
                    "path": str(log_path),
                    "max_bytes": "1024",
                    "backup_count": "3",
                },
            }
        },
        path=Path("config.yaml"),
    )

    settings = build_logging_settings(config)

    assert settings.level == logging.DEBUG
    assert settings.fmt == DEFAULT_FORMATS["simple"]
    assert settings.file.enabled is False
    assert settings.file.path == log_path
    assert settings.file.max_bytes == 1024
    assert settings.file.backup_count == 3


def test_configure_logging_sets_stream_and_rotating_file_handlers_without_duplicates(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "logs" / "bitdoze-bot.log"
    config = Config(
        data={
            "logging": {
                "level": "DEBUG",
                "file": {
                    "enabled": True,
                    "path": str(log_path),
                    "max_bytes": 2048,
                    "backup_count": 2,
                },
            }
        },
        path=Path("config.yaml"),
    )

    settings = build_logging_settings(config)
    configure_logging(settings)
    configure_logging(settings)

    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert len(root.handlers) == 2
    assert sum(isinstance(h, logging.StreamHandler) for h in root.handlers) == 2
    assert sum(isinstance(h, RotatingFileHandler) for h in root.handlers) == 1

    logging.getLogger("test.logger").info("hello")
    assert log_path.exists()


def test_configure_logging_from_config_invalid_level_uses_info() -> None:
    config = Config(
        data={"logging": {"level": "verbose", "file": {"enabled": False}}},
        path=Path("config.yaml"),
    )

    settings = configure_logging_from_config(config)

    assert settings.level == logging.INFO
    assert settings.level_fallback_used is True
