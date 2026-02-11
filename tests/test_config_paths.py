from __future__ import annotations

from pathlib import Path

from bitdoze_bot.config import Config, load_config
from bitdoze_bot.discord_bot import load_runtime_config
from bitdoze_bot.tool_permissions import load_tool_audit_config


def test_load_config_uses_absolute_path(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}", encoding="utf-8")

    config = load_config(config_path)

    assert config.path == config_path.resolve()
    assert config.base_dir == tmp_path.resolve()


def test_resolve_path_is_relative_to_config_directory(tmp_path: Path) -> None:
    config_path = tmp_path / "custom-home" / "config.yml"
    config = Config(data={}, path=config_path)

    assert config.resolve_path("logs/bot.log") == (config_path.parent / "logs/bot.log").resolve()
    assert config.resolve_path(None, default="data/bot.db") == (
        config_path.parent / "data/bot.db"
    ).resolve()
    assert config.resolve_optional_path(None) is None


def test_runtime_and_audit_paths_use_config_directory(tmp_path: Path) -> None:
    config_dir = tmp_path / "custom-home"
    config = Config(
        data={
            "monitoring": {"telemetry_path": "logs/runs.jsonl"},
            "tool_permissions": {"audit": {"path": "logs/tool-audit.jsonl"}},
        },
        path=config_dir / "config.yml",
    )

    runtime_cfg = load_runtime_config(config)
    assert runtime_cfg.monitor is not None
    assert runtime_cfg.monitor.telemetry_path == (config_dir / "logs/runs.jsonl").resolve()

    audit_cfg = load_tool_audit_config(config)
    assert audit_cfg.path == (config_dir / "logs/tool-audit.jsonl").resolve()
