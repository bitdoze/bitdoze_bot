from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from bitdoze_bot.config import Config
from bitdoze_bot.heartbeat import load_heartbeat_config, resolve_heartbeat_identity


def test_load_heartbeat_config_defaults() -> None:
    cfg = Config(data={}, path=Path("config.yaml"))
    hb = load_heartbeat_config(cfg)
    assert hb.enabled is True
    assert hb.interval_minutes == 30
    assert hb.quiet_ack == "HEARTBEAT_OK"
    assert hb.prompt_path == "workspace/HEARTBEAT.md"
    assert hb.session_scope == "isolated"
    assert hb.agent is None


def test_load_heartbeat_config_custom_values() -> None:
    cfg = Config(
        data={"heartbeat": {"session_scope": "shared", "agent": " heartbeat-agent "}},
        path=Path("config.yaml"),
    )
    hb = load_heartbeat_config(cfg)
    assert hb.session_scope == "shared"
    assert hb.agent == "heartbeat-agent"


def test_resolve_heartbeat_identity_shared() -> None:
    user_id, session_id = resolve_heartbeat_identity("shared")
    assert user_id == "heartbeat"
    assert session_id == "heartbeat"


def test_resolve_heartbeat_identity_isolated() -> None:
    now = datetime(2026, 2, 6, 10, 30, 0, tzinfo=timezone.utc)
    user_id, session_id = resolve_heartbeat_identity("isolated", now=now)
    assert user_id == "heartbeat:20260206T103000Z"
    assert session_id == "heartbeat:20260206T103000Z"
