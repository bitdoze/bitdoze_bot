from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from bitdoze_bot.config import Config
from bitdoze_bot.heartbeat import load_heartbeat_config, resolve_heartbeat_identity, run_heartbeat


def test_load_heartbeat_config_defaults() -> None:
    cfg = Config(data={}, path=Path("config.yaml"))
    hb = load_heartbeat_config(cfg)
    assert hb.enabled is True
    assert hb.interval_minutes == 30
    assert hb.quiet_ack == "HEARTBEAT_OK"
    assert hb.prompt_path == cfg.resolve_path(None, default="workspace/HEARTBEAT.md")
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


def test_run_heartbeat_uses_reasoning_content_for_quiet_ack() -> None:
    class DummyAgent:
        name = "heartbeat"

        def run(self, prompt: str, user_id: str, session_id: str):
            return type("Resp", (), {"content": "", "reasoning_content": "HEARTBEAT_OK"})()

    sent: list[str] = []

    async def _send_fn(content: str) -> None:
        sent.append(content)

    asyncio.run(run_heartbeat(DummyAgent(), "ping", "HEARTBEAT_OK", _send_fn))
    assert sent == []
