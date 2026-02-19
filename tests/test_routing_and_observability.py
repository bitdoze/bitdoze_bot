from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import asyncio
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from bitdoze_bot.config import Config
from bitdoze_bot.discord_bot import (
    _AccessControlSettings,
    _build_user_context_from_settings,
    _ContextSettings,
    _collect_delegation_paths,
    _build_response_input,
    _build_session_id,
    _ensure_token_metrics,
    _extract_metrics,
    _is_message_authorized,
    _needs_completion_retry,
    _parse_agent_hint,
    _resolve_context_paths_for_message,
    _run_agent,
    _select_agent_name,
    RuntimeConfig,
    load_runtime_config,
)


class _DummyChannel:
    def __init__(self, channel_id: int):
        self.id = channel_id


class _DummyAuthor:
    def __init__(self, user_id: int, roles: list[object] | None = None):
        self.id = user_id
        self.roles = roles or []


class _DummyGuild:
    def __init__(self, guild_id: int):
        self.id = guild_id


class _DummyMessage:
    def __init__(
        self,
        content: str,
        channel_id: int,
        user_id: int,
        guild_id: int | None = None,
        roles: list[object] | None = None,
    ):
        self.content = content
        self.channel = _DummyChannel(channel_id)
        self.author = _DummyAuthor(user_id, roles=roles)
        self.guild = _DummyGuild(guild_id) if guild_id is not None else None


class _DummyRole:
    def __init__(self, role_id: int):
        self.id = role_id


def test_parse_agent_hint() -> None:
    agent_hint, content = _parse_agent_hint("agent:architect plan the migration")
    assert agent_hint == "architect"
    assert content == "plan the migration"


def test_select_agent_name_from_rules() -> None:
    config = Config(
        data={
            "agents": {
                "routing": {
                    "rules": [
                        {
                            "agent": "delivery-team",
                            "contains": ["manage"],
                            "channel_ids": [],
                            "user_ids": [],
                            "guild_ids": [],
                        }
                    ]
                }
            }
        },
        path=Path("config.yaml"),
    )
    message = _DummyMessage("please manage this", channel_id=1, user_id=2, guild_id=3)
    assert _select_agent_name(config, message, None) == "delivery-team"


def test_collect_delegation_paths() -> None:
    response = SimpleNamespace(
        team_name="delivery-team",
        member_responses=[
            SimpleNamespace(agent_name="architect", member_responses=[]),
            SimpleNamespace(agent_name="software-engineer", member_responses=[]),
        ],
    )
    paths = _collect_delegation_paths(response)
    assert "delivery-team->architect" in paths
    assert "delivery-team->software-engineer" in paths


def test_build_response_input_adds_file_hint_for_file_requests() -> None:
    result = _build_response_input("CTX", "Please edit config.yaml and read README.md")
    assert isinstance(result, list)
    assert len(result) == 3
    assert result[1].role == "system"
    assert "use the 'file' tool functions" in (result[1].content or "")


def test_extract_metrics_accepts_dict_usage_shape() -> None:
    metrics = {
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 7,
        },
        "latency": 0.8,
    }
    extracted = _extract_metrics(metrics)
    assert extracted["input_tokens"] == 11
    assert extracted["output_tokens"] == 7
    assert extracted["total_tokens"] == 18
    assert extracted["latency"] == 0.8


def test_extract_metrics_accepts_object_shape() -> None:
    metrics = SimpleNamespace(
        input_tokens=5,
        output_tokens=3,
        total_tokens=8,
        time_to_first_token=0.2,
    )
    extracted = _extract_metrics(metrics)
    assert extracted["input_tokens"] == 5
    assert extracted["output_tokens"] == 3
    assert extracted["total_tokens"] == 8
    assert extracted["time_to_first_token"] == 0.2


def test_extract_metrics_accepts_response_usage_shape() -> None:
    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=13, completion_tokens=4),
        latency=1.2,
    )
    extracted = _extract_metrics(response)
    assert extracted["input_tokens"] == 13
    assert extracted["output_tokens"] == 4
    assert extracted["total_tokens"] == 17
    assert extracted["latency"] == 1.2


def test_ensure_token_metrics_estimates_when_missing_usage() -> None:
    metrics = _ensure_token_metrics({}, "hello world", "done")
    assert metrics["input_tokens"] > 0
    assert metrics["output_tokens"] > 0
    assert metrics["total_tokens"] == metrics["input_tokens"] + metrics["output_tokens"]
    assert metrics["token_estimated"] is True


def test_extract_metrics_aggregates_tokens_from_events() -> None:
    response = SimpleNamespace(
        metrics=None,
        events=[
            SimpleNamespace(input_tokens=20, output_tokens=5, total_tokens=25),
            SimpleNamespace(input_tokens=30, output_tokens=8, total_tokens=38),
        ],
    )
    extracted = _extract_metrics(response)
    assert extracted["input_tokens"] == 50
    assert extracted["output_tokens"] == 13
    assert extracted["total_tokens"] == 63
    assert extracted["token_source"] == "events"


def test_needs_completion_retry_for_placeholder_text() -> None:
    assert _needs_completion_retry("Let me check my configuration:")
    assert not _needs_completion_retry("Here are the exact changes and impacts:\n1. ...")


def test_run_agent_timeout_requests_cancellation(monkeypatch) -> None:
    class _SlowAgent:
        name = "slow"

        def run(self, *_args, **_kwargs):
            time.sleep(1.2)
            return SimpleNamespace(content="late")

    cancelled: list[str] = []

    def _fake_cancel(run_id: str) -> bool:
        cancelled.append(run_id)
        return True

    monkeypatch.setattr("bitdoze_bot.discord_bot._cancel_run", _fake_cancel)

    with pytest.raises(TimeoutError):
        asyncio.run(
            _run_agent(
                _SlowAgent(),
                "hello",
                None,
                "user-1",
                "session-1",
                RuntimeConfig(agent_timeout=1),
            )
        )

    assert len(cancelled) == 1
    assert cancelled[0].startswith("discord-")


def test_load_runtime_config_defaults_to_channel_user_session_strategy() -> None:
    cfg = load_runtime_config(Config(data={}, path=Path("config.yaml")))
    assert cfg.session_id_strategy == "channel_user"


def test_load_runtime_config_accepts_session_strategy_values() -> None:
    cfg = load_runtime_config(
        Config(data={"runtime": {"session_id_strategy": "user"}}, path=Path("config.yaml"))
    )
    assert cfg.session_id_strategy == "user"


def test_build_session_id_uses_requested_strategy() -> None:
    message = _DummyMessage("hello", channel_id=11, user_id=22, guild_id=33)
    assert _build_session_id(message, "channel") == "discord:33:channel:11"
    assert _build_session_id(message, "user") == "discord:33:user:22"
    assert _build_session_id(message, "channel_user") == "discord:33:channel:11:user:22"


def test_is_message_authorized_applies_user_channel_guild_and_role_filters() -> None:
    settings = _AccessControlSettings(
        allowed_user_ids=frozenset({22}),
        allowed_channel_ids=frozenset({11}),
        allowed_guild_ids=frozenset({33}),
        allowed_role_ids=frozenset({44}),
    )
    denied = _DummyMessage("hello", channel_id=11, user_id=22, guild_id=33, roles=[_DummyRole(55)])
    assert not _is_message_authorized(settings, denied)
    allowed = _DummyMessage("hello", channel_id=11, user_id=22, guild_id=33, roles=[_DummyRole(44)])
    assert _is_message_authorized(settings, allowed)


def test_resolve_context_paths_is_tenant_scoped() -> None:
    settings = _ContextSettings(
        use_workspace_context_files=True,
        user_path=Path("/tmp/global/USER.md"),
        memory_dir=Path("/tmp/global/memory"),
        long_memory_path=Path("/tmp/global/MEMORY.md"),
        scope_workspace_context_by_tenant=True,
        scoped_context_dir=Path("/tmp/scoped"),
        allow_global_context_in_guilds=False,
        main_session_scope="always",
        tzinfo=ZoneInfo("UTC"),
    )
    message = _DummyMessage("hello", channel_id=11, user_id=22, guild_id=33)
    user_path, memory_dir, long_memory_path = _resolve_context_paths_for_message(settings, message)
    assert user_path == Path("/tmp/scoped/guild-33/user-22/USER.md")
    assert memory_dir == Path("/tmp/scoped/guild-33/user-22/memory")
    assert long_memory_path == Path("/tmp/scoped/guild-33/user-22/MEMORY.md")


def test_build_user_context_skips_global_files_in_guild_when_not_scoped(tmp_path: Path) -> None:
    user_path = tmp_path / "global" / "USER.md"
    user_path.parent.mkdir(parents=True, exist_ok=True)
    user_path.write_text("secret", encoding="utf-8")
    settings = _ContextSettings(
        use_workspace_context_files=True,
        user_path=user_path,
        memory_dir=tmp_path / "global" / "memory",
        long_memory_path=tmp_path / "global" / "MEMORY.md",
        scope_workspace_context_by_tenant=False,
        scoped_context_dir=tmp_path / "scoped",
        allow_global_context_in_guilds=False,
        main_session_scope="always",
        tzinfo=ZoneInfo("UTC"),
    )
    message = _DummyMessage("hello", channel_id=11, user_id=22, guild_id=33)
    assert _build_user_context_from_settings(settings, message) == ""


def test_build_user_context_reads_scoped_files(tmp_path: Path) -> None:
    today = datetime.now(ZoneInfo("UTC")).date().isoformat()
    scoped_root = tmp_path / "scoped" / "guild-33" / "user-22"
    (scoped_root / "memory").mkdir(parents=True, exist_ok=True)
    (scoped_root / "USER.md").write_text("tenant user", encoding="utf-8")
    (scoped_root / "MEMORY.md").write_text("tenant memory", encoding="utf-8")
    (scoped_root / "memory" / f"{today}.md").write_text("daily", encoding="utf-8")
    settings = _ContextSettings(
        use_workspace_context_files=True,
        user_path=tmp_path / "global" / "USER.md",
        memory_dir=tmp_path / "global" / "memory",
        long_memory_path=tmp_path / "global" / "MEMORY.md",
        scope_workspace_context_by_tenant=True,
        scoped_context_dir=tmp_path / "scoped",
        allow_global_context_in_guilds=False,
        main_session_scope="always",
        tzinfo=ZoneInfo("UTC"),
    )
    message = _DummyMessage("hello", channel_id=11, user_id=22, guild_id=33)
    context = _build_user_context_from_settings(settings, message)
    assert "tenant user" in context
    assert "tenant memory" in context
