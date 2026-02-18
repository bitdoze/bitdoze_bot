from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

from bitdoze_bot.config import Config
from bitdoze_bot.discord_bot import (
    _collect_delegation_paths,
    _build_response_input,
    _ensure_token_metrics,
    _extract_metrics,
    _needs_completion_retry,
    _parse_agent_hint,
    _select_agent_name,
)


class _DummyChannel:
    def __init__(self, channel_id: int):
        self.id = channel_id


class _DummyAuthor:
    def __init__(self, user_id: int):
        self.id = user_id


class _DummyGuild:
    def __init__(self, guild_id: int):
        self.id = guild_id


class _DummyMessage:
    def __init__(self, content: str, channel_id: int, user_id: int, guild_id: int | None = None):
        self.content = content
        self.channel = _DummyChannel(channel_id)
        self.author = _DummyAuthor(user_id)
        self.guild = _DummyGuild(guild_id) if guild_id is not None else None


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
