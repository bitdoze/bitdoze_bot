from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

from bitdoze_bot.config import Config
from bitdoze_bot.discord_bot import (
    _collect_delegation_paths,
    _build_response_input,
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
