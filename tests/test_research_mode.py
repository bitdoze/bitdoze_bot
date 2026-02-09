from __future__ import annotations

import asyncio
from pathlib import Path

from bitdoze_bot.config import Config
from bitdoze_bot import discord_bot
from bitdoze_bot.discord_bot import (
    ResearchModeConfig,
    _RESEARCH_FAILURE_MESSAGE,
    _is_research_mode_request,
    _load_research_mode_config,
    _run_research_mode,
    _validate_research_response,
)


def test_load_research_mode_config_defaults() -> None:
    cfg = _load_research_mode_config(Config(data={}, path=Path("config.yaml")))

    assert cfg.enabled is True
    assert cfg.trigger_on_prefix is True
    assert cfg.trigger_on_research_agent is True
    assert cfg.research_agent_name == "research"
    assert cfg.min_sources == 3


def test_is_research_mode_request_by_prefix() -> None:
    cfg = ResearchModeConfig()
    assert _is_research_mode_request(cfg, "research: python http clients", "main") is True


def test_is_research_mode_request_by_routed_agent() -> None:
    cfg = ResearchModeConfig(trigger_on_prefix=False, trigger_on_research_agent=True)
    assert _is_research_mode_request(cfg, "summarize this", "research") is True


def test_validate_research_response_success() -> None:
    text = """TL;DR
+ short

Findings
- A

Risks
- B

Sources
- https://a.example.com/x
- https://b.example.com/y
- https://c.example.com/z
"""
    result = _validate_research_response(text, min_sources=3)

    assert result.valid is True
    assert result.missing_sections == ()
    assert result.source_count == 3


def test_validate_research_response_fails_when_sources_missing() -> None:
    text = """TL;DR: short
Findings:
- A
Risks:
- B
Sources:
- https://a.example.com/x
"""
    result = _validate_research_response(text, min_sources=3)

    assert result.valid is False
    assert result.missing_sections == ()
    assert result.source_count == 1


def test_run_research_mode_retries_once_and_returns_valid(monkeypatch) -> None:
    calls: list[str] = []
    responses = [
        "TL;DR\nshort\n\nFindings\n- one\n\nRisks\n- one\n\nSources\n- https://a.example.com",
        "TL;DR\nshort\n\nFindings\n- one\n\nRisks\n- one\n\nSources\n- https://a.example.com\n- https://b.example.com\n- https://c.example.com",
    ]

    async def fake_run_agent(agent, content, user_context, user_id, session_id, runtime_cfg=None):
        calls.append(content)
        return responses[len(calls) - 1]

    monkeypatch.setattr(discord_bot, "_run_agent", fake_run_agent)

    reply = asyncio.run(
        _run_research_mode(
            agent=object(),
            content="research: bitdoze",
            user_context=None,
            user_id="u",
            session_id="s",
            research_cfg=ResearchModeConfig(min_sources=3),
        )
    )

    assert len(calls) == 2
    assert reply.count("https://") == 3


def test_run_research_mode_returns_failure_after_retry(monkeypatch) -> None:
    calls = 0

    async def fake_run_agent(agent, content, user_context, user_id, session_id, runtime_cfg=None):
        nonlocal calls
        calls += 1
        return "TL;DR\nshort\n\nFindings\n- one\n\nRisks\n- one\n\nSources\n- https://a.example.com"

    monkeypatch.setattr(discord_bot, "_run_agent", fake_run_agent)

    reply = asyncio.run(
        _run_research_mode(
            agent=object(),
            content="research: bitdoze",
            user_context=None,
            user_id="u",
            session_id="s",
            research_cfg=ResearchModeConfig(min_sources=3),
        )
    )

    assert calls == 2
    assert reply == _RESEARCH_FAILURE_MESSAGE
