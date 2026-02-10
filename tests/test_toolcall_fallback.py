from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from bitdoze_bot import discord_bot
from bitdoze_bot.config import Config
from bitdoze_bot.discord_bot import (
    RuntimeConfig,
    _handle_toolcall_fallback,
    _parse_tool_calls,
    _run_tool_calls,
    load_runtime_config,
)


def test_parse_tool_calls_supports_plain_arg_key_format() -> None:
    text = (
        'Checking status...'
        '<tool_call>run_shell_command'
        '<arg_key>args</arg_key>'
        '<arg_value>["sudo","docker","ps","-a"]</arg_value>'
        '</tool_call>'
    )
    calls = _parse_tool_calls(text)
    assert calls == [{"name": "run_shell_command", "params": {"args": ["sudo", "docker", "ps", "-a"]}}]


def test_handle_toolcall_fallback_hides_markup_when_no_execution(monkeypatch) -> None:
    async def fake_run_tool_calls(agent, calls, denied_tools=None):
        return []

    monkeypatch.setattr(discord_bot, "_run_tool_calls", fake_run_tool_calls)
    reply = (
        "Checking container status."
        "<tool_call>run_shell_command<arg_key>args</arg_key>"
        "<arg_value>[\"docker\",\"ps\"]</arg_value></tool_call>"
    )
    output = asyncio.run(_handle_toolcall_fallback(object(), reply))
    assert "<tool_call>" not in output
    assert "Checking container status." in output
    assert "could not execute" in output.lower()


def test_load_runtime_config_default_fallback_denied_tools() -> None:
    cfg = load_runtime_config(Config(data={}, path=Path("config.yaml")))
    assert cfg.fallback_denied_tools == ("shell", "discord")


def test_load_runtime_config_custom_fallback_denied_tools() -> None:
    cfg = load_runtime_config(
        Config(data={"tool_fallback": {"denied_tools": []}}, path=Path("config.yaml"))
    )
    assert cfg.fallback_denied_tools == ()


def test_handle_toolcall_fallback_passes_runtime_denied_tools(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_run_tool_calls(agent, calls, denied_tools):
        captured["denied_tools"] = denied_tools
        return []

    monkeypatch.setattr(discord_bot, "_run_tool_calls", fake_run_tool_calls)
    reply = (
        "x"
        "<tool_call>run_shell_command<arg_key>args</arg_key>"
        "<arg_value>[\"docker\",\"ps\"]</arg_value></tool_call>"
    )
    cfg = RuntimeConfig(fallback_denied_tools=("discord",))
    asyncio.run(_handle_toolcall_fallback(object(), reply, cfg))
    assert captured["denied_tools"] == {"discord"}


def test_handle_toolcall_fallback_reports_no_output_without_summarizer(monkeypatch) -> None:
    async def fake_run_tool_calls(agent, calls, denied_tools=None):
        return [{"name": "run_shell_command", "output": "(no output)"}]

    monkeypatch.setattr(discord_bot, "_run_tool_calls", fake_run_tool_calls)
    reply = "<tool_call>run_shell_command<arg_key>args</arg_key><arg_value>[\"docker\",\"ps\"]</arg_value></tool_call>"
    out = asyncio.run(_handle_toolcall_fallback(object(), reply))
    assert "returned no output" in out.lower()
    assert "run_shell_command" in out


def test_run_tool_calls_uses_declared_entrypoint_when_attr_missing() -> None:
    class DummyTool:
        _bitdoze_tool_name = "shell"

        def get_functions(self):
            def _entrypoint(**kwargs):
                return f"ok:{kwargs.get('args')}"

            return {"run_shell_command": SimpleNamespace(entrypoint=_entrypoint)}

    class DummyAgent:
        tools = [DummyTool()]

    calls = [{"name": "run_shell_command", "params": {"args": ["docker", "ps"]}}]
    results = asyncio.run(_run_tool_calls(DummyAgent(), calls, denied_tools=set()))
    assert len(results) == 1
    assert results[0]["name"] == "run_shell_command"
    assert "docker" in results[0]["output"]
