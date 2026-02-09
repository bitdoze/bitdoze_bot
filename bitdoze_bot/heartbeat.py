from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Awaitable

from agno.agent import Agent

from bitdoze_bot.config import Config
from bitdoze_bot.tool_permissions import tool_runtime_context
from bitdoze_bot.utils import read_text_if_exists


@dataclass
class HeartbeatConfig:
    enabled: bool
    interval_minutes: int
    quiet_ack: str
    prompt_path: str
    session_scope: str
    agent: str | None


def load_heartbeat_config(config: Config) -> HeartbeatConfig:
    hb_cfg = config.get("heartbeat", default={})
    agent_value = hb_cfg.get("agent")
    agent = str(agent_value).strip() if agent_value is not None else None
    if not agent:
        agent = None
    session_scope = str(hb_cfg.get("session_scope", "isolated")).strip().lower()
    if session_scope not in {"shared", "isolated"}:
        session_scope = "isolated"
    return HeartbeatConfig(
        enabled=bool(hb_cfg.get("enabled", True)),
        interval_minutes=int(hb_cfg.get("interval_minutes", 30)),
        quiet_ack=str(hb_cfg.get("quiet_ack", "HEARTBEAT_OK")),
        prompt_path=str(hb_cfg.get("prompt_path", "workspace/HEARTBEAT.md")),
        session_scope=session_scope,
        agent=agent,
    )


def build_heartbeat_prompt(config: HeartbeatConfig) -> str:
    prompt = read_text_if_exists(config.prompt_path)
    if prompt:
        return prompt
    return (
        "Run a brief heartbeat check-in. If there is nothing to report, "
        f"respond with {config.quiet_ack} and nothing else."
    )


def resolve_heartbeat_identity(session_scope: str, now: datetime | None = None) -> tuple[str, str]:
    if session_scope == "shared":
        return "heartbeat", "heartbeat"
    current = now or datetime.now(tz=timezone.utc)
    stamp = current.strftime("%Y%m%dT%H%M%SZ")
    identity = f"heartbeat:{stamp}"
    return identity, identity


async def run_heartbeat(
    agent: Agent,
    prompt: str,
    quiet_ack: str,
    send_fn: Callable[[str], Awaitable[None]],
    session_scope: str = "isolated",
) -> None:
    user_id, session_id = resolve_heartbeat_identity(session_scope)
    agent_name = str(getattr(agent, "name", "unknown"))
    with tool_runtime_context(
        run_kind="heartbeat",
        user_id=user_id,
        session_id=session_id,
        agent_name=agent_name,
    ):
        response = await asyncio.to_thread(agent.run, prompt, user_id=user_id, session_id=session_id)
    content = getattr(response, "content", None) or str(response)
    if content.strip().startswith(quiet_ack):
        return
    await send_fn(content)
