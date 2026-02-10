from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable

from agno.agent import Agent

from bitdoze_bot.config import Config
from bitdoze_bot.tool_permissions import tool_runtime_context
from bitdoze_bot.utils import extract_response_text, read_text_if_exists

logger = logging.getLogger(__name__)

_DEFAULT_HEARTBEAT_TIMEOUT = 120


@dataclass
class HeartbeatConfig:
    enabled: bool
    interval_minutes: int
    quiet_ack: str
    prompt_path: str
    session_scope: str
    agent: str | None
    channel_id: int | None


def _safe_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def load_heartbeat_config(config: Config) -> HeartbeatConfig:
    hb_cfg = config.get("heartbeat", default={})
    agent_value = hb_cfg.get("agent")
    agent = str(agent_value).strip() if agent_value is not None else None
    if not agent:
        agent = None
    session_scope = str(hb_cfg.get("session_scope", "isolated")).strip().lower()
    if session_scope not in {"shared", "isolated"}:
        session_scope = "isolated"
    raw_channel_id = hb_cfg.get("channel_id")
    channel_id: int | None = None
    if raw_channel_id is not None:
        try:
            channel_id = int(raw_channel_id)
        except (TypeError, ValueError):
            channel_id = None
    return HeartbeatConfig(
        enabled=bool(hb_cfg.get("enabled", True)),
        interval_minutes=_safe_positive_int(hb_cfg.get("interval_minutes", 30), 30),
        quiet_ack=str(hb_cfg.get("quiet_ack", "HEARTBEAT_OK")),
        prompt_path=str(hb_cfg.get("prompt_path", "workspace/HEARTBEAT.md")),
        session_scope=session_scope,
        agent=agent,
        channel_id=channel_id,
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
    timeout: int | None = None,
) -> None:
    effective_timeout = timeout or _DEFAULT_HEARTBEAT_TIMEOUT
    user_id, session_id = resolve_heartbeat_identity(session_scope)
    agent_name = str(getattr(agent, "name", "unknown"))
    with tool_runtime_context(
        run_kind="heartbeat",
        user_id=user_id,
        session_id=session_id,
        agent_name=agent_name,
    ):
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(agent.run, prompt, user_id=user_id, session_id=session_id),
                timeout=effective_timeout,
            )
        except TimeoutError:
            logger.warning("Heartbeat run timed out after %ds", effective_timeout)
            return
    content = extract_response_text(response).strip()
    if not content:
        return
    if content.strip().startswith(quiet_ack):
        return
    await send_fn(content)
