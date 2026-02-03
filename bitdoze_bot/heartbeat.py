from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable, Awaitable

from agno.agent import Agent

from bitdoze_bot.config import Config
from bitdoze_bot.utils import read_text_if_exists


@dataclass
class HeartbeatConfig:
    enabled: bool
    interval_minutes: int
    quiet_ack: str
    prompt_path: str


def load_heartbeat_config(config: Config) -> HeartbeatConfig:
    hb_cfg = config.get("heartbeat", default={})
    return HeartbeatConfig(
        enabled=bool(hb_cfg.get("enabled", True)),
        interval_minutes=int(hb_cfg.get("interval_minutes", 30)),
        quiet_ack=str(hb_cfg.get("quiet_ack", "HEARTBEAT_OK")),
        prompt_path=str(hb_cfg.get("prompt_path", "workspace/HEARTBEAT.md")),
    )


def build_heartbeat_prompt(config: HeartbeatConfig) -> str:
    prompt = read_text_if_exists(config.prompt_path)
    if prompt:
        return prompt
    return (
        "Run a brief heartbeat check-in. If there is nothing to report, "
        f"respond with {config.quiet_ack} and nothing else."
    )


async def run_heartbeat(
    agent: Agent,
    prompt: str,
    quiet_ack: str,
    send_fn: Callable[[str], Awaitable[None]],
) -> None:
    response = await asyncio.to_thread(agent.run, prompt, user_id="heartbeat", session_id="heartbeat")
    content = getattr(response, "content", None) or str(response)
    if content.strip().startswith(quiet_ack):
        return
    await send_fn(content)
