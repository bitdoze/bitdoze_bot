from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Awaitable

from agno.agent import Agent

from bitdoze_bot.config import Config
from bitdoze_bot.run_monitor import RunMonitor
from bitdoze_bot.tool_permissions import tool_runtime_context
from bitdoze_bot.utils import extract_response_text, read_text_if_exists

logger = logging.getLogger(__name__)

_DEFAULT_HEARTBEAT_TIMEOUT = 120


@dataclass
class HeartbeatConfig:
    enabled: bool
    interval_minutes: int
    quiet_ack: str
    prompt_path: Path
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
        prompt_path=config.resolve_path(hb_cfg.get("prompt_path"), default="workspace/HEARTBEAT.md"),
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


def _extract_metrics(metrics: Any) -> dict[str, Any]:
    if metrics is None:
        return {}
    snapshot: dict[str, Any] = {}
    usage = getattr(metrics, "usage", None)
    if isinstance(metrics, dict):
        usage = metrics.get("usage", usage)
    for normalized_key, keys in (
        ("input_tokens", ("input_tokens", "prompt_tokens")),
        ("output_tokens", ("output_tokens", "completion_tokens")),
        ("total_tokens", ("total_tokens",)),
    ):
        value = None
        for key in keys:
            value = metrics.get(key) if isinstance(metrics, dict) else getattr(metrics, key, None)
            if value is not None:
                break
            if isinstance(usage, dict):
                value = usage.get(key)
            elif usage is not None:
                value = getattr(usage, key, None)
            if value is not None:
                break
        if value is not None:
            snapshot[normalized_key] = value
    if "total_tokens" not in snapshot and "input_tokens" in snapshot and "output_tokens" in snapshot:
        try:
            snapshot["total_tokens"] = int(snapshot["input_tokens"]) + int(snapshot["output_tokens"])
        except (TypeError, ValueError):
            pass
    return snapshot


def _estimate_tokens_from_text(value: str) -> int:
    text = value.strip()
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _ensure_token_metrics(metrics: dict[str, Any], prompt: str, output_text: str) -> dict[str, Any]:
    merged = dict(metrics)
    if (
        merged.get("input_tokens") is None
        or merged.get("output_tokens") is None
        or merged.get("total_tokens") is None
    ):
        in_tokens = _estimate_tokens_from_text(prompt)
        out_tokens = _estimate_tokens_from_text(output_text)
        merged["input_tokens"] = in_tokens
        merged["output_tokens"] = out_tokens
        merged["total_tokens"] = in_tokens + out_tokens
        merged["token_estimated"] = True
    else:
        merged["token_estimated"] = False
    return merged


async def run_heartbeat(
    agent: Agent,
    prompt: str,
    quiet_ack: str,
    send_fn: Callable[[str], Awaitable[None]],
    session_scope: str = "isolated",
    timeout: int | None = None,
    monitor: RunMonitor | None = None,
) -> None:
    effective_timeout = timeout or _DEFAULT_HEARTBEAT_TIMEOUT
    user_id, session_id = resolve_heartbeat_identity(session_scope)
    agent_name = str(getattr(agent, "name", "unknown"))
    monitor_token = None
    if monitor is not None:
        monitor_token = monitor.start(
            run_kind="heartbeat",
            target_name=agent_name,
            user_id=user_id,
            session_id=session_id,
            timeout_seconds=effective_timeout,
        )
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
            if monitor is not None:
                monitor.finish(monitor_token, status="timeout", error="heartbeat timeout")
            return
        except Exception as exc:  # noqa: BLE001
            if monitor is not None:
                monitor.finish(monitor_token, status="error", error=str(exc))
            raise
    content = extract_response_text(response).strip()
    metrics = _ensure_token_metrics(_extract_metrics(response), prompt, content)
    logger.info(
        "Heartbeat completed run_id=%s model=%s input_tokens=%s output_tokens=%s total_tokens=%s token_estimated=%s",
        getattr(response, "run_id", None),
        getattr(response, "model", None),
        metrics.get("input_tokens"),
        metrics.get("output_tokens"),
        metrics.get("total_tokens"),
        metrics.get("token_estimated"),
    )
    if monitor is not None:
        monitor.finish(
            monitor_token,
            status="completed",
            run_id=getattr(response, "run_id", None),
            model=getattr(response, "model", None),
            metrics=metrics or {},
        )
    if not content:
        return
    if content.strip().startswith(quiet_ack):
        return
    await send_fn(content)
