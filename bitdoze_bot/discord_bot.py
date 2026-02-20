from __future__ import annotations

import asyncio
import logging
import os
import re
from uuid import uuid4
from functools import lru_cache
from time import perf_counter
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from dataclasses import dataclass
from typing import Any, cast

import discord
from discord.ext import commands, tasks

from agno.run.agent import RunEvent, RunOutput
from agno.run.cancel import cancel_run as _cancel_run_global

from bitdoze_bot.agents import AgentRegistry, build_agents
from bitdoze_bot.cognee import CogneeClient, CogneeConfig, load_cognee_config
from bitdoze_bot.config import Config, load_config
from bitdoze_bot.cron import (
    build_cron_trigger,
    build_scheduler,
    get_cron_path,
    load_cron_config,
    run_cron_job,
)
from bitdoze_bot.heartbeat import (
    build_heartbeat_prompt,
    load_heartbeat_config,
    run_heartbeat,
)
from bitdoze_bot.discord_runtime_utils import (
    build_response_input as _build_response_input,
    collect_delegation_paths as _collect_delegation_paths,
    ensure_token_metrics as _ensure_token_metrics,
    extract_metrics as _extract_metrics,
    is_team_target as _is_team_target,
    needs_completion_retry as _needs_completion_retry,
    target_members as _target_members,
)
from bitdoze_bot.discord_tool_fallback import (
    parse_tool_calls as _parse_tool_calls,
    run_tool_calls as _run_tool_calls,
    strip_tool_call_markup as _strip_tool_call_markup,
)
from bitdoze_bot.run_monitor import RunMonitor
from bitdoze_bot.tool_permissions import ToolPermissionError, tool_runtime_context
from bitdoze_bot.utils import extract_response_text, parse_bool, read_text_if_exists

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeConfig:
    agent_timeout: int = 600
    cron_timeout: int = 600
    heartbeat_timeout: int = 120
    max_concurrent_runs: int = 4
    slow_run_threshold_seconds: int = 0
    streaming_enabled: bool = True
    streaming_edit_interval: float = 1.5
    watchdog_enabled: bool = True
    watchdog_threshold_seconds: int = 120
    watchdog_max_reports: int = 5
    fallback_denied_tools: tuple[str, ...] = ("shell", "discord")
    session_id_strategy: str = "channel_user"
    monitor: RunMonitor | None = None


def _safe_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _parse_non_negative_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def load_runtime_config(config: Config) -> RuntimeConfig:
    raw = config.get("runtime", default={})
    cfg = raw if isinstance(raw, dict) else {}
    monitoring_raw = config.get("monitoring", default={})
    monitoring_cfg = monitoring_raw if isinstance(monitoring_raw, dict) else {}
    fallback_raw = config.get("tool_fallback", default={})
    fallback_cfg = fallback_raw if isinstance(fallback_raw, dict) else {}
    session_id_strategy_raw = str(cfg.get("session_id_strategy", "channel_user")).strip().lower()
    if session_id_strategy_raw not in {"channel", "user", "channel_user"}:
        session_id_strategy_raw = "channel_user"
    denied_tools_cfg = fallback_cfg.get("denied_tools", ["shell", "discord"])
    denied_tools: tuple[str, ...] = ("shell", "discord")
    if isinstance(denied_tools_cfg, list):
        normalized = [str(item).strip().lower() for item in denied_tools_cfg if str(item).strip()]
        denied_tools = tuple(normalized)
    monitor = RunMonitor(
        enabled=parse_bool(monitoring_cfg.get("telemetry_enabled", True), True),
        telemetry_path=str(
            config.resolve_path(
                monitoring_cfg.get("telemetry_path"),
                default="logs/run-telemetry.jsonl",
            )
        ),
    )
    return RuntimeConfig(
        agent_timeout=_safe_positive_int(cfg.get("agent_timeout"), 600),
        cron_timeout=_safe_positive_int(cfg.get("cron_timeout"), 600),
        heartbeat_timeout=_safe_positive_int(cfg.get("heartbeat_timeout"), 120),
        max_concurrent_runs=_safe_positive_int(cfg.get("max_concurrent_runs"), 4),
        slow_run_threshold_seconds=_parse_non_negative_int(cfg.get("slow_run_threshold_seconds", 0), 0),
        streaming_enabled=parse_bool(cfg.get("streaming_enabled", True), True),
        streaming_edit_interval=max(float(cfg.get("streaming_edit_interval", 1.5) or 1.5), 0.5),
        watchdog_enabled=parse_bool(monitoring_cfg.get("watchdog_enabled", True), True),
        watchdog_threshold_seconds=max(
            _parse_non_negative_int(monitoring_cfg.get("watchdog_threshold_seconds", 120), 120),
            1,
        ),
        watchdog_max_reports=max(
            _parse_non_negative_int(monitoring_cfg.get("watchdog_max_reports", 5), 5),
            1,
        ),
        fallback_denied_tools=denied_tools,
        session_id_strategy=session_id_strategy_raw,
        monitor=monitor,
    )


def _load_access_control_settings(config: Config) -> _AccessControlSettings:
    discord_cfg = config.get("discord", default={})
    discord_map = discord_cfg if isinstance(discord_cfg, dict) else {}
    access_cfg = discord_map.get("access_control", {})
    access_map = access_cfg if isinstance(access_cfg, dict) else {}
    return _AccessControlSettings(
        allowed_user_ids=_coerce_int_set(access_map.get("allowed_user_ids")),
        allowed_channel_ids=_coerce_int_set(access_map.get("allowed_channel_ids")),
        allowed_guild_ids=_coerce_int_set(access_map.get("allowed_guild_ids")),
        allowed_role_ids=_coerce_int_set(access_map.get("allowed_role_ids")),
    )


@dataclass
class DiscordRuntime:
    config: Config
    agents: AgentRegistry
    runtime_cfg: RuntimeConfig


@dataclass(frozen=True)
class _CompiledRoutingRule:
    agent: str | None
    channel_ids: frozenset[int]
    user_ids: frozenset[int]
    guild_ids: frozenset[int]
    contains: tuple[str, ...]
    starts_with: tuple[str, ...]


@dataclass(frozen=True)
class _ContextSettings:
    use_workspace_context_files: bool
    user_path: Path
    memory_dir: Path
    long_memory_path: Path
    scope_workspace_context_by_tenant: bool
    scoped_context_dir: Path
    allow_global_context_in_guilds: bool
    main_session_scope: str
    tzinfo: ZoneInfo


@dataclass(frozen=True)
class _AccessControlSettings:
    allowed_user_ids: frozenset[int]
    allowed_channel_ids: frozenset[int]
    allowed_guild_ids: frozenset[int]
    allowed_role_ids: frozenset[int]


def _coerce_int_set(values: Any) -> frozenset[int]:
    if not isinstance(values, list):
        return frozenset()
    parsed: set[int] = set()
    for value in values:
        try:
            parsed.add(int(value))
        except (TypeError, ValueError):
            continue
    return frozenset(parsed)


def _normalize_tokens(values: Any) -> tuple[str, ...]:
    if not isinstance(values, list):
        return ()
    return tuple(str(value).lower() for value in values if str(value).strip())


def _compile_routing_rules(config: Config) -> tuple[_CompiledRoutingRule, ...]:
    routing = config.get("agents", "routing", default={})
    rules = routing.get("rules", []) if isinstance(routing, dict) else []
    compiled: list[_CompiledRoutingRule] = []
    for raw_rule in rules or []:
        if not isinstance(raw_rule, dict):
            continue
        compiled.append(
            _CompiledRoutingRule(
                agent=cast(str | None, raw_rule.get("agent")),
                channel_ids=_coerce_int_set(raw_rule.get("channel_ids")),
                user_ids=_coerce_int_set(raw_rule.get("user_ids")),
                guild_ids=_coerce_int_set(raw_rule.get("guild_ids")),
                contains=_normalize_tokens(raw_rule.get("contains")),
                starts_with=_normalize_tokens(raw_rule.get("starts_with")),
            )
        )
    return tuple(compiled)


def _load_context_settings(config: Config) -> _ContextSettings:
    context_cfg = config.get("context", default={})
    cfg = context_cfg if isinstance(context_cfg, dict) else {}
    tz_name = cfg.get("timezone_identifier", "UTC")
    try:
        tzinfo = ZoneInfo(tz_name)
    except Exception:  # noqa: BLE001
        tzinfo = ZoneInfo("UTC")
    return _ContextSettings(
        use_workspace_context_files=bool(cfg.get("use_workspace_context_files", True)),
        user_path=config.resolve_path(cfg.get("user_path"), default="workspace/USER.md"),
        memory_dir=config.resolve_path(cfg.get("memory_dir"), default="workspace/memory"),
        long_memory_path=config.resolve_path(cfg.get("long_memory_path"), default="workspace/MEMORY.md"),
        scope_workspace_context_by_tenant=parse_bool(cfg.get("scope_workspace_context_by_tenant", True), True),
        scoped_context_dir=config.resolve_path(cfg.get("scoped_context_dir"), default="workspace/context"),
        allow_global_context_in_guilds=parse_bool(cfg.get("allow_global_context_in_guilds", False), False),
        main_session_scope=str(cfg.get("main_session_scope", "dm_only")),
        tzinfo=tzinfo,
    )


def _is_message_authorized(settings: _AccessControlSettings, message: discord.Message) -> bool:
    if settings.allowed_user_ids and message.author.id not in settings.allowed_user_ids:
        return False
    if settings.allowed_channel_ids and message.channel.id not in settings.allowed_channel_ids:
        return False
    if message.guild is not None:
        if settings.allowed_guild_ids and message.guild.id not in settings.allowed_guild_ids:
            return False
        if settings.allowed_role_ids:
            role_ids = set(_extract_role_ids(message))
            if role_ids.isdisjoint(settings.allowed_role_ids):
                return False
    return True


def _build_session_id(message: discord.Message, strategy: str) -> str:
    guild_segment = str(message.guild.id) if message.guild else "dm"
    if strategy == "channel":
        return f"discord:{guild_segment}:channel:{message.channel.id}"
    if strategy == "user":
        return f"discord:{guild_segment}:user:{message.author.id}"
    return f"discord:{guild_segment}:channel:{message.channel.id}:user:{message.author.id}"


def _resolve_context_paths_for_message(
    settings: _ContextSettings,
    message: discord.Message,
) -> tuple[Path, Path, Path]:
    if not settings.scope_workspace_context_by_tenant:
        return settings.user_path, settings.memory_dir, settings.long_memory_path
    if message.guild is None:
        context_root = settings.scoped_context_dir / f"dm-user-{message.author.id}"
    else:
        context_root = settings.scoped_context_dir / f"guild-{message.guild.id}" / f"user-{message.author.id}"
    return (
        context_root / "USER.md",
        context_root / "memory",
        context_root / "MEMORY.md",
    )


@lru_cache(maxsize=8)
def _bot_mention_pattern(bot_user_id: int) -> re.Pattern[str]:
    return re.compile(rf"<@!?{bot_user_id}>")


def _strip_bot_mention(content: str, bot_user_id: int) -> str:
    cleaned = _bot_mention_pattern(bot_user_id).sub("", content)
    return cleaned.strip()


def _parse_agent_hint(content: str) -> tuple[str | None, str]:
    # Optional format: "agent:NAME <message>"
    lowered = content.lower()
    if lowered.startswith("agent:"):
        parts = content.split(maxsplit=1)
        if parts:
            agent_part = parts[0][len("agent:") :].strip()
            message_part = parts[1] if len(parts) > 1 else ""
            return agent_part or None, message_part.strip()
    return None, content


def _clean_cognee_result_item(item: str) -> str:
    raw = item.strip()
    if not raw:
        return ""

    # Stored Cognee memories include metadata headers followed by a blank line.
    # Prefer the body so injected context contains user facts, not IDs/timestamps.
    sections = raw.split("\n\n")
    if len(sections) >= 2:
        header = sections[0].lower()
        if "timestamp:" in header and ("user_id:" in header or "session_id:" in header):
            candidate = "\n\n".join(sections[1:]).strip()
            if candidate:
                raw = candidate

    cleaned = " ".join(raw.split())
    for marker in (
        "USER (summary):",
        "ASSISTANT (summary):",
        "role: user",
        "role: assistant",
    ):
        cleaned = cleaned.replace(marker, "").strip()
    return cleaned


_run_semaphore: asyncio.Semaphore | None = None
_run_semaphore_size: int | None = None


def _get_run_semaphore(max_concurrent: int = 4) -> asyncio.Semaphore:
    global _run_semaphore, _run_semaphore_size  # noqa: PLW0603
    if _run_semaphore is None or _run_semaphore_size != max_concurrent:
        _run_semaphore = asyncio.Semaphore(max_concurrent)
        _run_semaphore_size = max_concurrent
    return _run_semaphore


def _cancel_run(run_id: str) -> bool:
    try:
        return bool(_cancel_run_global(run_id))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to cancel run run_id=%s error=%s", run_id, exc)
        return False


async def _run_agent(
    agent,
    content: str,
    user_context: str | None,
    user_id: str,
    session_id: str,
    runtime_cfg: RuntimeConfig | None = None,
) -> str:
    cfg = runtime_cfg or RuntimeConfig()
    response_input = _build_response_input(user_context, content)
    target_name = _agent_name(agent)
    target_kind = "team" if _is_team_target(agent) else "agent"
    members = _target_members(agent)
    monitor_token = None
    if cfg.monitor is not None:
        monitor_token = cfg.monitor.start(
            run_kind="discord",
            target_name=target_name,
            user_id=user_id,
            session_id=session_id,
            timeout_seconds=cfg.agent_timeout,
        )
    logger.info(
        "Target run started kind=%s name=%s session_id=%s user_id=%s members=%s",
        target_kind,
        target_name,
        session_id,
        user_id,
        ", ".join(members) if members else "none",
    )
    started_at = perf_counter()
    run_id = f"discord-{uuid4().hex}"
    run_task: asyncio.Task[Any] | None = None
    try:
        async with _get_run_semaphore(cfg.max_concurrent_runs):
            run_task = asyncio.create_task(
                asyncio.to_thread(
                    agent.run,
                    response_input,
                    user_id=user_id,
                    session_id=session_id,
                    run_id=run_id,
                ),
            )
            response = await asyncio.wait_for(
                asyncio.shield(run_task),
                timeout=cfg.agent_timeout,
            )
    except TimeoutError as exc:
        cancelled = _cancel_run(run_id)
        logger.warning(
            "Target run timed out kind=%s name=%s run_id=%s cancel_requested=%s",
            target_kind,
            target_name,
            run_id,
            cancelled,
        )
        if run_task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(run_task), timeout=3)
            except Exception:  # noqa: BLE001
                pass
        if cfg.monitor is not None:
            cfg.monitor.finish(
                monitor_token,
                status="timeout",
                error=str(exc) or "agent timeout",
            )
        raise
    except Exception as exc:  # noqa: BLE001
        if cfg.monitor is not None:
            cfg.monitor.finish(
                monitor_token,
                status="error",
                error=str(exc),
            )
        raise
    elapsed_ms = int((perf_counter() - started_at) * 1000)
    run_id = getattr(response, "run_id", None)
    model = getattr(response, "model", None)
    output_text = extract_response_text(response)
    metrics = _ensure_token_metrics(_extract_metrics(response), response_input, output_text)
    delegation_paths = _collect_delegation_paths(response) if _is_team_target(agent) else []
    logger.info(
        (
            "Target run completed kind=%s name=%s run_id=%s model=%s elapsed_ms=%d "
            "input_tokens=%s output_tokens=%s total_tokens=%s token_estimated=%s metrics=%s delegation_paths=%s"
        ),
        target_kind,
        target_name,
        run_id,
        model,
        elapsed_ms,
        metrics.get("input_tokens"),
        metrics.get("output_tokens"),
        metrics.get("total_tokens"),
        metrics.get("token_estimated"),
        metrics or {},
        " | ".join(delegation_paths) if delegation_paths else "n/a",
    )
    if cfg.monitor is not None:
        cfg.monitor.finish(
            monitor_token,
            status="completed",
            run_id=run_id,
            model=model,
            metrics=metrics or {},
            extra={
                "target_kind": target_kind,
                "members": members,
                "elapsed_ms": elapsed_ms,
                "metrics": metrics or {},
                "delegation_paths": delegation_paths,
            },
        )
    return output_text


def _iter_stream_in_thread(agent, response_input, user_id: str, session_id: str, run_id: str):
    """Run agent.run(stream=True) in a thread and yield events via a queue."""
    import queue as queue_mod

    q: queue_mod.Queue = queue_mod.Queue()

    def _produce():
        try:
            for event in agent.run(
                response_input,
                stream=True,
                user_id=user_id,
                session_id=session_id,
                run_id=run_id,
            ):
                q.put(event)
        except Exception as exc:  # noqa: BLE001
            q.put(exc)
        finally:
            q.put(None)

    return q, _produce


async def _run_agent_streaming(
    agent,
    content: str,
    user_context: str | None,
    user_id: str,
    session_id: str,
    channel: discord.abc.Messageable,
    runtime_cfg: RuntimeConfig | None = None,
) -> tuple[str, discord.Message | None]:
    """Run agent with streaming, progressively editing a Discord message."""
    cfg = runtime_cfg or RuntimeConfig()
    response_input = _build_response_input(user_context, content)
    target_name = _agent_name(agent)
    monitor_token = None
    if cfg.monitor is not None:
        monitor_token = cfg.monitor.start(
            run_kind="discord",
            target_name=target_name,
            user_id=user_id,
            session_id=session_id,
            timeout_seconds=cfg.agent_timeout,
        )
    started_at = perf_counter()
    accumulated = ""
    sent_message: discord.Message | None = None
    last_edit_time: float = 0
    edit_interval = cfg.streaming_edit_interval
    final_response: Any = None
    run_id = f"discord-{uuid4().hex}"

    try:
        async with _get_run_semaphore(cfg.max_concurrent_runs):
            q, produce = _iter_stream_in_thread(
                agent,
                response_input,
                user_id,
                session_id,
                run_id,
            )
            loop = asyncio.get_event_loop()
            thread = loop.run_in_executor(None, produce)

            deadline = loop.time() + cfg.agent_timeout
            while True:
                if loop.time() > deadline:
                    _cancel_run(run_id)
                    raise TimeoutError("agent timeout")
                try:
                    import queue as queue_mod

                    event = await asyncio.wait_for(
                        loop.run_in_executor(None, q.get, True, 2.0),
                        timeout=3.0,
                    )
                except (TimeoutError, queue_mod.Empty):
                    continue

                if event is None:
                    break
                if isinstance(event, Exception):
                    raise event

                if isinstance(event, RunOutput):
                    final_response = event
                    text = extract_response_text(event)
                    if text:
                        accumulated = text
                    break

                event_type = getattr(event, "event", "")
                if event_type == RunEvent.run_content.value:
                    delta = getattr(event, "content", None)
                    if isinstance(delta, str):
                        accumulated += delta

                    now = perf_counter()
                    if accumulated and (now - last_edit_time) >= edit_interval:
                        display = accumulated[:1900]
                        if len(accumulated) > 1900:
                            display += "…"
                        try:
                            if sent_message is None:
                                sent_message = await channel.send(display)
                            else:
                                await sent_message.edit(content=display)
                            last_edit_time = now
                        except discord.HTTPException:
                            pass

            await thread
    except TimeoutError as exc:
        logger.warning(
            "Target run timed out kind=%s name=%s run_id=%s cancel_requested=true",
            "agent",
            target_name,
            run_id,
        )
        if cfg.monitor is not None:
            cfg.monitor.finish(monitor_token, status="timeout", error=str(exc) or "agent timeout")
        raise
    except Exception as exc:  # noqa: BLE001
        if cfg.monitor is not None:
            cfg.monitor.finish(monitor_token, status="error", error=str(exc))
        raise

    elapsed_ms = int((perf_counter() - started_at) * 1000)
    run_id = getattr(final_response, "run_id", None) if final_response else None
    model = getattr(final_response, "model", None) if final_response else None
    metrics = _extract_metrics(final_response) if final_response else {}
    if not accumulated:
        accumulated = extract_response_text(final_response) if final_response else ""
    metrics = _ensure_token_metrics(metrics, response_input, accumulated)
    logger.info(
        (
            "Target run completed kind=%s name=%s run_id=%s model=%s elapsed_ms=%d "
            "input_tokens=%s output_tokens=%s total_tokens=%s token_estimated=%s metrics=%s delegation_paths=%s"
        ),
        "agent",
        target_name,
        run_id,
        model,
        elapsed_ms,
        metrics.get("input_tokens"),
        metrics.get("output_tokens"),
        metrics.get("total_tokens"),
        metrics.get("token_estimated"),
        metrics or {},
        "n/a",
    )
    if cfg.monitor is not None:
        cfg.monitor.finish(
            monitor_token,
            status="completed",
            run_id=run_id,
            model=model,
            metrics=metrics or {},
            extra={
                "target_kind": "agent",
                "members": [],
                "elapsed_ms": elapsed_ms,
                "metrics": metrics or {},
                "delegation_paths": [],
                "streamed": True,
            },
        )

    # Final edit with complete content
    if sent_message is not None and accumulated:
        final_text = accumulated[:1900]
        try:
            await sent_message.edit(content=final_text)
        except discord.HTTPException:
            pass

    return accumulated, sent_message


def _agent_name(agent: Any, fallback: str | None = None) -> str:
    if hasattr(agent, "name"):
        return str(getattr(agent, "name"))
    return fallback or "unknown"


def _match_rule(rule: dict[str, Any], message: discord.Message) -> bool:
    channel_ids = rule.get("channel_ids")
    if channel_ids and message.channel.id not in set(int(x) for x in channel_ids):
        return False

    user_ids = rule.get("user_ids")
    if user_ids and message.author.id not in set(int(x) for x in user_ids):
        return False

    if message.guild is not None:
        guild_ids = rule.get("guild_ids")
        if guild_ids and message.guild.id not in set(int(x) for x in guild_ids):
            return False

    content = (message.content or "").lower()
    contains = rule.get("contains")
    if contains:
        if not any(str(token).lower() in content for token in contains):
            return False

    starts_with = rule.get("starts_with")
    if starts_with:
        if not any(content.startswith(str(token).lower()) for token in starts_with):
            return False

    return True


def _match_compiled_rule(
    rule: _CompiledRoutingRule,
    message: discord.Message,
    lower_content: str,
) -> bool:
    if rule.channel_ids and message.channel.id not in rule.channel_ids:
        return False
    if rule.user_ids and message.author.id not in rule.user_ids:
        return False
    if message.guild is not None and rule.guild_ids and message.guild.id not in rule.guild_ids:
        return False
    if rule.contains and not any(token in lower_content for token in rule.contains):
        return False
    if rule.starts_with and not any(lower_content.startswith(token) for token in rule.starts_with):
        return False
    return True


def _select_agent_name(
    config: Config,
    message: discord.Message,
    agent_hint: str | None,
) -> str | None:
    if agent_hint:
        return agent_hint

    routing = config.get("agents", "routing", default={})
    rules = routing.get("rules", []) or []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        if _match_rule(rule, message):
            return rule.get("agent")
    return None


def _select_agent_name_compiled(
    compiled_rules: tuple[_CompiledRoutingRule, ...],
    message: discord.Message,
) -> str | None:
    lower_content = (message.content or "").lower()
    for rule in compiled_rules:
        if _match_compiled_rule(rule, message, lower_content):
            return rule.agent
    return None


def _build_user_context_from_settings(settings: _ContextSettings, message: discord.Message) -> str:
    if not settings.use_workspace_context_files:
        return ""
    if (
        message.guild is not None
        and not settings.scope_workspace_context_by_tenant
        and not settings.allow_global_context_in_guilds
    ):
        return ""

    today = datetime.now(settings.tzinfo).date()
    yesterday = today - timedelta(days=1)
    user_path, memory_dir, long_memory_path = _resolve_context_paths_for_message(settings, message)

    sections: list[str] = []
    user_text = read_text_if_exists(user_path)
    if user_text:
        sections.append("USER:\n" + user_text)

    if memory_dir.exists():
        for day in (yesterday, today):
            daily_path = memory_dir / f"{day.isoformat()}.md"
            daily_text = read_text_if_exists(daily_path)
            if daily_text:
                sections.append(f"DAILY {day.isoformat()}:\n{daily_text}")

    include_long_memory = (
        settings.main_session_scope == "always"
        or (settings.main_session_scope == "dm_only" and message.guild is None)
    )
    if include_long_memory:
        long_memory_text = read_text_if_exists(long_memory_path)
        if long_memory_text:
            sections.append("MEMORY:\n" + long_memory_text)

    return "\n\n".join(sections).strip()


def _build_user_context(config: Config, message: discord.Message) -> str:
    return _build_user_context_from_settings(_load_context_settings(config), message)


def _extract_role_ids(message: discord.Message) -> list[int]:
    roles = getattr(message.author, "roles", None) or []
    role_ids: list[int] = []
    for role in roles:
        role_id = getattr(role, "id", None)
        if isinstance(role_id, int):
            role_ids.append(role_id)
    return role_ids


class DiscordAgentBot(commands.Bot):
    def __init__(self, runtime: DiscordRuntime) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix=commands.when_mentioned, intents=intents)
        self.runtime = runtime
        self.last_active_channel_id: int | None = None
        self._routing_rules = _compile_routing_rules(runtime.config)
        self._context_settings = _load_context_settings(runtime.config)
        self._access_control = _load_access_control_settings(runtime.config)
        self._toolkit_log_cache: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {}
        self._cognee_cfg: CogneeConfig = load_cognee_config(runtime.config)
        self._cognee_client: CogneeClient | None = (
            CogneeClient(self._cognee_cfg) if self._cognee_cfg.enabled else None
        )
        self._cognee_recall_failures = 0
        self._cognee_recall_backoff_until = 0.0
        self._background_tasks: set[asyncio.Task[Any]] = set()

        self.heartbeat_cfg = load_heartbeat_config(runtime.config)
        self._heartbeat_loop = tasks.loop(minutes=self.heartbeat_cfg.interval_minutes)(
            self._heartbeat_tick
        )
        self.cron_cfg = load_cron_config(runtime.config)
        self._cron_scheduler = build_scheduler()
        self._cron_path = get_cron_path(runtime.config)
        try:
            self._cron_last_mtime: float | None = self._cron_path.stat().st_mtime
        except FileNotFoundError:
            self._cron_last_mtime = None
        self._cron_watch_loop = tasks.loop(minutes=10)(self._cron_watch_tick)

    def _log_agent_toolkit_once(self, agent_name: str, agent: Any) -> None:
        if not logger.isEnabledFor(logging.INFO):
            return
        tools = getattr(agent, "tools", None) or []
        snapshot: list[tuple[str, tuple[str, ...]]] = []
        for tool in tools:
            tool_name = str(getattr(tool, "name", tool.__class__.__name__))
            if hasattr(tool, "get_functions"):
                functions = tuple(sorted(tool.get_functions().keys()))
            else:
                functions = ()
            snapshot.append((tool_name, functions))
        current = tuple(snapshot)
        if self._toolkit_log_cache.get(agent_name) == current:
            return
        self._toolkit_log_cache[agent_name] = current
        for tool_name, functions in current:
            logger.info(
                "Agent %s toolkit %s functions: %s",
                agent_name,
                tool_name,
                ", ".join(functions) or "none",
            )

    def _schedule_conversation_sync(
        self,
        *,
        user_id: str,
        session_id: str,
        user_message: str,
        assistant_message: str,
        agent_name: str,
        channel_id: int,
        guild_id: int | None,
    ) -> None:
        if self._cognee_client is None or not self._cognee_cfg.auto_sync_conversations:
            return
        client = self._cognee_client

        async def _sync() -> None:
            try:
                await asyncio.to_thread(
                    client.save_conversation_turn,
                    user_id=user_id,
                    session_id=session_id,
                    user_message=user_message,
                    assistant_message=assistant_message,
                    agent_name=agent_name,
                    channel_id=channel_id,
                    guild_id=guild_id,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Cognee conversation sync failed: %s", exc)

        task = asyncio.create_task(_sync())
        self._background_tasks.add(task)
        task.add_done_callback(lambda t: self._background_tasks.discard(t))

    async def _build_cognee_context_block(self, query: str) -> str:
        if self._cognee_client is None or not self._cognee_cfg.auto_recall_enabled:
            return ""
        client = self._cognee_client
        loop = asyncio.get_running_loop()
        if loop.time() < self._cognee_recall_backoff_until:
            return ""
        search_query = query.strip()
        if len(search_query) < 3:
            return ""
        try:
            results = await asyncio.to_thread(
                client.search,
                search_query,
                self._cognee_cfg.auto_recall_limit,
                max_payloads=2,
                max_paths=1,
                timeout_seconds=min(
                    self._cognee_cfg.timeout_seconds,
                    self._cognee_cfg.auto_recall_timeout_seconds,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            self._cognee_recall_failures += 1
            if self._cognee_recall_failures >= 2:
                self._cognee_recall_backoff_until = loop.time() + 120
            logger.warning("Cognee auto-recall failed: %s", exc)
            return ""
        self._cognee_recall_failures = 0
        if not results:
            return ""

        lines: list[str] = ["COGNEE MEMORY (auto-retrieved):"]
        total_chars = 0
        max_chars = self._cognee_cfg.auto_recall_max_chars
        inject_all = self._cognee_cfg.auto_recall_inject_all
        for item in results:
            cleaned = _clean_cognee_result_item(item)
            if not cleaned:
                continue
            snippet = cleaned if inject_all else cleaned[:420]
            if not inject_all and total_chars + len(snippet) > max_chars:
                break
            lines.append(f"- {snippet}")
            total_chars += len(snippet)
        if len(lines) == 1:
            return ""
        logger.info(
            "Cognee auto-recall injected items=%d available=%d inject_all=%s query=%s",
            len(lines) - 1,
            len(results),
            inject_all,
            search_query[:100],
        )
        return "\n".join(lines)

    async def setup_hook(self) -> None:
        if self.heartbeat_cfg.enabled:
            self._heartbeat_loop.start()
        logger.info(
            "Cron config loaded enabled=%s path=%s jobs=%d",
            self.cron_cfg.enabled,
            self._cron_path,
            len(self.cron_cfg.jobs),
        )
        if self.cron_cfg.enabled:
            self._configure_cron_jobs()
            if not self._cron_scheduler.running:
                self._cron_scheduler.start()
            logger.info(
                "Cron scheduler started path=%s jobs=%d",
                self._cron_path,
                len(self._cron_scheduler.get_jobs()),
            )
        self._cron_watch_loop.start()

    async def close(self) -> None:
        logger.info("Shutting down bot...")
        for task in list(self._background_tasks):
            task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()
        if self._heartbeat_loop.is_running():
            self._heartbeat_loop.cancel()
        if self._cron_watch_loop.is_running():
            self._cron_watch_loop.cancel()
        if self._cron_scheduler.running:
            self._cron_scheduler.shutdown(wait=False)
        await super().close()
        logger.info("Bot shutdown complete")

    async def on_ready(self) -> None:
        user = self.user
        if user is not None:
            logger.info("Logged in as %s (id: %s)", user, user.id)
        else:
            logger.info("Logged in (user not available yet)")
        if self._cognee_client is not None and self._cognee_cfg.enabled:
            client = self._cognee_client

            async def _ensure_dataset() -> None:
                try:
                    await asyncio.to_thread(client.ensure_dataset)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Cognee startup ensure_dataset failed: %s", exc)

            task = asyncio.create_task(_ensure_dataset())
            self._background_tasks.add(task)
            task.add_done_callback(lambda t: self._background_tasks.discard(t))

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return

        if message.guild is None:
            should_respond = True
        else:
            should_respond = self.user in message.mentions if self.user else False

        if not should_respond:
            return
        if not _is_message_authorized(self._access_control, message):
            logger.warning(
                "Blocked unauthorized message guild=%s channel=%s user=%s",
                message.guild.id if message.guild else "dm",
                message.channel.id,
                message.author.id,
            )
            await message.channel.send("You are not allowed to use this bot in this context.")
            return

        content = message.content or ""
        if self.user:
            content = _strip_bot_mention(content, self.user.id)
        content = content.strip()
        if not content:
            await message.channel.send("Mention me with a message to get a response.")
            return

        agent_hint, content = _parse_agent_hint(content)
        user_prompt = content
        selected_agent = agent_hint or _select_agent_name_compiled(self._routing_rules, message)
        if selected_agent is None:
            selected_agent = _select_agent_name(self.runtime.config, message, None)
        agent = self.runtime.agents.get(selected_agent)
        agent_name = _agent_name(agent, selected_agent)
        logger.info(
            "Active agent for message (guild=%s channel=%s user=%s): %s",
            message.guild.id if message.guild else "dm",
            message.channel.id,
            message.author.id,
            agent_name,
        )
        self._log_agent_toolkit_once(agent_name, agent)

        session_id = _build_session_id(message, self.runtime.runtime_cfg.session_id_strategy)
        user_id = f"discord:{message.author.id}"
        role_ids = _extract_role_ids(message)

        try:
            with tool_runtime_context(
                run_kind="discord",
                user_id=user_id,
                session_id=session_id,
                discord_user_id=message.author.id,
                channel_id=message.channel.id,
                guild_id=message.guild.id if message.guild else None,
                role_ids=role_ids,
                agent_name=agent_name,
            ):
                async with message.channel.typing():
                    user_context = _build_user_context_from_settings(self._context_settings, message)
                    cognee_block = await self._build_cognee_context_block(content)
                    if cognee_block:
                        user_context = f"{user_context}\n\n{cognee_block}".strip() if user_context else cognee_block
                    rt_cfg = self.runtime.runtime_cfg
                    streamed_message: discord.Message | None = None
                    if rt_cfg.streaming_enabled and not _is_team_target(agent):
                        reply, streamed_message = await _run_agent_streaming(
                            agent,
                            content,
                            user_context,
                            user_id,
                            session_id,
                            message.channel,
                            rt_cfg,
                        )
                        if "<tool_call>" in reply:
                            reply = await _handle_toolcall_fallback(agent, reply, rt_cfg)
                            streamed_message = None
                    else:
                        reply = await _await_with_slow_notice(
                            _run_agent(
                                agent,
                                content,
                                user_context,
                                user_id,
                                session_id,
                                rt_cfg,
                            ),
                            message.channel,
                            rt_cfg.slow_run_threshold_seconds,
                        )
                        if "<tool_call>" in reply:
                            reply = await _handle_toolcall_fallback(agent, reply, rt_cfg)
                    if _needs_completion_retry(reply):
                        logger.warning("Reply looks incomplete; retrying once for final answer")
                        retry_prompt = (
                            f"{content}\n\n"
                            "Return the final answer directly. "
                            "Do not describe your plan or say you will check/search."
                        )
                        reply = await _await_with_slow_notice(
                            _run_agent(
                                agent,
                                retry_prompt,
                                user_context,
                                user_id,
                                session_id,
                                rt_cfg,
                            ),
                            message.channel,
                            rt_cfg.slow_run_threshold_seconds,
                        )
                        if "<tool_call>" in reply:
                            reply = await _handle_toolcall_fallback(agent, reply, rt_cfg)
        except TimeoutError:
            await message.channel.send(
                "The request timed out and was cancelled to stop API token usage. Please try again with a simpler query."
            )
            return
        except ToolPermissionError as exc:
            await message.channel.send(str(exc)[:1900])
            return
        except Exception:  # noqa: BLE001
            error_id = uuid4().hex[:12]
            logger.exception(
                "Discord request failed error_id=%s guild=%s channel=%s user=%s",
                error_id,
                message.guild.id if message.guild else "dm",
                message.channel.id,
                message.author.id,
            )
            await message.channel.send(
                f"Something went wrong while processing your request. Reference ID: `{error_id}`"
            )
            return

        self.last_active_channel_id = message.channel.id
        if streamed_message is not None:
            # Already sent via streaming edits; send overflow chunks if needed
            if len(reply) > 1900:
                await _send_chunked(message.channel, reply[1900:])
        else:
            await _send_chunked(message.channel, reply)
        self._schedule_conversation_sync(
            user_id=user_id,
            session_id=session_id,
            user_message=user_prompt,
            assistant_message=reply,
            agent_name=agent_name,
            channel_id=message.channel.id,
            guild_id=message.guild.id if message.guild else None,
        )

    async def _heartbeat_tick(self) -> None:
        if not self.heartbeat_cfg.enabled:
            return

        channel_id = self.heartbeat_cfg.channel_id
        if channel_id is None:
            channel_id = self.last_active_channel_id
        if channel_id is None:
            return

        channel = self.get_channel(int(channel_id))
        if channel is None:
            try:
                channel = await self.fetch_channel(int(channel_id))
            except Exception:  # noqa: BLE001
                return

        target_name = self.heartbeat_cfg.agent
        resolved_target = target_name or self.runtime.agents.default_target
        logger.info("Active target for heartbeat: %s", resolved_target)
        agent = self.runtime.agents.get(target_name)
        prompt = build_heartbeat_prompt(self.heartbeat_cfg)

        async def send_fn(content: str) -> None:
            await _send_chunked(cast(discord.abc.Messageable, channel), content)

        monitor = self.runtime.runtime_cfg.monitor
        if self.runtime.runtime_cfg.watchdog_enabled and monitor is not None:
            stale = monitor.stale_runs(
                threshold_seconds=self.runtime.runtime_cfg.watchdog_threshold_seconds,
                max_items=self.runtime.runtime_cfg.watchdog_max_reports,
            )
            if stale:
                lines = [
                    "Watchdog: long-running tasks detected:",
                    *[
                        (
                            f"- `{item.target_name}` ({item.run_kind}) running for "
                            f"{item.age_seconds}s in session `{item.session_id}`"
                        )
                        for item in stale
                    ],
                ]
                await send_fn("\n".join(lines))

        await run_heartbeat(
            agent,
            prompt,
            self.heartbeat_cfg.quiet_ack,
            send_fn,
            session_scope=self.heartbeat_cfg.session_scope,
            timeout=self.runtime.runtime_cfg.heartbeat_timeout,
            monitor=monitor,
        )

    def _configure_cron_jobs(self) -> None:
        if not self.cron_cfg.jobs:
            logger.info("Cron enabled but no jobs are configured")
            return
        for job in self.cron_cfg.jobs:
            timezone = job.timezone or self.cron_cfg.default_timezone
            trigger = build_cron_trigger(job.cron, timezone)
            if trigger is None:
                continue
            scheduled = self._cron_scheduler.add_job(
                self._run_cron_job,
                trigger=trigger,
                args=[job],
                id=job.name,
                replace_existing=True,
            )
            logger.info(
                (
                    "Registered cron job name=%s cron=%s timezone=%s next_run=%s "
                    "session_scope=%s deliver=%s channel_id=%s"
                ),
                job.name,
                job.cron,
                timezone,
                getattr(scheduled, "next_run_time", None),
                job.session_scope,
                job.deliver,
                job.channel_id or self.cron_cfg.default_channel_id,
            )

    def _reload_cron(self, reason: str = "manual") -> None:
        logger.info("Reloading cron config reason=%s path=%s", reason, self._cron_path)
        self.cron_cfg = load_cron_config(self.runtime.config)
        if not self.cron_cfg.enabled:
            self._cron_scheduler.remove_all_jobs()
            logger.info("Cron disabled after reload; all scheduled jobs removed")
            return
        self._cron_scheduler.remove_all_jobs()
        self._configure_cron_jobs()
        if not self._cron_scheduler.running:
            self._cron_scheduler.start()
            logger.info("Cron scheduler started after reload")
        logger.info("Cron reload completed active_jobs=%d", len(self._cron_scheduler.get_jobs()))

    async def _cron_watch_tick(self) -> None:
        try:
            stat = self._cron_path.stat()
        except FileNotFoundError:
            if self._cron_last_mtime is not None:
                self._cron_last_mtime = None
                logger.info("Cron config file removed: %s", self._cron_path)
                self._reload_cron(reason="file_removed")
            return
        if self._cron_last_mtime is None:
            self._cron_last_mtime = stat.st_mtime
            logger.info("Cron config file detected: %s", self._cron_path)
            self._reload_cron(reason="file_created")
            return
        if stat.st_mtime > self._cron_last_mtime:
            self._cron_last_mtime = stat.st_mtime
            logger.info("Cron config file changed: %s", self._cron_path)
            self._reload_cron(reason="file_changed")

    async def _run_cron_job(self, job) -> None:
        channel_id = job.channel_id or self.cron_cfg.default_channel_id
        if channel_id is None:
            channel_id = self.last_active_channel_id
        send_fn = None
        if channel_id is not None:
            channel = self.get_channel(int(channel_id))
            if channel is None:
                try:
                    channel = await self.fetch_channel(int(channel_id))
                except Exception:  # noqa: BLE001
                    channel = None
            if channel is not None:
                async def send_fn(content: str) -> None:
                    await _send_chunked(cast(discord.abc.Messageable, channel), content)

        logger.info(
            "Active target for cron job '%s': %s",
            job.name,
            job.agent or self.runtime.agents.default_target,
        )
        agent = self.runtime.agents.get(job.agent)
        await run_cron_job(
            agent,
            job,
            send_fn,
            timeout=self.runtime.runtime_cfg.cron_timeout,
            monitor=self.runtime.runtime_cfg.monitor,
        )


async def _send_chunked(channel: discord.abc.Messageable, content: str) -> None:
    if not content:
        return
    max_len = 1900
    chunks: list[str] = []
    remaining = content
    while len(remaining) > max_len:
        split_at = remaining.rfind("\n", 0, max_len)
        if split_at <= 0:
            split_at = max_len
        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    if remaining:
        chunks.append(remaining)
    for i, chunk in enumerate(chunks):
        if i > 0:
            await asyncio.sleep(0.3)
        await channel.send(chunk)


async def _await_with_slow_notice(
    run_coro: Any,
    channel: discord.abc.Messageable,
    threshold_seconds: int,
) -> str:
    if threshold_seconds <= 0:
        return await run_coro

    task = asyncio.create_task(run_coro)
    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout=threshold_seconds)
    except TimeoutError:
        await _send_chunked(
            channel,
            "Still working on this. It is taking longer than usual, I'll send the result when ready.",
        )
        return await task


async def _handle_toolcall_fallback(
    agent, reply: str, runtime_cfg: RuntimeConfig | None = None
) -> str:
    cfg = runtime_cfg or RuntimeConfig()
    calls = _parse_tool_calls(reply)
    if not calls:
        return reply
    results = await _run_tool_calls(agent, calls, denied_tools=set(cfg.fallback_denied_tools))
    if not results:
        cleaned = _strip_tool_call_markup(reply)
        if cleaned:
            return (
                f"{cleaned}\n\n"
                "I could not execute that tool call in fallback mode."
            )
        return "I could not execute that tool call in fallback mode."
    if all(item.get("output", "").strip() == "(no output)" for item in results):
        names = ", ".join(item.get("name", "tool_call") for item in results)
        return (
            f"Executed tool call(s): {names}\n\n"
            "The command completed but returned no output."
        )
    lines = ["Tool results:"]
    for item in results:
        name = item.get("name", "tool_call").strip() or "tool_call"
        output = item.get("output", "").strip() or "(no output)"
        if len(output) > 1500:
            output = output[:1500].rstrip() + "\n...[truncated]"
        lines.append(f"- {name}:\n{output}")
    return "\n\n".join(lines)


def build_runtime(config_path: str) -> DiscordRuntime:
    config = load_config(config_path)
    agents = build_agents(config)
    runtime_cfg = load_runtime_config(config)
    return DiscordRuntime(config=config, agents=agents, runtime_cfg=runtime_cfg)


def run_bot(config_path: str) -> None:
    runtime = build_runtime(config_path)
    token_env = runtime.config.get("discord", "token_env", default="DISCORD_BOT_TOKEN")
    token = os.getenv(token_env, "").strip()
    if not token:
        raise ValueError(f"Missing Discord token env var: {token_env}")

    bot = DiscordAgentBot(runtime)
    bot.run(token)
