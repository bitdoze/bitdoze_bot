from __future__ import annotations

import asyncio
import logging
import os
import re
from time import perf_counter
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import json
from dataclasses import dataclass
from typing import Any, cast

import discord
from discord.ext import commands, tasks

from agno.run.agent import Message, RunEvent, RunContentEvent, RunOutput
from agno.team import Team

from bitdoze_bot.agents import AgentRegistry, build_agents
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
    monitor: RunMonitor | None = None


def _safe_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def load_runtime_config(config: Config) -> RuntimeConfig:
    raw = config.get("runtime", default={})
    cfg = raw if isinstance(raw, dict) else {}
    monitoring_raw = config.get("monitoring", default={})
    monitoring_cfg = monitoring_raw if isinstance(monitoring_raw, dict) else {}
    fallback_raw = config.get("tool_fallback", default={})
    fallback_cfg = fallback_raw if isinstance(fallback_raw, dict) else {}
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
        monitor=monitor,
    )


@dataclass
class DiscordRuntime:
    config: Config
    agents: AgentRegistry
    runtime_cfg: RuntimeConfig


@dataclass(frozen=True)
class ResearchModeConfig:
    enabled: bool = True
    trigger_on_prefix: bool = True
    trigger_on_research_agent: bool = True
    research_agent_name: str = "research"
    min_sources: int = 3


@dataclass(frozen=True)
class ResearchValidationResult:
    valid: bool
    missing_sections: tuple[str, ...]
    source_count: int
    min_sources: int


_RESEARCH_REQUIRED_SECTIONS = ("TL;DR", "Findings", "Risks", "Sources")
_RESEARCH_FAILURE_MESSAGE = (
    "I couldn't produce a valid research response in the required format. "
    "Please retry with a narrower topic or clearer request."
)


def _parse_non_negative_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _load_research_mode_config(config: Config) -> ResearchModeConfig:
    raw = config.get("research_mode", default={})
    cfg = raw if isinstance(raw, dict) else {}
    return ResearchModeConfig(
        enabled=parse_bool(cfg.get("enabled", True), True),
        trigger_on_prefix=parse_bool(cfg.get("trigger_on_prefix", True), True),
        trigger_on_research_agent=parse_bool(cfg.get("trigger_on_research_agent", True), True),
        research_agent_name=str(cfg.get("research_agent_name", "research")).strip() or "research",
        min_sources=max(_parse_non_negative_int(cfg.get("min_sources", 3), 3), 1),
    )


def _is_research_prefixed(content: str) -> bool:
    return content.lower().startswith("research:")


def _strip_research_prefix(content: str) -> str:
    if not _is_research_prefixed(content):
        return content.strip()
    return content[len("research:") :].strip()


def _is_research_mode_request(
    cfg: ResearchModeConfig,
    content: str,
    selected_target_name: str | None,
) -> bool:
    if not cfg.enabled:
        return False
    by_prefix = cfg.trigger_on_prefix and _is_research_prefixed(content)
    by_target = (
        cfg.trigger_on_research_agent
        and bool(selected_target_name)
        and str(selected_target_name).strip().lower() == cfg.research_agent_name.lower()
    )
    return by_prefix or by_target


def _build_research_prompt(query: str, min_sources: int) -> str:
    return (
        "Research request:\n"
        f"{query.strip()}\n\n"
        "Return exactly these sections in this order:\n"
        "TL;DR\n"
        "Findings\n"
        "Risks\n"
        "Sources\n\n"
        "Requirements:\n"
        f"- In Sources, include at least {min_sources} unique http(s) URLs.\n"
        "- Keep Findings and Risks concise bullet lists.\n"
    )


def _section_header_pattern(section: str) -> re.Pattern[str]:
    return re.compile(
        rf"^\s*(?:#+\s*)?{re.escape(section)}\s*:?\s*$|^\s*(?:#+\s*)?{re.escape(section)}\s*:\s+\S",
        flags=re.IGNORECASE | re.MULTILINE,
    )


def _extract_sources_block(text: str) -> str:
    sources_start = re.search(r"(?im)^\s*(?:#+\s*)?Sources\s*:?", text)
    if not sources_start:
        return ""
    return text[sources_start.end() :]


def _extract_unique_http_urls(text: str) -> set[str]:
    raw_urls = re.findall(r"https?://\S+", text, flags=re.IGNORECASE)
    cleaned: set[str] = set()
    for url in raw_urls:
        cleaned.add(url.rstrip(".,);:!?]}'\""))
    return cleaned


def _validate_research_response(text: str, min_sources: int) -> ResearchValidationResult:
    missing_sections: list[str] = []
    for section in _RESEARCH_REQUIRED_SECTIONS:
        if not _section_header_pattern(section).search(text):
            missing_sections.append(section)
    sources_block = _extract_sources_block(text)
    unique_urls = _extract_unique_http_urls(sources_block)
    valid = not missing_sections and len(unique_urls) >= min_sources
    return ResearchValidationResult(
        valid=valid,
        missing_sections=tuple(missing_sections),
        source_count=len(unique_urls),
        min_sources=min_sources,
    )


def _build_research_retry_prompt(query: str, validation: ResearchValidationResult) -> str:
    issues: list[str] = []
    if validation.missing_sections:
        issues.append("missing sections: " + ", ".join(validation.missing_sections))
    if validation.source_count < validation.min_sources:
        issues.append(
            f"insufficient source URLs in Sources: {validation.source_count}/{validation.min_sources}"
        )
    issue_summary = "; ".join(issues) if issues else "format mismatch"
    return (
        "Retry the research answer and strictly follow this format.\n"
        f"Validation issues: {issue_summary}\n\n"
        "Return exactly these sections in this order:\n"
        "TL;DR\n"
        "Findings\n"
        "Risks\n"
        "Sources\n\n"
        "Requirements:\n"
        f"- Sources must include at least {validation.min_sources} unique http(s) URLs.\n"
        "- Do not include any extra sections.\n\n"
        f"Research request:\n{query.strip()}\n"
    )


def _strip_bot_mention(content: str, bot_user_id: int) -> str:
    pattern = re.compile(rf"<@!?{bot_user_id}>")
    cleaned = pattern.sub("", content)
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


def _build_response_input(user_context: str | None, content: str) -> list[Message] | str:
    file_task_hint = (
        "For local file/code operations, use the 'file' tool functions "
        "(list_files, search_files, read_file, read_file_chunk, save_file, replace_file_chunk) "
        "instead of shell or github tools. File paths are relative to the workspace root, "
        "so use 'USER.md' (not 'workspace/USER.md'). Use save_file(contents=..., file_name=...)."
    )
    lower_content = content.lower()
    file_keywords = (
        "file",
        "folder",
        "directory",
        "read ",
        "open ",
        "edit ",
        "update ",
        "change ",
        "rewrite ",
        "search in",
        "find in",
        "codebase",
        ".py",
        ".js",
        ".ts",
        ".md",
        ".yaml",
        ".yml",
        ".json",
    )
    needs_file_hint = any(token in lower_content for token in file_keywords)

    if user_context:
        messages: list[Message] = [Message(role="system", content=user_context)]
        if needs_file_hint:
            messages.append(Message(role="system", content=file_task_hint))
        messages.append(Message(role="user", content=content))
        return messages
    if needs_file_hint:
        return [
            Message(role="system", content=file_task_hint),
            Message(role="user", content=content),
        ]
    return content


def _is_team_target(target: Any) -> bool:
    return isinstance(target, Team)


def _target_members(target: Any) -> list[str]:
    names: list[str] = []
    for member in getattr(target, "members", None) or []:
        names.append(_agent_name(member, "unknown"))
    return names


def _extract_metrics(metrics: Any) -> dict[str, Any]:
    if metrics is None:
        return {}
    # RunOutput carries primary metrics on `.metrics`; extract from there first.
    if not isinstance(metrics, dict):
        nested_metrics = getattr(metrics, "metrics", None)
        if nested_metrics is not None and nested_metrics is not metrics:
            snapshot = _extract_metrics(nested_metrics)
        else:
            snapshot = {}
    else:
        snapshot = {}

    # Aggregate token usage across model requests when available (useful for
    # streaming + tool calls where final response metrics may be partial/missing).
    events = getattr(metrics, "events", None)
    if isinstance(events, list):
        input_sum = 0
        output_sum = 0
        total_sum = 0
        seen = False
        for event in events:
            in_tok = getattr(event, "input_tokens", None)
            out_tok = getattr(event, "output_tokens", None)
            tot_tok = getattr(event, "total_tokens", None)
            if in_tok is None and out_tok is None and tot_tok is None:
                continue
            seen = True
            input_sum += int(in_tok or 0)
            output_sum += int(out_tok or 0)
            total_sum += int(tot_tok or 0)
        if seen:
            if input_sum > 0:
                snapshot["input_tokens"] = input_sum
            if output_sum > 0:
                snapshot["output_tokens"] = output_sum
            if total_sum > 0:
                snapshot["total_tokens"] = total_sum
            snapshot["token_source"] = "events"

    usage = getattr(metrics, "usage", None)
    if isinstance(metrics, dict):
        usage = metrics.get("usage", usage)
        provider_data = metrics.get("model_provider_data")
    else:
        provider_data = getattr(metrics, "model_provider_data", None)
    if isinstance(provider_data, dict):
        usage = provider_data.get("usage", usage)
        if usage is None and isinstance(provider_data.get("response"), dict):
            usage = provider_data["response"].get("usage")
    field_aliases: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("input_tokens", ("input_tokens", "prompt_tokens")),
        ("output_tokens", ("output_tokens", "completion_tokens")),
        ("total_tokens", ("total_tokens",)),
        ("latency", ("latency",)),
        ("time_to_first_token", ("time_to_first_token",)),
    )
    for normalized_key, candidates in field_aliases:
        value = None
        for key in candidates:
            if isinstance(metrics, dict):
                value = metrics.get(key)
            else:
                value = getattr(metrics, key, None)
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
    if (
        "total_tokens" not in snapshot
        and "input_tokens" in snapshot
        and "output_tokens" in snapshot
    ):
        try:
            snapshot["total_tokens"] = int(snapshot["input_tokens"]) + int(snapshot["output_tokens"])
        except (TypeError, ValueError):
            pass
    return snapshot


def _estimate_tokens_from_text(value: str) -> int:
    text = value.strip()
    if not text:
        return 0
    # Simple fallback heuristic when provider usage is unavailable.
    return max(1, (len(text) + 3) // 4)


def _estimate_input_tokens(response_input: Any) -> int:
    if isinstance(response_input, str):
        return _estimate_tokens_from_text(response_input)
    if isinstance(response_input, list):
        parts: list[str] = []
        for item in response_input:
            content = getattr(item, "content", None)
            if isinstance(content, str):
                parts.append(content)
        return _estimate_tokens_from_text("\n".join(parts))
    return _estimate_tokens_from_text(str(response_input))


def _ensure_token_metrics(metrics: dict[str, Any], response_input: Any, output_text: str) -> dict[str, Any]:
    merged = dict(metrics)
    if (
        merged.get("input_tokens") is None
        or merged.get("output_tokens") is None
        or merged.get("total_tokens") is None
    ):
        in_tokens = _estimate_input_tokens(response_input)
        out_tokens = _estimate_tokens_from_text(output_text)
        merged["input_tokens"] = in_tokens
        merged["output_tokens"] = out_tokens
        merged["total_tokens"] = in_tokens + out_tokens
        merged["token_estimated"] = True
    else:
        merged["token_estimated"] = False
    return merged


def _collect_delegation_paths(node: Any, prefix: str = "") -> list[str]:
    node_name = (
        getattr(node, "team_name", None)
        or getattr(node, "agent_name", None)
        or getattr(node, "name", None)
        or "unknown"
    )
    current = f"{prefix}->{node_name}" if prefix else str(node_name)
    member_responses = getattr(node, "member_responses", None) or []
    if not member_responses:
        return [current]
    paths: list[str] = []
    for child in member_responses:
        paths.extend(_collect_delegation_paths(child, current))
    return paths


_run_semaphore: asyncio.Semaphore | None = None
_run_semaphore_size: int | None = None


def _get_run_semaphore(max_concurrent: int = 4) -> asyncio.Semaphore:
    global _run_semaphore, _run_semaphore_size  # noqa: PLW0603
    if _run_semaphore is None or _run_semaphore_size != max_concurrent:
        _run_semaphore = asyncio.Semaphore(max_concurrent)
        _run_semaphore_size = max_concurrent
    return _run_semaphore


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
    try:
        async with _get_run_semaphore(cfg.max_concurrent_runs):
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    agent.run,
                    response_input,
                    user_id=user_id,
                    session_id=session_id,
                ),
                timeout=cfg.agent_timeout,
            )
    except TimeoutError as exc:
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


def _iter_stream_in_thread(agent, response_input, user_id: str, session_id: str):
    """Run agent.run(stream=True) in a thread and yield events via a queue."""
    import queue as queue_mod

    q: queue_mod.Queue = queue_mod.Queue()

    def _produce():
        try:
            for event in agent.run(
                response_input, stream=True, user_id=user_id, session_id=session_id
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

    try:
        async with _get_run_semaphore(cfg.max_concurrent_runs):
            q, produce = _iter_stream_in_thread(
                agent, response_input, user_id, session_id
            )
            loop = asyncio.get_event_loop()
            thread = loop.run_in_executor(None, produce)

            deadline = loop.time() + cfg.agent_timeout
            while True:
                if loop.time() > deadline:
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


def _build_user_context(config: Config, message: discord.Message) -> str:
    context_cfg = config.get("context", default={})
    use_workspace_context_files = bool(context_cfg.get("use_workspace_context_files", True))
    if not use_workspace_context_files:
        return ""

    user_path = config.resolve_path(context_cfg.get("user_path"), default="workspace/USER.md")
    memory_dir = config.resolve_path(context_cfg.get("memory_dir"), default="workspace/memory")
    long_memory_path = config.resolve_path(
        context_cfg.get("long_memory_path"),
        default="workspace/MEMORY.md",
    )
    main_session_scope = context_cfg.get("main_session_scope", "dm_only")
    tz_name = context_cfg.get("timezone_identifier", "UTC")

    try:
        tzinfo = ZoneInfo(tz_name)
    except Exception:  # noqa: BLE001
        tzinfo = ZoneInfo("UTC")

    today = datetime.now(tzinfo).date()
    yesterday = today - timedelta(days=1)

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
        main_session_scope == "always"
        or (main_session_scope == "dm_only" and message.guild is None)
    )
    if include_long_memory:
        long_memory_text = read_text_if_exists(long_memory_path)
        if long_memory_text:
            sections.append("MEMORY:\n" + long_memory_text)

    return "\n\n".join(sections).strip()


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

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return

        if message.guild is None:
            should_respond = True
        else:
            should_respond = self.user in message.mentions if self.user else False

        if not should_respond:
            return

        content = message.content or ""
        if self.user:
            content = _strip_bot_mention(content, self.user.id)
        content = content.strip()
        if not content:
            await message.channel.send("Mention me with a message to get a response.")
            return

        agent_hint, content = _parse_agent_hint(content)
        selected_agent = _select_agent_name(self.runtime.config, message, agent_hint)
        agent = self.runtime.agents.get(selected_agent)
        agent_name = _agent_name(agent, selected_agent)
        research_cfg = _load_research_mode_config(self.runtime.config)
        research_mode_active = _is_research_mode_request(research_cfg, content, agent_name)
        logger.info(
            "Active agent for message (guild=%s channel=%s user=%s): %s",
            message.guild.id if message.guild else "dm",
            message.channel.id,
            message.author.id,
            agent_name,
        )
        logger.info(
            "Research mode active=%s trigger_prefix=%s trigger_research_agent=%s target=%s",
            research_mode_active,
            _is_research_prefixed(content),
            agent_name.strip().lower() == research_cfg.research_agent_name.lower(),
            research_cfg.research_agent_name,
        )
        tools = getattr(agent, "tools", None) or []
        for tool in tools:
            tool_name = getattr(tool, "name", tool.__class__.__name__)
            if hasattr(tool, "get_functions"):
                tool_functions = sorted(tool.get_functions().keys())
            else:
                tool_functions = []
            logger.info(
                "Agent %s toolkit %s functions: %s",
                agent_name,
                tool_name,
                ", ".join(tool_functions) or "none",
            )

        session_id = f"discord:{message.guild.id if message.guild else 'dm'}:{message.channel.id}"
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
                    user_context = _build_user_context(self.runtime.config, message)
                    rt_cfg = self.runtime.runtime_cfg
                    streamed_message: discord.Message | None = None
                    if research_mode_active:
                        reply = await _await_with_slow_notice(
                            _run_research_mode(
                                agent,
                                content,
                                user_context,
                                user_id,
                                session_id,
                                research_cfg,
                                rt_cfg,
                            ),
                            message.channel,
                            rt_cfg.slow_run_threshold_seconds,
                        )
                    elif rt_cfg.streaming_enabled and not _is_team_target(agent):
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
                "The request timed out. Please try again with a simpler query."
            )
            return
        except ToolPermissionError as exc:
            await message.channel.send(str(exc)[:1900])
            return
        except Exception as exc:  # noqa: BLE001
            error_msg = f"Error: {exc}"
            await message.channel.send(error_msg[:1900])
            return

        self.last_active_channel_id = message.channel.id
        if streamed_message is not None:
            # Already sent via streaming edits; send overflow chunks if needed
            if len(reply) > 1900:
                await _send_chunked(message.channel, reply[1900:])
        else:
            await _send_chunked(message.channel, reply)

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


def _parse_tool_calls(text: str) -> list[dict[str, Any]]:
    def _parse_param_value(raw: str) -> Any:
        value = raw.strip()
        if value.isdigit():
            return int(value)
        try:
            return json.loads(value)
        except Exception:
            return value

    calls: list[dict[str, Any]] = []
    for block in re.findall(r"<tool_call>(.*?)</tool_call>", text, flags=re.DOTALL):
        func_match = re.search(r"<function=([a-zA-Z0-9_]+)>", block)
        if func_match:
            func_name = func_match.group(1)
        else:
            # Supports plain style: <tool_call>fn_name<arg_key>..</arg_key>...</tool_call>
            plain_name_match = re.match(r"\s*([a-zA-Z_][a-zA-Z0-9_]*)", block)
            if not plain_name_match:
                continue
            func_name = plain_name_match.group(1)
        params: dict[str, Any] = {}

        for p_match in re.findall(r"<parameter=([a-zA-Z0-9_]+)>(.*?)</parameter>", block, flags=re.DOTALL):
            key = p_match[0]
            params[key] = _parse_param_value(p_match[1])

        for p_match in re.finditer(
            r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
            block,
            flags=re.DOTALL,
        ):
            key = p_match.group(1).strip()
            if not key:
                continue
            params[key] = _parse_param_value(p_match.group(2))

        calls.append({"name": func_name, "params": params})
    return calls


def _strip_tool_call_markup(text: str) -> str:
    cleaned = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL).strip()
    return cleaned


def _needs_completion_retry(reply: str) -> bool:
    text = reply.strip()
    if not text:
        return True
    if len(text) > 260:
        return False
    lowered = text.lower()
    if "<tool_call>" in lowered:
        return False
    if lowered.endswith(":"):
        return True
    placeholder_prefixes = (
        "let me ",
        "i will ",
        "i'll ",
        "i am going to ",
        "i'm going to ",
        "one moment",
        "hold on",
    )
    return any(lowered.startswith(prefix) for prefix in placeholder_prefixes)


def _get_declared_functions(tool: Any) -> set[str]:
    get_functions = getattr(tool, "get_functions", None)
    if callable(get_functions):
        funcs = get_functions()
        if isinstance(funcs, dict):
            return set(funcs.keys())
    return set()


async def _run_tool_calls(
    agent,
    calls: list[dict[str, Any]],
    denied_tools: set[str] | None = None,
) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    denied = {"shell", "discord"} if denied_tools is None else denied_tools
    tools = getattr(agent, "tools", None) or []
    for call in calls:
        func_name = str(call.get("name", ""))
        params = cast(dict[str, Any], call.get("params", {}))
        target = None
        for tool in tools:
            tool_name = getattr(tool, "_bitdoze_tool_name", "") or ""
            if tool_name.lower() in denied:
                continue
            get_functions = getattr(tool, "get_functions", None)
            declared_funcs = get_functions() if callable(get_functions) else {}
            if not isinstance(declared_funcs, dict):
                declared_funcs = {}
            declared = set(declared_funcs.keys())
            if declared and func_name not in declared:
                continue
            if not declared:
                continue

            # Preferred: call the tool's declared function entrypoint directly.
            declared_fn = declared_funcs.get(func_name)
            entrypoint = getattr(declared_fn, "entrypoint", None)
            if callable(entrypoint):
                target = entrypoint
                break

            # Fallback: call method attribute by name when available.
            fn = getattr(tool, func_name, None)
            if callable(fn):
                target = fn
                break
        if target is None:
            logger.warning("Tool fallback: function '%s' not found or denied", func_name)
            continue
        try:
            output = await asyncio.to_thread(target, **params)
        except Exception as exc:  # noqa: BLE001
            logger.error("Tool fallback: function '%s' failed: %s", func_name, exc)
            results.append({"name": func_name, "output": f"Error: {exc}"})
            continue
        output_text = str(output).strip()
        if not output_text:
            output_text = "(no output)"
        results.append({"name": func_name, "output": output_text})
    return results


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


async def _run_research_mode(
    agent,
    content: str,
    user_context: str | None,
    user_id: str,
    session_id: str,
    research_cfg: ResearchModeConfig,
    runtime_cfg: RuntimeConfig | None = None,
) -> str:
    query = _strip_research_prefix(content)
    prompt = _build_research_prompt(query or content, research_cfg.min_sources)
    reply = await _run_agent(agent, prompt, user_context, user_id, session_id, runtime_cfg)
    if "<tool_call>" in reply:
        reply = await _handle_toolcall_fallback(agent, reply, runtime_cfg)

    first_validation = _validate_research_response(reply, research_cfg.min_sources)
    if first_validation.valid:
        return reply

    logger.warning(
        "Research mode validation failed; retrying once missing_sections=%s source_count=%d min_sources=%d",
        ", ".join(first_validation.missing_sections) or "none",
        first_validation.source_count,
        first_validation.min_sources,
    )
    retry_prompt = _build_research_retry_prompt(query or content, first_validation)
    retry_reply = await _run_agent(agent, retry_prompt, user_context, user_id, session_id, runtime_cfg)
    if "<tool_call>" in retry_reply:
        retry_reply = await _handle_toolcall_fallback(agent, retry_reply, runtime_cfg)

    second_validation = _validate_research_response(retry_reply, research_cfg.min_sources)
    if second_validation.valid:
        return retry_reply

    logger.error(
        "Research mode retry failed missing_sections=%s source_count=%d min_sources=%d",
        ", ".join(second_validation.missing_sections) or "none",
        second_validation.source_count,
        second_validation.min_sources,
    )
    return _RESEARCH_FAILURE_MESSAGE


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
