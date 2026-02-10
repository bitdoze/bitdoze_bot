from __future__ import annotations

import asyncio
import logging
import os
import re
from time import perf_counter
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import json
from dataclasses import dataclass
from typing import Any, cast

import discord
from discord.ext import commands, tasks

from agno.run.agent import Message
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
from bitdoze_bot.tool_permissions import ToolPermissionError, tool_runtime_context
from bitdoze_bot.utils import extract_response_text, parse_bool, read_text_if_exists

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeConfig:
    agent_timeout: int = 600
    cron_timeout: int = 600
    heartbeat_timeout: int = 120
    max_concurrent_runs: int = 4


def _safe_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def load_runtime_config(config: Config) -> RuntimeConfig:
    raw = config.get("runtime", default={})
    cfg = raw if isinstance(raw, dict) else {}
    return RuntimeConfig(
        agent_timeout=_safe_positive_int(cfg.get("agent_timeout"), 600),
        cron_timeout=_safe_positive_int(cfg.get("cron_timeout"), 600),
        heartbeat_timeout=_safe_positive_int(cfg.get("heartbeat_timeout"), 120),
        max_concurrent_runs=_safe_positive_int(cfg.get("max_concurrent_runs"), 4),
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
    if user_context:
        return [
            Message(role="system", content=user_context),
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
    snapshot: dict[str, Any] = {}
    for key in (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "latency",
        "time_to_first_token",
    ):
        value = getattr(metrics, key, None)
        if value is not None:
            snapshot[key] = value
    return snapshot


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
    logger.info(
        "Target run started kind=%s name=%s session_id=%s user_id=%s members=%s",
        target_kind,
        target_name,
        session_id,
        user_id,
        ", ".join(members) if members else "none",
    )
    started_at = perf_counter()
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
    elapsed_ms = int((perf_counter() - started_at) * 1000)
    run_id = getattr(response, "run_id", None)
    model = getattr(response, "model", None)
    metrics = _extract_metrics(getattr(response, "metrics", None))
    delegation_paths = _collect_delegation_paths(response) if _is_team_target(agent) else []
    logger.info(
        "Target run completed kind=%s name=%s run_id=%s model=%s elapsed_ms=%d metrics=%s delegation_paths=%s",
        target_kind,
        target_name,
        run_id,
        model,
        elapsed_ms,
        metrics or {},
        " | ".join(delegation_paths) if delegation_paths else "n/a",
    )
    return extract_response_text(response)


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

    user_path = context_cfg.get("user_path", "workspace/USER.md")
    memory_dir = Path(context_cfg.get("memory_dir", "workspace/memory"))
    long_memory_path = context_cfg.get("long_memory_path", "workspace/MEMORY.md")
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
        self._cron_last_mtime: float | None = None
        self._cron_watch_loop = tasks.loop(minutes=10)(self._cron_watch_tick)

    async def setup_hook(self) -> None:
        if self.heartbeat_cfg.enabled:
            self._heartbeat_loop.start()
        if self.cron_cfg.enabled:
            self._configure_cron_jobs()
            self._cron_scheduler.start()
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
                    if research_mode_active:
                        reply = await _run_research_mode(
                            agent,
                            content,
                            user_context,
                            user_id,
                            session_id,
                            research_cfg,
                            rt_cfg,
                        )
                    else:
                        reply = await _run_agent(
                            agent,
                            content,
                            user_context,
                            user_id,
                            session_id,
                            rt_cfg,
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

        await run_heartbeat(
            agent,
            prompt,
            self.heartbeat_cfg.quiet_ack,
            send_fn,
            session_scope=self.heartbeat_cfg.session_scope,
            timeout=self.runtime.runtime_cfg.heartbeat_timeout,
        )

    def _configure_cron_jobs(self) -> None:
        if not self.cron_cfg.jobs:
            return
        for job in self.cron_cfg.jobs:
            timezone = job.timezone or self.cron_cfg.default_timezone
            trigger = build_cron_trigger(job.cron, timezone)
            if trigger is None:
                continue
            self._cron_scheduler.add_job(
                self._run_cron_job,
                trigger=trigger,
                args=[job],
                id=job.name,
                replace_existing=True,
            )

    def _reload_cron(self) -> None:
        self.cron_cfg = load_cron_config(self.runtime.config)
        if not self.cron_cfg.enabled:
            self._cron_scheduler.remove_all_jobs()
            return
        self._cron_scheduler.remove_all_jobs()
        self._configure_cron_jobs()

    async def _cron_watch_tick(self) -> None:
        try:
            stat = self._cron_path.stat()
        except FileNotFoundError:
            if self._cron_last_mtime is not None:
                self._cron_last_mtime = None
                self._reload_cron()
            return
        if self._cron_last_mtime is None:
            self._cron_last_mtime = stat.st_mtime
            return
        if stat.st_mtime > self._cron_last_mtime:
            self._cron_last_mtime = stat.st_mtime
            self._reload_cron()

    async def _run_cron_job(self, job) -> None:
        channel_id = job.channel_id or self.cron_cfg.default_channel_id
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
        await run_cron_job(agent, job, send_fn, timeout=self.runtime.runtime_cfg.cron_timeout)


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


def _parse_tool_calls(text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for block in re.findall(r"<tool_call>(.*?)</tool_call>", text, flags=re.DOTALL):
        func_match = re.search(r"<function=([a-zA-Z0-9_]+)>", block)
        if not func_match:
            continue
        func_name = func_match.group(1)
        params: dict[str, Any] = {}
        for p_match in re.findall(r"<parameter=([a-zA-Z0-9_]+)>(.*?)</parameter>", block, flags=re.DOTALL):
            key = p_match[0]
            value = p_match[1].strip()
            if value.isdigit():
                params[key] = int(value)
            else:
                try:
                    params[key] = json.loads(value)
                except Exception:
                    params[key] = value
        calls.append({"name": func_name, "params": params})
    return calls


_FALLBACK_DENIED_TOOLS: set[str] = {"shell", "discord"}


def _get_declared_functions(tool: Any) -> set[str]:
    get_functions = getattr(tool, "get_functions", None)
    if callable(get_functions):
        funcs = get_functions()
        if isinstance(funcs, dict):
            return set(funcs.keys())
    return set()


async def _run_tool_calls(agent, calls: list[dict[str, Any]]) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    tools = getattr(agent, "tools", None) or []
    for call in calls:
        func_name = str(call.get("name", ""))
        params = cast(dict[str, Any], call.get("params", {}))
        target = None
        for tool in tools:
            tool_name = getattr(tool, "_bitdoze_tool_name", "") or ""
            if tool_name.lower() in _FALLBACK_DENIED_TOOLS:
                continue
            declared = _get_declared_functions(tool)
            if declared and func_name not in declared:
                continue
            if not declared:
                continue
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
        results.append({"name": func_name, "output": str(output)})
    return results


async def _handle_toolcall_fallback(
    agent, reply: str, runtime_cfg: RuntimeConfig | None = None
) -> str:
    cfg = runtime_cfg or RuntimeConfig()
    calls = _parse_tool_calls(reply)
    if not calls:
        return reply
    results = await _run_tool_calls(agent, calls)
    if not results:
        return reply
    prompt = "Summarize these tool results for Discord. Use bullets and include links when present.\n\n"
    for item in results:
        prompt += f"{item['name']}:\n{item['output']}\n\n"
    response = await asyncio.wait_for(
        asyncio.to_thread(agent.run, prompt, user_id="tool_fallback", session_id="tool_fallback"),
        timeout=cfg.agent_timeout,
    )
    return extract_response_text(response)


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
