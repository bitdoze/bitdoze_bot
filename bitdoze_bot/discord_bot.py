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
from bitdoze_bot.utils import read_text_if_exists

logger = logging.getLogger(__name__)


@dataclass
class DiscordRuntime:
    config: Config
    agents: AgentRegistry


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
    return hasattr(target, "members")


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


async def _run_agent(
    agent,
    content: str,
    user_context: str | None,
    user_id: str,
    session_id: str,
) -> str:
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
    response = await asyncio.to_thread(
        agent.run,
        response_input,
        user_id=user_id,
        session_id=session_id,
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
    return getattr(response, "content", None) or str(response)


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

    async def on_ready(self) -> None:
        user = self.user
        if user is not None:
            print(f"Logged in as {user} (id: {user.id})")
        else:
            print("Logged in (user not available yet)")

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
        logger.info(
            "Active agent for message (guild=%s channel=%s user=%s): %s",
            message.guild.id if message.guild else "dm",
            message.channel.id,
            message.author.id,
            agent_name,
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

        try:
            async with message.channel.typing():
                user_context = _build_user_context(self.runtime.config, message)
                reply = await _run_agent(
                    agent,
                    content,
                    user_context,
                    user_id,
                    session_id,
                )
                if "<tool_call>" in reply:
                    reply = await _handle_toolcall_fallback(agent, reply)
        except Exception as exc:  # noqa: BLE001
            await message.channel.send(f"Error: {exc}")
            return

        self.last_active_channel_id = message.channel.id
        await _send_chunked(message.channel, reply)

    async def _heartbeat_tick(self) -> None:
        if not self.heartbeat_cfg.enabled:
            return

        channel_id = self.runtime.config.get("heartbeat", "channel_id", default=None)
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

        logger.info("Active target for heartbeat: %s", self.runtime.agents.default_target)
        agent = self.runtime.agents.get()
        prompt = build_heartbeat_prompt(self.heartbeat_cfg)

        async def send_fn(content: str) -> None:
            await _send_chunked(cast(discord.abc.Messageable, channel), content)

        await run_heartbeat(agent, prompt, self.heartbeat_cfg.quiet_ack, send_fn)

    def _configure_cron_jobs(self) -> None:
        if not self.cron_cfg.jobs:
            return
        for job in self.cron_cfg.jobs:
            timezone = job.timezone or self.cron_cfg.default_timezone
            trigger = build_cron_trigger(job.cron, timezone)
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
        await run_cron_job(agent, job, send_fn)


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
    for chunk in chunks:
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


async def _run_tool_calls(agent, calls: list[dict[str, Any]]) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    tools = getattr(agent, "tools", None) or []
    for call in calls:
        func_name = str(call.get("name", ""))
        params = cast(dict[str, Any], call.get("params", {}))
        target = None
        for tool in tools:
            fn = getattr(tool, func_name, None)
            if callable(fn):
                target = fn
                break
        if target is None:
            logger.error("Tool function not found: %s", func_name)
            continue
        output = await asyncio.to_thread(target, **params)
        results.append({"name": func_name, "output": str(output)})
    return results


async def _handle_toolcall_fallback(agent, reply: str) -> str:
    calls = _parse_tool_calls(reply)
    if not calls:
        return reply
    results = await _run_tool_calls(agent, calls)
    if not results:
        return reply
    prompt = "Summarize these tool results for Discord. Use bullets and include links when present.\n\n"
    for item in results:
        prompt += f"{item['name']}:\n{item['output']}\n\n"
    response = await asyncio.to_thread(agent.run, prompt, user_id="tool_fallback", session_id="tool_fallback")
    return getattr(response, "content", None) or str(response)


def build_runtime(config_path: str) -> DiscordRuntime:
    config = load_config(config_path)
    agents = build_agents(config)
    return DiscordRuntime(config=config, agents=agents)


def run_bot(config_path: str) -> None:
    runtime = build_runtime(config_path)
    token_env = runtime.config.get("discord", "token_env", default="DISCORD_BOT_TOKEN")
    token = os.getenv(token_env, "").strip()
    if not token:
        raise ValueError(f"Missing Discord token env var: {token_env}")

    bot = DiscordAgentBot(runtime)
    bot.run(token)
