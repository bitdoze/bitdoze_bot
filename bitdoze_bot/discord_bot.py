from __future__ import annotations

import asyncio
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from dataclasses import dataclass
from typing import Any

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
        print(f"Logged in as {self.user} (id: {self.user.id})")

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

        session_id = f"discord:{message.guild.id if message.guild else 'dm'}:{message.channel.id}"
        user_id = f"discord:{message.author.id}"

        try:
            async with message.channel.typing():
                user_context = _build_user_context(self.runtime.config, message)
                if user_context:
                    response_input = [
                        Message(role="system", content=user_context),
                        Message(role="user", content=content),
                    ]
                else:
                    response_input = content
                response = await asyncio.to_thread(
                    agent.run,
                    response_input,
                    user_id=user_id,
                    session_id=session_id,
                )
                reply = getattr(response, "content", None) or str(response)
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

        agent = self.runtime.agents.get()
        prompt = build_heartbeat_prompt(self.heartbeat_cfg)

        async def send_fn(content: str) -> None:
            await _send_chunked(channel, content)

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
                    await _send_chunked(channel, content)

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
