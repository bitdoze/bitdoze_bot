from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Awaitable

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from zoneinfo import ZoneInfo

import yaml

from bitdoze_bot.config import Config


@dataclass(frozen=True)
class CronJobConfig:
    name: str
    cron: str
    timezone: str | None
    agent: str | None
    message: str
    deliver: bool
    channel_id: int | None
    session_scope: str


@dataclass(frozen=True)
class CronConfig:
    enabled: bool
    default_timezone: str
    default_channel_id: int | None
    jobs: list[CronJobConfig]


def load_cron_config(config: Config) -> CronConfig:
    cron_cfg = config.get("cron", default={})
    cron_path = cron_cfg.get("path")
    if cron_path:
        try:
            raw = Path(cron_path).read_text(encoding="utf-8")
            loaded = yaml.safe_load(raw) or {}
            if isinstance(loaded, dict):
                cron_cfg = loaded
        except FileNotFoundError:
            cron_cfg = {}
    enabled = bool(cron_cfg.get("enabled", False))
    default_tz = str(cron_cfg.get("timezone", "UTC"))
    default_channel_id = cron_cfg.get("channel_id")
    if default_channel_id is not None:
        default_channel_id = int(default_channel_id)

    jobs_cfg = cron_cfg.get("jobs", []) or []
    jobs: list[CronJobConfig] = []
    for raw in jobs_cfg:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name", "cron_job"))
        cron = str(raw.get("cron", ""))
        if not cron:
            continue
        timezone = raw.get("timezone")
        agent = raw.get("agent")
        message = str(raw.get("message", ""))
        deliver = bool(raw.get("deliver", True))
        channel_id = raw.get("channel_id")
        if channel_id is not None:
            channel_id = int(channel_id)
        session_scope = str(raw.get("session_scope", "isolated"))
        jobs.append(
            CronJobConfig(
                name=name,
                cron=cron,
                timezone=timezone,
                agent=agent,
                message=message,
                deliver=deliver,
                channel_id=channel_id,
                session_scope=session_scope,
            )
        )

    return CronConfig(
        enabled=enabled,
        default_timezone=default_tz,
        default_channel_id=default_channel_id,
        jobs=jobs,
    )


def build_scheduler() -> AsyncIOScheduler:
    return AsyncIOScheduler()


async def run_cron_job(
    agent,
    job: CronJobConfig,
    send_fn: Callable[[str], Awaitable[None]] | None,
) -> None:
    session_id = "cron:isolated"
    if job.session_scope == "main":
        session_id = "cron:main"
    response = await asyncio.to_thread(
        agent.run,
        job.message,
        user_id="cron",
        session_id=session_id,
    )
    content = getattr(response, "content", None) or str(response)
    if job.deliver and send_fn is not None:
        await send_fn(content)


def build_cron_trigger(cron_expr: str, tz: str) -> CronTrigger:
    tzinfo = ZoneInfo(tz)
    return CronTrigger.from_crontab(cron_expr, timezone=tzinfo)


def get_cron_path(config: Config) -> Path:
    cron_cfg = config.get("cron", default={})
    cron_path = cron_cfg.get("path", "workspace/CRON.yaml")
    return Path(cron_path)
