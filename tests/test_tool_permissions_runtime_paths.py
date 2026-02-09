from __future__ import annotations

import asyncio
import json
from pathlib import Path

from bitdoze_bot.config import Config
from bitdoze_bot.cron import CronJobConfig, run_cron_job
from bitdoze_bot.heartbeat import run_heartbeat
from bitdoze_bot.tool_permissions import ToolPermissionManager


class DummyTool:
    def get_functions(self):
        return {"ping": self.ping}

    def ping(self, value: str) -> str:
        return f"ok:{value}"


class DummyAgent:
    def __init__(self, tool):
        self.name = "main"
        self._tool = tool

    def run(self, prompt: str, user_id: str, session_id: str):
        self._tool.ping(prompt)
        return type("Resp", (), {"content": "done"})()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_cron_tool_events_are_audited(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    config = Config(
        data={
            "tool_permissions": {
                "enabled": True,
                "default_effect": "allow",
                "rules": [],
                "audit": {"enabled": True, "path": str(audit_path)},
            }
        },
        path=Path("config.yaml"),
    )
    manager = ToolPermissionManager.from_config(config)
    tool = manager.wrap_tool(DummyTool(), tool_name="shell", agent_name_getter=lambda: "main")
    agent = DummyAgent(tool)
    job = CronJobConfig(
        name="j",
        cron="* * * * *",
        timezone=None,
        agent=None,
        message="hello",
        deliver=False,
        channel_id=None,
        session_scope="isolated",
    )

    asyncio.run(run_cron_job(agent, job, send_fn=None))

    entries = _read_jsonl(audit_path)
    assert len(entries) == 2
    assert {entry["outcome"] for entry in entries} == {"allowed", "executed"}
    assert all(entry["run_kind"] == "cron" for entry in entries)


def test_heartbeat_tool_events_are_audited(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    config = Config(
        data={
            "tool_permissions": {
                "enabled": True,
                "default_effect": "allow",
                "rules": [],
                "audit": {"enabled": True, "path": str(audit_path)},
            }
        },
        path=Path("config.yaml"),
    )
    manager = ToolPermissionManager.from_config(config)
    tool = manager.wrap_tool(DummyTool(), tool_name="file", agent_name_getter=lambda: "main")
    agent = DummyAgent(tool)

    async def _send_fn(content: str) -> None:
        return None

    asyncio.run(run_heartbeat(agent, "beat", "HEARTBEAT_OK", _send_fn, session_scope="isolated"))

    entries = _read_jsonl(audit_path)
    assert len(entries) == 2
    assert {entry["outcome"] for entry in entries} == {"allowed", "executed"}
    assert all(entry["run_kind"] == "heartbeat" for entry in entries)
