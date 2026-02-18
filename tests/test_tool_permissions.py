from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from bitdoze_bot.config import Config
from bitdoze_bot.tool_permissions import (
    ToolPermissionError,
    ToolPermissionManager,
    tool_runtime_context,
)


class DummyTool:
    def get_functions(self):
        return {
            "ping": self.ping,
            "fail": self.fail,
        }

    def ping(self, text: str, token: str | None = None) -> str:
        return f"pong:{text}:{token or ''}"

    def fail(self) -> str:
        raise RuntimeError("boom")


class DummyFileCompatTool:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.last_contents: str | None = None
        self.last_file_name: str | None = None

    def get_functions(self):
        return {"save_file": self.save_file}

    def save_file(self, contents: str, file_name: str) -> str:
        self.last_contents = contents
        self.last_file_name = file_name
        return "ok"


class _DummyGithubFile:
    sha = "sha-from-repo"


class _DummyGithubRepo:
    def get_contents(self, path: str, ref: str | None = None):
        return _DummyGithubFile()


class _DummyGithubClient:
    def get_repo(self, repo_name: str):
        return _DummyGithubRepo()


class DummyGithubCompatTool:
    def __init__(self):
        self.g = _DummyGithubClient()
        self.last_kwargs: dict[str, str] | None = None

    def get_functions(self):
        return {"update_file": self.update_file}

    def update_file(
        self,
        repo_name: str,
        path: str,
        content: str,
        message: str,
        sha: str,
        branch: str | None = None,
    ) -> str:
        self.last_kwargs = {
            "repo_name": repo_name,
            "path": path,
            "content": content,
            "message": message,
            "sha": sha,
            "branch": branch or "",
        }
        return "ok"


def _load_lines(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in raw if line.strip()]


def test_permission_deny_overrides_allow(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    config = Config(
        data={
            "tool_permissions": {
                "enabled": True,
                "default_effect": "allow",
                "rules": [
                    {"effect": "allow", "tools": ["shell"], "role_ids": [1]},
                    {"effect": "deny", "tools": ["shell"], "channel_ids": [9]},
                ],
                "audit": {"enabled": True, "path": str(log_path)},
            }
        },
        path=Path("config.yaml"),
    )
    manager = ToolPermissionManager.from_config(config)
    tool = manager.wrap_tool(DummyTool(), tool_name="shell", agent_name_getter=lambda: "main")

    with tool_runtime_context(
        run_kind="discord",
        discord_user_id=11,
        channel_id=9,
        role_ids=[1],
        agent_name="main",
    ):
        with pytest.raises(ToolPermissionError):
            tool.ping("x")

    entries = _load_lines(log_path)
    assert entries[-1]["outcome"] == "blocked"
    assert entries[-1]["reason"] == "matched_deny_rule"


def test_default_allow_when_no_rules(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    config = Config(
        data={
            "tool_permissions": {
                "enabled": True,
                "default_effect": "allow",
                "rules": [],
                "audit": {"enabled": True, "path": str(log_path)},
            }
        },
        path=Path("config.yaml"),
    )
    manager = ToolPermissionManager.from_config(config)
    tool = manager.wrap_tool(DummyTool(), tool_name="file", agent_name_getter=lambda: "main")

    with tool_runtime_context(run_kind="discord", discord_user_id=1, channel_id=2, agent_name="main"):
        result = tool.ping("ok")

    assert result.startswith("pong:ok")
    entries = _load_lines(log_path)
    assert [item["outcome"] for item in entries] == ["allowed", "executed"]


def test_wrap_tool_keeps_get_functions_compatible(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    config = Config(
        data={
            "tool_permissions": {
                "enabled": True,
                "default_effect": "allow",
                "rules": [],
                "audit": {"enabled": True, "path": str(log_path)},
            }
        },
        path=Path("config.yaml"),
    )
    manager = ToolPermissionManager.from_config(config)
    tool = manager.wrap_tool(DummyTool(), tool_name="file", agent_name_getter=lambda: "main")

    functions = tool.get_functions()
    assert "ping" in functions
    assert callable(functions["ping"])

    with tool_runtime_context(run_kind="discord", discord_user_id=1, agent_name="main"):
        assert tool.ping("compat").startswith("pong:compat")


def test_blocked_message_is_user_friendly(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    config = Config(
        data={
            "tool_permissions": {
                "enabled": True,
                "default_effect": "deny",
                "rules": [],
                "audit": {"enabled": True, "path": str(log_path)},
            }
        },
        path=Path("config.yaml"),
    )
    manager = ToolPermissionManager.from_config(config)
    tool = manager.wrap_tool(DummyTool(), tool_name="github", agent_name_getter=lambda: "research")

    with tool_runtime_context(run_kind="discord", discord_user_id=7, agent_name="research"):
        with pytest.raises(ToolPermissionError) as exc_info:
            tool.ping("x")

    assert "I can't use the 'github' tool" in str(exc_info.value)


def test_audit_log_concurrent_writes_and_redaction(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    config = Config(
        data={
            "tool_permissions": {
                "enabled": True,
                "default_effect": "allow",
                "rules": [],
                "audit": {
                    "enabled": True,
                    "path": str(log_path),
                    "include_arguments": True,
                    "redacted_keys": ["token"],
                },
            }
        },
        path=Path("config.yaml"),
    )
    manager = ToolPermissionManager.from_config(config)
    tool = manager.wrap_tool(DummyTool(), tool_name="shell", agent_name_getter=lambda: "main")

    def _run_once(i: int) -> str:
        with tool_runtime_context(
            run_kind="discord",
            discord_user_id=100 + i,
            channel_id=10,
            guild_id=20,
            role_ids=[30],
            agent_name="main",
        ):
            return tool.ping(f"v{i}", token="secret")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_run_once, range(10)))

    entries = _load_lines(log_path)
    assert len(entries) == 20
    assert {item["outcome"] for item in entries} == {"allowed", "executed"}
    assert all(item["tool"] == "shell" for item in entries)
    # include_arguments=true: token value must be redacted
    assert all(item.get("kwargs", {}).get("token") == "***REDACTED***" for item in entries)


def test_file_tool_argument_aliases_and_workspace_prefix_are_normalized(tmp_path: Path) -> None:
    config = Config(
        data={
            "tool_permissions": {
                "enabled": False,
                "audit": {"enabled": False},
            }
        },
        path=Path("config.yaml"),
    )
    manager = ToolPermissionManager.from_config(config)
    tool = manager.wrap_tool(
        DummyFileCompatTool(base_dir=tmp_path / "workspace"),
        tool_name="file",
        agent_name_getter=lambda: "main",
    )

    with tool_runtime_context(run_kind="discord", agent_name="main"):
        result = tool.save_file(content="hello", file_name="workspace/USER.md")

    assert result == "ok"
    assert tool.last_contents == "hello"
    assert tool.last_file_name == "USER.md"


def test_file_tool_absolute_root_path_is_remapped_to_workspace_relative(tmp_path: Path) -> None:
    config = Config(
        data={
            "tool_permissions": {
                "enabled": False,
                "audit": {"enabled": False},
            }
        },
        path=Path("config.yaml"),
    )
    manager = ToolPermissionManager.from_config(config)
    tool = manager.wrap_tool(
        DummyFileCompatTool(base_dir=tmp_path / "workspace"),
        tool_name="file",
        agent_name_getter=lambda: "main",
    )

    with tool_runtime_context(run_kind="discord", agent_name="main"):
        result = tool.save_file(contents="hello", file_name="/USER.md")

    assert result == "ok"
    assert tool.last_file_name == "USER.md"


def test_github_update_argument_aliases_and_defaults_are_normalized(tmp_path: Path) -> None:
    config = Config(
        data={
            "tool_permissions": {
                "enabled": False,
                "audit": {"enabled": False},
            }
        },
        path=Path("config.yaml"),
    )
    manager = ToolPermissionManager.from_config(config)
    tool = manager.wrap_tool(
        DummyGithubCompatTool(),
        tool_name="github",
        agent_name_getter=lambda: "main",
    )

    with tool_runtime_context(run_kind="discord", agent_name="main"):
        result = tool.update_file(
            repo_name="owner/repo",
            file_name="USER.md",
            contents="updated",
            branch="main",
        )

    assert result == "ok"
    assert tool.last_kwargs is not None
    assert tool.last_kwargs["repo_name"] == "owner/repo"
    assert tool.last_kwargs["path"] == "USER.md"
    assert tool.last_kwargs["content"] == "updated"
    assert tool.last_kwargs["sha"] == "sha-from-repo"
    assert tool.last_kwargs["message"] == "Update USER.md via bitdoze-bot"
