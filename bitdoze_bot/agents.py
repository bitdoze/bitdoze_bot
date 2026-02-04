from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.learn import (
    DecisionLogConfig,
    EntityMemoryConfig,
    LearnedKnowledgeConfig,
    LearningMachine,
    LearningMode,
    SessionContextConfig,
    UserMemoryConfig,
    UserProfileConfig,
)
from agno.models.openai.like import OpenAILike
from agno.skills import LocalSkills, SkillLoader, Skills
from agno.session import SessionSummaryManager
from agno.team import Team
from agno.tools.github import GithubTools
from agno.tools.hackernews import HackerNewsTools
from agno.tools.discord import DiscordTools
from agno.tools.file import FileTools
from agno.tools.shell import ShellTools
from agno.tools.websearch import WebSearchTools
from agno.tools.website import WebsiteTools
from agno.tools.youtube import YouTubeTools

from bitdoze_bot.bridge_tools import BridgeTools
from bitdoze_bot.collab_tools import CollaborationTools
from bitdoze_bot.compat_tools import FileCompatTools
from bitdoze_bot.config import Config
from bitdoze_bot.task_store import TaskBoardStore
from bitdoze_bot.task_tools import TaskBoardTools
from bitdoze_bot.utils import read_text_if_exists


@dataclass(frozen=True)
class AgentRegistry:
    agents: dict[str, Any]
    default_agent: str
    standalone_agents: dict[str, Agent]
    teams: dict[str, Team]

    def get(self, name: str | None = None) -> Any:
        if name and name in self.agents:
            return self.agents[name]
        return self.agents[self.default_agent]


def _require_env(var_name: str) -> str:
    value = os.getenv(var_name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {var_name}")
    return value


def _build_model(config: Config, overrides: dict[str, Any] | None = None) -> OpenAILike:
    model_cfg = dict(config.get("model", default={}))
    if overrides:
        model_cfg.update(overrides)

    provider = str(model_cfg.get("provider", "openai_like")).lower()
    if provider not in {"openai_like", "openai-like"}:
        raise ValueError(f"Unsupported model provider: {provider}")

    model_id = model_cfg.get("id", "gpt-4o-mini")
    base_url = model_cfg.get("base_url")
    api_key = str(model_cfg.get("api_key", "")).strip()
    if not api_key:
        api_key_env = model_cfg.get("api_key_env", "OPENAI_API_KEY")
        api_key = _require_env(api_key_env)
    structured_outputs = model_cfg.get("structured_outputs", "auto")

    model = OpenAILike(
        id=model_id,
        api_key=api_key,
        base_url=base_url,
    )
    if structured_outputs in {False, "false", "off", "disabled"}:
        model.supports_native_structured_outputs = False
        model.supports_json_schema_outputs = False
    return model


def _build_tools(config: Config) -> dict[str, Any]:
    tools: dict[str, Any] = {}
    tool_cfg = config.get("toolkits", default={})

    web_cfg = tool_cfg.get("web_search", {})
    if web_cfg.get("enabled", True):
        backend = web_cfg.get("backend")
        tools["web_search"] = WebSearchTools(backend=backend) if backend else WebSearchTools()

    hackernews_cfg = tool_cfg.get("hackernews", {})
    if hackernews_cfg.get("enabled", True):
        tools["hackernews"] = HackerNewsTools()

    website_cfg = tool_cfg.get("website", {})
    if website_cfg.get("enabled", True):
        tools["website"] = WebsiteTools()

    github_cfg = tool_cfg.get("github", {})
    if github_cfg.get("enabled", True):
        token_env = github_cfg.get("token_env", "GITHUB_ACCESS_TOKEN")
        access_token = os.getenv(token_env, "").strip() or None
        base_url = github_cfg.get("base_url")
        tools["github"] = GithubTools(access_token=access_token, base_url=base_url)

    youtube_cfg = tool_cfg.get("youtube", {})
    if youtube_cfg.get("enabled", True):
        languages = youtube_cfg.get("languages")
        tools["youtube"] = YouTubeTools(languages=languages) if languages else YouTubeTools()

    file_cfg = tool_cfg.get("file", {})
    if file_cfg.get("enabled", True):
        base_dir = Path(file_cfg.get("base_dir", "workspace"))
        base_dir.mkdir(parents=True, exist_ok=True)
        tools["file"] = FileTools(base_dir=base_dir)

    shell_cfg = tool_cfg.get("shell", {})
    if shell_cfg.get("enabled", True):
        base_dir = Path(shell_cfg.get("base_dir", "workspace"))
        base_dir.mkdir(parents=True, exist_ok=True)
        tools["shell"] = ShellTools(base_dir=base_dir)

    discord_cfg = tool_cfg.get("discord", {})
    if discord_cfg.get("enabled", True):
        token_env = config.get("discord", "token_env", default="DISCORD_BOT_TOKEN")
        bot_token = _require_env(token_env)
        tools["discord"] = DiscordTools(bot_token=bot_token)

    collaboration_cfg = tool_cfg.get("collaboration", {})
    if collaboration_cfg.get("enabled", True):
        specs_dir = _resolve_config_path(config, config.get("context", "specs_dir", default="workspace/team/specs"))
        handoffs_dir = _resolve_config_path(
            config,
            config.get("context", "handoffs_dir", default="workspace/team/handoffs"),
        )
        tools["collaboration"] = CollaborationTools(specs_dir=specs_dir, handoffs_dir=handoffs_dir)

    bridge_cfg = tool_cfg.get("bridge", {})
    if bridge_cfg.get("enabled", True):
        workspace_dir = _resolve_config_path(
            config,
            config.get("toolkits", "file", "base_dir", default="workspace"),
        )
        tasks_dir = _resolve_config_path(config, config.get("context", "tasks_dir", default="workspace/tasks"))
        skills_dirs = [_resolve_config_path(config, p) for p in config.get("skills", "directories", default=[]) or []]
        tools["bridge"] = BridgeTools(
            workspace_dir=workspace_dir,
            tasks_dir=tasks_dir,
            default_actor="system",
            websearch_tools=tools.get("web_search"),
            website_tools=tools.get("website"),
            hackernews_tools=tools.get("hackernews"),
            youtube_tools=tools.get("youtube"),
            github_tools=tools.get("github"),
            collaboration_tools=tools.get("collaboration"),
            skill_dirs=skills_dirs,
        )

    return tools


def _resolve_config_path(config: Config, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (config.path.parent / path).resolve()


def _parse_external_agent_config(config_path: Path) -> dict[str, Any]:
    raw = config_path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Agent config must be a YAML mapping: {config_path}")
    return data


def _ensure_shared_team_workspace(config: Config) -> None:
    specs_dir = _resolve_config_path(config, config.get("context", "specs_dir", default="workspace/team/specs"))
    handoffs_dir = _resolve_config_path(
        config,
        config.get("context", "handoffs_dir", default="workspace/team/handoffs"),
    )
    specs_dir.mkdir(parents=True, exist_ok=True)
    handoffs_dir.mkdir(parents=True, exist_ok=True)

    team_readme = _resolve_config_path(config, config.get("context", "team_readme_path", default="workspace/team/README.md"))
    team_readme.parent.mkdir(parents=True, exist_ok=True)
    if not team_readme.exists():
        team_readme.write_text(
            "# Team Workspace\n\n"
            "Shared workspace for cross-agent collaboration.\n\n"
            "## Shared Paths\n"
            "- Specs: `workspace/team/specs/`\n"
            "- Handoffs: `workspace/team/handoffs/`\n\n"
            "Architect writes design/specification docs in specs.\n"
            "Software engineer reads specs and writes implementation handoffs.\n",
            encoding="utf-8",
        )

    # Backward-compatible path some runs may still reference.
    legacy_specs_dir = _resolve_config_path(config, "workspace/specifications")
    legacy_specs_dir.mkdir(parents=True, exist_ok=True)
    legacy_readme = legacy_specs_dir / "README.md"
    if not legacy_readme.exists():
        legacy_readme.write_text(
            "Legacy compatibility path. Prefer `workspace/team/specs/`.\n",
            encoding="utf-8",
        )


def _discover_external_agents(config: Config) -> list[dict[str, Any]]:
    agents_cfg = config.get("agents", default={})
    team_root_raw = agents_cfg.get("team_directory", "workspace/agents")
    team_root = _resolve_config_path(config, team_root_raw)
    if not team_root.exists():
        return []

    discovered: list[dict[str, Any]] = []
    for entry in sorted(team_root.iterdir(), key=lambda item: item.name.lower()):
        if not entry.is_dir():
            continue

        config_file = None
        for candidate_name in ("agent-config.yml", "agent-config.yaml"):
            candidate = entry / candidate_name
            if candidate.exists():
                config_file = candidate
                break
        if config_file is None:
            continue

        external_cfg = _parse_external_agent_config(config_file)
        if not external_cfg.get("enabled", True):
            continue

        instructions_file = None
        for candidate_name in ("agents.md", "AGENTS.md"):
            candidate = entry / candidate_name
            if candidate.exists():
                instructions_file = candidate
                break

        model_cfg = external_cfg.get("model", {}) or {}
        model_override: dict[str, Any] = {}
        for field_name in ("provider", "id", "base_url", "api_key", "api_key_env", "structured_outputs"):
            if field_name in model_cfg:
                model_override[field_name] = model_cfg[field_name]
        for field_name in ("provider", "id", "base_url", "api_key", "api_key_env", "structured_outputs"):
            if field_name in external_cfg:
                model_override[field_name] = external_cfg[field_name]

        workspace_raw = external_cfg.get("workspace", "workspace")
        workspace_dir = (entry / workspace_raw).resolve()

        discovered.append(
            {
                "name": external_cfg.get("name", entry.name),
                "tools": external_cfg.get("tools"),
                "skills": external_cfg.get("skills", []),
                "learning": external_cfg.get("learning", {}),
                "instructions": external_cfg.get("instructions", ""),
                "instructions_file": str(instructions_file) if instructions_file else "",
                "workspace_dir": str(workspace_dir),
                "model_override": model_override,
            }
        )

    return discovered


def _coerce_learning_mode(value: str) -> LearningMode:
    try:
        return LearningMode(str(value).strip().lower())
    except ValueError as exc:
        raise ValueError(
            f"Unsupported learning mode '{value}'. Supported: always, agentic, propose, hitl."
        ) from exc


def _build_learning_store(
    store_name: str,
    store_value: Any,
    default_mode: LearningMode,
    config_cls: type,
) -> Any:
    if isinstance(store_value, bool):
        if not store_value:
            return False
        return config_cls(mode=default_mode)
    if isinstance(store_value, str):
        return config_cls(mode=_coerce_learning_mode(store_value))
    if isinstance(store_value, dict):
        enabled = bool(store_value.get("enabled", True))
        if not enabled:
            return False
        mode_value = store_value.get("mode", default_mode.value)
        mode = _coerce_learning_mode(mode_value)
        extra = {k: v for k, v in store_value.items() if k not in {"enabled", "mode"}}
        return config_cls(mode=mode, **extra)
    raise ValueError(
        f"Invalid learning store config for '{store_name}'. Use bool, mode string, or mapping."
    )


def _build_learning(config: Config, db: SqliteDb, model: OpenAILike, agent_def: dict[str, Any]) -> tuple[Any, bool]:
    learning_cfg = dict(config.get("learning", default={}) or {})
    agent_learning_cfg = agent_def.get("learning", {})
    if isinstance(agent_learning_cfg, dict):
        learning_cfg.update(agent_learning_cfg)

    if not learning_cfg or not learning_cfg.get("enabled", False):
        return None, True

    default_mode = _coerce_learning_mode(learning_cfg.get("mode", "always"))
    stores_cfg = learning_cfg.get("stores", {}) or {}

    store_builders: dict[str, tuple[type, bool]] = {
        "user_profile": (UserProfileConfig, True),
        "user_memory": (UserMemoryConfig, True),
        "session_context": (SessionContextConfig, False),
        "entity_memory": (EntityMemoryConfig, False),
        "learned_knowledge": (LearnedKnowledgeConfig, False),
        "decision_log": (DecisionLogConfig, False),
    }
    store_values: dict[str, Any] = {}
    for store_name, (config_cls, default_enabled) in store_builders.items():
        raw_value = stores_cfg.get(store_name, default_enabled)
        store_values[store_name] = _build_learning_store(
            store_name=store_name,
            store_value=raw_value,
            default_mode=default_mode,
            config_cls=config_cls,
        )

    learning = LearningMachine(
        db=db,
        model=model,
        user_profile=store_values["user_profile"],
        user_memory=store_values["user_memory"],
        session_context=store_values["session_context"],
        entity_memory=store_values["entity_memory"],
        learned_knowledge=store_values["learned_knowledge"],
        decision_log=store_values["decision_log"],
    )
    add_learnings_to_context = bool(learning_cfg.get("add_to_context", True))
    return learning, add_learnings_to_context


def _resolve_instructions(config: Config, extra_parts: list[str] | None = None) -> str:
    instructions_parts: list[str] = []
    base_instructions = config.get("agents", "base_instructions", default="")
    if base_instructions:
        instructions_parts.append(base_instructions)

    agents_path = config.get("context", "agents_path", default="workspace/AGENTS.md")
    agents_text = read_text_if_exists(agents_path)
    if agents_text:
        instructions_parts.append("AGENTS:\n" + agents_text)

    soul_path = config.get("soul", "path", default="workspace/SOUL.md")
    soul_text = read_text_if_exists(soul_path)
    if soul_text:
        instructions_parts.append("SOUL:\n" + soul_text)

    tasks_path = config.get("context", "tasks_path", default="workspace/tasks/README.md")
    tasks_text = read_text_if_exists(tasks_path)
    if tasks_text:
        instructions_parts.append("TASKS:\n" + tasks_text)

    team_path = config.get("context", "team_readme_path", default="workspace/team/README.md")
    team_text = read_text_if_exists(team_path)
    if team_text:
        instructions_parts.append("TEAM WORKSPACE:\n" + team_text)

    file_base = _resolve_config_path(config, config.get("toolkits", "file", "base_dir", default="workspace"))
    shell_base = _resolve_config_path(config, config.get("toolkits", "shell", "base_dir", default="workspace"))
    tasks_dir = _resolve_config_path(config, config.get("context", "tasks_dir", default="workspace/tasks"))
    specs_dir = _resolve_config_path(config, config.get("context", "specs_dir", default="workspace/team/specs"))
    handoffs_dir = _resolve_config_path(config, config.get("context", "handoffs_dir", default="workspace/team/handoffs"))
    skills_dirs = [_resolve_config_path(config, p) for p in config.get("skills", "directories", default=[]) or []]
    skills_lines = "\n".join(f"- `{path}`" for path in skills_dirs) if skills_dirs else "- (none configured)"
    instructions_parts.append(
        "WORKING DIRECTORIES:\n"
        f"- repo_root: `{config.path.parent.resolve()}`\n"
        f"- file_tools_base_dir: `{file_base}`\n"
        f"- shell_tools_base_dir: `{shell_base}`\n"
        f"- tasks_dir: `{tasks_dir}`\n"
        f"- specs_dir: `{specs_dir}`\n"
        f"- handoffs_dir: `{handoffs_dir}`\n"
        "- skills_dirs:\n"
        f"{skills_lines}\n\n"
        "Tool-call rule: call `bridge.get_available_tools` first and use exact function names from that output."
    )

    if extra_parts:
        instructions_parts.extend(part for part in extra_parts if part)

    return "\n\n".join(part for part in instructions_parts if part).strip()


def _build_agent_tools(
    config: Config,
    shared_tools: dict[str, Any],
    agent_def: dict[str, Any],
) -> list[Any]:
    def _dedupe(items: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            ordered.append(item)
        return ordered

    tool_names = agent_def.get("tools")
    bridge_enabled = bool(config.get("toolkits", "bridge", "enabled", default=True))
    if tool_names:
        selected_names = [name for name in tool_names if name in shared_tools]
        if "tasks" in tool_names:
            selected_names.append("tasks")
        if bridge_enabled and "bridge" not in selected_names:
            selected_names.append("bridge")
    else:
        selected_names = list(shared_tools.keys())
        if bool(config.get("toolkits", "tasks", "enabled", default=True)):
            selected_names.append("tasks")
        if bridge_enabled and "bridge" not in selected_names:
            selected_names.append("bridge")
    selected_names = _dedupe(selected_names)

    workspace_dir_raw = agent_def.get("workspace_dir")
    workspace_dir = Path(workspace_dir_raw) if workspace_dir_raw else None
    if workspace_dir is None and config.get("agents", "per_agent_workspace", default=False):
        workspace_dir = _resolve_config_path(config, Path("workspace/agents") / agent_def.get("name", "agent"))

    if workspace_dir:
        workspace_dir.mkdir(parents=True, exist_ok=True)

    tasks_enabled = bool(config.get("toolkits", "tasks", "enabled", default=True))
    tasks_dir = _resolve_config_path(config, config.get("context", "tasks_dir", default="workspace/tasks"))

    agent_tools: list[Any] = []
    for tool_name in selected_names:
        if tasks_enabled and tool_name == "tasks":
            actor_name = str(agent_def.get("name", "agent"))
            agent_tools.append(TaskBoardTools(tasks_dir=tasks_dir, default_actor=actor_name))
            continue
        if bridge_enabled and tool_name == "bridge":
            actor_name = str(agent_def.get("name", "agent"))
            default_file_base = _resolve_config_path(
                config,
                config.get("toolkits", "file", "base_dir", default="workspace"),
            )
            default_file_base.mkdir(parents=True, exist_ok=True)
            bridge_workspace_dir = workspace_dir.resolve() if workspace_dir else default_file_base
            agent_tools.append(
                BridgeTools(
                    workspace_dir=bridge_workspace_dir,
                    tasks_dir=tasks_dir,
                    default_actor=actor_name,
                    websearch_tools=shared_tools.get("web_search"),
                    website_tools=shared_tools.get("website"),
                    hackernews_tools=shared_tools.get("hackernews"),
                    youtube_tools=shared_tools.get("youtube"),
                    github_tools=shared_tools.get("github"),
                    collaboration_tools=shared_tools.get("collaboration"),
                    skill_dirs=[_resolve_config_path(config, p) for p in config.get("skills", "directories", default=[]) or []],
                )
            )
            continue
        if workspace_dir and tool_name == "file":
            agent_tools.append(FileTools(base_dir=workspace_dir))
            agent_tools.append(FileCompatTools(base_dir=workspace_dir))
            continue
        if workspace_dir and tool_name == "shell":
            agent_tools.append(ShellTools(base_dir=workspace_dir))
            continue
        agent_tools.append(shared_tools[tool_name])
        if tool_name == "file":
            default_file_base = _resolve_config_path(
                config,
                config.get("toolkits", "file", "base_dir", default="workspace"),
            )
            default_file_base.mkdir(parents=True, exist_ok=True)
            agent_tools.append(FileCompatTools(base_dir=default_file_base))
    return agent_tools


def _build_team_tools(
    config: Config,
    shared_tools: dict[str, Any],
    team_def: dict[str, Any],
) -> list[Any] | None:
    def _dedupe(items: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            ordered.append(item)
        return ordered

    tool_names = team_def.get("tools")
    if not tool_names:
        tool_names = []
    tasks_enabled = bool(config.get("toolkits", "tasks", "enabled", default=True))
    bridge_enabled = bool(config.get("toolkits", "bridge", "enabled", default=True))
    tasks_dir = _resolve_config_path(config, config.get("context", "tasks_dir", default="workspace/tasks"))
    if bridge_enabled and "bridge" not in tool_names:
        tool_names = list(tool_names) + ["bridge"]
    tool_names = _dedupe(list(tool_names))
    built_tools: list[Any] = []
    for tool_name in tool_names:
        if tasks_enabled and tool_name == "tasks":
            built_tools.append(TaskBoardTools(tasks_dir=tasks_dir, default_actor=str(team_def.get("name", "team"))))
            continue
        if bridge_enabled and tool_name == "bridge":
            default_file_base = _resolve_config_path(
                config,
                config.get("toolkits", "file", "base_dir", default="workspace"),
            )
            default_file_base.mkdir(parents=True, exist_ok=True)
            built_tools.append(
                BridgeTools(
                    workspace_dir=default_file_base,
                    tasks_dir=tasks_dir,
                    default_actor=str(team_def.get("name", "team")),
                    websearch_tools=shared_tools.get("web_search"),
                    website_tools=shared_tools.get("website"),
                    hackernews_tools=shared_tools.get("hackernews"),
                    youtube_tools=shared_tools.get("youtube"),
                    github_tools=shared_tools.get("github"),
                    collaboration_tools=shared_tools.get("collaboration"),
                    skill_dirs=[_resolve_config_path(config, p) for p in config.get("skills", "directories", default=[]) or []],
                )
            )
            continue
        if tool_name in shared_tools:
            built_tools.append(shared_tools[tool_name])
    return built_tools


def _build_teams(
    config: Config,
    db: SqliteDb,
    shared_tools: dict[str, Any],
    standalone_agents: dict[str, Agent],
) -> dict[str, Team]:
    teams_cfg = config.get("teams", default={}) or {}
    if not teams_cfg.get("enabled", False):
        return {}

    base_instructions = teams_cfg.get("base_instructions", "")
    definitions = teams_cfg.get("definitions", []) or []
    registry: dict[str, Team] = {}
    for team_def in definitions:
        name = team_def.get("name")
        if not name:
            continue
        if name in standalone_agents:
            raise ValueError(f"Team name '{name}' conflicts with an existing agent name.")

        member_names = list(team_def.get("members", []) or [])
        members = [standalone_agents[m] for m in member_names if m in standalone_agents]
        if not members:
            continue

        team_model = _build_model(config, team_def.get("model_override", {}))
        team_tools = _build_team_tools(config, shared_tools, team_def)

        extra_instruction_parts: list[str] = []
        if base_instructions:
            extra_instruction_parts.append(str(base_instructions))
        if team_def.get("instructions"):
            extra_instruction_parts.append(str(team_def.get("instructions")))
        instructions = _resolve_instructions(config, extra_parts=extra_instruction_parts)

        team = Team(
            name=name,
            members=members,
            model=team_model,
            db=db,
            markdown=bool(team_def.get("markdown", True)),
            instructions=instructions or None,
            tools=team_tools,
            respond_directly=bool(team_def.get("respond_directly", False)),
            determine_input_for_members=bool(team_def.get("determine_input_for_members", True)),
            delegate_to_all_members=bool(team_def.get("delegate_to_all_members", False)),
            add_member_tools_to_context=bool(team_def.get("add_member_tools_to_context", True)),
            share_member_interactions=bool(team_def.get("share_member_interactions", True)),
            add_datetime_to_context=bool(config.get("context", "add_datetime", default=True)),
            timezone_identifier=config.get("context", "timezone_identifier", default=None),
            add_history_to_context=bool(team_def.get("add_history_to_context", True)),
            num_history_runs=team_def.get("num_history_runs", 6),
            search_session_history=bool(team_def.get("search_session_history", True)),
            num_history_sessions=team_def.get("num_history_sessions", 3),
            enable_session_summaries=bool(team_def.get("enable_session_summaries", True)),
        )
        registry[name] = team
    return registry


def build_agents(config: Config) -> AgentRegistry:
    tasks_dir_raw = config.get("context", "tasks_dir", default="workspace/tasks")
    tasks_dir = _resolve_config_path(config, tasks_dir_raw)
    TaskBoardStore(tasks_dir=tasks_dir).ensure_workspace()
    _ensure_shared_team_workspace(config)
    shared_tools = _build_tools(config)

    memory_cfg = config.get("memory", default={})
    db_file = memory_cfg.get("db_file", "data/bitdoze.db")
    db_path = Path(db_file)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = SqliteDb(db_file=str(db_path))

    agents_cfg = config.get("agents", default={})
    default_agent = agents_cfg.get("default", "main")
    default_team = config.get("teams", "default", default=None)
    definitions = list(agents_cfg.get("definitions", []) or [])
    definitions.extend(_discover_external_agents(config))

    standalone_registry: dict[str, Agent] = {}
    if not definitions:
        definitions = [{"name": default_agent}]

    for agent_def in definitions:
        name = agent_def.get("name", default_agent)
        model_overrides = agent_def.get("model_override", {})
        model = _build_model(config, model_overrides)

        agent_tools = _build_agent_tools(config, shared_tools, agent_def)

        extra_instruction_parts: list[str] = []
        instructions_file = agent_def.get("instructions_file")
        if instructions_file:
            instructions_text = read_text_if_exists(instructions_file)
            if instructions_text:
                extra_instruction_parts.append("TEAM AGENT INSTRUCTIONS:\n" + instructions_text)
        inline_instructions = agent_def.get("instructions", "")
        if inline_instructions:
            extra_instruction_parts.append(str(inline_instructions))
        instructions = _resolve_instructions(config, extra_parts=extra_instruction_parts)

        skills_cfg = config.get("skills", default={})
        skills_enabled = bool(skills_cfg.get("enabled", False))
        skills_loader = None
        if skills_enabled:
            base_dirs = [Path(p) for p in skills_cfg.get("directories", [])]
            skill_names = list(agent_def.get("skills", []) or [])
            loaders: list[SkillLoader] = []
            if skill_names:
                for base_dir in base_dirs:
                    for skill_name in skill_names:
                        candidate = base_dir / skill_name
                        if candidate.exists():
                            loaders.append(LocalSkills(str(candidate)))
            else:
                for base_dir in base_dirs:
                    if base_dir.exists():
                        loaders.append(LocalSkills(str(base_dir)))
            if loaders:
                skills_loader = Skills(loaders=loaders)
        memory_mode = memory_cfg.get("mode", "automatic")
        add_history = bool(memory_cfg.get("add_history_to_context", True))
        num_history_runs = memory_cfg.get("num_history_runs")
        read_chat_history = bool(memory_cfg.get("read_chat_history", True))
        search_session_history = bool(memory_cfg.get("search_session_history", True))
        num_history_sessions = memory_cfg.get("num_history_sessions")
        add_memories = memory_cfg.get("add_memories_to_context")
        enable_session_summaries = bool(memory_cfg.get("enable_session_summaries", True))
        add_session_summary = bool(memory_cfg.get("add_session_summary_to_context", True))
        summary_prompt = memory_cfg.get("summary_prompt")
        summary_request_message = memory_cfg.get("summary_request_message")
        add_datetime = bool(config.get("context", "add_datetime", default=True))
        timezone_identifier = config.get("context", "timezone_identifier", default=None)

        session_summary_manager = None
        if enable_session_summaries:
            session_summary_manager = SessionSummaryManager(
                model=model,
                session_summary_prompt=summary_prompt,
                summary_request_message=summary_request_message
                or "Return a JSON object with keys summary and topics.",
            )
        learning, add_learnings_to_context = _build_learning(config, db, model, agent_def)

        agent = Agent(
            name=name,
            model=model,
            tools=agent_tools,
            instructions=instructions or None,
            db=db,
            markdown=True,
            update_memory_on_run=(memory_mode == "automatic"),
            enable_agentic_memory=(memory_mode == "agentic"),
            add_memories_to_context=add_memories,
            add_history_to_context=add_history,
            num_history_runs=num_history_runs,
            read_chat_history=read_chat_history,
            search_session_history=search_session_history,
            num_history_sessions=num_history_sessions,
            enable_session_summaries=enable_session_summaries,
            add_session_summary_to_context=add_session_summary,
            session_summary_manager=session_summary_manager,
            add_datetime_to_context=add_datetime,
            timezone_identifier=timezone_identifier,
            skills=skills_loader,
            learning=learning,
            add_learnings_to_context=add_learnings_to_context,
        )
        standalone_registry[name] = agent

    teams_registry = _build_teams(config, db, shared_tools, standalone_registry)
    registry: dict[str, Any] = {}
    registry.update(standalone_registry)
    registry.update(teams_registry)

    if default_agent not in registry and default_team in registry:
        default_agent = str(default_team)
    if default_agent not in registry:
        default_agent = next(iter(registry.keys()))

    return AgentRegistry(
        agents=registry,
        default_agent=default_agent,
        standalone_agents=standalone_registry,
        teams=teams_registry,
    )
