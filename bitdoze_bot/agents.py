from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Literal, cast

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
from agno.session import SessionSummaryManager
from agno.skills import LocalSkills, SkillLoader, Skills
from agno.team import Team
from agno.tools.discord import DiscordTools
from agno.tools.file import FileTools
from agno.tools.github import GithubTools
from agno.tools.hackernews import HackerNewsTools
from agno.tools.shell import ShellTools
from agno.tools.websearch import WebSearchTools
from agno.tools.website import WebsiteTools
from agno.tools.youtube import YouTubeTools

from bitdoze_bot.config import Config
from bitdoze_bot.tool_permissions import ToolPermissionManager, get_tool_runtime_context
from bitdoze_bot.utils import read_text_if_exists

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentRegistry:
    agents: dict[str, Agent]
    teams: dict[str, Team]
    default_target: str
    aliases: dict[str, str]

    def _resolve_name(self, name: str | None) -> str | None:
        if not name:
            return None
        return self.aliases.get(name, name)

    def get(self, name: str | None = None) -> Any:
        resolved = self._resolve_name(name)
        if resolved:
            if resolved in self.teams:
                return self.teams[resolved]
            if resolved in self.agents:
                return self.agents[resolved]
        if self.default_target in self.teams:
            return self.teams[self.default_target]
        return self.agents[self.default_target]

    def has(self, name: str) -> bool:
        resolved = self._resolve_name(name)
        if not resolved:
            return False
        return resolved in self.agents or resolved in self.teams


def _require_env(var_name: str) -> str:
    value = os.getenv(var_name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {var_name}")
    return value


def _build_model(config: Config, overrides: Any = None) -> OpenAILike:
    model_cfg = dict(config.get("model", default={}))
    if isinstance(overrides, dict):
        model_cfg.update(cast(dict[str, Any], overrides))

    provider = str(model_cfg.get("provider", "openai_like")).lower()
    if provider not in {"openai_like", "openai-like"}:
        raise ValueError(f"Unsupported model provider: {provider}")

    model_id = model_cfg.get("id", "gpt-4o-mini")
    base_url = model_cfg.get("base_url")
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
        token_env = str(github_cfg.get("token_env", "GITHUB_ACCESS_TOKEN") or "GITHUB_ACCESS_TOKEN")
        access_token = os.getenv(token_env, "").strip()
        base_url = github_cfg.get("base_url")
        if access_token:
            tools["github"] = GithubTools(access_token=access_token, base_url=base_url)
        else:
            logger.warning(
                "Github toolkit enabled but env var '%s' is empty; skipping github toolkit",
                token_env,
            )

    youtube_cfg = tool_cfg.get("youtube", {})
    if youtube_cfg.get("enabled", True):
        languages = youtube_cfg.get("languages")
        tools["youtube"] = YouTubeTools(languages=languages) if languages else YouTubeTools()

    file_cfg = tool_cfg.get("file", {})
    if file_cfg.get("enabled", True):
        base_dir = config.resolve_path(file_cfg.get("base_dir"), default="workspace")
        base_dir.mkdir(parents=True, exist_ok=True)
        tools["file"] = FileTools(base_dir=base_dir)

    shell_cfg = tool_cfg.get("shell", {})
    if shell_cfg.get("enabled", True):
        base_dir = config.resolve_path(shell_cfg.get("base_dir"), default="workspace")
        base_dir.mkdir(parents=True, exist_ok=True)
        tools["shell"] = ShellTools(base_dir=base_dir)

    discord_cfg = tool_cfg.get("discord", {})
    if discord_cfg.get("enabled", True):
        token_env = config.get("discord", "token_env", default="DISCORD_BOT_TOKEN")
        bot_token = _require_env(token_env)
        tools["discord"] = DiscordTools(bot_token=bot_token)

    return tools


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


def _build_learning(config: Config, db: SqliteDb, model: OpenAILike, cfg: dict[str, Any]) -> tuple[Any, bool]:
    learning_cfg = dict(config.get("learning", default={}) or {})
    cfg_learning = cfg.get("learning", {})
    if isinstance(cfg_learning, dict):
        learning_cfg.update(cfg_learning)

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


def _resolve_instructions(
    config: Config,
    extra: str = "",
    agent_instructions_path: str | None = None,
) -> str:
    instructions_parts: list[str] = []
    base_instructions = config.get("agents", "base_instructions", default="")
    if base_instructions:
        instructions_parts.append(base_instructions)

    agents_path = config.resolve_path(
        config.get("context", "agents_path", default=None),
        default="workspace/AGENTS.md",
    )
    agents_text = read_text_if_exists(agents_path)
    if agents_text:
        instructions_parts.append("WORKSPACE AGENTS:\n" + agents_text)

    soul_path = config.resolve_path(config.get("soul", "path", default=None), default="workspace/SOUL.md")
    soul_text = read_text_if_exists(soul_path)
    if soul_text:
        instructions_parts.append("SOUL:\n" + soul_text)

    if agent_instructions_path:
        agent_text = read_text_if_exists(config.resolve_path(agent_instructions_path))
        if agent_text:
            instructions_parts.append("AGENT INSTRUCTIONS:\n" + agent_text)

    if extra:
        instructions_parts.append(extra)

    return "\n\n".join(part for part in instructions_parts if part).strip()


def _load_workspace_agent_definitions(config: Config) -> list[dict[str, Any]]:
    workspace_dir = config.resolve_path(
        config.get("agents", "workspace_dir", default=None),
        default="workspace/agents",
    )
    if not workspace_dir.exists():
        return []

    definitions: list[dict[str, Any]] = []
    for child in sorted(workspace_dir.iterdir()):
        if not child.is_dir():
            continue
        cfg_path = child / "agent.yaml"
        raw_cfg: dict[str, Any] = {}
        if cfg_path.exists():
            loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"Invalid workspace agent config: {cfg_path}")
            raw_cfg = dict(loaded)

        if raw_cfg.get("enabled", True) is False:
            continue

        name = str(raw_cfg.get("name") or child.name)
        model_cfg = raw_cfg.get("model", {}) or {}
        if not isinstance(model_cfg, dict):
            raise ValueError(f"Invalid model config in {cfg_path}")

        definition: dict[str, Any] = {
            "name": name,
            "model_override": dict(model_cfg),
            "tools": raw_cfg.get("tools"),
            "skills": raw_cfg.get("skills", []),
            "instructions": raw_cfg.get("instructions", ""),
            "learning": raw_cfg.get("learning", {}),
            "workspace": {
                "path": str(child),
                "instructions_path": str(child / "AGENTS.md"),
            },
        }
        definitions.append(definition)

    return definitions


def _merge_agent_definitions(
    configured_defs: list[dict[str, Any]],
    workspace_defs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for item in configured_defs:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        merged[name] = dict(item)
    for item in workspace_defs:
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        merged[name] = dict(item)
    return list(merged.values())


def _build_memory_options(
    config: Config,
    model: OpenAILike,
    local_cfg: Any = None,
) -> dict[str, Any]:
    memory_cfg = dict(config.get("memory", default={}) or {})
    if isinstance(local_cfg, dict):
        memory_cfg.update(cast(dict[str, Any], local_cfg))

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

    session_summary_manager = None
    if enable_session_summaries:
        session_summary_manager = SessionSummaryManager(
            model=model,
            session_summary_prompt=summary_prompt,
            summary_request_message=summary_request_message
            or "Return a JSON object with keys summary and topics.",
        )

    return {
        "memory_mode": memory_mode,
        "add_history_to_context": add_history,
        "num_history_runs": num_history_runs,
        "read_chat_history": read_chat_history,
        "search_session_history": search_session_history,
        "num_history_sessions": num_history_sessions,
        "add_memories_to_context": add_memories,
        "enable_session_summaries": enable_session_summaries,
        "add_session_summary_to_context": add_session_summary,
        "session_summary_manager": session_summary_manager,
    }


def _coerce_bool(cfg: dict[str, Any], key: str, default: bool) -> bool:
    if key not in cfg:
        return default
    return bool(cfg[key])


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return cast(dict[str, Any], value)
    return {}


def _coerce_debug_level(value: Any, default: Literal[1, 2] = 1) -> Literal[1, 2]:
    if value == 2:
        return 2
    if value == 1:
        return 1
    return default


def build_agents(config: Config) -> AgentRegistry:
    tools = _build_tools(config)
    permission_manager = ToolPermissionManager.from_config(config)
    for tool_name, tool in list(tools.items()):
        tools[tool_name] = permission_manager.wrap_tool(
            tool=tool,
            tool_name=tool_name,
            agent_name_getter=lambda: get_tool_runtime_context().agent_name or "unknown",
        )

    memory_cfg = config.get("memory", default={})
    db_file = memory_cfg.get("db_file", "data/bitdoze.db")
    db_path = config.resolve_path(db_file, default="data/bitdoze.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = SqliteDb(db_file=str(db_path))

    add_datetime = bool(config.get("context", "add_datetime", default=True))
    timezone_identifier = config.get("context", "timezone_identifier", default=None)

    agents_cfg = config.get("agents", default={})
    configured_defs = agents_cfg.get("definitions", []) or []
    workspace_defs = _load_workspace_agent_definitions(config)
    definitions = _merge_agent_definitions(configured_defs, workspace_defs)

    if not definitions:
        default_agent_name = str(agents_cfg.get("default", "main"))
        definitions = [{"name": default_agent_name}]

    registry: dict[str, Agent] = {}
    for agent_def in definitions:
        name = str(agent_def.get("name", "")).strip()
        if not name:
            continue

        model_overrides: dict[str, Any] = _coerce_mapping(agent_def.get("model_override"))
        model = _build_model(config, model_overrides)

        explicit_agent_tools = "tools" in agent_def and agent_def.get("tools") is not None
        if explicit_agent_tools:
            tool_names = list(agent_def.get("tools") or [])
            agent_tool_names = [t for t in tool_names if t in tools]
            agent_tools = [tools[t] for t in agent_tool_names]
        else:
            agent_tool_names = list(tools.keys())
            agent_tools = list(tools.values())

        workspace_cfg: dict[str, Any] = _coerce_mapping(agent_def.get("workspace"))
        agent_instructions_path = workspace_cfg.get("instructions_path")
        instructions = _resolve_instructions(
            config,
            agent_def.get("instructions", ""),
            agent_instructions_path=agent_instructions_path,
        )

        skills_cfg = config.get("skills", default={})
        skills_enabled = bool(skills_cfg.get("enabled", False))
        skills_loader = None
        if skills_enabled:
            base_dirs = [config.resolve_path(p) for p in skills_cfg.get("directories", [])]
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

        agent_memory_cfg: dict[str, Any] = _coerce_mapping(agent_def.get("memory"))
        mem_options = _build_memory_options(config, model, agent_memory_cfg)
        learning, add_learnings_to_context = _build_learning(config, db, model, agent_def)

        agent = Agent(
            name=name,
            model=model,
            tools=agent_tools,
            instructions=instructions or None,
            db=db,
            markdown=True,
            update_memory_on_run=(mem_options["memory_mode"] == "automatic"),
            enable_agentic_memory=(mem_options["memory_mode"] == "agentic"),
            add_memories_to_context=mem_options["add_memories_to_context"],
            add_history_to_context=mem_options["add_history_to_context"],
            num_history_runs=mem_options["num_history_runs"],
            read_chat_history=mem_options["read_chat_history"],
            search_session_history=mem_options["search_session_history"],
            num_history_sessions=mem_options["num_history_sessions"],
            enable_session_summaries=mem_options["enable_session_summaries"],
            add_session_summary_to_context=mem_options["add_session_summary_to_context"],
            session_summary_manager=mem_options["session_summary_manager"],
            add_datetime_to_context=add_datetime,
            timezone_identifier=timezone_identifier,
            skills=skills_loader,
            learning=learning,
            add_learnings_to_context=add_learnings_to_context,
            metadata={"workspace": workspace_cfg.get("path")} if workspace_cfg else None,
        )
        registry[name] = agent
        logger.info("Agent '%s' tools: %s", name, ", ".join(agent_tool_names) or "none")

    teams_cfg = config.get("teams", default={}) or {}
    team_definitions = teams_cfg.get("definitions", []) or []
    team_registry: dict[str, Team] = {}

    for team_def in team_definitions:
        if not isinstance(team_def, dict):
            continue
        team_name = str(team_def.get("name", "")).strip()
        if not team_name:
            continue
        member_names = [str(m).strip() for m in (team_def.get("members") or []) if str(m).strip()]
        if not member_names:
            raise ValueError(f"Team '{team_name}' has no members")

        members: list[Any] = []
        for member_name in member_names:
            if member_name in registry:
                members.append(registry[member_name])
                continue
            if member_name in team_registry:
                members.append(team_registry[member_name])
                continue
            raise ValueError(f"Team '{team_name}' member '{member_name}' not found")

        team_model_override: dict[str, Any] = _coerce_mapping(team_def.get("model_override"))
        model = _build_model(config, team_model_override)
        team_memory_cfg: dict[str, Any] = _coerce_mapping(team_def.get("memory"))
        mem_options = _build_memory_options(config, model, team_memory_cfg)
        explicit_team_tools = "tools" in team_def and team_def.get("tools") is not None
        if explicit_team_tools:
            team_tool_names = list(team_def.get("tools") or [])
            selected_team_tool_names = [t for t in team_tool_names if t in tools]
            team_tools = [tools[t] for t in selected_team_tool_names]
        else:
            selected_team_tool_names = list(tools.keys())
            team_tools = list(tools.values())

        instructions = _resolve_instructions(config, team_def.get("instructions", ""))

        team = Team(
            name=team_name,
            model=model,
            members=members,
            tools=team_tools,
            instructions=instructions or None,
            respond_directly=_coerce_bool(team_def, "respond_directly", True),
            determine_input_for_members=_coerce_bool(team_def, "determine_input_for_members", True),
            delegate_to_all_members=_coerce_bool(team_def, "delegate_to_all_members", False),
            share_member_interactions=_coerce_bool(team_def, "share_member_interactions", False),
            show_members_responses=_coerce_bool(team_def, "show_members_responses", False),
            add_member_tools_to_context=_coerce_bool(team_def, "add_member_tools_to_context", True),
            add_team_history_to_members=_coerce_bool(team_def, "add_team_history_to_members", True),
            num_team_history_runs=int(team_def.get("num_team_history_runs", 3)),
            db=db,
            markdown=True,
            update_memory_on_run=(mem_options["memory_mode"] == "automatic"),
            enable_agentic_memory=(mem_options["memory_mode"] == "agentic"),
            add_memories_to_context=mem_options["add_memories_to_context"],
            add_history_to_context=mem_options["add_history_to_context"],
            num_history_runs=mem_options["num_history_runs"],
            read_chat_history=mem_options["read_chat_history"],
            search_session_history=mem_options["search_session_history"],
            num_history_sessions=mem_options["num_history_sessions"],
            enable_session_summaries=mem_options["enable_session_summaries"],
            add_session_summary_to_context=mem_options["add_session_summary_to_context"],
            session_summary_manager=mem_options["session_summary_manager"],
            add_datetime_to_context=add_datetime,
            timezone_identifier=timezone_identifier,
            stream_member_events=_coerce_bool(team_def, "stream_member_events", True),
            debug_mode=_coerce_bool(team_def, "debug_mode", False),
            debug_level=_coerce_debug_level(team_def.get("debug_level")),
        )
        team_registry[team_name] = team
        logger.info(
            "Team '%s' members: %s | tools: %s",
            team_name,
            ", ".join(member_names),
            ", ".join(selected_team_tool_names) or "none",
        )

    aliases = dict(agents_cfg.get("aliases", {}) or {})
    default_target = str(teams_cfg.get("default") or agents_cfg.get("default", "main"))
    if default_target in aliases:
        default_target = aliases[default_target]

    if default_target not in registry and default_target not in team_registry:
        if team_registry:
            default_target = next(iter(team_registry.keys()))
        else:
            default_target = next(iter(registry.keys()))

    logger.info("Default target: %s", default_target)
    return AgentRegistry(
        agents=registry,
        teams=team_registry,
        default_target=default_target,
        aliases=aliases,
    )
