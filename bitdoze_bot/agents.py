from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai.like import OpenAILike
from agno.skills import LocalSkills, Skills
from agno.session import SessionSummaryManager
from agno.tools.github import GithubTools
from agno.tools.hackernews import HackerNewsTools
from agno.tools.discord import DiscordTools
from agno.tools.file import FileTools
from agno.tools.shell import ShellTools
from agno.tools.websearch import WebSearchTools
from agno.tools.website import WebsiteTools
from agno.tools.youtube import YouTubeTools

from bitdoze_bot.config import Config
from bitdoze_bot.utils import read_text_if_exists


@dataclass(frozen=True)
class AgentRegistry:
    agents: dict[str, Agent]
    default_agent: str

    def get(self, name: str | None = None) -> Agent:
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

    return tools


def _resolve_instructions(
    config: Config,
    extra: str = "",
) -> str:
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

    if extra:
        instructions_parts.append(extra)

    return "\n\n".join(part for part in instructions_parts if part).strip()


def build_agents(config: Config) -> AgentRegistry:
    tools = _build_tools(config)

    memory_cfg = config.get("memory", default={})
    db_file = memory_cfg.get("db_file", "data/bitdoze.db")
    db_path = Path(db_file)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = SqliteDb(db_file=str(db_path))

    agents_cfg = config.get("agents", default={})
    default_agent = agents_cfg.get("default", "main")
    definitions = agents_cfg.get("definitions", []) or []

    registry: dict[str, Agent] = {}
    if not definitions:
        definitions = [{"name": default_agent}]

    for agent_def in definitions:
        name = agent_def.get("name", default_agent)
        model_overrides = agent_def.get("model_override", {})
        model = _build_model(config, model_overrides)

        tool_names = agent_def.get("tools")

        if tool_names:
            agent_tools = [tools[t] for t in tool_names if t in tools]
        else:
            agent_tools = list(tools.values())

        instructions = _resolve_instructions(config, agent_def.get("instructions", ""))

        skills_cfg = config.get("skills", default={})
        skills_enabled = bool(skills_cfg.get("enabled", False))
        skills_loader = None
        if skills_enabled:
            base_dirs = [Path(p) for p in skills_cfg.get("directories", [])]
            skill_names = list(agent_def.get("skills", []) or [])
            loaders: list[LocalSkills] = []
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
        )
        registry[name] = agent

    if default_agent not in registry:
        default_agent = next(iter(registry.keys()))

    return AgentRegistry(agents=registry, default_agent=default_agent)
