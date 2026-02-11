from __future__ import annotations

from pathlib import Path

from agno.tools.hackernews import HackerNewsTools

from bitdoze_bot.agents import build_agents
from bitdoze_bot.config import load_config


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_build_agents_with_workspace_and_team(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TEST_MODEL_KEY", "test-key")

    workspace_dir = tmp_path / "workspace" / "agents"
    _write(
        workspace_dir / "architect" / "agent.yaml",
        """
name: architect
model:
  id: test-model
  base_url: https://example.test/v1
  api_key_env: TEST_MODEL_KEY
skills: []
""".strip(),
    )
    _write(
        workspace_dir / "architect" / "AGENTS.md",
        "Architect instructions",
    )
    _write(
        workspace_dir / "software-engineer" / "agent.yaml",
        """
name: software-engineer
model:
  id: test-model
  base_url: https://example.test/v1
  api_key_env: TEST_MODEL_KEY
skills: []
""".strip(),
    )
    _write(
        workspace_dir / "software-engineer" / "AGENTS.md",
        "Engineer instructions",
    )

    config_path = tmp_path / "config.yaml"
    db_file = tmp_path / "data" / "bot.db"
    config_path.write_text(
        f"""
model:
  provider: openai_like
  id: test-model
  base_url: https://example.test/v1
  api_key_env: TEST_MODEL_KEY
  structured_outputs: off

memory:
  db_file: {db_file}
  mode: automatic
  add_history_to_context: false
  read_chat_history: false
  search_session_history: false
  add_memories_to_context: false
  enable_session_summaries: false
  add_session_summary_to_context: false

learning:
  enabled: false

toolkits:
  web_search: {{ enabled: false }}
  hackernews: {{ enabled: false }}
  website: {{ enabled: false }}
  github: {{ enabled: false }}
  youtube: {{ enabled: false }}
  file: {{ enabled: false }}
  shell: {{ enabled: false }}
  discord: {{ enabled: false }}

context:
  add_datetime: false

skills:
  enabled: false

agents:
  workspace_dir: {workspace_dir}
  aliases:
    software-enginner: software-engineer
  definitions:
    - name: main
  default: main

teams:
  default: delivery-team
  definitions:
    - name: delivery-team
      members: [architect, software-engineer]
      respond_directly: true
      determine_input_for_members: true
      delegate_to_all_members: true
""".strip(),
        encoding="utf-8",
    )

    registry = build_agents(load_config(config_path))

    assert set(registry.agents.keys()) == {"main", "architect", "software-engineer"}
    assert set(registry.teams.keys()) == {"delivery-team"}
    assert registry.default_target == "delivery-team"

    resolved = registry.get("software-enginner")
    assert getattr(resolved, "name", None) == "software-engineer"


def test_build_agents_explicit_empty_tools(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TEST_MODEL_KEY", "test-key")

    config_path = tmp_path / "config.yaml"
    db_file = tmp_path / "data" / "bot.db"
    config_path.write_text(
        f"""
model:
  provider: openai_like
  id: test-model
  base_url: https://example.test/v1
  api_key_env: TEST_MODEL_KEY
  structured_outputs: off

memory:
  db_file: {db_file}
  mode: automatic
  add_history_to_context: false
  read_chat_history: false
  search_session_history: false
  add_memories_to_context: false
  enable_session_summaries: false
  add_session_summary_to_context: false

learning:
  enabled: false

toolkits:
  web_search: {{ enabled: false }}
  hackernews: {{ enabled: false }}
  website: {{ enabled: false }}
  github: {{ enabled: false }}
  youtube: {{ enabled: false }}
  file: {{ enabled: false }}
  shell: {{ enabled: false }}
  discord: {{ enabled: false }}

context:
  add_datetime: false

skills:
  enabled: false

agents:
  workspace_dir: {tmp_path / "workspace" / "agents"}
  definitions:
    - name: heartbeat
      tools: []
  default: heartbeat
""".strip(),
        encoding="utf-8",
    )

    registry = build_agents(load_config(config_path))
    heartbeat = registry.get("heartbeat")
    assert getattr(heartbeat, "name", None) == "heartbeat"
    assert getattr(heartbeat, "tools", None) == []


def test_workspace_agent_missing_tools_uses_default_toolset(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TEST_MODEL_KEY", "test-key")

    workspace_dir = tmp_path / "workspace" / "agents"
    _write(
        workspace_dir / "architect" / "agent.yaml",
        """
name: architect
model:
  id: test-model
  base_url: https://example.test/v1
  api_key_env: TEST_MODEL_KEY
skills: []
""".strip(),
    )
    _write(
        workspace_dir / "architect" / "AGENTS.md",
        "Architect instructions",
    )

    config_path = tmp_path / "config.yaml"
    db_file = tmp_path / "data" / "bot.db"
    config_path.write_text(
        f"""
model:
  provider: openai_like
  id: test-model
  base_url: https://example.test/v1
  api_key_env: TEST_MODEL_KEY
  structured_outputs: off

memory:
  db_file: {db_file}
  mode: automatic
  add_history_to_context: false
  read_chat_history: false
  search_session_history: false
  add_memories_to_context: false
  enable_session_summaries: false
  add_session_summary_to_context: false

learning:
  enabled: false

toolkits:
  web_search: {{ enabled: false }}
  hackernews: {{ enabled: true }}
  website: {{ enabled: false }}
  github: {{ enabled: false }}
  youtube: {{ enabled: false }}
  file: {{ enabled: false }}
  shell: {{ enabled: false }}
  discord: {{ enabled: false }}

context:
  add_datetime: false

skills:
  enabled: false

agents:
  workspace_dir: {workspace_dir}
  definitions:
    - name: main
  default: architect
""".strip(),
        encoding="utf-8",
    )

    registry = build_agents(load_config(config_path))
    architect = registry.get("architect")
    tools = getattr(architect, "tools", None) or []
    assert len(tools) == 1
    assert isinstance(tools[0], HackerNewsTools)


def test_github_tool_missing_token_is_skipped(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TEST_MODEL_KEY", "test-key")
    monkeypatch.delenv("MISSING_GH_TOKEN", raising=False)

    config_path = tmp_path / "config.yaml"
    db_file = tmp_path / "data" / "bot.db"
    config_path.write_text(
        f"""
model:
  provider: openai_like
  id: test-model
  base_url: https://example.test/v1
  api_key_env: TEST_MODEL_KEY
  structured_outputs: off

memory:
  db_file: {db_file}
  mode: automatic
  add_history_to_context: false
  read_chat_history: false
  search_session_history: false
  add_memories_to_context: false
  enable_session_summaries: false
  add_session_summary_to_context: false

learning:
  enabled: false

toolkits:
  web_search: {{ enabled: false }}
  hackernews: {{ enabled: false }}
  website: {{ enabled: false }}
  github:
    enabled: true
    token_env: MISSING_GH_TOKEN
  youtube: {{ enabled: false }}
  file: {{ enabled: false }}
  shell: {{ enabled: false }}
  discord: {{ enabled: false }}

context:
  add_datetime: false

skills:
  enabled: false

agents:
  workspace_dir: {tmp_path / "workspace" / "agents"}
  definitions:
    - name: main
  default: main
""".strip(),
        encoding="utf-8",
    )

    registry = build_agents(load_config(config_path))
    main_agent = registry.get("main")
    tools = getattr(main_agent, "tools", None) or []
    assert tools == []
