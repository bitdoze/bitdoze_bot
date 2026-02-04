from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bitdoze_bot.agents import build_agents
from bitdoze_bot.bridge_tools import BridgeTools
from bitdoze_bot.config import load_config


class AgentsRegistryTests(unittest.TestCase):
    def test_external_agents_and_team_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agents_dir = workspace / "agents"
            (agents_dir / "architect").mkdir(parents=True, exist_ok=True)
            (agents_dir / "software-engineer").mkdir(parents=True, exist_ok=True)

            (agents_dir / "architect" / "agent-config.yml").write_text(
                "name: architect\n"
                "tools: []\n"
                "workspace: workspace\n",
                encoding="utf-8",
            )
            (agents_dir / "software-engineer" / "agent-config.yml").write_text(
                "name: software-engineer\n"
                "tools: []\n"
                "workspace: workspace\n",
                encoding="utf-8",
            )
            (workspace / "AGENTS.md").write_text("global agent instructions", encoding="utf-8")
            (workspace / "SOUL.md").write_text("soul", encoding="utf-8")

            config_path = root / "config.yaml"
            config_path.write_text(
                "model:\n"
                "  provider: openai_like\n"
                "  id: gpt-4o-mini\n"
                "  api_key: dummy\n"
                "toolkits:\n"
                "  web_search: {enabled: false}\n"
                "  hackernews: {enabled: false}\n"
                "  website: {enabled: false}\n"
                "  github: {enabled: false}\n"
                "  youtube: {enabled: false}\n"
                "  file: {enabled: false}\n"
                "  shell: {enabled: false}\n"
                "  discord: {enabled: false}\n"
                "  tasks: {enabled: false}\n"
                "context:\n"
                "  tasks_dir: workspace/tasks\n"
                "  tasks_path: workspace/tasks/README.md\n"
                "  agents_path: workspace/AGENTS.md\n"
                "soul:\n"
                "  path: workspace/SOUL.md\n"
                "agents:\n"
                "  default: build-team\n"
                "  team_directory: workspace/agents\n"
                "  definitions: []\n"
                "teams:\n"
                "  enabled: true\n"
                "  default: build-team\n"
                "  definitions:\n"
                "    - name: build-team\n"
                "      members: [architect, software-engineer]\n",
                encoding="utf-8",
            )

            cfg = load_config(config_path)
            registry = build_agents(cfg)

            self.assertIn("architect", registry.agents)
            self.assertIn("software-engineer", registry.agents)
            self.assertIn("build-team", registry.agents)
            self.assertEqual(registry.default_agent, "build-team")
            self.assertIn("build-team", registry.teams)

    def test_external_agent_workspace_is_used_by_bridge_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            agents_dir = workspace / "agents"
            (agents_dir / "architect").mkdir(parents=True, exist_ok=True)
            (root / "skills" / "demo").mkdir(parents=True, exist_ok=True)
            (root / "skills" / "demo" / "SKILL.md").write_text("# demo", encoding="utf-8")

            (agents_dir / "architect" / "agent-config.yml").write_text(
                "name: architect\n"
                "tools: [file, shell, bridge]\n"
                "workspace: ../../..\n",
                encoding="utf-8",
            )
            (workspace / "AGENTS.md").write_text("global agent instructions", encoding="utf-8")
            (workspace / "SOUL.md").write_text("soul", encoding="utf-8")

            config_path = root / "config.yaml"
            config_path.write_text(
                "model:\n"
                "  provider: openai_like\n"
                "  id: gpt-4o-mini\n"
                "  api_key: dummy\n"
                "toolkits:\n"
                "  web_search: {enabled: false}\n"
                "  hackernews: {enabled: false}\n"
                "  website: {enabled: false}\n"
                "  github: {enabled: false}\n"
                "  youtube: {enabled: false}\n"
                "  file: {enabled: true, base_dir: workspace}\n"
                "  shell: {enabled: true, base_dir: workspace}\n"
                "  discord: {enabled: false}\n"
                "  tasks: {enabled: false}\n"
                "  collaboration: {enabled: false}\n"
                "  bridge: {enabled: true}\n"
                "context:\n"
                "  tasks_dir: workspace/tasks\n"
                "  tasks_path: workspace/tasks/README.md\n"
                "  agents_path: workspace/AGENTS.md\n"
                "soul:\n"
                "  path: workspace/SOUL.md\n"
                "agents:\n"
                "  default: architect\n"
                "  team_directory: workspace/agents\n"
                "  definitions: []\n"
                "teams:\n"
                "  enabled: false\n"
                "skills:\n"
                "  enabled: true\n"
                "  directories: [skills]\n",
                encoding="utf-8",
            )

            cfg = load_config(config_path)
            registry = build_agents(cfg)
            architect = registry.standalone_agents["architect"]

            bridge = next(tool for tool in architect.tools or [] if isinstance(tool, BridgeTools))
            self.assertEqual(bridge.file_tools.base_dir, root.resolve())
            listed = bridge.list_files()
            self.assertIn("skills", listed)


if __name__ == "__main__":
    unittest.main()
