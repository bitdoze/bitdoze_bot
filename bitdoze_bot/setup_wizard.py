from __future__ import annotations

import argparse
import getpass
import os
import re
import shutil
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_HOME_DIR = Path("~/.bitdoze-bot")
DEFAULT_CONFIG_NAMES = ("config.yaml", "config.yml")
DEFAULT_ENV_NAME = ".env"
DEFAULT_SERVICE_NAME = "bitdoze-bot.service"
ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

WORKSPACE_FILE_TEMPLATES: dict[str, str] = {
    "AGENTS.md": """# Workspace Instructions

You are operating with a home workspace rooted at ~/.bitdoze-bot.
Keep answers practical and concise. Explain changes and validation when editing files.

## Workspace Map
- AGENTS.md: Global behavior and operational rules for all agents.
- USER.md: User profile and onboarding state. Keep this up to date.
- MEMORY.md: Long-term notes and durable decisions.
- memory/YYYY-MM-DD.md: Daily memory logs.
- SOUL.md: Persona and communication style anchor.
- HEARTBEAT.md: Prompt used by heartbeat runs.
- CRON.yaml: Scheduled jobs configuration (cron expressions + actions).
- agents/<name>/agent.yaml: Agent-specific model/tools/skills config.
- agents/<name>/AGENTS.md: Agent-specific instructions.

## File Tool Paths
- The `file` toolkit base directory is already this `workspace/` folder.
- Use relative paths like `USER.md`, `agents/architect/agent.yaml`, `memory/2026-02-11.md`.
- Do not prefix with `workspace/` in file tool calls.
- Use `save_file(contents=..., file_name=...)` argument names exactly.

## First-Use Onboarding
On first direct interaction, check USER.md.
If onboarding_status is pending or user profile fields are empty, ask onboarding questions:
1. Preferred name.
2. Discord username and how they want to be addressed.
3. Timezone and preferred active hours.
4. Preferred response style (short/detailed, technical level).
5. Whether they want proactive scheduled updates.
6. If yes, what cadence and what topic/channel.

After onboarding:
- Update USER.md with captured details.
- Set onboarding_status to completed.
- Summarize what was captured in one short confirmation message.

## Cron Awareness
- Use CRON.yaml for scheduled jobs.
- Explain cron schedules in plain language before applying changes.
- Ask for explicit confirmation before creating or changing recurring jobs.
- Include timezone and destination channel in every cron proposal.
""",
    "USER.md": """# User Profile

onboarding_status: pending
preferred_name:
discord_username:
timezone:
active_hours:
response_style:
technical_level:
scheduled_updates_enabled:
scheduled_updates_preferences:
notes:
""",
    "MEMORY.md": """# Long-Term Memory

Track long-term preferences and project decisions here.
""",
    "SOUL.md": """# Soul

You are Bitdoze Bot: practical, clear, and execution-focused.
""",
    "HEARTBEAT.md": """Run a brief health check.
If there is nothing to report, respond exactly: HEARTBEAT_OK
If there are issues, respond in 1-3 short bullets.
""",
    "CRON.yaml": """# Configure recurring scheduled runs for the bot.
# Cron format: "minute hour day month day-of-week"
# Example: "0 9 * * *" runs daily at 09:00 in configured timezone.
enabled: false
timezone: UTC
channel_id: null
jobs: []
""",
}

AGENT_TEMPLATE = """name: {name}
enabled: true
model:
  id: {model_id}
  base_url: {base_url}
  api_key_env: {api_key_env}
skills: []
"""

AGENT_INSTRUCTIONS: dict[str, str] = {
    "architect": """You are the architect.
Focus on system design, tradeoffs, interfaces, and rollout sequencing.
Return clear implementation plans that engineers can execute.
""",
    "software-engineer": """You are the software engineer.
Implement the plan with safe, tested changes.
Keep diffs focused and explain validation steps.
""",
}


@dataclass(frozen=True)
class SetupAnswers:
    home_dir: Path
    config_path: Path
    env_path: Path
    discord_token_env: str
    discord_token_value: str | None
    configure_github: bool
    github_token_env: str | None
    github_token_value: str | None
    configure_model: bool
    model_api_key_env: str | None
    model_api_key_value: str | None
    model_id: str | None
    model_base_url: str | None
    timezone_identifier: str
    create_workspace_files: bool
    create_workspace_agents: bool
    configure_service: bool
    service_workdir: Path | None
    install_user_service: bool
    service_user_dir: Path | None
    setup_pgvector: bool = False
    knowledge_backend: str = "lancedb"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_home_dir(override: str | None = None) -> Path:
    raw = override or os.getenv("BITDOZE_BOT_HOME") or str(DEFAULT_HOME_DIR)
    return Path(raw).expanduser().resolve()


def _select_config_path(home_dir: Path) -> Path:
    for name in DEFAULT_CONFIG_NAMES:
        candidate = home_dir / name
        if candidate.exists():
            return candidate
    return home_dir / DEFAULT_CONFIG_NAMES[0]


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Invalid YAML mapping: {path}")
    return dict(loaded)


def _load_base_config(config_path: Path) -> dict[str, Any]:
    if config_path.exists():
        return _load_yaml_mapping(config_path)
    if config_path.name == "config.yaml":
        legacy = config_path.with_name("config.yml")
        if legacy.exists():
            return _load_yaml_mapping(legacy)
    template = _repo_root() / "config.example.yaml"
    if template.exists():
        return _load_yaml_mapping(template)
    return {}


def _archive_legacy_config_if_needed(config_path: Path) -> Path | None:
    if config_path.name != "config.yaml":
        return None
    legacy = config_path.with_name("config.yml")
    if not legacy.exists():
        return None
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = legacy.with_name(f"{legacy.name}.bak.{ts}")
    legacy.rename(backup)
    return backup


def _get_nested(mapping: dict[str, Any], keys: tuple[str, ...], default: Any) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _set_nested(mapping: dict[str, Any], keys: tuple[str, ...], value: Any) -> None:
    current = mapping
    for key in keys[:-1]:
        next_value = current.get(key)
        if not isinstance(next_value, dict):
            next_value = {}
            current[key] = next_value
        current = next_value
    current[keys[-1]] = value


def _normalize_rel_paths(config: dict[str, Any]) -> None:
    _set_nested(config, ("monitoring", "telemetry_path"), "logs/run-telemetry.jsonl")
    _set_nested(config, ("logging", "file", "path"), "logs/bitdoze-bot.log")
    _set_nested(config, ("tool_permissions", "audit", "path"), "logs/tool-audit.jsonl")
    _set_nested(config, ("memory", "db_file"), "data/bitdoze.db")
    _set_nested(config, ("toolkits", "file", "base_dir"), "workspace")
    _set_nested(config, ("toolkits", "shell", "base_dir"), "workspace")
    _set_nested(config, ("heartbeat", "prompt_path"), "workspace/HEARTBEAT.md")
    _set_nested(config, ("cron", "path"), "workspace/CRON.yaml")
    _set_nested(config, ("soul", "path"), "workspace/SOUL.md")
    _set_nested(config, ("context", "agents_path"), "workspace/AGENTS.md")
    _set_nested(config, ("context", "user_path"), "workspace/USER.md")
    _set_nested(config, ("context", "memory_dir"), "workspace/memory")
    _set_nested(config, ("context", "long_memory_path"), "workspace/MEMORY.md")
    _set_nested(config, ("agents", "workspace_dir"), "workspace/agents")
    _set_nested(config, ("skills", "directories"), ["skills"])


def _remove_delivery_team_references(config: dict[str, Any]) -> None:
    teams = _get_nested(config, ("teams",), {})
    if isinstance(teams, dict):
        teams["definitions"] = []
        teams.pop("default", None)

    rules = _get_nested(config, ("agents", "routing", "rules"), [])
    if isinstance(rules, list):
        filtered = []
        for item in rules:
            if not isinstance(item, dict):
                continue
            if str(item.get("agent", "")).strip().lower() == "delivery-team":
                continue
            filtered.append(item)
        _set_nested(config, ("agents", "routing", "rules"), filtered)

    _set_nested(config, ("agents", "default"), "main")


def _parse_env_text(raw: str) -> tuple[list[str], dict[str, str]]:
    order: list[str] = []
    data: dict[str, str] = {}
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        clean_key = key.strip()
        if not clean_key or not ENV_VAR_NAME_RE.match(clean_key):
            continue
        if clean_key not in data:
            order.append(clean_key)
        data[clean_key] = value.strip()
    return order, data


def _load_env_defaults() -> tuple[list[str], dict[str, str]]:
    template = _repo_root() / ".env.example"
    if not template.exists():
        return ([], {})
    return _parse_env_text(template.read_text(encoding="utf-8"))


def _write_env_file(path: Path, updates: dict[str, str]) -> None:
    base_order, base_values = _load_env_defaults()
    existing_order: list[str] = []
    existing_values: dict[str, str] = {}
    if path.exists():
        existing_order, existing_values = _parse_env_text(path.read_text(encoding="utf-8"))

    merged = dict(base_values)
    merged.update(existing_values)
    merged.update(updates)

    order = list(base_order)
    for key in existing_order:
        if key not in order:
            order.append(key)
    for key in updates:
        if key not in order:
            order.append(key)
    for key in merged:
        if key not in order:
            order.append(key)

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Bitdoze Bot environment",
        "# Generated/updated by setup wizard.",
        "",
    ]
    for key in order:
        lines.append(f"{key}={merged.get(key, '')}")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _ensure_home_dirs(home_dir: Path) -> None:
    for rel in ("logs", "data", "workspace", "workspace/memory", "workspace/agents", "skills"):
        (home_dir / rel).mkdir(parents=True, exist_ok=True)


def _write_if_missing(path: Path, content: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def _create_workspace_files(home_dir: Path) -> None:
    workspace_dir = home_dir / "workspace"
    for filename, content in WORKSPACE_FILE_TEMPLATES.items():
        _write_if_missing(workspace_dir / filename, content)


def _setup_pgvector_docker(home_dir: Path) -> None:
    """Start PgVector via docker compose using the project's docker-compose.yml."""
    import os
    import shutil
    import subprocess

    compose_file = _repo_root() / "docker-compose.yml"
    if not compose_file.exists():
        print("WARNING: docker-compose.yml not found, skipping PgVector setup")
        return

    docker_bin = shutil.which("docker")
    if not docker_bin:
        print("WARNING: docker not found in PATH, skipping PgVector setup")
        print(f"  Run manually: docker compose -f {compose_file} up -d")
        return

    pgdata_dir = home_dir / "data" / "pgdata"
    pgdata_dir.mkdir(parents=True, exist_ok=True)
    print(f"Starting PgVector (data at {pgdata_dir})...")
    docker_cmd = [docker_bin, "compose", "-f", str(compose_file), "up", "-d"]
    docker_sock = Path("/var/run/docker.sock")
    should_use_sudo = (
        os.geteuid() != 0 and docker_sock.exists() and not os.access(docker_sock, os.W_OK)
    )
    if should_use_sudo:
        sudo_bin = shutil.which("sudo")
        if sudo_bin:
            print("Docker socket requires elevated permissions; retrying with sudo...")
            docker_cmd = [sudo_bin, *docker_cmd]
        else:
            print("WARNING: Docker socket needs elevated permissions but sudo was not found")
            print(f"  Run manually: sudo docker compose -f {compose_file} up -d")
            return

    try:
        subprocess.run(
            docker_cmd,
            check=True,
            timeout=120,
        )
        print("PgVector started successfully on port 5532")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"WARNING: Failed to start PgVector: {exc}")
        print(f"  Run manually: docker compose -f {compose_file} up -d")


def _create_workspace_agents(
    home_dir: Path,
    model_id: str | None,
    base_url: str | None,
    api_key_env: str | None,
) -> None:
    agents_dir = home_dir / "workspace" / "agents"
    resolved_model_id = model_id or "gpt-4o-mini"
    resolved_api_key_env = api_key_env or "OPENAI_API_KEY"
    normalized_base_url = base_url or "null"
    for name, instructions in AGENT_INSTRUCTIONS.items():
        agent_dir = agents_dir / name
        _write_if_missing(
            agent_dir / "agent.yaml",
            AGENT_TEMPLATE.format(
                name=name,
                model_id=resolved_model_id,
                base_url=normalized_base_url,
                api_key_env=resolved_api_key_env,
            ),
        )
        _write_if_missing(agent_dir / "AGENTS.md", instructions)


def _render_systemd_service(answers: SetupAnswers, workdir: Path) -> str:
    main_py = workdir / "main.py"
    uv_path = shutil.which("uv")
    if uv_path:
        exec_start = f"{Path(uv_path).resolve()} run {main_py}"
    else:
        exec_start = f"/usr/bin/env uv run {main_py}"
    return (
        "[Unit]\n"
        "Description=Bitdoze Bot\n"
        "After=network.target\n\n"
        "[Service]\n"
        f"WorkingDirectory={workdir}\n"
        f"ExecStart={exec_start}\n"
        "SuccessExitStatus=143 SIGTERM\n"
        "Restart=always\n"
        "RestartSec=5\n"
        "Environment=PYTHONUNBUFFERED=1\n"
        f"Environment=BITDOZE_BOT_HOME={answers.home_dir}\n"
        f"EnvironmentFile={answers.env_path}\n\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


def _write_systemd_service_files(answers: SetupAnswers) -> list[Path]:
    workdir = answers.service_workdir or _repo_root()
    workdir = workdir.expanduser().resolve()
    service_text = _render_systemd_service(answers, workdir)

    written: list[Path] = []
    home_service = answers.home_dir / DEFAULT_SERVICE_NAME
    home_service.parent.mkdir(parents=True, exist_ok=True)
    home_service.write_text(service_text, encoding="utf-8")
    written.append(home_service)

    if answers.install_user_service:
        user_dir = answers.service_user_dir
        if user_dir is None:
            user_dir = Path("~/.config/systemd/user").expanduser()
        user_unit = user_dir / DEFAULT_SERVICE_NAME
        user_unit.parent.mkdir(parents=True, exist_ok=True)
        user_unit.write_text(service_text, encoding="utf-8")
        written.append(user_unit)

    return written


def _apply_answers_to_config(config: dict[str, Any], answers: SetupAnswers) -> dict[str, Any]:
    updated = dict(config)
    _normalize_rel_paths(updated)

    _set_nested(updated, ("discord", "token_env"), answers.discord_token_env)
    if answers.configure_github and answers.github_token_env:
        _set_nested(updated, ("toolkits", "github", "token_env"), answers.github_token_env)
    _set_nested(updated, ("context", "timezone_identifier"), answers.timezone_identifier)
    if answers.configure_model:
        if answers.model_api_key_env:
            _set_nested(updated, ("model", "api_key_env"), answers.model_api_key_env)
        if answers.model_id:
            _set_nested(updated, ("model", "id"), answers.model_id)
        _set_nested(updated, ("model", "base_url"), answers.model_base_url)
        base_url_value = str(answers.model_base_url or "").strip().lower()
        if "openrouter.ai" in base_url_value:
            _set_nested(updated, ("model", "structured_outputs"), False)
    else:
        existing_base_url = str(_get_nested(updated, ("model", "base_url"), "") or "").strip().lower()
        existing_structured = _get_nested(updated, ("model", "structured_outputs"), "auto")
        if "openrouter.ai" in existing_base_url and existing_structured in {None, "auto", "AUTO"}:
            _set_nested(updated, ("model", "structured_outputs"), False)

    if not answers.create_workspace_agents:
        _remove_delivery_team_references(updated)

    # Knowledge base settings
    _set_nested(updated, ("knowledge", "backend"), answers.knowledge_backend)
    if answers.setup_pgvector:
        _set_nested(updated, ("knowledge", "enabled"), True)
        _set_nested(updated, ("knowledge", "backend"), "pgvector")

    return updated


def run_setup(answers: SetupAnswers) -> list[Path]:
    _ensure_home_dirs(answers.home_dir)

    if answers.create_workspace_files:
        _create_workspace_files(answers.home_dir)
    if answers.create_workspace_agents:
        _create_workspace_agents(
            answers.home_dir,
            model_id=answers.model_id,
            base_url=answers.model_base_url,
            api_key_env=answers.model_api_key_env,
        )

    base_config = _load_base_config(answers.config_path)
    updated_config = _apply_answers_to_config(base_config, answers)
    answers.config_path.parent.mkdir(parents=True, exist_ok=True)
    answers.config_path.write_text(
        yaml.safe_dump(updated_config, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    _archive_legacy_config_if_needed(answers.config_path)

    env_updates: dict[str, str] = {}
    if answers.discord_token_value is not None:
        env_updates[answers.discord_token_env] = answers.discord_token_value
    if (
        answers.configure_github
        and answers.github_token_env is not None
        and answers.github_token_value is not None
    ):
        env_updates[answers.github_token_env] = answers.github_token_value
    if (
        answers.configure_model
        and answers.model_api_key_env is not None
        and answers.model_api_key_value is not None
    ):
        env_updates[answers.model_api_key_env] = answers.model_api_key_value
    _write_env_file(answers.env_path, env_updates)

    # Setup PgVector Docker if requested
    if answers.setup_pgvector:
        _setup_pgvector_docker(answers.home_dir)

    if answers.configure_service:
        return _write_systemd_service_files(answers)
    return []


def _prompt_text(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    entered = input(f"{prompt}{suffix}: ").strip()
    if entered:
        return entered
    return default or ""


def _prompt_secret(prompt: str) -> str:
    return getpass.getpass(f"{prompt}: ").strip()


def _prompt_yes_no(prompt: str, default: bool = True) -> bool:
    default_str = "Y/n" if default else "y/N"
    entered = input(f"{prompt} [{default_str}]: ").strip().lower()
    if not entered:
        return default
    if entered in {"y", "yes"}:
        return True
    if entered in {"n", "no"}:
        return False
    return default


def _interactive_answers(home_dir_override: str | None = None) -> SetupAnswers:
    default_home = _default_home_dir(home_dir_override)
    home_input = _prompt_text("Home directory", str(default_home))
    home_dir = Path(home_input).expanduser().resolve()
    config_default = _select_config_path(home_dir)
    config_input = _prompt_text("Config path", str(config_default))
    config_path = Path(config_input).expanduser().resolve()
    env_input = _prompt_text("Env file path", str(home_dir / DEFAULT_ENV_NAME))
    env_path = Path(env_input).expanduser().resolve()

    base_config = _load_base_config(config_path)
    default_discord_env = str(
        _get_nested(base_config, ("discord", "token_env"), "DISCORD_BOT_TOKEN")
        or "DISCORD_BOT_TOKEN"
    )
    default_github_env = str(
        _get_nested(base_config, ("toolkits", "github", "token_env"), "GITHUB_ACCESS_TOKEN")
        or "GITHUB_ACCESS_TOKEN"
    )
    default_api_env = str(
        _get_nested(base_config, ("model", "api_key_env"), "OPENAI_API_KEY")
        or "OPENAI_API_KEY"
    )
    default_model_id = str(_get_nested(base_config, ("model", "id"), "gpt-4o-mini"))
    default_base_url = _get_nested(base_config, ("model", "base_url"), "")
    if default_base_url is None:
        default_base_url = ""
    default_tz = str(_get_nested(base_config, ("context", "timezone_identifier"), "UTC"))

    discord_token_env = default_discord_env
    discord_token_value = _prompt_secret(
        f"Discord bot token (stored as {discord_token_env}; leave blank to keep existing)"
    )
    configure_github = _prompt_yes_no("Configure GitHub token now", default=False)
    github_token_env: str | None = None
    github_token_value: str | None = None
    if configure_github:
        github_token_env = default_github_env
        github_token_value = _prompt_secret(
            f"GitHub token (stored as {github_token_env}; leave blank to keep existing)"
        )
    configure_model = _prompt_yes_no("Configure model/API settings now", default=False)
    model_api_key_env: str | None = None
    model_api_key_value: str | None = None
    model_id: str | None = None
    model_base_url: str | None = None
    if configure_model:
        model_api_key_env = default_api_env
        model_api_key_value = _prompt_secret(
            f"Model API key (stored as {model_api_key_env}; leave blank to keep existing)"
        )
        model_id = _prompt_text("Model id", default_model_id)
        base_url_input = _prompt_text("Model base_url (blank for null)", str(default_base_url))
        model_base_url = base_url_input.strip() or None
    timezone_identifier = _prompt_text("Timezone identifier", default_tz)
    create_workspace_files = _prompt_yes_no("Create workspace starter files", default=True)
    create_workspace_agents = _prompt_yes_no(
        "Create starter workspace agents (architect + software-engineer)",
        default=True,
    )
    configure_service = _prompt_yes_no(
        "Create/update systemd user service files",
        default=True,
    )
    service_workdir: Path | None = None
    install_user_service = False
    service_user_dir: Path | None = None
    if configure_service:
        workdir_input = _prompt_text("Service working directory", str(_repo_root()))
        service_workdir = Path(workdir_input).expanduser().resolve()
        install_user_service = _prompt_yes_no(
            "Install unit in ~/.config/systemd/user",
            default=True,
        )
        if install_user_service:
            service_user_dir = Path("~/.config/systemd/user").expanduser()

    setup_pgvector = _prompt_yes_no(
        "Setup PostgreSQL + PgVector via Docker (for knowledge base)",
        default=False,
    )
    knowledge_backend = "lancedb"
    if setup_pgvector:
        knowledge_backend = "pgvector"
    else:
        use_knowledge = _prompt_yes_no(
            "Enable knowledge base with LanceDb (file-based, no Docker needed)",
            default=False,
        )
        if use_knowledge:
            knowledge_backend = "lancedb"

    return SetupAnswers(
        home_dir=home_dir,
        config_path=config_path,
        env_path=env_path,
        discord_token_env=discord_token_env,
        discord_token_value=discord_token_value or None,
        configure_github=configure_github,
        github_token_env=github_token_env,
        github_token_value=github_token_value or None,
        configure_model=configure_model,
        model_api_key_env=model_api_key_env,
        model_api_key_value=model_api_key_value or None,
        model_id=model_id,
        model_base_url=model_base_url,
        timezone_identifier=timezone_identifier,
        create_workspace_files=create_workspace_files,
        create_workspace_agents=create_workspace_agents,
        configure_service=configure_service,
        service_workdir=service_workdir,
        install_user_service=install_user_service,
        service_user_dir=service_user_dir,
        setup_pgvector=setup_pgvector,
        knowledge_backend=knowledge_backend,
    )


def _defaults_answers(home_dir_override: str | None = None) -> SetupAnswers:
    home_dir = _default_home_dir(home_dir_override)
    config_path = _select_config_path(home_dir)
    env_path = home_dir / DEFAULT_ENV_NAME
    base_config = _load_base_config(config_path)
    discord_env = str(
        _get_nested(base_config, ("discord", "token_env"), "DISCORD_BOT_TOKEN")
        or "DISCORD_BOT_TOKEN"
    )
    timezone_identifier = str(_get_nested(base_config, ("context", "timezone_identifier"), "UTC"))
    return SetupAnswers(
        home_dir=home_dir,
        config_path=config_path,
        env_path=env_path,
        discord_token_env=discord_env,
        discord_token_value=None,
        configure_github=False,
        github_token_env=None,
        github_token_value=None,
        configure_model=False,
        model_api_key_env=None,
        model_api_key_value=None,
        model_id=None,
        model_base_url=None,
        timezone_identifier=timezone_identifier,
        create_workspace_files=True,
        create_workspace_agents=True,
        configure_service=True,
        service_workdir=_repo_root(),
        install_user_service=True,
        service_user_dir=Path("~/.config/systemd/user").expanduser(),
        setup_pgvector=False,
        knowledge_backend="lancedb",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Bitdoze Bot setup wizard")
    parser.add_argument(
        "--home-dir",
        default=None,
        help="Target home directory (default: $BITDOZE_BOT_HOME or ~/.bitdoze-bot)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Use defaults without interactive prompts",
    )
    args = parser.parse_args()

    if args.yes:
        answers = _defaults_answers(args.home_dir)
    else:
        answers = _interactive_answers(args.home_dir)

    service_paths = run_setup(answers)

    print("")
    print("Setup complete.")
    print(f"- Home: {answers.home_dir}")
    print(f"- Config: {answers.config_path}")
    print(f"- Env: {answers.env_path}")
    if service_paths:
        print("- Service file(s):")
        for path in service_paths:
            print(f"  - {path}")
        if answers.install_user_service:
            print("To enable/start service:")
            print("  systemctl --user daemon-reload")
            print(f"  systemctl --user enable --now {DEFAULT_SERVICE_NAME}")
    print("Next: review .env values, then run `uv run main.py`.")


if __name__ == "__main__":
    main()
