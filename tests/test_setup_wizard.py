from __future__ import annotations

from pathlib import Path

import yaml

from bitdoze_bot.setup_wizard import SetupAnswers, run_setup


def _load_yaml(path: Path) -> dict:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    assert isinstance(loaded, dict)
    return loaded


def test_run_setup_creates_config_env_workspace_and_agents(tmp_path: Path) -> None:
    home_dir = tmp_path / "home"
    answers = SetupAnswers(
        home_dir=home_dir,
        config_path=home_dir / "config.yaml",
        env_path=home_dir / ".env",
        discord_token_env="DISCORD_BOT_TOKEN",
        discord_token_value="discord-token-value",
        configure_github=True,
        github_token_env="GITHUB_ACCESS_TOKEN",
        github_token_value="gh-token-value",
        configure_model=True,
        model_api_key_env="OPENAI_API_KEY",
        model_api_key_value="model-key-value",
        model_id="gpt-4o-mini",
        model_base_url="https://api.example.test/v1",
        timezone_identifier="UTC",
        create_workspace_files=True,
        create_workspace_agents=True,
        configure_service=True,
        service_workdir=tmp_path / "repo-root",
        install_user_service=True,
        service_user_dir=home_dir / ".config" / "systemd" / "user",
    )

    run_setup(answers)

    assert (home_dir / "config.yaml").exists()
    assert (home_dir / ".env").exists()
    assert (home_dir / "workspace" / "AGENTS.md").exists()
    assert (home_dir / "workspace" / "USER.md").exists()
    assert (home_dir / "workspace" / "SOUL.md").exists()
    assert (home_dir / "workspace" / "agents" / "architect" / "agent.yaml").exists()
    assert (home_dir / "workspace" / "agents" / "software-engineer" / "agent.yaml").exists()
    assert (home_dir / "bitdoze-bot.service").exists()
    assert (home_dir / ".config" / "systemd" / "user" / "bitdoze-bot.service").exists()

    config = _load_yaml(home_dir / "config.yaml")
    assert config["discord"]["token_env"] == "DISCORD_BOT_TOKEN"
    assert config["toolkits"]["github"]["token_env"] == "GITHUB_ACCESS_TOKEN"
    assert config["model"]["api_key_env"] == "OPENAI_API_KEY"
    assert config["model"]["id"] == "gpt-4o-mini"
    assert config["model"]["base_url"] == "https://api.example.test/v1"
    assert config["memory"]["db_file"] == "data/bitdoze.db"
    assert config["agents"]["workspace_dir"] == "workspace/agents"
    assert config["skills"]["directories"] == ["skills"]

    env_raw = (home_dir / ".env").read_text(encoding="utf-8")
    assert "DISCORD_BOT_TOKEN=discord-token-value" in env_raw
    assert "GITHUB_ACCESS_TOKEN=gh-token-value" in env_raw
    assert "OPENAI_API_KEY=model-key-value" in env_raw
    service_raw = (home_dir / "bitdoze-bot.service").read_text(encoding="utf-8")
    assert f"Environment=BITDOZE_BOT_HOME={home_dir}" in service_raw
    assert f"EnvironmentFile={home_dir / '.env'}" in service_raw
    assert f"WorkingDirectory={tmp_path / 'repo-root'}" in service_raw
    assert "ExecStart=" in service_raw
    assert "uv run" in service_raw
    agents_md = (home_dir / "workspace" / "AGENTS.md").read_text(encoding="utf-8")
    assert "Workspace Map" in agents_md
    assert "First-Use Onboarding" in agents_md
    assert "Cron Awareness" in agents_md
    user_md = (home_dir / "workspace" / "USER.md").read_text(encoding="utf-8")
    assert "onboarding_status: pending" in user_md


def test_run_setup_without_workspace_agents_disables_delivery_team(tmp_path: Path) -> None:
    home_dir = tmp_path / "home"
    config_path = home_dir / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "id": "kept-model",
                    "base_url": "https://kept.example.test/v1",
                    "api_key_env": "KEPT_MODEL_KEY",
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    answers = SetupAnswers(
        home_dir=home_dir,
        config_path=config_path,
        env_path=home_dir / ".env",
        discord_token_env="DISCORD_BOT_TOKEN",
        discord_token_value=None,
        configure_github=False,
        github_token_env=None,
        github_token_value=None,
        configure_model=False,
        model_api_key_env=None,
        model_api_key_value=None,
        model_id=None,
        model_base_url=None,
        timezone_identifier="UTC",
        create_workspace_files=True,
        create_workspace_agents=False,
        configure_service=False,
        service_workdir=None,
        install_user_service=False,
        service_user_dir=None,
    )

    run_setup(answers)

    config = _load_yaml(home_dir / "config.yaml")
    assert config["agents"]["default"] == "main"
    assert config["model"]["id"] == "kept-model"
    assert config["model"]["base_url"] == "https://kept.example.test/v1"
    assert config["model"]["api_key_env"] == "KEPT_MODEL_KEY"
    if "teams" in config:
        assert config["teams"].get("definitions", []) == []
    rules = config.get("agents", {}).get("routing", {}).get("rules", [])
    assert all(rule.get("agent") != "delivery-team" for rule in rules if isinstance(rule, dict))
    assert not (home_dir / "workspace" / "agents" / "architect" / "agent.yaml").exists()
    assert not (home_dir / "bitdoze-bot.service").exists()


def test_run_setup_ignores_invalid_env_keys_when_rewriting(tmp_path: Path) -> None:
    home_dir = tmp_path / "home"
    env_path = home_dir / ".env"
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text(
        "\n".join(
            [
                "# old",
                "DISCORD_BOT_TOKEN=old-discord",
                "invalid.key=should-be-removed",
                "ALSO-INVALID=removed-too",
                "CUSTOM_KEEP=1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    answers = SetupAnswers(
        home_dir=home_dir,
        config_path=home_dir / "config.yaml",
        env_path=env_path,
        discord_token_env="DISCORD_BOT_TOKEN",
        discord_token_value=None,
        configure_github=False,
        github_token_env=None,
        github_token_value=None,
        configure_model=False,
        model_api_key_env=None,
        model_api_key_value=None,
        model_id=None,
        model_base_url=None,
        timezone_identifier="UTC",
        create_workspace_files=False,
        create_workspace_agents=False,
        configure_service=False,
        service_workdir=None,
        install_user_service=False,
        service_user_dir=None,
    )

    run_setup(answers)

    env_raw = env_path.read_text(encoding="utf-8")
    assert "invalid.key=" not in env_raw
    assert "ALSO-INVALID=" not in env_raw
    assert "CUSTOM_KEEP=1" in env_raw


def test_run_setup_migrates_legacy_config_yml_to_yaml(tmp_path: Path) -> None:
    home_dir = tmp_path / "home"
    legacy = home_dir / "config.yml"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text(
        yaml.safe_dump(
            {
                "model": {"id": "legacy-model"},
                "discord": {"token_env": "DISCORD_BOT_TOKEN"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    answers = SetupAnswers(
        home_dir=home_dir,
        config_path=home_dir / "config.yaml",
        env_path=home_dir / ".env",
        discord_token_env="DISCORD_BOT_TOKEN",
        discord_token_value=None,
        configure_github=False,
        github_token_env=None,
        github_token_value=None,
        configure_model=False,
        model_api_key_env=None,
        model_api_key_value=None,
        model_id=None,
        model_base_url=None,
        timezone_identifier="UTC",
        create_workspace_files=False,
        create_workspace_agents=False,
        configure_service=False,
        service_workdir=None,
        install_user_service=False,
        service_user_dir=None,
    )

    run_setup(answers)

    assert (home_dir / "config.yaml").exists()
    assert not (home_dir / "config.yml").exists()
    backups = sorted(home_dir.glob("config.yml.bak.*"))
    assert backups
    cfg = _load_yaml(home_dir / "config.yaml")
    assert cfg["model"]["id"] == "legacy-model"
