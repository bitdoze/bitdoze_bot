from __future__ import annotations

from pathlib import Path

from main import _bootstrap_home, _resolve_config_path, _resolve_env_path


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_bootstrap_home_creates_seed_files_and_directories(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    home_dir = tmp_path / "home"

    _write(repo_root / "config.example.yaml", "model: {}\n")
    _write(repo_root / ".env.example", "DISCORD_BOT_TOKEN=\n")
    _write(repo_root / "workspace" / "AGENTS.md", "workspace defaults\n")
    _write(repo_root / "skills" / "web-research" / "SKILL.md", "name: web-research\n")

    _bootstrap_home(home_dir, repo_root)

    assert (home_dir / "config.yaml").exists()
    assert (home_dir / ".env").exists()
    assert (home_dir / "workspace" / "AGENTS.md").exists()
    assert (home_dir / "skills" / "web-research" / "SKILL.md").exists()
    assert (home_dir / "logs").is_dir()
    assert (home_dir / "data").is_dir()


def test_default_paths_prefer_home_configuration(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    home_dir = tmp_path / "home"
    home_dir.mkdir(parents=True, exist_ok=True)
    repo_root.mkdir(parents=True, exist_ok=True)

    _write(repo_root / "config.yaml", "model: {}\n")
    _write(repo_root / ".env", "A=repo\n")
    _write(home_dir / "config.yaml", "model: {}\n")
    _write(home_dir / "config.yml", "model: {legacy: true}\n")
    _write(home_dir / ".env", "A=home\n")

    config_path = _resolve_config_path(None, home_dir, repo_root)
    env_path = _resolve_env_path(None, config_path, home_dir, repo_root)

    assert config_path == home_dir / "config.yaml"
    assert env_path == home_dir / ".env"
