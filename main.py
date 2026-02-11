from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

from bitdoze_bot.config import load_config
from bitdoze_bot.discord_bot import run_bot
from bitdoze_bot.logging_setup import configure_logging_from_config


HOME_DIR_ENV_VAR = "BITDOZE_BOT_HOME"
DEFAULT_HOME_DIR = Path("~/.bitdoze-bot")
DEFAULT_HOME_CONFIG_CANDIDATES = ("config.yaml", "config.yml")
DEFAULT_HOME_ENV_NAME = ".env"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _resolve_home_dir(arg_value: str | None) -> Path:
    raw = arg_value or os.getenv(HOME_DIR_ENV_VAR) or str(DEFAULT_HOME_DIR)
    return Path(raw).expanduser().resolve()


def _copy_tree_if_missing(source: Path, target: Path) -> None:
    if not source.exists() or target.exists():
        return
    shutil.copytree(source, target)


def _bootstrap_home(home_dir: Path, repo_root: Path) -> None:
    home_dir.mkdir(parents=True, exist_ok=True)
    for folder in ("logs", "data"):
        (home_dir / folder).mkdir(parents=True, exist_ok=True)

    defaults = (
        (repo_root / "config.example.yaml", home_dir / "config.yaml"),
        (repo_root / ".env.example", home_dir / ".env"),
    )
    for source, target in defaults:
        if source.exists() and not target.exists():
            shutil.copy2(source, target)

    workspace_target = home_dir / "workspace"
    skills_target = home_dir / "skills"
    _copy_tree_if_missing(repo_root / "workspace", workspace_target)
    _copy_tree_if_missing(repo_root / "skills", skills_target)
    workspace_target.mkdir(parents=True, exist_ok=True)
    skills_target.mkdir(parents=True, exist_ok=True)


def _resolve_config_path(arg_value: str | None, home_dir: Path, repo_root: Path) -> Path:
    if arg_value:
        return Path(arg_value).expanduser().resolve()

    for name in DEFAULT_HOME_CONFIG_CANDIDATES:
        candidate = home_dir / name
        if candidate.exists():
            return candidate

    fallback = repo_root / "config.yaml"
    if fallback.exists():
        return fallback
    return home_dir / "config.yaml"


def _resolve_env_path(arg_value: str | None, config_path: Path, home_dir: Path, repo_root: Path) -> Path:
    if arg_value:
        return Path(arg_value).expanduser().resolve()

    candidates = (
        config_path.parent / DEFAULT_HOME_ENV_NAME,
        home_dir / DEFAULT_HOME_ENV_NAME,
        repo_root / DEFAULT_HOME_ENV_NAME,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return config_path.parent / DEFAULT_HOME_ENV_NAME


def _load_env_layers(env_path: Path, home_dir: Path, repo_root: Path) -> None:
    layers = (
        env_path,
        home_dir / DEFAULT_HOME_ENV_NAME,
        repo_root / DEFAULT_HOME_ENV_NAME,
    )
    seen: set[Path] = set()
    for layer in layers:
        try:
            resolved = layer.resolve()
        except OSError:
            resolved = layer
        if not layer.exists() or resolved in seen:
            continue
        load_dotenv(layer, override=False)
        seen.add(resolved)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bitdoze Bot (Agno + Discord)")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config YAML file (default: auto-detect in ~/.bitdoze-bot)",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="Path to .env file (default: auto-detect in config/home directory)",
    )
    parser.add_argument(
        "--home-dir",
        default=None,
        help=(
            "Home customization directory "
            "(default: $BITDOZE_BOT_HOME or ~/.bitdoze-bot)"
        ),
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    home_dir = _resolve_home_dir(args.home_dir)
    _bootstrap_home(home_dir, repo_root)

    config_path = _resolve_config_path(args.config, home_dir, repo_root)
    env_path = _resolve_env_path(args.env_file, config_path, home_dir, repo_root)
    _load_env_layers(env_path, home_dir, repo_root)

    config = load_config(config_path)
    configure_logging_from_config(config)
    run_bot(str(config_path))


if __name__ == "__main__":
    main()
