import argparse
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

from bitdoze_bot.discord_bot import run_bot


def _configure_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "bitdoze-bot.log"

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(stream_handler)
    root.addHandler(file_handler)


def main() -> None:
    _configure_logging(Path("logs"))
    parser = argparse.ArgumentParser(description="Bitdoze Bot (Agno + Discord)")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML file (default: config.yaml)",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file (default: .env)",
    )
    args = parser.parse_args()
    default_env = Path(".env")
    if default_env.exists():
        load_dotenv(default_env)
    env_path = Path(args.env_file)
    if env_path.exists() and env_path != default_env:
        load_dotenv(env_path)
    run_bot(args.config)


if __name__ == "__main__":
    main()
