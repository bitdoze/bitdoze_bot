import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from bitdoze_bot.discord_bot import run_bot


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
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
