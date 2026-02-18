#!/usr/bin/env python3
"""Setup and seed the knowledge base for bitdoze-bot.

This script:
1. Initializes the vector DB (LanceDb or PgVector based on config)
2. Loads documents from workspace/knowledge/ directory into the knowledge base
3. Can be re-run to add new documents

Usage:
    python scripts/setup_knowledge.py                    # uses ~/.bitdoze-bot/config.yaml
    python scripts/setup_knowledge.py --config /path     # custom config
    python scripts/setup_knowledge.py --backend lancedb  # override backend
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup knowledge base for bitdoze-bot")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--home-dir", default=None, help="Home directory (default: ~/.bitdoze-bot)")
    parser.add_argument("--backend", default=None, choices=["lancedb", "pgvector"], help="Override backend")
    parser.add_argument("--docs-dir", default=None, help="Directory with documents to load")
    args = parser.parse_args()

    home_dir = Path(args.home_dir or "~/.bitdoze-bot").expanduser().resolve()

    # Load env
    env_path = home_dir / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    # Load config
    from bitdoze_bot.config import load_config

    config_path = args.config or str(home_dir / "config.yaml")
    if not Path(config_path).exists():
        config_path = str(ROOT / "config.example.yaml")
    config = load_config(config_path)

    kb_cfg = config.get("knowledge", default={}) or {}
    backend = args.backend or kb_cfg.get("backend", "lancedb")
    embedder_id = kb_cfg.get("embedder", "text-embedding-3-small")

    from agno.knowledge import Knowledge
    from agno.knowledge.embedder.openai import OpenAIEmbedder

    embedder = OpenAIEmbedder(id=embedder_id)

    if backend == "pgvector":
        try:
            from agno.vectordb.pgvector import PgVector, SearchType
        except ImportError:
            print("ERROR: pgvector not installed. Run: uv pip install pgvector psycopg[binary]")
            sys.exit(1)
        db_url = kb_cfg.get("db_url") or os.getenv("PGVECTOR_DB_URL", "")
        if not db_url:
            print("ERROR: Set knowledge.db_url in config or PGVECTOR_DB_URL env var")
            sys.exit(1)
        print(f"Using PgVector backend: {db_url.split('@')[-1] if '@' in db_url else db_url}")
        knowledge = Knowledge(
            name="Bitdoze Knowledge",
            vector_db=PgVector(
                db_url=db_url,
                table_name=kb_cfg.get("table_name", "bitdoze_knowledge"),
                search_type=SearchType.hybrid,
                embedder=embedder,
            ),
        )
        learnings = Knowledge(
            name="Bitdoze Learnings",
            vector_db=PgVector(
                db_url=db_url,
                table_name=kb_cfg.get("learnings_table_name", "bitdoze_learnings"),
                search_type=SearchType.hybrid,
                embedder=embedder,
            ),
        )
    else:
        try:
            from agno.vectordb.lancedb import LanceDb, SearchType
        except ImportError:
            print("ERROR: lancedb not installed. Run: uv pip install lancedb tantivy")
            sys.exit(1)
        lance_uri = str(config.resolve_path(kb_cfg.get("lance_uri"), default="data/lancedb"))
        print(f"Using LanceDb backend: {lance_uri}")
        knowledge = Knowledge(
            name="Bitdoze Knowledge",
            vector_db=LanceDb(
                uri=lance_uri,
                table_name=kb_cfg.get("table_name", "bitdoze_knowledge"),
                search_type=SearchType.hybrid,
                embedder=embedder,
            ),
        )
        learnings = Knowledge(
            name="Bitdoze Learnings",
            vector_db=LanceDb(
                uri=lance_uri,
                table_name=kb_cfg.get("learnings_table_name", "bitdoze_learnings"),
                search_type=SearchType.hybrid,
                embedder=embedder,
            ),
        )

    # Load documents from docs directory
    docs_dir = Path(args.docs_dir or home_dir / "workspace" / "knowledge")
    if not docs_dir.exists():
        docs_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created knowledge directory: {docs_dir}")
        print("Add .md, .txt, or .pdf files there and re-run this script.")
        print("\nKnowledge base initialized (empty). Set knowledge.enabled=true in config.")
        return

    loaded = 0
    for filepath in sorted(docs_dir.rglob("*")):
        if filepath.is_dir():
            continue
        suffix = filepath.suffix.lower()
        if suffix in (".md", ".txt"):
            text = filepath.read_text(encoding="utf-8").strip()
            if text:
                print(f"  Loading: {filepath.relative_to(docs_dir)}")
                knowledge.insert(text=text)
                loaded += 1
        elif suffix == ".pdf":
            print(f"  Loading PDF: {filepath.relative_to(docs_dir)}")
            knowledge.insert(path=str(filepath))
            loaded += 1
        else:
            print(f"  Skipping unsupported format: {filepath.name}")

    print(f"\nLoaded {loaded} document(s) into knowledge base.")
    print("Set knowledge.enabled=true in your config.yaml to activate.")


if __name__ == "__main__":
    main()
