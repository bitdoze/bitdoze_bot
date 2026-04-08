"""Embedder builder with configurable endpoint support.

Provides build_embedder() to create an OpenAIEmbedder with optional
endpoint customization. Falls back to main model's base_url and api_key_env
when not specified in embedder config.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from agno.knowledge.embedder.openai import OpenAIEmbedder

from bitdoze_bot.config import Config

logger = logging.getLogger(__name__)


def _require_env(var_name: str) -> str:
    """Get required environment variable or raise error."""
    value = os.getenv(var_name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {var_name}")
    return value


def build_embedder(config: Config) -> OpenAIEmbedder:
    """Build an OpenAIEmbedder from config.

    Reads knowledge.embedder config which can be:
    - A string (model id): uses main model's base_url and api_key_env
    - A mapping with id, base_url, api_key_env for custom endpoint

    Falls back to main model's base_url and api_key_env when not specified
    in embedder config.

    Args:
        config: Application config object.

    Returns:
        Configured OpenAIEmbedder instance.
    """
    kb_cfg = config.get("knowledge", default={}) or {}
    model_cfg = config.get("model", default={}) or {}

    # Get embedder config (string or mapping)
    embedder_cfg = kb_cfg.get("embedder", "text-embedding-3-small")

    # Parse embedder config
    if isinstance(embedder_cfg, str):
        # String format: just model id
        embedder_id = embedder_cfg
        embedder_base_url = model_cfg.get("base_url")
        embedder_api_key_env = model_cfg.get("api_key_env", "OPENAI_API_KEY")
    elif isinstance(embedder_cfg, dict):
        # Mapping format: id, base_url, api_key_env
        embedder_id = embedder_cfg.get("id", "text-embedding-3-small")
        embedder_base_url = embedder_cfg.get("base_url", model_cfg.get("base_url"))
        embedder_api_key_env = embedder_cfg.get(
            "api_key_env", model_cfg.get("api_key_env", "OPENAI_API_KEY")
        )
    else:
        # Fallback to default
        logger.warning(
            "Unexpected embedder config type: %s, using defaults",
            type(embedder_cfg).__name__,
        )
        embedder_id = "text-embedding-3-small"
        embedder_base_url = model_cfg.get("base_url")
        embedder_api_key_env = model_cfg.get("api_key_env", "OPENAI_API_KEY")

    # Get API key from environment
    api_key = _require_env(embedder_api_key_env)

    logger.info(
        "Embedder id=%s base_url=%s api_key_env=%s",
        embedder_id,
        embedder_base_url or "default",
        embedder_api_key_env,
    )

    return OpenAIEmbedder(
        id=embedder_id,
        api_key=api_key,
        base_url=embedder_base_url,
    )