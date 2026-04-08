from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from bitdoze_bot.config import Config
from bitdoze_bot.embedder import build_embedder


def test_embedder_string_uses_model_endpoint(tmp_path: Path) -> None:
    config = Config(
        data={"model": {"base_url": "https://api.example.com/v1", "api_key_env": "MODEL_KEY"},
              "knowledge": {"embedder": "text-embedding-3-small"}},
        path=tmp_path / "config.yaml",
    )
    with patch.dict(os.environ, {"MODEL_KEY": "test-key"}):
        e = build_embedder(config)
    assert e.id == "text-embedding-3-small"
    assert e.base_url == "https://api.example.com/v1"


def test_embedder_dict_custom_endpoint(tmp_path: Path) -> None:
    config = Config(
        data={"knowledge": {"embedder": {"id": "custom-model", "base_url": "https://embed.com/v1", "api_key_env": "EMBED_KEY"}}},
        path=tmp_path / "config.yaml",
    )
    with patch.dict(os.environ, {"EMBED_KEY": "embed-key"}):
        e = build_embedder(config)
    assert e.id == "custom-model"
    assert e.base_url == "https://embed.com/v1"


def test_embedder_dict_partial_fallback(tmp_path: Path) -> None:
    config = Config(
        data={"model": {"base_url": "https://api.example.com/v1", "api_key_env": "MODEL_KEY"},
              "knowledge": {"embedder": {"id": "text-embedding-3-large"}}},
        path=tmp_path / "config.yaml",
    )
    with patch.dict(os.environ, {"MODEL_KEY": "model-key"}):
        e = build_embedder(config)
    assert e.id == "text-embedding-3-large"
    assert e.base_url == "https://api.example.com/v1"


def test_embedder_missing_api_key_raises(tmp_path: Path) -> None:
    config = Config(data={"model": {"api_key_env": "MISSING_KEY"}}, path=tmp_path / "config.yaml")
    try:
        build_embedder(config)
        raise AssertionError("Expected ValueError")
    except ValueError as e:
        assert "MISSING_KEY" in str(e)