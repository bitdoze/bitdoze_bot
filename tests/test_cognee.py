from __future__ import annotations

from pathlib import Path

from bitdoze_bot.cognee import CogneeClient, load_cognee_config
from bitdoze_bot.config import Config


class _DummyResponse:
    def __init__(self, status_code: int, text: str = "", payload: object | None = None) -> None:
        self.status_code = status_code
        self.text = text
        self._payload = payload

    def json(self) -> object:
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def test_load_cognee_config_from_memory_section(monkeypatch) -> None:
    monkeypatch.setenv("COGNEE_API_TOKEN", "token-123")
    config = Config(
        data={
            "memory": {
                "cognee": {
                    "enabled": True,
                    "base_url": "http://localhost:8000",
                    "user": "bitdoze-bot@example.com",
                    "dataset": "bitdoze-user-profile",
                    "auto_sync_conversations": True,
                    "auto_recall_enabled": True,
                    "auto_recall_limit": 4,
                    "auto_recall_timeout_seconds": 2,
                    "auto_recall_max_chars": 1500,
                    "auto_recall_inject_all": True,
                    "timeout_seconds": 9,
                }
            }
        },
        path=Path("config.yaml"),
    )

    cfg = load_cognee_config(config)
    assert cfg.enabled is True
    assert cfg.base_url == "http://localhost:8000"
    assert cfg.user == "bitdoze-bot@example.com"
    assert cfg.dataset == "bitdoze-user-profile"
    assert cfg.auth_token == "token-123"
    assert cfg.timeout_seconds == 9
    assert cfg.auto_recall_enabled is True
    assert cfg.auto_recall_limit == 4
    assert cfg.auto_recall_timeout_seconds == 2
    assert cfg.auto_recall_max_chars == 1500
    assert cfg.auto_recall_inject_all is True


def test_cognee_add_memory_tries_multiple_paths(monkeypatch) -> None:
    config = Config(
        data={
            "memory": {
                "cognee": {
                    "enabled": True,
                    "base_url": "http://localhost:8000",
                    "upsert_paths": ["/not-found", "/ingest-ok"],
                    "dataset_paths": ["/dataset-ok"],
                }
            }
        },
        path=Path("config.yaml"),
    )
    client = CogneeClient(load_cognee_config(config))

    called_urls: list[str] = []

    def fake_request(self, method, url, timeout, headers, **kwargs):  # noqa: ANN001
        called_urls.append(url)
        if url.endswith("/dataset-ok"):
            return _DummyResponse(200, payload={"ok": True})
        if url.endswith("/not-found"):
            return _DummyResponse(404, text="not found")
        if url.endswith("/ingest-ok"):
            return _DummyResponse(200, payload={"ok": True})
        return _DummyResponse(500, text="unexpected")

    monkeypatch.setattr("requests.Session.request", fake_request)

    assert client.add_memory("hello memory", metadata={"type": "test"}) is True
    assert any(url.endswith("/not-found") for url in called_urls)
    assert any(url.endswith("/ingest-ok") for url in called_urls)


def test_cognee_search_skips_empty_success_and_finds_non_empty(monkeypatch) -> None:
    config = Config(
        data={
            "memory": {
                "cognee": {
                    "enabled": True,
                    "base_url": "http://localhost:8000",
                    "search_paths": ["/search-a", "/search-b"],
                }
            }
        },
        path=Path("config.yaml"),
    )
    client = CogneeClient(load_cognee_config(config))

    def fake_request(self, method, url, timeout, headers, **kwargs):  # noqa: ANN001
        if url.endswith("/search-a"):
            return _DummyResponse(200, payload={"results": []})
        if url.endswith("/search-b"):
            return _DummyResponse(200, payload={"results": [{"text": "remember this"}]})
        return _DummyResponse(500, text="unexpected")

    monkeypatch.setattr("requests.Session.request", fake_request)
    results = client.search("remember", limit=3)
    assert results == ["remember this"]


def test_cognee_search_prefers_chunks_search_type(monkeypatch) -> None:
    config = Config(
        data={
            "memory": {
                "cognee": {
                    "enabled": True,
                    "base_url": "http://localhost:8000",
                    "search_paths": ["/api/v1/search"],
                }
            }
        },
        path=Path("config.yaml"),
    )
    client = CogneeClient(load_cognee_config(config))
    captured_payloads: list[dict[str, object]] = []

    def fake_request(self, method, url, timeout, headers, **kwargs):  # noqa: ANN001
        payload = kwargs.get("json") or {}
        if isinstance(payload, dict):
            captured_payloads.append(payload)
        if payload.get("searchType") == "CHUNKS":
            return _DummyResponse(200, payload={"results": [{"text": "chunk hit"}]})
        return _DummyResponse(200, payload={"results": []})

    monkeypatch.setattr(client, "ensure_dataset", lambda: True)
    monkeypatch.setattr("requests.Session.request", fake_request)
    results = client.search("remember", limit=3)
    assert results == ["chunk hit"]
    assert captured_payloads
    assert captured_payloads[0].get("searchType") == "CHUNKS"


def test_cognee_search_extracts_from_verbose_context_result(monkeypatch) -> None:
    config = Config(
        data={"memory": {"cognee": {"enabled": True, "base_url": "http://localhost:8000"}}},
        path=Path("config.yaml"),
    )
    client = CogneeClient(load_cognee_config(config))

    def fake_request(self, method, url, timeout, headers, **kwargs):  # noqa: ANN001
        payload = [
            {
                "text_result": [{"text": "CANARY_FROM_VERBOSE"}],
                "context_result": "CANARY_FROM_VERBOSE extra",
            }
        ]
        return _DummyResponse(200, payload=payload)

    monkeypatch.setattr(client, "ensure_dataset", lambda: True)
    monkeypatch.setattr("requests.Session.request", fake_request)
    results = client.search("canary", limit=3)
    assert any("CANARY_FROM_VERBOSE" in item for item in results)


def test_cognee_search_ensures_dataset_before_query(monkeypatch) -> None:
    config = Config(
        data={"memory": {"cognee": {"enabled": True, "base_url": "http://localhost:8000"}}},
        path=Path("config.yaml"),
    )
    client = CogneeClient(load_cognee_config(config))
    called = {"ensure": 0}

    def fake_ensure_dataset() -> bool:
        called["ensure"] += 1
        return True

    def fake_request(self, method, url, timeout, headers, **kwargs):  # noqa: ANN001
        return _DummyResponse(200, payload={"results": [{"text": "match"}]})

    monkeypatch.setattr(client, "ensure_dataset", fake_ensure_dataset)
    monkeypatch.setattr("requests.Session.request", fake_request)
    results = client.search("query", limit=1)
    assert called["ensure"] == 1
    assert results == ["match"]


def test_cognee_save_conversation_turn_chunks_large_payload(monkeypatch) -> None:
    config = Config(
        data={"memory": {"cognee": {"enabled": True}}},
        path=Path("config.yaml"),
    )
    client = CogneeClient(load_cognee_config(config))

    captured_payloads: list[str] = []

    def fake_add_memory(text: str, metadata=None) -> bool:  # noqa: ANN001
        captured_payloads.append(text)
        return True

    monkeypatch.setattr(client, "add_memory", fake_add_memory)
    ok = client.save_conversation_turn(
        user_id="u1",
        session_id="s1",
        user_message="U" * 4200,
        assistant_message="A" * 4300,
        agent_name="main",
        channel_id=123,
        guild_id=None,
    )
    assert ok is True
    # Summary + at least 3 chunks for each side with default chunking.
    assert len(captured_payloads) >= 7


def test_cognee_add_memory_deduplicates_recent_identical_content(monkeypatch) -> None:
    config = Config(
        data={"memory": {"cognee": {"enabled": True, "dataset_paths": ["/dataset-ok"]}}},
        path=Path("config.yaml"),
    )
    client = CogneeClient(load_cognee_config(config))
    calls = {"count": 0}

    def fake_request(self, method, url, timeout, headers, **kwargs):  # noqa: ANN001
        calls["count"] += 1
        if url.endswith("/dataset-ok"):
            return _DummyResponse(200, payload={"ok": True})
        if url.endswith("/api/v1/add"):
            return _DummyResponse(200, payload={"ok": True})
        return _DummyResponse(200, payload={"ok": True})

    monkeypatch.setattr("requests.Session.request", fake_request)
    assert client.add_memory("duplicate text", metadata={"type": "test"}) is True
    first_count = calls["count"]
    assert client.add_memory("duplicate text", metadata={"type": "test"}) is True
    assert calls["count"] == first_count


def test_cognee_add_memory_triggers_cognify_with_cooldown(monkeypatch) -> None:
    config = Config(
        data={
            "memory": {
                "cognee": {
                    "enabled": True,
                    "dataset_paths": ["/dataset-ok"],
                    "auto_cognify_after_write": True,
                    "cognify_cooldown_seconds": 3600,
                }
            }
        },
        path=Path("config.yaml"),
    )
    client = CogneeClient(load_cognee_config(config))
    called_urls: list[str] = []

    def fake_request(self, method, url, timeout, headers, **kwargs):  # noqa: ANN001
        called_urls.append(url)
        if url.endswith("/dataset-ok"):
            return _DummyResponse(200, payload={"ok": True})
        if url.endswith("/api/v1/add"):
            return _DummyResponse(200, payload={"status": "ok"})
        if url.endswith("/api/v1/cognify"):
            return _DummyResponse(200, payload={"status": "ok"})
        return _DummyResponse(500, text="unexpected")

    monkeypatch.setattr("requests.Session.request", fake_request)

    assert client.add_memory("first text", metadata={"type": "test"}) is True
    assert client.add_memory("second text", metadata={"type": "test"}) is True

    cognify_calls = [url for url in called_urls if url.endswith("/api/v1/cognify")]
    assert len(cognify_calls) == 1
