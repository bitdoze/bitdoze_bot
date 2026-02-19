from __future__ import annotations

import logging
import os
import json
from hashlib import sha1
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from time import sleep
from typing import Any

import requests

from bitdoze_bot.config import Config
from bitdoze_bot.utils import parse_bool

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CogneeConfig:
    enabled: bool = False
    base_url: str = "http://localhost:8000"
    user: str = ""
    dataset: str = "bitdoze-user-profile"
    timeout_seconds: int = 8
    auto_sync_conversations: bool = True
    auto_recall_enabled: bool = True
    auto_recall_limit: int = 5
    auto_recall_timeout_seconds: int = 3
    auto_recall_max_chars: int = 2000
    auto_recall_inject_all: bool = False
    max_turn_chars: int = 6000
    auth_token_env: str = "COGNEE_API_TOKEN"
    auth_token: str = ""
    upsert_paths: tuple[str, ...] = (
        "/api/v1/add",
        "/api/v1/memify",
    )
    search_paths: tuple[str, ...] = (
        "/api/v1/search",
    )
    dataset_paths: tuple[str, ...] = (
        "/api/v1/datasets",
    )


def _coerce_paths(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(value, list):
        return default
    paths = [str(item).strip() for item in value if str(item).strip()]
    if not paths:
        return default
    return tuple(paths)


def _safe_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def load_cognee_config(config: Config) -> CogneeConfig:
    root_raw = config.get("cognee", default={})
    root_cfg = root_raw if isinstance(root_raw, dict) else {}
    mem_raw = config.get("memory", "cognee", default={})
    mem_cfg = mem_raw if isinstance(mem_raw, dict) else {}

    merged: dict[str, Any] = dict(mem_cfg)
    merged.update(root_cfg)

    auth_token_env = str(merged.get("auth_token_env", "COGNEE_API_TOKEN") or "COGNEE_API_TOKEN")
    auth_token = os.getenv(auth_token_env, "").strip()

    return CogneeConfig(
        enabled=parse_bool(merged.get("enabled", False), False),
        base_url=str(merged.get("base_url", "http://localhost:8000")).rstrip("/"),
        user=str(merged.get("user", "")).strip(),
        dataset=str(merged.get("dataset", "bitdoze-user-profile")).strip() or "bitdoze-user-profile",
        timeout_seconds=_safe_positive_int(merged.get("timeout_seconds", 8), 8),
        auto_sync_conversations=parse_bool(merged.get("auto_sync_conversations", True), True),
        auto_recall_enabled=parse_bool(merged.get("auto_recall_enabled", True), True),
        auto_recall_limit=max(_safe_positive_int(merged.get("auto_recall_limit", 5), 5), 1),
        auto_recall_timeout_seconds=max(
            _safe_positive_int(merged.get("auto_recall_timeout_seconds", 3), 3),
            1,
        ),
        auto_recall_max_chars=max(
            _safe_positive_int(merged.get("auto_recall_max_chars", 2000), 2000),
            256,
        ),
        auto_recall_inject_all=parse_bool(merged.get("auto_recall_inject_all", False), False),
        max_turn_chars=max(_safe_positive_int(merged.get("max_turn_chars", 6000), 6000), 512),
        auth_token_env=auth_token_env,
        auth_token=auth_token,
        upsert_paths=_coerce_paths(merged.get("upsert_paths"), CogneeConfig.upsert_paths),
        search_paths=_coerce_paths(merged.get("search_paths"), CogneeConfig.search_paths),
        dataset_paths=_coerce_paths(merged.get("dataset_paths"), CogneeConfig.dataset_paths),
    )


class CogneeClient:
    def __init__(self, config: CogneeConfig) -> None:
        self._cfg = config
        self._session = requests.Session()
        self._dataset_ready = False
        self._dataset_last_attempt_at = 0.0
        self._dataset_retry_cooldown_seconds = 30.0
        self._dataset_id: str | None = None
        self._recent_content_hashes: dict[str, float] = {}
        self._dedup_ttl_seconds = 6 * 60 * 60
        self._dedup_max_items = 2048

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self._cfg.auth_token:
            headers["Authorization"] = f"Bearer {self._cfg.auth_token}"
        return headers

    def _url(self, path: str) -> str:
        return f"{self._cfg.base_url}/{path.lstrip('/')}"

    def _request(
        self,
        method: str,
        path: str,
        *,
        timeout_seconds: int | float | None = None,
        **kwargs: Any,
    ) -> requests.Response:
        return self._session.request(
            method=method.upper(),
            url=self._url(path),
            timeout=timeout_seconds if timeout_seconds is not None else self._cfg.timeout_seconds,
            headers=self._headers(),
            **kwargs,
        )

    def _first_success(self, method: str, paths: tuple[str, ...], **kwargs: Any) -> tuple[bool, str]:
        last_error = "no endpoint responded successfully"
        for path in paths:
            try:
                response = self._request(method, path, **kwargs)
            except requests.RequestException as exc:
                last_error = str(exc)
                continue
            if 200 <= response.status_code < 300:
                return True, path
            if response.status_code in {404, 405}:
                continue
            last_error = f"{path}: HTTP {response.status_code} {response.text[:200]}"
        return False, last_error

    def ensure_dataset(self) -> bool:
        started_at = perf_counter()
        now = perf_counter()
        if self._dataset_ready:
            return True
        if now - self._dataset_last_attempt_at < self._dataset_retry_cooldown_seconds:
            return False
        self._dataset_last_attempt_at = now
        payloads = [
            {"name": self._cfg.dataset},
            {"dataset": self._cfg.dataset},
            {"name": self._cfg.dataset, "user": self._cfg.user},
            {"dataset": self._cfg.dataset, "user": self._cfg.user},
        ]
        for payload in payloads:
            ok, detail = self._first_success("POST", self._cfg.dataset_paths, json=payload)
            if ok:
                elapsed_ms = int((perf_counter() - started_at) * 1000)
                logger.info(
                    "Cognee ensure_dataset success dataset=%s elapsed_ms=%d",
                    self._cfg.dataset,
                    elapsed_ms,
                )
                self._dataset_ready = True
                self._dataset_id = self._resolve_dataset_id()
                return True
            if "HTTP 409" in detail:
                elapsed_ms = int((perf_counter() - started_at) * 1000)
                logger.info(
                    "Cognee ensure_dataset already_exists dataset=%s elapsed_ms=%d",
                    self._cfg.dataset,
                    elapsed_ms,
                )
                self._dataset_ready = True
                self._dataset_id = self._resolve_dataset_id()
                return True
        elapsed_ms = int((perf_counter() - started_at) * 1000)
        logger.warning(
            "Cognee ensure_dataset failed dataset=%s elapsed_ms=%d reason=%s",
            self._cfg.dataset,
            elapsed_ms,
            detail,
        )
        return False

    def _resolve_dataset_id(self) -> str | None:
        try:
            response = self._request("GET", "/api/v1/datasets")
        except requests.RequestException:
            return None
        if not (200 <= response.status_code < 300):
            return None
        payload = _json_or_empty(response)
        if not isinstance(payload, list):
            return None
        for item in payload:
            if not isinstance(item, dict):
                continue
            if str(item.get("name", "")).strip() != self._cfg.dataset:
                continue
            dataset_id = item.get("id")
            if isinstance(dataset_id, str) and dataset_id.strip():
                return dataset_id
        return None

    def add_memory(self, text: str, metadata: dict[str, Any] | None = None) -> bool:
        started_at = perf_counter()
        if not text.strip():
            return False
        self.ensure_dataset()
        metadata_payload = dict(metadata or {})
        if self._cfg.user:
            metadata_payload.setdefault("user", self._cfg.user)

        # Keep writes lightweight; on failure we also retry with a shorter payload.
        full_text = text.strip()
        content_hash = sha1(full_text.encode("utf-8")).hexdigest()
        if self._is_recent_duplicate(content_hash):
            logger.info(
                "Cognee add_memory deduplicated dataset=%s hash=%s chars=%d",
                self._cfg.dataset,
                content_hash[:12],
                len(full_text),
            )
            return True
        metadata_payload.setdefault("content_hash", content_hash)
        short_text = full_text[:1200]
        dataset_id = self._dataset_id or self._resolve_dataset_id()
        self._dataset_id = dataset_id or self._dataset_id

        detail = "no endpoint responded successfully"
        # Preferred path for this Cognee build: multipart /api/v1/add
        form_data: dict[str, Any] = {"datasetName": self._cfg.dataset}
        if dataset_id:
            form_data["datasetId"] = dataset_id
        if metadata_payload:
            form_data["metadata"] = json.dumps(metadata_payload, ensure_ascii=True, separators=(",", ":"))
        files = [
            ("data", ("memory.txt", full_text.encode("utf-8"), "text/plain")),
        ]
        try:
            response = self._request("POST", "/api/v1/add", data=form_data, files=files)
            if 200 <= response.status_code < 300:
                elapsed_ms = int((perf_counter() - started_at) * 1000)
                logger.info(
                    "Cognee add_memory success dataset=%s chars=%d attempt=%d elapsed_ms=%d",
                    self._cfg.dataset,
                    len(text),
                    1,
                    elapsed_ms,
                )
                self._remember_hash(content_hash)
                return True
            detail = f"/api/v1/add: HTTP {response.status_code} {response.text[:200]}"
        except requests.RequestException as exc:
            detail = str(exc)

        payloads = [
            {
                "data": full_text,
                "dataset_name": self._cfg.dataset,
                "dataset_id": dataset_id,
                "run_in_background": True,
            },
            {
                "data": short_text,
                "dataset_name": self._cfg.dataset,
                "dataset_id": dataset_id,
                "run_in_background": True,
            },
            {"dataset": self._cfg.dataset, "user": self._cfg.user, "text": full_text, "metadata": metadata_payload},
            {"dataset": self._cfg.dataset, "content": full_text, "metadata": metadata_payload},
            {"dataset": self._cfg.dataset, "documents": [{"text": full_text, "metadata": metadata_payload}]},
            {"text": full_text, "metadata": metadata_payload},
            {"content": full_text, "metadata": metadata_payload},
        ]
        for attempt in (1, 2):
            for payload in payloads:
                ok, detail = self._first_success("POST", self._cfg.upsert_paths, json=payload)
                if ok:
                    elapsed_ms = int((perf_counter() - started_at) * 1000)
                    logger.info(
                        "Cognee add_memory success dataset=%s chars=%d attempt=%d elapsed_ms=%d",
                        self._cfg.dataset,
                        len(text),
                        attempt,
                        elapsed_ms,
                    )
                    self._remember_hash(content_hash)
                    return True
            if attempt == 1:
                # Short pause before retry to ride out transient disconnects.
                sleep(0.25)
        elapsed_ms = int((perf_counter() - started_at) * 1000)
        logger.warning(
            "Cognee add_memory failed dataset=%s chars=%d attempts=2 elapsed_ms=%d reason=%s",
            self._cfg.dataset,
            len(text),
            elapsed_ms,
            detail,
        )
        return False

    def search(
        self,
        query: str,
        limit: int = 5,
        *,
        max_payloads: int | None = None,
        max_paths: int | None = None,
        timeout_seconds: int | float | None = None,
    ) -> list[str]:
        started_at = perf_counter()
        if not query.strip():
            return []
        # Ensure dataset exists for first-recall scenarios (before any write occurred).
        self.ensure_dataset()
        limit = max(int(limit), 1)
        payloads = [
            {"query": query, "datasets": [self._cfg.dataset], "topK": limit, "onlyContext": True},
            {"dataset": self._cfg.dataset, "user": self._cfg.user, "query": query, "limit": limit},
            {"dataset": self._cfg.dataset, "query": query, "top_k": limit},
            {"query": query, "limit": limit},
        ]
        payload_candidates = payloads[: max_payloads] if max_payloads is not None else payloads
        path_candidates = self._cfg.search_paths[: max_paths] if max_paths is not None else self._cfg.search_paths
        detail = "no endpoint responded successfully"
        had_success_response = False
        for payload in payload_candidates:
            for path in path_candidates:
                try:
                    response = self._request("POST", path, json=payload, timeout_seconds=timeout_seconds)
                except requests.RequestException as exc:
                    detail = str(exc)
                    continue
                if response.status_code in {404, 405}:
                    detail = f"{path}: HTTP {response.status_code}"
                    continue
                if not (200 <= response.status_code < 300):
                    detail = f"{path}: HTTP {response.status_code} {response.text[:200]}"
                    continue
                had_success_response = True
                results = _extract_text_results(_json_or_empty(response))
                if not results:
                    detail = f"{path}: HTTP {response.status_code} empty results"
                    continue
                elapsed_ms = int((perf_counter() - started_at) * 1000)
                logger.info(
                    "Cognee search success dataset=%s query=%s limit=%d results=%d elapsed_ms=%d",
                    self._cfg.dataset,
                    query[:80],
                    limit,
                    len(results),
                    elapsed_ms,
                )
                deduped: list[str] = []
                seen: set[str] = set()
                for item in results:
                    normalized = " ".join(item.split())
                    if not normalized or normalized in seen:
                        continue
                    seen.add(normalized)
                    deduped.append(normalized)
                    if len(deduped) >= limit:
                        break
                return deduped
        elapsed_ms = int((perf_counter() - started_at) * 1000)
        if had_success_response:
            logger.info(
                "Cognee search no_matches dataset=%s query=%s limit=%d elapsed_ms=%d",
                self._cfg.dataset,
                query[:80],
                limit,
                elapsed_ms,
            )
            return []
        logger.warning(
            "Cognee search failed dataset=%s query=%s limit=%d elapsed_ms=%d reason=%s",
            self._cfg.dataset,
            query[:80],
            limit,
            elapsed_ms,
            detail,
        )
        return []

    def save_conversation_turn(
        self,
        *,
        user_id: str,
        session_id: str,
        user_message: str,
        assistant_message: str,
        agent_name: str = "",
        channel_id: int | None = None,
        guild_id: int | None = None,
    ) -> bool:
        started_at = perf_counter()
        max_chars = self._cfg.max_turn_chars
        user_text = user_message.strip()[:max_chars]
        assistant_text = assistant_message.strip()[:max_chars]
        if not user_text and not assistant_text:
            return False
        timestamp = datetime.now(timezone.utc).isoformat()
        base_metadata = {
            "type": "conversation_turn",
            "user_id": user_id,
            "session_id": session_id,
            "agent_name": agent_name or "unknown",
            "channel_id": channel_id,
            "guild_id": guild_id,
            "timestamp": timestamp,
        }
        records: list[tuple[str, dict[str, Any]]] = []
        turn_summary = (
            f"timestamp: {timestamp}\n"
            f"user_id: {user_id}\n"
            f"session_id: {session_id}\n"
            f"agent: {agent_name or 'unknown'}\n"
            f"channel_id: {channel_id if channel_id is not None else 'unknown'}\n"
            f"guild_id: {guild_id if guild_id is not None else 'dm'}\n\n"
            f"USER (summary):\n{user_text[:700]}\n\n"
            f"ASSISTANT (summary):\n{assistant_text[:900]}\n"
        )
        records.append((turn_summary, dict(base_metadata)))

        for role, role_text in (("user", user_text), ("assistant", assistant_text)):
            if not role_text:
                continue
            chunks = _chunk_text(role_text, chunk_size=1400, overlap=120)
            for idx, chunk in enumerate(chunks, 1):
                memory_text = (
                    f"timestamp: {timestamp}\n"
                    f"user_id: {user_id}\n"
                    f"session_id: {session_id}\n"
                    f"agent: {agent_name or 'unknown'}\n"
                    f"role: {role}\n"
                    f"chunk: {idx}/{len(chunks)}\n\n"
                    f"{chunk}\n"
                )
                metadata = dict(base_metadata)
                metadata.update(
                    {
                        "type": "conversation_chunk",
                        "role": role,
                        "chunk_index": idx,
                        "chunks_total": len(chunks),
                    }
                )
                records.append((memory_text, metadata))

        successful_writes = 0
        for memory_text, metadata in records:
            if self.add_memory(memory_text, metadata=metadata):
                successful_writes += 1
        ok = successful_writes > 0
        elapsed_ms = int((perf_counter() - started_at) * 1000)
        logger.info(
            "Cognee save_conversation_turn status=%s dataset=%s user_id=%s session_id=%s writes=%d/%d elapsed_ms=%d",
            "ok" if ok else "failed",
            self._cfg.dataset,
            user_id,
            session_id,
            successful_writes,
            len(records),
            elapsed_ms,
        )
        return ok

    def _is_recent_duplicate(self, content_hash: str) -> bool:
        now = perf_counter()
        last_seen = self._recent_content_hashes.get(content_hash)
        if last_seen is None:
            return False
        return now - last_seen < self._dedup_ttl_seconds

    def _remember_hash(self, content_hash: str) -> None:
        now = perf_counter()
        self._recent_content_hashes[content_hash] = now
        if len(self._recent_content_hashes) <= self._dedup_max_items:
            return
        cutoff = now - self._dedup_ttl_seconds
        stale_keys = [key for key, seen_at in self._recent_content_hashes.items() if seen_at < cutoff]
        for key in stale_keys:
            self._recent_content_hashes.pop(key, None)
        if len(self._recent_content_hashes) <= self._dedup_max_items:
            return
        # If still over limit, drop the oldest entries.
        ordered = sorted(self._recent_content_hashes.items(), key=lambda kv: kv[1])
        for key, _ in ordered[: len(self._recent_content_hashes) - self._dedup_max_items]:
            self._recent_content_hashes.pop(key, None)


def _json_or_empty(response: requests.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return {}


def _extract_text_results(payload: Any) -> list[str]:
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        for key in ("results", "items", "data", "matches"):
            if isinstance(payload.get(key), list):
                items = payload[key]
                break
        else:
            items = [payload]
    else:
        return []

    texts: list[str] = []
    seen: set[str] = set()
    for item in items:
        for text in _extract_text_candidates(item):
            normalized = " ".join(text.split())
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            texts.append(normalized)
    return texts


def _extract_text_candidates(value: Any, *, depth: int = 0) -> list[str]:
    if depth > 3:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, list):
        results: list[str] = []
        for item in value:
            results.extend(_extract_text_candidates(item, depth=depth + 1))
        return results
    if not isinstance(value, dict):
        text = str(value).strip()
        return [text] if text else []

    results: list[str] = []
    preferred_keys = (
        "text",
        "content",
        "memory",
        "document",
        "value",
        "context",
        "snippet",
        "chunk",
        "message",
        "answer",
    )
    for key in preferred_keys:
        if key in value:
            results.extend(_extract_text_candidates(value[key], depth=depth + 1))
    if results:
        return results
    for key, nested in value.items():
        if key.lower() == "metadata":
            continue
        results.extend(_extract_text_candidates(nested, depth=depth + 1))
    return results


def _chunk_text(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    if chunk_size <= 0:
        return [cleaned]
    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        overlap = max(chunk_size // 5, 0)
    chunks: list[str] = []
    step = max(chunk_size - overlap, 1)
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(cleaned):
            break
        start += step
    return chunks
