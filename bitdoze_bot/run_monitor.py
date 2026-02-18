from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class ActiveRun:
    token: str
    run_kind: str
    target_name: str
    user_id: str
    session_id: str
    started_monotonic: float
    started_at_iso: str
    timeout_seconds: int | None
    stale_reported: bool = False


@dataclass(frozen=True)
class StaleRun:
    run_kind: str
    target_name: str
    user_id: str
    session_id: str
    age_seconds: int
    started_at: str


class RunMonitor:
    def __init__(self, enabled: bool = True, telemetry_path: str = "logs/run-telemetry.jsonl") -> None:
        self.enabled = enabled
        self.telemetry_path = Path(telemetry_path)
        self._lock = threading.Lock()
        self._active: dict[str, ActiveRun] = {}

    def start(
        self,
        run_kind: str,
        target_name: str,
        user_id: str,
        session_id: str,
        timeout_seconds: int | None = None,
    ) -> str:
        token = str(uuid4())
        current = ActiveRun(
            token=token,
            run_kind=run_kind,
            target_name=target_name,
            user_id=user_id,
            session_id=session_id,
            started_monotonic=monotonic(),
            started_at_iso=datetime.now(tz=timezone.utc).isoformat(),
            timeout_seconds=timeout_seconds,
        )
        with self._lock:
            self._active[token] = current
        return token

    def finish(
        self,
        token: str | None,
        *,
        status: str,
        run_id: str | None = None,
        model: str | None = None,
        error: str | None = None,
        metrics: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if token is None:
            return
        with self._lock:
            current = self._active.pop(token, None)
        if current is None:
            return

        elapsed_ms = int((monotonic() - current.started_monotonic) * 1000)
        event: dict[str, Any] = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "status": status,
            "run_kind": current.run_kind,
            "target_name": current.target_name,
            "user_id": current.user_id,
            "session_id": current.session_id,
            "started_at": current.started_at_iso,
            "elapsed_ms": elapsed_ms,
            "timeout_seconds": current.timeout_seconds,
            "run_id": run_id,
            "model": model,
            "error": error,
        }
        if metrics:
            event["metrics"] = metrics
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                if key in metrics and metrics[key] is not None:
                    event[key] = metrics[key]
        if extra:
            event["extra"] = extra
        self._append_event(event)

    def stale_runs(self, threshold_seconds: int, max_items: int = 5) -> list[StaleRun]:
        if threshold_seconds <= 0:
            return []
        now_monotonic = monotonic()
        stale: list[StaleRun] = []
        with self._lock:
            for active in self._active.values():
                age_seconds = int(now_monotonic - active.started_monotonic)
                if age_seconds < threshold_seconds:
                    continue
                if active.stale_reported:
                    continue
                active.stale_reported = True
                stale.append(
                    StaleRun(
                        run_kind=active.run_kind,
                        target_name=active.target_name,
                        user_id=active.user_id,
                        session_id=active.session_id,
                        age_seconds=age_seconds,
                        started_at=active.started_at_iso,
                    )
                )
                if len(stale) >= max_items:
                    break
        return stale

    def _append_event(self, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
            with self.telemetry_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True))
                handle.write("\n")
        except OSError:
            logger.exception("Failed writing run telemetry to %s", self.telemetry_path)
