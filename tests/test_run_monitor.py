from __future__ import annotations

import json
import time
from pathlib import Path

from bitdoze_bot.run_monitor import RunMonitor


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_run_monitor_writes_telemetry(tmp_path: Path) -> None:
    telemetry = tmp_path / "run-telemetry.jsonl"
    monitor = RunMonitor(enabled=True, telemetry_path=str(telemetry))

    token = monitor.start(
        run_kind="discord",
        target_name="main",
        user_id="u1",
        session_id="s1",
        timeout_seconds=60,
    )
    monitor.finish(token, status="completed", run_id="r1", model="m1")

    events = _read_jsonl(telemetry)
    assert len(events) == 1
    assert events[0]["status"] == "completed"
    assert events[0]["run_kind"] == "discord"
    assert events[0]["target_name"] == "main"
    assert events[0]["run_id"] == "r1"


def test_run_monitor_stale_runs_report_once(tmp_path: Path) -> None:
    monitor = RunMonitor(enabled=False, telemetry_path=str(tmp_path / "unused.jsonl"))
    monitor.start(
        run_kind="discord",
        target_name="main",
        user_id="u1",
        session_id="s1",
        timeout_seconds=60,
    )
    time.sleep(1.1)
    first = monitor.stale_runs(threshold_seconds=1, max_items=5)
    assert len(first) == 1
    assert first[0].run_kind == "discord"
    assert first[0].target_name == "main"

    second = monitor.stale_runs(threshold_seconds=1, max_items=5)
    assert second == []


def test_run_monitor_persists_token_fields_from_metrics(tmp_path: Path) -> None:
    telemetry = tmp_path / "run-telemetry.jsonl"
    monitor = RunMonitor(enabled=True, telemetry_path=str(telemetry))

    token = monitor.start(
        run_kind="discord",
        target_name="main",
        user_id="u1",
        session_id="s1",
        timeout_seconds=60,
    )
    monitor.finish(
        token,
        status="completed",
        run_id="r2",
        model="m2",
        metrics={"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
    )

    events = _read_jsonl(telemetry)
    assert len(events) == 1
    assert events[0]["input_tokens"] == 11
    assert events[0]["output_tokens"] == 7
    assert events[0]["total_tokens"] == 18
