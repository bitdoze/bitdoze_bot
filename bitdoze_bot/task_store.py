from __future__ import annotations

import json
import os
import re
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import fcntl
import yaml

CURRENT_TASKS_SCHEMA_VERSION = 2
TASK_STATUSES = {"todo", "in_progress", "blocked", "done"}
TASK_TRANSITIONS: dict[str, set[str]] = {
    "todo": {"todo", "in_progress", "blocked"},
    "in_progress": {"in_progress", "blocked", "done", "todo"},
    "blocked": {"blocked", "in_progress", "todo"},
    "done": {"done", "in_progress"},
}
POLICY_ROLES = {"architect", "software-engineer"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class TaskBoardStore:
    def __init__(self, tasks_dir: Path) -> None:
        self.tasks_dir = Path(tasks_dir).resolve()
        self.board_path = self.tasks_dir / "board.yaml"
        self.schema_path = self.tasks_dir / "board.v2.schema.json"
        self.readme_path = self.tasks_dir / "README.md"
        self.notes_dir = self.tasks_dir / "notes"
        self.events_path = self.tasks_dir / "events.log"
        self.lock_path = self.tasks_dir / ".board.lock"

    def ensure_workspace(self) -> None:
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.notes_dir.mkdir(parents=True, exist_ok=True)
        if not self.board_path.exists():
            self.board_path.write_text(
                "# yaml-language-server: $schema=./board.v2.schema.json\n"
                "version: 2\n"
                "updated_at: null\n"
                "tasks: []\n",
                encoding="utf-8",
            )
        if not self.schema_path.exists():
            self.schema_path.write_text(
                "{\n"
                '  "$schema": "https://json-schema.org/draft/2020-12/schema",\n'
                '  "title": "Bitdoze Tasks Board",\n'
                '  "type": "object",\n'
                '  "required": ["version", "updated_at", "tasks"],\n'
                '  "additionalProperties": false,\n'
                '  "properties": {\n'
                '    "version": {"type": "integer"},\n'
                '    "updated_at": {"type": ["string", "null"]},\n'
                '    "tasks": {\n'
                '      "type": "array",\n'
                '      "items": {\n'
                '        "type": "object",\n'
                '        "required": ["id", "title", "status", "owner"],\n'
                '        "additionalProperties": false,\n'
                '        "properties": {\n'
                '          "id": {"type": "string", "minLength": 1},\n'
                '          "title": {"type": "string", "minLength": 1},\n'
                '          "status": {"type": "string", "enum": ["todo", "in_progress", "blocked", "done"]},\n'
                '          "owner": {"type": "string", "minLength": 1},\n'
                '          "depends_on": {"type": "array", "items": {"type": "string"}, "default": []},\n'
                '          "updated_at": {"type": ["string", "null"]},\n'
                '          "notes": {"type": "array", "items": {"type": "string"}, "default": []},\n'
                '          "blocked_reason": {"type": "string", "default": ""}\n'
                "        }\n"
                "      }\n"
                "    }\n"
                "  }\n"
                "}\n",
                encoding="utf-8",
            )
        if not self.readme_path.exists():
            self.readme_path.write_text(
                "# Team Tasks\n\n"
                "Shared task board for all agents.\n\n"
                "## Lifecycle\n"
                "- `todo` -> `in_progress` -> `done`\n"
                "- Any open task can become `blocked` with `blocked_reason`\n\n"
                "## Team Policy\n"
                "- `architect`: planning, decomposition, assignment, review\n"
                "- `software-engineer`: implementation and testing execution\n"
                "- `software-engineer` can only mutate tasks owned by `software-engineer`\n"
                "- Every board write appends an event to `events.log`\n",
                encoding="utf-8",
            )
        self.events_path.touch(exist_ok=True)
        with self._locked():
            board = self._read_board_unlocked()
            self._write_board_unlocked(board)

    def list_tasks(self, *, status: str | None = None, owner: str | None = None) -> list[dict[str, Any]]:
        with self._locked():
            board = self._read_board_unlocked()
            tasks = [deepcopy(task) for task in board["tasks"]]
        if status:
            tasks = [task for task in tasks if task["status"] == status]
        if owner:
            tasks = [task for task in tasks if task["owner"] == owner]
        return tasks

    def get_task(self, task_id: str) -> dict[str, Any]:
        with self._locked():
            board = self._read_board_unlocked()
            task = self._find_task(board, task_id)
            return deepcopy(task)

    def create_task(
        self,
        *,
        task_id: str,
        title: str,
        owner: str,
        actor: str,
        depends_on: list[str] | None = None,
        notes: list[str] | None = None,
    ) -> dict[str, Any]:
        with self._locked():
            board = self._read_board_unlocked()
            resolved_task_id = self._allocate_task_id_unlocked(board=board, requested_id=task_id.strip())
            self._authorize_create(actor=actor, owner=owner)
            task = {
                "id": resolved_task_id,
                "title": title.strip(),
                "status": "todo",
                "owner": owner.strip(),
                "depends_on": list(depends_on or []),
                "updated_at": utc_now_iso(),
                "notes": list(notes or []),
                "blocked_reason": "",
            }
            board["tasks"].append(task)
            self._write_board_unlocked(board)
            self._append_event_unlocked(
                action="create_task",
                actor=actor,
                task_id=resolved_task_id,
                details={"owner": owner, "title": title, "requested_id": task_id.strip()},
            )
            return deepcopy(task)

    def assign_owner(self, *, task_id: str, owner: str, actor: str) -> dict[str, Any]:
        with self._locked():
            board = self._read_board_unlocked()
            task = self._find_task(board, task_id)
            self._authorize_mutation(actor=actor, task=task, action="assign_owner")
            task["owner"] = owner.strip()
            task["updated_at"] = utc_now_iso()
            self._write_board_unlocked(board)
            self._append_event_unlocked(
                action="assign_owner",
                actor=actor,
                task_id=task_id,
                details={"owner": owner},
            )
            return deepcopy(task)

    def update_status(
        self,
        *,
        task_id: str,
        status: str,
        actor: str,
        blocked_reason: str | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        with self._locked():
            board = self._read_board_unlocked()
            task = self._find_task(board, task_id)
            self._authorize_mutation(actor=actor, task=task, action="update_status")
            normalized_status = status.strip()
            if normalized_status not in TASK_STATUSES:
                raise ValueError(f"Invalid task status: {normalized_status}")
            if normalized_status not in TASK_TRANSITIONS[task["status"]]:
                raise ValueError(
                    f"Invalid transition {task['status']} -> {normalized_status} for task '{task_id}'."
                )
            if normalized_status == "blocked":
                if not blocked_reason or not blocked_reason.strip():
                    raise ValueError("blocked_reason is required when status is 'blocked'.")
                task["blocked_reason"] = blocked_reason.strip()
            else:
                task["blocked_reason"] = ""
            task["status"] = normalized_status
            task["updated_at"] = utc_now_iso()
            if note and note.strip():
                task["notes"].append(f"[{utc_now_iso()}] {actor}: {note.strip()}")
            self._write_board_unlocked(board)
            self._append_event_unlocked(
                action="update_status",
                actor=actor,
                task_id=task_id,
                details={"status": normalized_status, "blocked_reason": task["blocked_reason"]},
            )
            return deepcopy(task)

    def add_note(self, *, task_id: str, note: str, actor: str) -> dict[str, Any]:
        with self._locked():
            board = self._read_board_unlocked()
            task = self._find_task(board, task_id)
            self._authorize_mutation(actor=actor, task=task, action="add_note")
            if not note.strip():
                raise ValueError("note must be a non-empty string.")
            task["notes"].append(f"[{utc_now_iso()}] {actor}: {note.strip()}")
            task["updated_at"] = utc_now_iso()
            self._write_board_unlocked(board)
            self._append_event_unlocked(
                action="add_note",
                actor=actor,
                task_id=task_id,
                details={"note": note.strip()},
            )
            return deepcopy(task)

    def set_dependencies(self, *, task_id: str, depends_on: list[str], actor: str) -> dict[str, Any]:
        with self._locked():
            board = self._read_board_unlocked()
            task = self._find_task(board, task_id)
            self._authorize_mutation(actor=actor, task=task, action="set_dependencies")
            task["depends_on"] = list(depends_on)
            task["updated_at"] = utc_now_iso()
            self._write_board_unlocked(board)
            self._append_event_unlocked(
                action="set_dependencies",
                actor=actor,
                task_id=task_id,
                details={"depends_on": depends_on},
            )
            return deepcopy(task)

    def _read_board_unlocked(self) -> dict[str, Any]:
        raw = self.board_path.read_text(encoding="utf-8")
        loaded = yaml.safe_load(raw) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Invalid tasks board at {self.board_path}: top-level must be a mapping.")
        migrated = self._migrate_board(loaded)
        self._validate_board(migrated)
        return migrated

    def _write_board_unlocked(self, board: dict[str, Any]) -> None:
        self._validate_board(board)
        board_to_save = deepcopy(board)
        board_to_save["updated_at"] = utc_now_iso()
        serialized = yaml.safe_dump(board_to_save, sort_keys=False, allow_unicode=False)
        with os.fdopen(
            os.open(self.board_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644),
            "w",
            encoding="utf-8",
        ) as handle:
            handle.write("# yaml-language-server: $schema=./board.v2.schema.json\n")
            handle.write(serialized)

    def _migrate_board(self, board: dict[str, Any]) -> dict[str, Any]:
        migrated = deepcopy(board)
        version = migrated.get("version", 1)
        if not isinstance(version, int):
            raise ValueError(f"Invalid tasks board at {self.board_path}: 'version' must be an integer.")
        if version > CURRENT_TASKS_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported board schema version {version}. "
                f"Current supported version is {CURRENT_TASKS_SCHEMA_VERSION}."
            )

        tasks = migrated.get("tasks")
        if tasks is None:
            tasks = []
        if not isinstance(tasks, list):
            raise ValueError(f"Invalid tasks board at {self.board_path}: 'tasks' must be a list.")
        normalized_tasks: list[dict[str, Any]] = []
        for task in tasks:
            if not isinstance(task, dict):
                raise ValueError(f"Invalid tasks board at {self.board_path}: each task must be a mapping.")
            raw_depends_on = task.get("depends_on", []) or []
            if isinstance(raw_depends_on, list):
                depends_on = [str(item) for item in raw_depends_on]
            else:
                depends_on = [str(raw_depends_on)]

            raw_notes = task.get("notes", []) or []
            note_lines: list[str] = []
            if isinstance(raw_notes, list):
                for item in raw_notes:
                    if isinstance(item, str):
                        note_lines.append(item)
                    elif isinstance(item, dict):
                        for key, value in item.items():
                            note_lines.append(f"{key}: {value}")
                    else:
                        note_lines.append(str(item))
            elif isinstance(raw_notes, str):
                note_lines = [raw_notes]
            else:
                note_lines = [str(raw_notes)]
            normalized_tasks.append(
                {
                    "id": task.get("id"),
                    "title": task.get("title"),
                    "status": task.get("status", "todo"),
                    "owner": task.get("owner"),
                    "depends_on": depends_on,
                    "updated_at": task.get("updated_at"),
                    "notes": note_lines,
                    "blocked_reason": str(task.get("blocked_reason") or ""),
                }
            )
        migrated["tasks"] = normalized_tasks
        migrated["updated_at"] = migrated.get("updated_at")

        if version < 2:
            migrated["version"] = 2
        return migrated

    def _validate_board(self, board: dict[str, Any]) -> None:
        version = board.get("version")
        if version != CURRENT_TASKS_SCHEMA_VERSION:
            raise ValueError(
                f"Invalid tasks board at {self.board_path}: expected version {CURRENT_TASKS_SCHEMA_VERSION}, got {version}."
            )

        updated_at = board.get("updated_at")
        if updated_at is not None and not isinstance(updated_at, str):
            raise ValueError(f"Invalid tasks board at {self.board_path}: 'updated_at' must be null or string.")

        tasks = board.get("tasks")
        if not isinstance(tasks, list):
            raise ValueError(f"Invalid tasks board at {self.board_path}: 'tasks' must be a list.")

        seen_ids: set[str] = set()
        for index, task in enumerate(tasks):
            prefix = f"Invalid tasks board at {self.board_path}: tasks[{index}]"
            if not isinstance(task, dict):
                raise ValueError(f"{prefix} must be a mapping.")
            for required in ("id", "title", "status", "owner"):
                value = task.get(required)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{prefix}.{required} must be a non-empty string.")

            task_id = str(task["id"])
            if task_id in seen_ids:
                raise ValueError(f"{prefix}.id '{task_id}' is duplicated.")
            seen_ids.add(task_id)

            status = str(task["status"])
            if status not in TASK_STATUSES:
                raise ValueError(f"{prefix}.status '{status}' is invalid.")

            blocked_reason = task.get("blocked_reason", "")
            if not isinstance(blocked_reason, str):
                raise ValueError(f"{prefix}.blocked_reason must be a string.")
            if status == "blocked" and not blocked_reason.strip():
                raise ValueError(f"{prefix}.blocked_reason is required when status is 'blocked'.")
            if status != "blocked" and blocked_reason.strip():
                raise ValueError(f"{prefix}.blocked_reason must be empty unless status is 'blocked'.")

            depends_on = task.get("depends_on", [])
            if not isinstance(depends_on, list) or not all(isinstance(item, str) for item in depends_on):
                raise ValueError(f"{prefix}.depends_on must be a list of strings.")

            notes = task.get("notes", [])
            if not isinstance(notes, list) or not all(isinstance(item, str) for item in notes):
                raise ValueError(f"{prefix}.notes must be a list of strings.")

            task_updated_at = task.get("updated_at")
            if task_updated_at is not None and not isinstance(task_updated_at, str):
                raise ValueError(f"{prefix}.updated_at must be null or string.")

        for index, task in enumerate(tasks):
            for dep_id in task.get("depends_on", []):
                if dep_id not in seen_ids:
                    raise ValueError(
                        f"Invalid tasks board at {self.board_path}: tasks[{index}].depends_on references "
                        f"unknown id '{dep_id}'."
                    )

    def _find_task(self, board: dict[str, Any], task_id: str) -> dict[str, Any]:
        for task in board["tasks"]:
            if task["id"] == task_id:
                return task
        raise ValueError(f"Task not found: {task_id}")

    def _allocate_task_id_unlocked(self, board: dict[str, Any], requested_id: str) -> str:
        existing_ids = {str(task.get("id", "")).strip() for task in board.get("tasks", []) if task.get("id")}
        if requested_id not in existing_ids:
            return requested_id

        numeric_match = re.match(r"^(.*?)(\d+)$", requested_id)
        if numeric_match:
            prefix = numeric_match.group(1)
            number_str = numeric_match.group(2)
            width = len(number_str)
            next_number = int(number_str) + 1
            while True:
                candidate = f"{prefix}{next_number:0{width}d}"
                if candidate not in existing_ids:
                    return candidate
                next_number += 1

        suffix = 2
        while True:
            candidate = f"{requested_id}-{suffix}"
            if candidate not in existing_ids:
                return candidate
            suffix += 1

    def _authorize_create(self, *, actor: str, owner: str) -> None:
        if actor == "software-engineer" and owner != "software-engineer":
            raise ValueError("software-engineer can only create tasks owned by software-engineer.")

    def _authorize_mutation(self, *, actor: str, task: dict[str, Any], action: str) -> None:
        if not actor:
            raise ValueError(f"actor is required for task action '{action}'.")
        if actor not in POLICY_ROLES:
            return
        if actor == "architect":
            return
        if actor == "software-engineer" and task.get("owner") != "software-engineer":
            raise ValueError("software-engineer can only modify tasks owned by software-engineer.")

    def _append_event_unlocked(
        self,
        *,
        action: str,
        actor: str,
        task_id: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        entry = {
            "timestamp": utc_now_iso(),
            "action": action,
            "actor": actor,
            "task_id": task_id,
            "details": details or {},
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        with open(self.lock_path, "a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
