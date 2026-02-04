from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agno.tools import Toolkit

from bitdoze_bot.task_store import TaskBoardStore


class TaskBoardTools(Toolkit):
    def __init__(self, tasks_dir: Path, default_actor: str | None = None) -> None:
        self.store = TaskBoardStore(tasks_dir=tasks_dir)
        self.default_actor = (default_actor or "").strip()
        self.store.ensure_workspace()
        super().__init__(
            name="task_board_tools",
            tools=[
                self.list_tasks,
                self.get_task,
                self.create_task,
                self.assign_owner,
                self.update_status,
                self.add_note,
                self.set_dependencies,
            ],
        )

    def list_tasks(self, status: str | None = None, owner: str | None = None) -> dict[str, Any]:
        return {"tasks": self.store.list_tasks(status=status, owner=owner)}

    def get_task(self, task_id: str) -> dict[str, Any]:
        return {"task": self.store.get_task(task_id=task_id)}

    def _resolve_actor(self, actor: str | None) -> str:
        resolved = (actor or self.default_actor).strip()
        if not resolved:
            raise ValueError("actor is required for task operations.")
        return resolved

    def _coerce_string_list(self, value: list[str] | str | None, field_name: str) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            if not all(isinstance(item, str) for item in value):
                raise ValueError(f"{field_name} must contain only strings.")
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            if stripped.startswith("["):
                parsed = json.loads(stripped)
                if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
                    return parsed
                raise ValueError(f"{field_name} must be a JSON array of strings.")
            return [stripped]
        raise ValueError(f"{field_name} must be a list of strings.")

    def create_task(
        self,
        task_id: str,
        title: str,
        owner: str,
        actor: str | None = None,
        depends_on: list[str] | str | None = None,
        notes: list[str] | str | None = None,
    ) -> dict[str, Any]:
        task = self.store.create_task(
            task_id=task_id,
            title=title,
            owner=owner,
            actor=self._resolve_actor(actor),
            depends_on=self._coerce_string_list(depends_on, "depends_on"),
            notes=self._coerce_string_list(notes, "notes"),
        )
        return {"task": task}

    def assign_owner(self, task_id: str, owner: str, actor: str | None = None) -> dict[str, Any]:
        return {"task": self.store.assign_owner(task_id=task_id, owner=owner, actor=self._resolve_actor(actor))}

    def update_status(
        self,
        task_id: str,
        status: str,
        actor: str | None = None,
        blocked_reason: str | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        task = self.store.update_status(
            task_id=task_id,
            status=status,
            actor=self._resolve_actor(actor),
            blocked_reason=blocked_reason,
            note=note,
        )
        return {"task": task}

    def add_note(self, task_id: str, note: str, actor: str | None = None) -> dict[str, Any]:
        return {"task": self.store.add_note(task_id=task_id, note=note, actor=self._resolve_actor(actor))}

    def set_dependencies(
        self,
        task_id: str,
        depends_on: list[str] | str,
        actor: str | None = None,
    ) -> dict[str, Any]:
        return {
            "task": self.store.set_dependencies(
                task_id=task_id,
                depends_on=self._coerce_string_list(depends_on, "depends_on"),
                actor=self._resolve_actor(actor),
            )
        }
