from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from bitdoze_bot.task_store import TaskBoardStore


class TaskBoardStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tasks_dir = Path(self.tmpdir.name) / "workspace" / "tasks"
        self.store = TaskBoardStore(tasks_dir=self.tasks_dir)
        self.store.ensure_workspace()

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_create_and_list_tasks(self) -> None:
        created = self.store.create_task(
            task_id="task-100",
            title="Create architecture plan",
            owner="architect",
            actor="architect",
        )
        self.assertEqual(created["status"], "todo")
        listed = self.store.list_tasks(owner="architect")
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["id"], "task-100")

    def test_blocked_status_requires_reason(self) -> None:
        self.store.create_task(
            task_id="task-200",
            title="Implement endpoint",
            owner="software-engineer",
            actor="software-engineer",
        )
        self.store.update_status(
            task_id="task-200",
            status="in_progress",
            actor="software-engineer",
        )
        with self.assertRaises(ValueError):
            self.store.update_status(
                task_id="task-200",
                status="blocked",
                actor="software-engineer",
            )

    def test_dependency_validation(self) -> None:
        with self.assertRaises(ValueError):
            self.store.create_task(
                task_id="task-300",
                title="Depends on missing task",
                owner="architect",
                actor="architect",
                depends_on=["task-missing"],
            )

    def test_role_policy_for_engineer(self) -> None:
        self.store.create_task(
            task_id="task-400",
            title="Plan architecture",
            owner="architect",
            actor="architect",
        )
        with self.assertRaises(ValueError):
            self.store.add_note(
                task_id="task-400",
                note="Attempted unauthorized edit",
                actor="software-engineer",
            )

    def test_migrates_v1_board_to_v2(self) -> None:
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        board_path = self.tasks_dir / "board.yaml"
        board_path.write_text(
            "version: 1\n"
            "updated_at: null\n"
            "tasks:\n"
            "  - id: task-500\n"
            "    title: Legacy task\n"
            "    status: todo\n"
            "    owner: architect\n",
            encoding="utf-8",
        )
        self.store.ensure_workspace()
        loaded = self.store.get_task("task-500")
        self.assertIn("blocked_reason", loaded)
        self.assertEqual(loaded["blocked_reason"], "")
        raw = board_path.read_text(encoding="utf-8")
        parsed = yaml.safe_load(raw.split("\n", 1)[1])
        self.assertEqual(parsed["version"], 2)

    def test_migrates_non_string_notes_items(self) -> None:
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        board_path = self.tasks_dir / "board.yaml"
        board_path.write_text(
            "version: 2\n"
            "updated_at: null\n"
            "tasks:\n"
            "  - id: task-501\n"
            "    title: Mixed notes\n"
            "    status: todo\n"
            "    owner: architect\n"
            "    notes:\n"
            "      - message: note with colon\n",
            encoding="utf-8",
        )
        self.store.ensure_workspace()
        loaded = self.store.get_task("task-501")
        self.assertEqual(loaded["notes"], ["message: note with colon"])

    def test_writes_audit_events(self) -> None:
        self.store.create_task(
            task_id="task-600",
            title="Audit test",
            owner="architect",
            actor="architect",
        )
        events = (self.tasks_dir / "events.log").read_text(encoding="utf-8").strip().splitlines()
        self.assertGreaterEqual(len(events), 1)

    def test_duplicate_task_id_is_auto_incremented(self) -> None:
        first = self.store.create_task(
            task_id="task-001",
            title="First",
            owner="architect",
            actor="architect",
        )
        second = self.store.create_task(
            task_id="task-001",
            title="Second",
            owner="architect",
            actor="architect",
        )
        self.assertEqual(first["id"], "task-001")
        self.assertEqual(second["id"], "task-002")


if __name__ == "__main__":
    unittest.main()
