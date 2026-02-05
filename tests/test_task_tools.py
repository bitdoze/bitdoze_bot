from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bitdoze_bot.task_tools import TaskBoardTools


class TaskBoardToolsTests(unittest.TestCase):
    def test_default_actor_is_used(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = TaskBoardTools(tasks_dir=Path(tmp) / "workspace" / "tasks", default_actor="architect")
            result = tools.create_task(
                task_id="task-777",
                title="Created by default actor",
                owner="architect",
            )
            self.assertEqual(result["task"]["id"], "task-777")

    def test_accepts_stringified_list_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = TaskBoardTools(tasks_dir=Path(tmp) / "workspace" / "tasks", default_actor="architect")
            tools.create_task(
                task_id="task-778",
                title="String list args",
                owner="architect",
                depends_on="[]",
                notes='["created via string"]',
            )
            task = tools.get_task("task-778")["task"]
            self.assertEqual(task["notes"], ["created via string"])

    def test_duplicate_id_is_renamed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = TaskBoardTools(tasks_dir=Path(tmp) / "workspace" / "tasks", default_actor="architect")
            first = tools.create_task(task_id="task-001", title="One", owner="architect")
            second = tools.create_task(task_id="task-001", title="Two", owner="architect")
            self.assertEqual(first["task"]["id"], "task-001")
            self.assertEqual(second["task"]["id"], "task-002")


if __name__ == "__main__":
    unittest.main()
