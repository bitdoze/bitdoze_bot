from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bitdoze_bot.bridge_tools import BridgeTools


class BridgeToolsTests(unittest.TestCase):
    def test_common_aliases_exist_and_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = BridgeTools(
                workspace_dir=root / "workspace",
                tasks_dir=root / "workspace" / "tasks",
                default_actor="architect",
            )
            available = tools.get_available_tools()
            self.assertIn("run_shell_command", available["tools"])
            self.assertIn("create_task", available["tools"])
            self.assertIn("get_task", available["tools"])
            self.assertIn("get_skill_instructions", available["tools"])
            self.assertIn("file_read", available["tools"])

            tools.save_file(file_name="hello.txt", contents="hi")
            self.assertEqual(tools.read_file(file_name="hello.txt"), "hi")
            self.assertIn("hello.txt", tools.list_files())

            task = tools.create_task(task_id="task-999", title="Bridge task", owner="architect")
            self.assertEqual(task["task"]["id"], "task-999")
            fetched = tools.get_task(task_id="task-999")
            self.assertEqual(fetched["task"]["id"], "task-999")

    def test_shell_and_file_compatibility_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = BridgeTools(
                workspace_dir=root / "workspace",
                tasks_dir=root / "workspace" / "tasks",
                default_actor="architect",
            )
            tools.save_file(path="workspace/notes.txt", contents="hello")
            text = tools.read_file(path="workspace/notes.txt")
            self.assertEqual(text, "hello")

            listed = tools.list_files(path="workspace", pattern="*.txt")
            self.assertIn("notes.txt", listed)

            output = tools.run_shell_command(args=["ls -la"])
            self.assertIn("notes.txt", output)

            pwd = tools.run_shell_command(args="[pwd]")
            self.assertIn("workspace", pwd)


if __name__ == "__main__":
    unittest.main()
