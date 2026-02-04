from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bitdoze_bot.collab_tools import CollaborationTools


class CollaborationToolsTests(unittest.TestCase):
    def test_write_and_read_spec_and_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = CollaborationTools(
                specs_dir=root / "workspace" / "team" / "specs",
                handoffs_dir=root / "workspace" / "team" / "handoffs",
            )
            tools.write_spec(name="feature-login", content="Spec content", actor="architect")
            spec = tools.read_spec("feature-login")
            self.assertIn("Spec content", spec)

            tools.write_handoff(task_id="task-123", content="Implemented", actor="software-engineer")
            handoff = tools.read_handoff("task-123")
            self.assertIn("Implemented", handoff)

    def test_save_file_alias_in_team_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = CollaborationTools(
                specs_dir=root / "workspace" / "team" / "specs",
                handoffs_dir=root / "workspace" / "team" / "handoffs",
            )
            tools.save_file(
                contents="hello",
                file_name="specs/alias-test.md",
            )
            text = tools.read_file(file_name="specs/alias-test.md")
            self.assertEqual(text, "hello")
            listed = tools.list_files()
            self.assertIn("specs", listed)


if __name__ == "__main__":
    unittest.main()
