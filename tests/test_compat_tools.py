from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bitdoze_bot.compat_tools import FileCompatTools


class FileCompatToolsTests(unittest.TestCase):
    def test_save_and_read_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = FileCompatTools(base_dir=Path(tmp))
            tools.save_file(contents="abc", file_name="x.txt")
            text = tools.read_file(file_name="x.txt")
            self.assertEqual(text, "abc")


if __name__ == "__main__":
    unittest.main()
