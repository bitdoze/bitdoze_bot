from __future__ import annotations

from pathlib import Path
from typing import Any

from agno.tools import Toolkit
from agno.tools.file import FileTools


class FileCompatTools(Toolkit):
    """Compatibility aliases for common file function names used by LLMs."""

    def __init__(self, base_dir: Path) -> None:
        self.file_tools = FileTools(base_dir=base_dir)
        super().__init__(
            name="file_compat_tools",
            tools=[self.write_file, self.save_file, self.read_file],
        )

    def write_file(
        self,
        file_name: str | None = None,
        content: str | None = None,
        contents: str | None = None,
        path: str | None = None,
        overwrite: bool = True,
        encoding: str = "utf-8",
        **_: Any,
    ) -> str:
        target = (file_name or path or "").strip()
        if not target:
            raise ValueError("write_file requires file_name (or path).")
        body = contents if contents is not None else content
        if body is None:
            raise ValueError("write_file requires content (or contents).")
        return self.file_tools.save_file(
            contents=str(body),
            file_name=target,
            overwrite=overwrite,
            encoding=encoding,
        )

    def save_file(
        self,
        contents: str,
        file_name: str,
        overwrite: bool = True,
        encoding: str = "utf-8",
        **_: Any,
    ) -> str:
        return self.file_tools.save_file(
            contents=contents,
            file_name=file_name,
            overwrite=overwrite,
            encoding=encoding,
        )

    def read_file(
        self,
        file_name: str | None = None,
        path: str | None = None,
        encoding: str = "utf-8",
        **_: Any,
    ) -> str:
        target = (file_name or path or "").strip()
        if not target:
            raise ValueError("read_file requires file_name (or path).")
        return self.file_tools.read_file(file_name=target, encoding=encoding)
