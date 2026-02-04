from __future__ import annotations

import shlex
import re
from pathlib import Path
from typing import Any

from agno.tools import Toolkit
from agno.tools.file import FileTools
from agno.tools.shell import ShellTools


def _sanitize_name(name: str) -> str:
    value = name.strip()
    if not value:
        raise ValueError("name must be a non-empty string.")
    value = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-")
    if not value:
        raise ValueError("name must contain at least one valid character.")
    return value


class CollaborationTools(Toolkit):
    """Shared team workspace for specs and handoffs."""

    def __init__(self, specs_dir: Path, handoffs_dir: Path) -> None:
        self.specs_dir = Path(specs_dir).resolve()
        self.handoffs_dir = Path(handoffs_dir).resolve()
        self.team_root = self.specs_dir.parent.resolve()
        self.file_tools = FileTools(base_dir=self.team_root)
        self.shell_tools = ShellTools(base_dir=self.team_root)
        self.specs_dir.mkdir(parents=True, exist_ok=True)
        self.handoffs_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(
            name="collaboration_tools",
            tools=[
                self.write_spec,
                self.read_spec,
                self.list_specs,
                self.write_handoff,
                self.read_handoff,
                self.list_handoffs,
                self.save_file,
                self.read_file,
                self.list_files,
                self.run_shell_command,
            ],
        )

    def _safe_team_path(self, raw_path: str) -> Path:
        candidate = (self.team_root / raw_path).resolve()
        if self.team_root not in [candidate, *candidate.parents]:
            raise ValueError("Path escapes shared team workspace.")
        return candidate

    def write_spec(
        self,
        name: str,
        content: str,
        actor: str | None = None,
        overwrite: bool = True,
    ) -> str:
        spec_name = _sanitize_name(name)
        if not spec_name.endswith(".md"):
            spec_name = f"{spec_name}.md"
        path = self.specs_dir / spec_name
        if path.exists() and not overwrite:
            raise ValueError(f"Spec already exists: {spec_name}")
        text = content
        if actor and actor.strip():
            text = f"<!-- author: {actor.strip()} -->\n\n{content}"
        path.write_text(text, encoding="utf-8")
        return f"Saved spec: {path}"

    def read_spec(self, name: str) -> str:
        spec_name = _sanitize_name(name)
        if not spec_name.endswith(".md"):
            spec_name = f"{spec_name}.md"
        path = self.specs_dir / spec_name
        if not path.exists():
            raise ValueError(f"Spec not found: {spec_name}")
        return path.read_text(encoding="utf-8")

    def list_specs(self) -> dict[str, Any]:
        files = sorted(p.name for p in self.specs_dir.glob("*.md") if p.is_file())
        return {"specs": files}

    def write_handoff(
        self,
        task_id: str,
        content: str,
        actor: str | None = None,
        append: bool = True,
    ) -> str:
        handoff_name = f"{_sanitize_name(task_id)}.md"
        path = self.handoffs_dir / handoff_name
        mode = "a" if append else "w"
        entry = content.strip()
        if actor and actor.strip():
            entry = f"[{actor.strip()}]\n{entry}"
        with path.open(mode, encoding="utf-8") as handle:
            if append and path.stat().st_size > 0:
                handle.write("\n\n")
            handle.write(entry)
            handle.write("\n")
        return f"Saved handoff: {path}"

    def read_handoff(self, task_id: str) -> str:
        handoff_name = f"{_sanitize_name(task_id)}.md"
        path = self.handoffs_dir / handoff_name
        if not path.exists():
            raise ValueError(f"Handoff not found for task: {task_id}")
        return path.read_text(encoding="utf-8")

    def list_handoffs(self) -> dict[str, Any]:
        files = sorted(p.name for p in self.handoffs_dir.glob("*.md") if p.is_file())
        return {"handoffs": files}

    def save_file(
        self,
        contents: str,
        file_name: str | None = None,
        path: str | None = None,
        overwrite: bool = True,
        encoding: str = "utf-8",
    ) -> str:
        target = (file_name or path or "").strip()
        if not target:
            raise ValueError("save_file requires file_name (or path).")
        out = self._safe_team_path(target)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and not overwrite:
            raise ValueError(f"File already exists: {target}")
        out.write_text(contents, encoding=encoding)
        return f"Saved file: {out}"

    def read_file(
        self,
        file_name: str | None = None,
        path: str | None = None,
        encoding: str = "utf-8",
    ) -> str:
        target = (file_name or path or "").strip()
        if not target:
            raise ValueError("read_file requires file_name (or path).")
        src = self._safe_team_path(target)
        if not src.exists():
            raise ValueError(f"File not found: {target}")
        return src.read_text(encoding=encoding)

    def list_files(self) -> str:
        return self.file_tools.list_files()

    def run_shell_command(self, args: list[str] | str, tail: int = 120) -> str:
        resolved_args = args
        if isinstance(args, str):
            resolved_args = shlex.split(args)
        if not isinstance(resolved_args, list) or not all(isinstance(item, str) for item in resolved_args):
            raise ValueError("run_shell_command args must be a list[str] or shell string.")
        return self.shell_tools.run_shell_command(args=resolved_args, tail=tail)
