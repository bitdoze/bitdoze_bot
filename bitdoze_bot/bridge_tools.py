from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from agno.tools import Toolkit
from agno.tools.file import FileTools
from agno.tools.github import GithubTools
from agno.tools.hackernews import HackerNewsTools
from agno.tools.shell import ShellTools
from agno.tools.websearch import WebSearchTools
from agno.tools.website import WebsiteTools
from agno.tools.youtube import YouTubeTools

from bitdoze_bot.collab_tools import CollaborationTools
from bitdoze_bot.task_tools import TaskBoardTools


class BridgeTools(Toolkit):
    """Compatibility bridge for common tool names models frequently call."""

    def __init__(
        self,
        workspace_dir: Path,
        tasks_dir: Path,
        default_actor: str = "system",
        websearch_tools: WebSearchTools | None = None,
        website_tools: WebsiteTools | None = None,
        hackernews_tools: HackerNewsTools | None = None,
        youtube_tools: YouTubeTools | None = None,
        github_tools: GithubTools | None = None,
        collaboration_tools: CollaborationTools | None = None,
        skill_dirs: list[Path] | None = None,
    ) -> None:
        self.file_tools = FileTools(base_dir=workspace_dir)
        self.shell_tools = ShellTools(base_dir=workspace_dir)
        self.task_tools = TaskBoardTools(tasks_dir=tasks_dir, default_actor=default_actor)
        self.websearch_tools = websearch_tools
        self.website_tools = website_tools
        self.hackernews_tools = hackernews_tools
        self.youtube_tools = youtube_tools
        self.github_tools = github_tools
        self.collaboration_tools = collaboration_tools
        self.skill_dirs = [Path(p).resolve() for p in (skill_dirs or [])]
        super().__init__(
            name="bridge_tools",
            tools=[
                self.get_available_tools,
                self.read_file,
                self.save_file,
                self.write_file,
                self.list_files,
                self.run_shell_command,
                self.create_task,
                self.update_status,
                self.add_note,
                self.list_tasks,
                self.search_web,
                self.web_search,
                self.search_news,
                self.read_url,
                self.read_website,
                self.get_top_hackernews_stories,
                self.get_hackernews_top_stories,
                self.get_youtube_video_data,
                self.get_youtube_video_captions,
                self.get_youtube_video_timestamps,
                self.list_repositories,
                self.search_repositories,
                self.get_file_content,
                self.create_issue,
                self.create_pull_request,
                self.write_spec,
                self.read_spec,
                self.list_specs,
                self.write_handoff,
                self.read_handoff,
                self.list_handoffs,
                self.get_task,
                self.file_read,
                self.get_skill_instructions,
                self.get_skill_script,
            ],
        )

    def get_available_tools(self) -> dict[str, Any]:
        return {
            "tools": [
                "read_file",
                "save_file",
                "write_file",
                "list_files",
                "run_shell_command",
                "create_task",
                "update_status",
                "add_note",
                "list_tasks",
                "search_web",
                "web_search",
                "search_news",
                "read_url",
                "read_website",
                "get_top_hackernews_stories",
                "get_youtube_video_data",
                "get_youtube_video_captions",
                "list_repositories",
                "search_repositories",
                "get_file_content",
                "create_issue",
                "create_pull_request",
                "write_spec",
                "read_spec",
                "list_specs",
                "write_handoff",
                "read_handoff",
                "list_handoffs",
                "get_task",
                "file_read",
                "get_skill_instructions",
                "get_skill_script",
            ]
        }

    def _require(self, tool: Any, name: str) -> Any:
        if tool is None:
            raise ValueError(f"{name} is unavailable for this agent.")
        return tool

    def read_file(self, file_name: str | None = None, path: str | None = None, encoding: str = "utf-8") -> str:
        target = (file_name or path or "").strip()
        if not target:
            raise ValueError("read_file requires file_name (or path).")
        return self.file_tools.read_file(file_name=target, encoding=encoding)

    def save_file(
        self,
        contents: str | None = None,
        content: str | None = None,
        file_name: str | None = None,
        path: str | None = None,
        overwrite: bool = True,
        encoding: str = "utf-8",
    ) -> str:
        target = (file_name or path or "").strip()
        if not target:
            raise ValueError("save_file requires file_name (or path).")
        body = contents if contents is not None else content
        if body is None:
            raise ValueError("save_file requires contents (or content).")
        return self.file_tools.save_file(
            contents=str(body),
            file_name=target,
            overwrite=overwrite,
            encoding=encoding,
        )

    def write_file(self, file_name: str | None = None, content: str | None = None, **kwargs: Any) -> str:
        return self.save_file(file_name=file_name, content=content, **kwargs)

    def list_files(self) -> str:
        return self.file_tools.list_files()

    def run_shell_command(self, args: list[str] | str, tail: int = 120) -> str:
        resolved_args = args
        if isinstance(args, str):
            resolved_args = shlex.split(args)
        if not isinstance(resolved_args, list) or not all(isinstance(item, str) for item in resolved_args):
            raise ValueError("run_shell_command args must be a list[str] or shell string.")
        return self.shell_tools.run_shell_command(args=resolved_args, tail=tail)

    def create_task(
        self,
        task_id: str,
        title: str,
        owner: str,
        depends_on: list[str] | str | None = None,
        notes: list[str] | str | None = None,
        actor: str | None = None,
    ) -> dict[str, Any]:
        return self.task_tools.create_task(
            task_id=task_id,
            title=title,
            owner=owner,
            depends_on=depends_on,
            notes=notes,
            actor=actor,
        )

    def update_status(
        self,
        task_id: str,
        status: str,
        blocked_reason: str | None = None,
        note: str | None = None,
        actor: str | None = None,
    ) -> dict[str, Any]:
        return self.task_tools.update_status(
            task_id=task_id,
            status=status,
            blocked_reason=blocked_reason,
            note=note,
            actor=actor,
        )

    def add_note(self, task_id: str, note: str, actor: str | None = None) -> dict[str, Any]:
        return self.task_tools.add_note(task_id=task_id, note=note, actor=actor)

    def list_tasks(self, status: str | None = None, owner: str | None = None) -> dict[str, Any]:
        return self.task_tools.list_tasks(status=status, owner=owner)

    def get_task(self, task_id: str) -> dict[str, Any]:
        return self.task_tools.get_task(task_id=task_id)

    def file_read(self, file_name: str | None = None, path: str | None = None, encoding: str = "utf-8") -> str:
        return self.read_file(file_name=file_name, path=path, encoding=encoding)

    def search_web(self, query: str, num_results: int = 5) -> str:
        tool = self._require(self.websearch_tools, "web_search")
        return tool.web_search(query=query, max_results=num_results)

    def web_search(self, query: str, num_results: int = 5) -> str:
        return self.search_web(query=query, num_results=num_results)

    def search_news(self, query: str, num_results: int = 5) -> str:
        tool = self._require(self.websearch_tools, "web_search")
        return tool.search_news(query=query, max_results=num_results)

    def read_url(self, url: str, max_links: int = 0, max_depth: int = 0) -> str:
        tool = self._require(self.website_tools, "website")
        return tool.read_url(url=url)

    def read_website(self, url: str, max_links: int = 0, max_depth: int = 0) -> str:
        return self.read_url(url=url, max_links=max_links, max_depth=max_depth)

    def get_top_hackernews_stories(self, num_stories: int = 10) -> str:
        tool = self._require(self.hackernews_tools, "hackernews")
        return tool.get_top_hackernews_stories(num_stories=num_stories)

    def get_hackernews_top_stories(self, num_stories: int = 10) -> str:
        return self.get_top_hackernews_stories(num_stories=num_stories)

    def get_youtube_video_data(self, url: str) -> str:
        tool = self._require(self.youtube_tools, "youtube")
        return tool.get_youtube_video_data(url=url)

    def get_youtube_video_captions(self, url: str, languages: list[str] | None = None) -> str:
        tool = self._require(self.youtube_tools, "youtube")
        return tool.get_youtube_video_captions(url=url)

    def get_youtube_video_timestamps(self, url: str, minutes: int = 5) -> str:
        tool = self._require(self.youtube_tools, "youtube")
        return tool.get_video_timestamps(url=url)

    def list_repositories(self, user: str | None = None) -> str:
        tool = self._require(self.github_tools, "github")
        return tool.list_repositories()

    def search_repositories(self, query: str) -> str:
        tool = self._require(self.github_tools, "github")
        return tool.search_repositories(query=query)

    def get_file_content(
        self,
        owner: str | None = None,
        repo: str | None = None,
        repo_name: str | None = None,
        path: str = "",
        branch: str = "main",
    ) -> str:
        tool = self._require(self.github_tools, "github")
        resolved_repo = repo_name or repo
        if not resolved_repo:
            raise ValueError("get_file_content requires repo or repo_name.")
        return tool.get_file_content(repo_name=resolved_repo, path=path, ref=branch)

    def create_issue(
        self,
        owner: str | None = None,
        repo: str | None = None,
        repo_name: str | None = None,
        title: str = "",
        body: str = "",
    ) -> str:
        tool = self._require(self.github_tools, "github")
        resolved_repo = repo_name or repo
        if not resolved_repo:
            raise ValueError("create_issue requires repo or repo_name.")
        return tool.create_issue(repo_name=resolved_repo, title=title, body=body)

    def create_pull_request(
        self,
        title: str,
        body: str,
        head: str,
        owner: str | None = None,
        repo: str | None = None,
        repo_name: str | None = None,
        base: str = "main",
    ) -> str:
        tool = self._require(self.github_tools, "github")
        resolved_repo = repo_name or repo
        if not resolved_repo:
            raise ValueError("create_pull_request requires repo or repo_name.")
        return tool.create_pull_request(
            repo_name=resolved_repo,
            title=title,
            body=body,
            head=head,
            base=base,
        )

    def write_spec(self, name: str, content: str, actor: str | None = None, overwrite: bool = True) -> str:
        tool = self._require(self.collaboration_tools, "collaboration")
        return tool.write_spec(name=name, content=content, actor=actor, overwrite=overwrite)

    def read_spec(self, name: str) -> str:
        tool = self._require(self.collaboration_tools, "collaboration")
        return tool.read_spec(name=name)

    def list_specs(self) -> dict[str, Any]:
        tool = self._require(self.collaboration_tools, "collaboration")
        return tool.list_specs()

    def write_handoff(self, task_id: str, content: str, actor: str | None = None, append: bool = True) -> str:
        tool = self._require(self.collaboration_tools, "collaboration")
        return tool.write_handoff(task_id=task_id, content=content, actor=actor, append=append)

    def read_handoff(self, task_id: str) -> str:
        tool = self._require(self.collaboration_tools, "collaboration")
        return tool.read_handoff(task_id=task_id)

    def list_handoffs(self) -> dict[str, Any]:
        tool = self._require(self.collaboration_tools, "collaboration")
        return tool.list_handoffs()

    def _resolve_skill_dir(self, skill_name: str) -> Path:
        normalized = skill_name.strip()
        if not normalized:
            raise ValueError("skill_name is required.")
        for base in self.skill_dirs:
            candidate = base / normalized
            if candidate.exists() and candidate.is_dir():
                return candidate
        raise ValueError(f"Skill not found: {skill_name}")

    def get_skill_instructions(self, skill_name: str) -> str:
        skill_dir = self._resolve_skill_dir(skill_name)
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            raise ValueError(f"Skill file not found: {skill_file}")
        return skill_file.read_text(encoding="utf-8")

    def get_skill_script(self, skill_name: str, script_path: str | None = None) -> str:
        skill_dir = self._resolve_skill_dir(skill_name)
        scripts_dir = skill_dir / "scripts"
        if script_path is None or not script_path.strip():
            if not scripts_dir.exists():
                return "No scripts directory found for this skill."
            files = sorted(str(p.relative_to(skill_dir)) for p in scripts_dir.rglob("*") if p.is_file())
            return "\n".join(files) if files else "No scripts found for this skill."
        target = (skill_dir / script_path).resolve()
        if skill_dir not in [target, *target.parents]:
            raise ValueError("script_path escapes skill directory.")
        if not target.exists() or not target.is_file():
            raise ValueError(f"Script not found: {script_path}")
        return target.read_text(encoding="utf-8")
