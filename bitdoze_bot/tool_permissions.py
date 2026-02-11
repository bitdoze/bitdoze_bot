from __future__ import annotations

import contextlib
import contextvars
import functools
import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from bitdoze_bot.config import Config
from bitdoze_bot.utils import parse_bool


class ToolPermissionError(PermissionError):
    pass


@dataclass(frozen=True)
class ToolPermissionRule:
    effect: str
    channel_ids: tuple[int, ...] = ()
    role_ids: tuple[int, ...] = ()
    user_ids: tuple[int, ...] = ()
    guild_ids: tuple[int, ...] = ()
    agents: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolPermissionConfig:
    enabled: bool = False
    default_effect: str = "allow"
    rules: tuple[ToolPermissionRule, ...] = ()


@dataclass(frozen=True)
class ToolAuditConfig:
    enabled: bool = True
    path: Path = Path("logs/tool-audit.jsonl")
    include_arguments: bool = False
    redacted_keys: tuple[str, ...] = ("token", "secret", "password", "api_key", "authorization")


@dataclass(frozen=True)
class ToolRuntimeContext:
    run_kind: str
    user_id: str | None
    session_id: str | None
    discord_user_id: int | None
    channel_id: int | None
    guild_id: int | None
    role_ids: tuple[int, ...]
    agent_name: str | None


@dataclass(frozen=True)
class ToolPermissionDecision:
    allowed: bool
    reason: str


_CURRENT_CONTEXT: contextvars.ContextVar[ToolRuntimeContext] = contextvars.ContextVar(
    "tool_runtime_context",
    default=ToolRuntimeContext(
        run_kind="unknown",
        user_id=None,
        session_id=None,
        discord_user_id=None,
        channel_id=None,
        guild_id=None,
        role_ids=(),
        agent_name=None,
    ),
)


def _to_int_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list | tuple):
        return ()
    out: list[int] = []
    for item in value:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return tuple(out)


def _to_str_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return ()
    out: list[str] = []
    for item in value:
        text = str(item).strip().lower()
        if text:
            out.append(text)
    return tuple(out)


def load_tool_permission_config(config: Config) -> ToolPermissionConfig:
    raw = config.get("tool_permissions", default={})
    cfg = raw if isinstance(raw, dict) else {}

    rules_raw = cfg.get("rules", []) or []
    rules: list[ToolPermissionRule] = []
    for rule in rules_raw:
        if not isinstance(rule, dict):
            continue
        effect = str(rule.get("effect", "")).strip().lower()
        if effect not in {"allow", "deny"}:
            continue
        rules.append(
            ToolPermissionRule(
                effect=effect,
                channel_ids=_to_int_tuple(rule.get("channel_ids")),
                role_ids=_to_int_tuple(rule.get("role_ids")),
                user_ids=_to_int_tuple(rule.get("user_ids")),
                guild_ids=_to_int_tuple(rule.get("guild_ids")),
                agents=_to_str_tuple(rule.get("agents")),
                tools=_to_str_tuple(rule.get("tools")),
            )
        )

    default_effect = str(cfg.get("default_effect", "allow")).strip().lower()
    if default_effect not in {"allow", "deny"}:
        default_effect = "allow"

    return ToolPermissionConfig(
        enabled=parse_bool(cfg.get("enabled", False), False),
        default_effect=default_effect,
        rules=tuple(rules),
    )


def load_tool_audit_config(config: Config) -> ToolAuditConfig:
    raw = config.get("tool_permissions", "audit", default={})
    cfg = raw if isinstance(raw, dict) else {}
    redacted = _to_str_tuple(cfg.get("redacted_keys"))
    return ToolAuditConfig(
        enabled=parse_bool(cfg.get("enabled", True), True),
        path=config.resolve_path(cfg.get("path"), default="logs/tool-audit.jsonl"),
        include_arguments=parse_bool(cfg.get("include_arguments", False), False),
        redacted_keys=redacted
        or ("token", "secret", "password", "api_key", "authorization"),
    )


@contextlib.contextmanager
def tool_runtime_context(
    *,
    run_kind: str,
    user_id: str | None = None,
    session_id: str | None = None,
    discord_user_id: int | None = None,
    channel_id: int | None = None,
    guild_id: int | None = None,
    role_ids: list[int] | tuple[int, ...] | None = None,
    agent_name: str | None = None,
):
    context = ToolRuntimeContext(
        run_kind=run_kind,
        user_id=user_id,
        session_id=session_id,
        discord_user_id=discord_user_id,
        channel_id=channel_id,
        guild_id=guild_id,
        role_ids=tuple(role_ids or ()),
        agent_name=agent_name,
    )
    token = _CURRENT_CONTEXT.set(context)
    try:
        yield
    finally:
        _CURRENT_CONTEXT.reset(token)


def get_tool_runtime_context() -> ToolRuntimeContext:
    return _CURRENT_CONTEXT.get()


def _strip_workspace_prefix(value: str) -> str:
    normalized = value.strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    while normalized.startswith("workspace/"):
        normalized = normalized[len("workspace/") :]
    if normalized == "workspace":
        return "."
    return normalized


def _normalize_file_path_arg(raw_value: Any, tool_instance: Any) -> Any:
    if not isinstance(raw_value, str):
        return raw_value
    value = raw_value.strip()
    if not value:
        return value

    base_dir = getattr(tool_instance, "base_dir", None)
    if isinstance(base_dir, Path):
        candidate = Path(value).expanduser()
        if candidate.is_absolute():
            try:
                rel = candidate.resolve().relative_to(base_dir.resolve())
                value = rel.as_posix() or "."
            except ValueError:
                return raw_value

    return _strip_workspace_prefix(value)


def _resolve_github_sha(
    *,
    tool_instance: Any,
    repo_name: str,
    path: str,
    branch: str | None,
) -> str | None:
    github_client = getattr(tool_instance, "g", None)
    if github_client is None:
        return None
    try:
        repo = github_client.get_repo(repo_name)
        file_obj = repo.get_contents(path, ref=branch) if branch else repo.get_contents(path)
        if isinstance(file_obj, list):
            return None
        sha = getattr(file_obj, "sha", None)
        if isinstance(sha, str):
            clean_sha = sha.strip()
            if clean_sha:
                return clean_sha
    except Exception:  # noqa: BLE001
        return None
    return None


def _normalize_tool_kwargs(
    *,
    tool_name: str,
    function_name: str,
    func: Callable[..., Any],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in kwargs.items():
        clean_key = str(key).strip()
        if not clean_key:
            continue
        normalized[clean_key] = value

    tool_instance = getattr(func, "__self__", None)
    tool_name_lower = tool_name.strip().lower()
    function_name_lower = function_name.strip().lower()

    if tool_name_lower == "file":
        if "contents" not in normalized and "content" in normalized:
            normalized["contents"] = normalized.pop("content")
        if "file_name" in normalized:
            normalized["file_name"] = _normalize_file_path_arg(
                normalized["file_name"], tool_instance
            )
        if "directory" in normalized:
            normalized["directory"] = _normalize_file_path_arg(
                normalized["directory"], tool_instance
            )
        if "pattern" in normalized and isinstance(normalized["pattern"], str):
            normalized["pattern"] = _strip_workspace_prefix(normalized["pattern"])

    if tool_name_lower == "github":
        if "repo_name" not in normalized and "repo" in normalized:
            normalized["repo_name"] = normalized.pop("repo")
        if "path" not in normalized and "file_name" in normalized:
            normalized["path"] = normalized.pop("file_name")
        if "content" not in normalized and "contents" in normalized:
            normalized["content"] = normalized.pop("contents")
        if function_name_lower == "update_file":
            message = str(normalized.get("message", "")).strip()
            if not message:
                file_path = str(normalized.get("path", "")).strip() or "file"
                normalized["message"] = f"Update {file_path} via bitdoze-bot"
            sha = str(normalized.get("sha", "")).strip()
            repo_name = str(normalized.get("repo_name", "")).strip()
            path = str(normalized.get("path", "")).strip()
            branch = str(normalized.get("branch", "")).strip() or None
            if not sha and repo_name and path:
                resolved_sha = _resolve_github_sha(
                    tool_instance=tool_instance,
                    repo_name=repo_name,
                    path=path,
                    branch=branch,
                )
                if resolved_sha:
                    normalized["sha"] = resolved_sha

    return normalized


class ToolAuditLogger:
    def __init__(self, config: ToolAuditConfig):
        self._config = config
        self._lock = threading.Lock()

    def _sanitize(self, value: Any) -> Any:
        if isinstance(value, dict):
            sanitized: dict[str, Any] = {}
            for key, item in value.items():
                if str(key).strip().lower() in self._config.redacted_keys:
                    sanitized[str(key)] = "***REDACTED***"
                else:
                    sanitized[str(key)] = self._sanitize(item)
            return sanitized
        if isinstance(value, list | tuple):
            return [self._sanitize(item) for item in value]
        if value is None or isinstance(value, bool | int | float | str):
            return value
        text = repr(value)
        if len(text) > 500:
            return text[:497] + "..."
        return text

    def write(
        self,
        *,
        outcome: str,
        tool: str,
        function: str,
        decision_reason: str,
        error: str | None = None,
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        if not self._config.enabled:
            return

        ctx = get_tool_runtime_context()
        event: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "outcome": outcome,
            "reason": decision_reason,
            "agent": ctx.agent_name,
            "tool": tool,
            "function": function,
            "run_kind": ctx.run_kind,
            "user_id": ctx.user_id,
            "session_id": ctx.session_id,
            "discord_user_id": ctx.discord_user_id,
            "guild_id": ctx.guild_id,
            "channel_id": ctx.channel_id,
            "role_ids": list(ctx.role_ids),
        }
        if error is not None:
            event["error"] = error
        if self._config.include_arguments:
            event["args"] = self._sanitize(list(args or ()))
            event["kwargs"] = self._sanitize(kwargs or {})

        payload = json.dumps(event, ensure_ascii=True)
        with self._lock:
            self._config.path.parent.mkdir(parents=True, exist_ok=True)
            with self._config.path.open("a", encoding="utf-8") as handle:
                handle.write(payload + "\n")


class ToolPermissionManager:
    def __init__(self, permission_cfg: ToolPermissionConfig, audit_cfg: ToolAuditConfig):
        self._permission_cfg = permission_cfg
        self._audit = ToolAuditLogger(audit_cfg)

    @classmethod
    def from_config(cls, config: Config) -> ToolPermissionManager:
        return cls(load_tool_permission_config(config), load_tool_audit_config(config))

    def _rule_matches(self, rule: ToolPermissionRule, agent_name: str, tool_name: str) -> bool:
        ctx = get_tool_runtime_context()
        agent = agent_name.strip().lower()
        tool = tool_name.strip().lower()

        if rule.channel_ids and (ctx.channel_id is None or ctx.channel_id not in rule.channel_ids):
            return False
        if rule.guild_ids and (ctx.guild_id is None or ctx.guild_id not in rule.guild_ids):
            return False
        if rule.user_ids and (ctx.discord_user_id is None or ctx.discord_user_id not in rule.user_ids):
            return False
        if rule.role_ids and not any(role_id in set(ctx.role_ids) for role_id in rule.role_ids):
            return False
        if rule.agents and agent not in rule.agents:
            return False
        if rule.tools and tool not in rule.tools:
            return False
        return True

    def decide(self, agent_name: str, tool_name: str) -> ToolPermissionDecision:
        if not self._permission_cfg.enabled:
            return ToolPermissionDecision(allowed=True, reason="permissions_disabled")

        matched = [
            rule
            for rule in self._permission_cfg.rules
            if self._rule_matches(rule, agent_name=agent_name, tool_name=tool_name)
        ]
        if any(rule.effect == "deny" for rule in matched):
            return ToolPermissionDecision(allowed=False, reason="matched_deny_rule")
        if any(rule.effect == "allow" for rule in matched):
            return ToolPermissionDecision(allowed=True, reason="matched_allow_rule")
        if self._permission_cfg.default_effect == "deny":
            return ToolPermissionDecision(allowed=False, reason="default_deny")
        return ToolPermissionDecision(allowed=True, reason="default_allow")

    def wrap_callable(
        self,
        *,
        agent_name_getter: Callable[[], str],
        tool_name: str,
        function_name: str,
        func: Callable[..., Any],
    ) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            agent_name = agent_name_getter()
            normalized_kwargs = _normalize_tool_kwargs(
                tool_name=tool_name,
                function_name=function_name,
                func=func,
                kwargs=kwargs,
            )
            decision = self.decide(agent_name=agent_name, tool_name=tool_name)
            if not decision.allowed:
                self._audit.write(
                    outcome="blocked",
                    tool=tool_name,
                    function=function_name,
                    decision_reason=decision.reason,
                    args=args,
                    kwargs=normalized_kwargs,
                )
                raise ToolPermissionError(
                    f"I can't use the '{tool_name}' tool in this context."
                )

            self._audit.write(
                outcome="allowed",
                tool=tool_name,
                function=function_name,
                decision_reason=decision.reason,
                args=args,
                kwargs=normalized_kwargs,
            )
            try:
                result = func(*args, **normalized_kwargs)
            except Exception as exc:  # noqa: BLE001
                self._audit.write(
                    outcome="failed",
                    tool=tool_name,
                    function=function_name,
                    decision_reason=decision.reason,
                    error=f"{exc.__class__.__name__}: {exc}",
                    args=args,
                    kwargs=normalized_kwargs,
                )
                raise

            self._audit.write(
                outcome="executed",
                tool=tool_name,
                function=function_name,
                decision_reason=decision.reason,
                args=args,
                kwargs=normalized_kwargs,
            )
            return result

        return wrapped

    def wrap_tool(self, tool: Any, tool_name: str, agent_name_getter: Callable[[], str]) -> Any:
        if getattr(tool, "_bitdoze_policy_wrapped", False):
            return tool

        get_functions = getattr(tool, "get_functions", None)
        if callable(get_functions):
            functions = get_functions() or {}
            if isinstance(functions, dict):
                for function_name in functions.keys():
                    target = getattr(tool, function_name, None)
                    if callable(target):
                        setattr(
                            tool,
                            function_name,
                            self.wrap_callable(
                                agent_name_getter=agent_name_getter,
                                tool_name=tool_name,
                                function_name=function_name,
                                func=target,
                            ),
                        )

        setattr(tool, "_bitdoze_policy_wrapped", True)
        setattr(tool, "_bitdoze_tool_name", tool_name)
        return tool
