from __future__ import annotations

from typing import Any

from agno.run.agent import Message
from agno.team import Team


def build_response_input(user_context: str | None, content: str) -> list[Message] | str:
    file_task_hint = (
        "For local file/code operations, use the 'file' tool functions "
        "(list_files, search_files, read_file, read_file_chunk, save_file, replace_file_chunk) "
        "instead of shell or github tools. File paths are relative to the workspace root, "
        "so use 'USER.md' (not 'workspace/USER.md'). Use save_file(contents=..., file_name=...)."
    )
    lower_content = content.lower()
    file_keywords = (
        "file",
        "folder",
        "directory",
        "read ",
        "open ",
        "edit ",
        "update ",
        "change ",
        "rewrite ",
        "search in",
        "find in",
        "codebase",
        ".py",
        ".js",
        ".ts",
        ".md",
        ".yaml",
        ".yml",
        ".json",
    )
    needs_file_hint = any(token in lower_content for token in file_keywords)

    if user_context:
        messages: list[Message] = [Message(role="system", content=user_context)]
        if needs_file_hint:
            messages.append(Message(role="system", content=file_task_hint))
        messages.append(Message(role="user", content=content))
        return messages
    if needs_file_hint:
        return [
            Message(role="system", content=file_task_hint),
            Message(role="user", content=content),
        ]
    return content


def is_team_target(target: Any) -> bool:
    return isinstance(target, Team)


def target_members(target: Any) -> list[str]:
    names: list[str] = []
    for member in getattr(target, "members", None) or []:
        names.append(_agent_name(member, "unknown"))
    return names


def _agent_name(agent: Any, fallback: str | None = None) -> str:
    if hasattr(agent, "name"):
        return str(getattr(agent, "name"))
    return fallback or "unknown"


def extract_metrics(metrics: Any) -> dict[str, Any]:
    if metrics is None:
        return {}
    if not isinstance(metrics, dict):
        nested_metrics = getattr(metrics, "metrics", None)
        if nested_metrics is not None and nested_metrics is not metrics:
            snapshot = extract_metrics(nested_metrics)
        else:
            snapshot = {}
    else:
        snapshot = {}

    events = getattr(metrics, "events", None)
    if isinstance(events, list):
        input_sum = 0
        output_sum = 0
        total_sum = 0
        seen = False
        for event in events:
            in_tok = getattr(event, "input_tokens", None)
            out_tok = getattr(event, "output_tokens", None)
            tot_tok = getattr(event, "total_tokens", None)
            if in_tok is None and out_tok is None and tot_tok is None:
                continue
            seen = True
            input_sum += int(in_tok or 0)
            output_sum += int(out_tok or 0)
            total_sum += int(tot_tok or 0)
        if seen:
            if input_sum > 0:
                snapshot["input_tokens"] = input_sum
            if output_sum > 0:
                snapshot["output_tokens"] = output_sum
            if total_sum > 0:
                snapshot["total_tokens"] = total_sum
            snapshot["token_source"] = "events"

    usage = getattr(metrics, "usage", None)
    if isinstance(metrics, dict):
        usage = metrics.get("usage", usage)
        provider_data = metrics.get("model_provider_data")
    else:
        provider_data = getattr(metrics, "model_provider_data", None)
    if isinstance(provider_data, dict):
        usage = provider_data.get("usage", usage)
        if usage is None and isinstance(provider_data.get("response"), dict):
            usage = provider_data["response"].get("usage")
    field_aliases: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("input_tokens", ("input_tokens", "prompt_tokens")),
        ("output_tokens", ("output_tokens", "completion_tokens")),
        ("total_tokens", ("total_tokens",)),
        ("latency", ("latency",)),
        ("time_to_first_token", ("time_to_first_token",)),
    )
    for normalized_key, candidates in field_aliases:
        value = None
        for key in candidates:
            if isinstance(metrics, dict):
                value = metrics.get(key)
            else:
                value = getattr(metrics, key, None)
            if value is not None:
                break
            if isinstance(usage, dict):
                value = usage.get(key)
            elif usage is not None:
                value = getattr(usage, key, None)
            if value is not None:
                break
        if value is not None:
            snapshot[normalized_key] = value
    if (
        "total_tokens" not in snapshot
        and "input_tokens" in snapshot
        and "output_tokens" in snapshot
    ):
        try:
            snapshot["total_tokens"] = int(snapshot["input_tokens"]) + int(snapshot["output_tokens"])
        except (TypeError, ValueError):
            pass
    return snapshot


def estimate_tokens_from_text(value: str) -> int:
    text = value.strip()
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def estimate_input_tokens(response_input: Any) -> int:
    if isinstance(response_input, str):
        return estimate_tokens_from_text(response_input)
    if isinstance(response_input, list):
        parts: list[str] = []
        for item in response_input:
            content = getattr(item, "content", None)
            if isinstance(content, str):
                parts.append(content)
        return estimate_tokens_from_text("\n".join(parts))
    return estimate_tokens_from_text(str(response_input))


def ensure_token_metrics(metrics: dict[str, Any], response_input: Any, output_text: str) -> dict[str, Any]:
    merged = dict(metrics)
    if (
        merged.get("input_tokens") is None
        or merged.get("output_tokens") is None
        or merged.get("total_tokens") is None
    ):
        in_tokens = estimate_input_tokens(response_input)
        out_tokens = estimate_tokens_from_text(output_text)
        merged["input_tokens"] = in_tokens
        merged["output_tokens"] = out_tokens
        merged["total_tokens"] = in_tokens + out_tokens
        merged["token_estimated"] = True
    else:
        merged["token_estimated"] = False
    return merged


def collect_delegation_paths(node: Any, prefix: str = "") -> list[str]:
    node_name = (
        getattr(node, "team_name", None)
        or getattr(node, "agent_name", None)
        or getattr(node, "name", None)
        or "unknown"
    )
    current = f"{prefix}->{node_name}" if prefix else str(node_name)
    member_responses = getattr(node, "member_responses", None) or []
    if not member_responses:
        return [current]
    paths: list[str] = []
    for child in member_responses:
        paths.extend(collect_delegation_paths(child, current))
    return paths


def needs_completion_retry(reply: str) -> bool:
    text = reply.strip()
    if not text:
        return True
    if len(text) > 260:
        return False
    lowered = text.lower()
    if "<tool_call>" in lowered:
        return False
    if lowered.endswith(":"):
        return True
    placeholder_prefixes = (
        "let me ",
        "i will ",
        "i'll ",
        "i am going to ",
        "i'm going to ",
        "one moment",
        "hold on",
    )
    return any(lowered.startswith(prefix) for prefix in placeholder_prefixes)
