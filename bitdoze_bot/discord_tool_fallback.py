from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, cast

logger = logging.getLogger(__name__)


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    def _parse_param_value(raw: str) -> Any:
        value = raw.strip()
        if value.isdigit():
            return int(value)
        try:
            return json.loads(value)
        except Exception:
            return value

    calls: list[dict[str, Any]] = []
    for block in re.findall(r"<tool_call>(.*?)</tool_call>", text, flags=re.DOTALL):
        func_match = re.search(r"<function=([a-zA-Z0-9_]+)>", block)
        if func_match:
            func_name = func_match.group(1)
        else:
            plain_name_match = re.match(r"\s*([a-zA-Z_][a-zA-Z0-9_]*)", block)
            if not plain_name_match:
                continue
            func_name = plain_name_match.group(1)
        params: dict[str, Any] = {}

        for p_match in re.findall(r"<parameter=([a-zA-Z0-9_]+)>(.*?)</parameter>", block, flags=re.DOTALL):
            key = p_match[0]
            params[key] = _parse_param_value(p_match[1])

        for p_match in re.finditer(
            r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
            block,
            flags=re.DOTALL,
        ):
            key = p_match.group(1).strip()
            if not key:
                continue
            params[key] = _parse_param_value(p_match.group(2))

        calls.append({"name": func_name, "params": params})
    return calls


def strip_tool_call_markup(text: str) -> str:
    return re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL).strip()


def get_declared_functions(tool: Any) -> set[str]:
    get_functions = getattr(tool, "get_functions", None)
    if callable(get_functions):
        funcs = get_functions()
        if isinstance(funcs, dict):
            return set(funcs.keys())
    return set()


async def run_tool_calls(
    agent,
    calls: list[dict[str, Any]],
    denied_tools: set[str] | None = None,
) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    denied = {"shell", "discord"} if denied_tools is None else denied_tools
    tools = getattr(agent, "tools", None) or []
    for call in calls:
        func_name = str(call.get("name", ""))
        params = cast(dict[str, Any], call.get("params", {}))
        target = None
        for tool in tools:
            tool_name = getattr(tool, "_bitdoze_tool_name", "") or ""
            if tool_name.lower() in denied:
                continue
            get_functions = getattr(tool, "get_functions", None)
            declared_funcs = get_functions() if callable(get_functions) else {}
            if not isinstance(declared_funcs, dict):
                declared_funcs = {}
            declared = set(declared_funcs.keys())
            if declared and func_name not in declared:
                continue
            if not declared:
                continue

            declared_fn = declared_funcs.get(func_name)
            fn = getattr(tool, func_name, None)
            if declared_fn is not None and callable(fn):
                target = fn
                break
        if target is None:
            logger.warning("Tool fallback: function '%s' not found or denied", func_name)
            continue
        try:
            output = await asyncio.to_thread(target, **params)
        except Exception as exc:  # noqa: BLE001
            logger.error("Tool fallback: function '%s' failed: %s", func_name, exc)
            results.append({"name": func_name, "output": f"Error: {exc}"})
            continue
        output_text = str(output).strip()
        if not output_text:
            output_text = "(no output)"
        results.append({"name": func_name, "output": output_text})
    return results
