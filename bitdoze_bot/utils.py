from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def read_text_if_exists(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.exists():
        return ""
    try:
        return file_path.read_text(encoding="utf-8").strip()
    except OSError:
        logger.warning("Failed to read file: %s", file_path)
        return ""


def parse_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def strip_thinking_tags(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(
        r"<\s*(think|thinking)\b[^>]*>.*?<\s*/\s*\1\s*>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def extract_response_text(response: Any) -> str:
    if isinstance(response, str):
        return strip_thinking_tags(response)

    content = getattr(response, "content", None)
    if isinstance(content, str) and content.strip():
        return strip_thinking_tags(content)

    reasoning_content = getattr(response, "reasoning_content", None)
    if isinstance(reasoning_content, str) and reasoning_content.strip():
        return strip_thinking_tags(reasoning_content)

    messages = getattr(response, "messages", None)
    if isinstance(messages, list):
        for message in reversed(messages):
            msg_content = getattr(message, "content", None)
            if isinstance(msg_content, str) and msg_content.strip():
                return strip_thinking_tags(msg_content)
            msg_reasoning = getattr(message, "reasoning_content", None)
            if isinstance(msg_reasoning, str) and msg_reasoning.strip():
                return strip_thinking_tags(msg_reasoning)

    return ""
