from __future__ import annotations

from types import SimpleNamespace

from bitdoze_bot.utils import extract_response_text, strip_thinking_tags


def test_strip_thinking_tags_removes_think_blocks() -> None:
    text = "<think>internal reasoning</think>\n\nFinal answer"
    assert strip_thinking_tags(text) == "Final answer"


def test_strip_thinking_tags_supports_thinking_variant_and_case() -> None:
    text = "<THINKING>\nsecret\n</THINKING>\nVisible"
    assert strip_thinking_tags(text) == "Visible"


def test_extract_response_text_strips_think_from_content() -> None:
    response = SimpleNamespace(content="<think>hidden</think>Hello", reasoning_content=None, messages=None)
    assert extract_response_text(response) == "Hello"


def test_extract_response_text_strips_think_from_message_fallback() -> None:
    messages = [SimpleNamespace(content="<thinking>private</thinking>\nHi", reasoning_content=None)]
    response = SimpleNamespace(content="", reasoning_content="", messages=messages)
    assert extract_response_text(response) == "Hi"
