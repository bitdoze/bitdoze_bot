from __future__ import annotations

from bitdoze_bot.discovery_tools import DiscoveryTools


class _FakeKnowledge:
    def __init__(self) -> None:
        self.insert_calls: list[dict] = []
        self.search_calls: list[dict] = []

    def insert(self, **kwargs):
        self.insert_calls.append(kwargs)

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return []


def test_discovery_tools_uses_current_knowledge_api() -> None:
    knowledge = _FakeKnowledge()
    tools = DiscoveryTools(knowledge=knowledge, embedder_id="text-embedding-3-small")

    save_msg = tools.save_discovery("Test title", "Test content")
    search_msg = tools.search_discoveries("test query", num_results=3)

    assert "Discovery saved" in save_msg
    assert search_msg == "No matching discoveries found."
    assert knowledge.insert_calls
    assert knowledge.insert_calls[0]["name"] == "Test title"
    assert "# Test title" in knowledge.insert_calls[0]["text_content"]
    assert knowledge.search_calls
    assert knowledge.search_calls[0] == {"query": "test query", "max_results": 3}
