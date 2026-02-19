from __future__ import annotations

import logging

from agno.tools import Toolkit

from bitdoze_bot.cognee import CogneeClient

logger = logging.getLogger(__name__)


class CogneeTools(Toolkit):
    """Tools for persistent memory retrieval/storage via Cognee."""

    def __init__(
        self,
        client: CogneeClient,
        *,
        save_enabled: bool = True,
        search_enabled: bool = True,
    ) -> None:
        super().__init__(name="cognee")
        self._client = client
        if save_enabled:
            self.register(self.save_memory)
        if search_enabled:
            self.register(self.recall_memory)

    def save_memory(self, title: str, content: str) -> str:
        title_clean = title.strip()
        content_clean = content.strip()
        if not title_clean or not content_clean:
            return "Both title and content are required."
        text = f"# {title_clean}\n\n{content_clean}"
        ok = self._client.add_memory(
            text,
            metadata={"type": "manual_memory", "title": title_clean},
        )
        if ok:
            return f"Saved memory: {title_clean}"
        return "Failed to save memory to Cognee."

    def recall_memory(self, query: str, limit: int = 5) -> str:
        query_clean = query.strip()
        if not query_clean:
            return "Query is required."
        try:
            results = self._client.search(query_clean, limit=limit)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cognee search failed: %s", exc)
            return f"Cognee search failed: {exc}"
        if not results:
            return "No matching memories found in Cognee."
        lines = []
        for idx, item in enumerate(results[: max(limit, 1)], 1):
            lines.append(f"**{idx}.** {item[:700]}")
        return "\n\n".join(lines)
