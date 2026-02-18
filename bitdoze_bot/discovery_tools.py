"""Self-improvement tools for the agent.

Provides save_discovery and search_discoveries so the agent can
build and query its own knowledge base over time.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from agno.tools import Toolkit

logger = logging.getLogger(__name__)


class DiscoveryTools(Toolkit):
    """Tools that let the agent save and search its own discoveries."""

    def __init__(
        self,
        knowledge: Any,
        embedder_id: str | None = None,
        save_enabled: bool = True,
        search_enabled: bool = True,
    ) -> None:
        super().__init__(name="discoveries")
        self._knowledge = knowledge
        self._embedder_id = embedder_id
        if save_enabled:
            self.register(self.save_discovery)
        if search_enabled:
            self.register(self.search_discoveries)

    def save_discovery(self, title: str, content: str) -> str:
        """Save a discovery or learning to the knowledge base.

        Use this when you:
        - Fix an error and learn something reusable
        - Discover where information lives
        - Learn a user preference or correction
        - Find a pattern that will help with future queries

        Args:
            title: Short descriptive title for the discovery.
            content: Detailed description of what was learned.

        Returns:
            Confirmation message.
        """
        try:
            doc_text = f"# {title}\n\n{content}"
            self._knowledge.insert(
                name=title,
                text_content=doc_text,
                metadata={"source": "discovery_tools"},
            )
            logger.info(
                "Discovery saved title=%s chars=%d embedder=%s",
                title,
                len(doc_text),
                self._embedder_id or "unknown",
            )
            return f"Discovery saved: {title}"
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to save discovery: %s", exc)
            return f"Failed to save discovery: {exc}"

    def search_discoveries(self, query: str, num_results: int = 5) -> str:
        """Search previous discoveries and learnings.

        Use this before answering questions to check if you've
        already learned something relevant.

        Args:
            query: Search query describing what you're looking for.
            num_results: Maximum number of results to return.

        Returns:
            Matching discoveries or a message if none found.
        """
        try:
            logger.info(
                "Discovery search query=%s num_results=%d embedder=%s",
                query[:120],
                num_results,
                self._embedder_id or "unknown",
            )
            results = self._knowledge.search(query=query, max_results=num_results)
            logger.info(
                "Discovery search completed query=%s returned=%d",
                query[:120],
                len(results) if isinstance(results, list) else 0,
            )
            if not results:
                return "No matching discoveries found."
            parts: list[str] = []
            for i, doc in enumerate(results, 1):
                text = getattr(doc, "content", None) or str(doc)
                parts.append(f"**{i}.** {text[:500]}")
            return "\n\n".join(parts)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to search discoveries: %s", exc)
            return f"Failed to search discoveries: {exc}"
