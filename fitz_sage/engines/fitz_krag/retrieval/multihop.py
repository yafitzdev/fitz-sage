# fitz_sage/engines/fitz_krag/retrieval/multihop.py
"""
Multi-hop retrieval controller for KRAG.

Iterative retrieve → read → evaluate → bridge cycle adapted for
KRAG's address-based architecture (Address/ReadResult instead of Chunk).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fitz_sage.engines.fitz_krag.types import ReadResult

if TYPE_CHECKING:
    from fitz_sage.engines.fitz_krag.retrieval.reader import ContentReader
    from fitz_sage.engines.fitz_krag.retrieval.router import RetrievalRouter
    from fitz_sage.llm.factory import ChatFactory

logger = logging.getLogger(__name__)


class KragHopController:
    """
    Multi-hop retrieval adapted for KRAG's address-based architecture.

    Iterates: retrieve addresses → read content → evaluate sufficiency →
    extract bridge questions → re-retrieve with bridge queries.
    """

    def __init__(
        self,
        router: "RetrievalRouter",
        reader: "ContentReader",
        chat_factory: "ChatFactory",
        max_hops: int = 2,
        top_read: int = 5,
    ):
        self._router = router
        self._reader = reader
        self._chat_factory = chat_factory
        self._max_hops = max_hops
        self._top_read = top_read

    def execute(
        self,
        query: str,
        profile: Any = None,
    ) -> list[ReadResult]:
        """
        Run iterative multi-hop retrieval.

        Args:
            query: Original user query
            profile: RetrievalProfile with pre-computed gates and signals

        Returns:
            Accumulated read results across all hops
        """
        all_results: list[ReadResult] = []
        seen_keys: set[tuple[str, str]] = set()
        current_query = query

        for hop in range(self._max_hops):
            # 1. Retrieve addresses
            addresses = self._router.retrieve(current_query, profile)
            if not addresses:
                break

            # Filter out already-seen addresses
            new_addresses = []
            for addr in addresses:
                key = (addr.source_id, addr.location)
                if key not in seen_keys:
                    seen_keys.add(key)
                    new_addresses.append(addr)

            if not new_addresses:
                break

            # 2. Read content
            results = self._reader.read(new_addresses, self._top_read)
            all_results.extend(results)

            # 3. Evaluate sufficiency
            if self._is_sufficient(query, all_results):
                logger.debug(f"Multi-hop: sufficient evidence at hop {hop + 1}")
                break

            # 4. Extract bridge questions
            bridge_questions = self._extract_bridge(query, all_results)
            if not bridge_questions:
                logger.debug(f"Multi-hop: no bridge questions at hop {hop + 1}")
                break

            current_query = bridge_questions[0]
            logger.debug(f"Multi-hop: bridge query = '{current_query[:80]}'")

        return all_results

    def _is_sufficient(self, query: str, results: list[ReadResult]) -> bool:
        """Evaluate if current evidence is sufficient to answer the query."""
        if not results:
            return False

        context = self._build_context(results)
        prompt = (
            "Given this question and retrieved evidence, determine if there is "
            "enough information to answer the question.\n\n"
            f"Question: {query}\n\n"
            f"Retrieved evidence:\n{context}\n\n"
            "Respond with ONLY 'SUFFICIENT' or 'INSUFFICIENT'."
        )

        try:
            chat = self._chat_factory("fast")
            response = chat.chat([{"role": "user", "content": prompt}])
            response_upper = response.upper()
            return "SUFFICIENT" in response_upper and "INSUFFICIENT" not in response_upper
        except Exception as e:
            logger.warning(f"Evidence evaluation failed: {e}")
            return True  # Assume sufficient on error to avoid infinite loops

    def _extract_bridge(self, query: str, results: list[ReadResult]) -> list[str]:
        """Generate bridge questions to fill evidence gaps."""
        context = self._build_context(results)
        prompt = (
            "You're helping answer a question. The current evidence is missing information.\n\n"
            f"Original question: {query}\n\n"
            f"Current evidence:\n{context}\n\n"
            "What specific follow-up question would help find the missing information?\n"
            'Return ONLY a JSON array: ["query1", "query2"]\n'
            "If no clear gaps, return: []"
        )

        try:
            import json

            chat = self._chat_factory("fast")
            response = chat.chat([{"role": "user", "content": prompt}])
            text = response.strip()
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                parsed = json.loads(text[start:end])
                if isinstance(parsed, list):
                    return [str(q) for q in parsed[:2] if isinstance(q, str) and q.strip()]
        except Exception as e:
            logger.warning(f"Bridge extraction failed: {e}")

        return []

    def _build_context(self, results: list[ReadResult], max_chars: int = 5000) -> str:
        """Build context string from read results."""
        parts: list[str] = []
        total = 0
        for r in results:
            content = r.content[:500]
            if total + len(content) > max_chars:
                break
            parts.append(f"[{r.file_path}] {content}")
            total += len(content) + 20
        return "\n\n".join(parts)
