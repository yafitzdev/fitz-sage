# fitz_ai/engines/fitz_krag/ingestion/enricher.py
"""
KRAG-specific enrichment — extract keywords and entities from symbols/sections.

Reuses the LLM-based enrichment pattern from the shared enrichment bus but
adapted for KRAG's data model (symbol dicts + section dicts instead of Chunks).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fitz_ai.llm.providers.base import ChatProvider

logger = logging.getLogger(__name__)


class KragEnricher:
    """Batch LLM enrichment for KRAG symbols and sections."""

    def __init__(self, chat: "ChatProvider", batch_size: int = 15):
        self._chat = chat
        self._batch_size = batch_size

    def enrich_symbols(self, symbol_dicts: list[dict[str, Any]]) -> None:
        """Enrich symbol dicts in-place with keywords and entities."""
        for i in range(0, len(symbol_dicts), self._batch_size):
            batch = symbol_dicts[i : i + self._batch_size]
            items = [
                {
                    "name": s.get("name", ""),
                    "content": s.get("summary", "") or f"{s.get('kind', '')} {s.get('name', '')}",
                }
                for s in batch
            ]
            enriched = self._enrich_batch(items)
            for j, enrichment in enumerate(enriched):
                batch[j]["keywords"] = enrichment.get("keywords", [])
                batch[j]["entities"] = enrichment.get("entities", [])

    def enrich_sections(self, section_dicts: list[dict[str, Any]]) -> None:
        """Enrich section dicts in-place with keywords and entities."""
        for i in range(0, len(section_dicts), self._batch_size):
            batch = section_dicts[i : i + self._batch_size]
            items = [
                {
                    "name": s.get("title", ""),
                    "content": (s.get("summary", "") or s.get("content", ""))[:500],
                }
                for s in batch
            ]
            enriched = self._enrich_batch(items)
            for j, enrichment in enumerate(enriched):
                batch[j]["keywords"] = enrichment.get("keywords", [])
                batch[j]["entities"] = enrichment.get("entities", [])

    def _enrich_batch(self, items: list[dict[str, str]]) -> list[dict[str, Any]]:
        """Run a single LLM call to extract keywords + entities for a batch."""
        parts = []
        for i, item in enumerate(items):
            parts.append(f"Item {i + 1}: '{item['name']}'\n{item['content']}")
        prompt = "\n\n".join(parts)

        try:
            response = self._chat.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "Extract keywords and entities from each item.\n"
                            "Keywords: exact-match identifiers (function names, class names, "
                            "technical terms, IDs, abbreviations).\n"
                            "Entities: named entities with types "
                            '(e.g., {"name": "PostgreSQL", "type": "technology"}).\n\n'
                            "Return a JSON array with one object per item:\n"
                            '[{"keywords": ["kw1", "kw2"], '
                            '"entities": [{"name": "X", "type": "T"}]}, ...]'
                        ),
                    },
                    {"role": "user", "content": prompt},
                ]
            )
            return self._parse_response(response, len(items))
        except Exception as e:
            logger.warning(f"Enrichment batch failed: {e}")
            return [{"keywords": [], "entities": []} for _ in items]

    def _parse_response(self, response: str, expected_count: int) -> list[dict[str, Any]]:
        """Parse LLM response into list of enrichment dicts."""
        try:
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                text = text.rsplit("```", 1)[0]
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                parsed = json.loads(text[start:end])
                if isinstance(parsed, list) and len(parsed) >= expected_count:
                    return parsed[:expected_count]
        except (json.JSONDecodeError, IndexError):
            pass

        return [{"keywords": [], "entities": []} for _ in range(expected_count)]
