# fitz_ai/ingestion/enrichment/modules/chunk/entities.py
"""Entity extraction module for the enrichment bus."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fitz_ai.ingestion.enrichment.modules.base import EnrichmentModule
from fitz_ai.ingestion.enrichment.prompts import load_chunk_prompt

if TYPE_CHECKING:
    from fitz_ai.core.chunk import Chunk


class EntityModule(EnrichmentModule):
    """Extracts named entities (people, organizations, concepts, etc.)."""

    @property
    def name(self) -> str:
        return "entities"

    @property
    def json_key(self) -> str:
        return "entities"

    def prompt_instruction(self) -> str:
        return load_chunk_prompt("entities")

    def parse_result(self, data: Any) -> list[dict[str, str]]:
        if not isinstance(data, list):
            return []

        entities = []
        for item in data:
            if isinstance(item, dict) and "name" in item:
                entities.append(
                    {
                        "name": str(item.get("name", "")),
                        "type": str(item.get("type", "unknown")),
                    }
                )
        return entities

    def apply_to_chunk(self, chunk: "Chunk", result: Any) -> None:
        if result:
            chunk.metadata["entities"] = result


__all__ = ["EntityModule"]
