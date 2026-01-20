# fitz_ai/ingestion/enrichment/modules/chunk/summary.py
"""Summary extraction module for the enrichment bus."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fitz_ai.ingestion.enrichment.modules.base import EnrichmentModule
from fitz_ai.ingestion.enrichment.prompts import load_chunk_prompt

if TYPE_CHECKING:
    from fitz_ai.core.chunk import Chunk


class SummaryModule(EnrichmentModule):
    """Extracts a searchable summary for each chunk."""

    @property
    def name(self) -> str:
        return "summary"

    @property
    def json_key(self) -> str:
        return "summary"

    def prompt_instruction(self) -> str:
        return load_chunk_prompt("summary")

    def parse_result(self, data: Any) -> str:
        if isinstance(data, str):
            return data.strip()
        return str(data).strip() if data else ""

    def apply_to_chunk(self, chunk: "Chunk", result: Any) -> None:
        if result:
            chunk.metadata["summary"] = result


__all__ = ["SummaryModule"]
