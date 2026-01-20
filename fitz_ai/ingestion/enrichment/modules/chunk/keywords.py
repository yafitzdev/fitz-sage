# fitz_ai/ingestion/enrichment/modules/chunk/keywords.py
"""Keyword extraction module for the enrichment bus."""

from __future__ import annotations

from typing import Any

from fitz_ai.ingestion.enrichment.modules.base import EnrichmentModule
from fitz_ai.ingestion.enrichment.prompts import load_chunk_prompt


class KeywordModule(EnrichmentModule):
    """
    Extracts exact-match keywords/identifiers.

    Keywords are collected separately (not attached to chunks) and
    saved to VocabularyStore for exact-match retrieval.
    """

    @property
    def name(self) -> str:
        return "keywords"

    @property
    def json_key(self) -> str:
        return "keywords"

    def prompt_instruction(self) -> str:
        return load_chunk_prompt("keywords")

    def parse_result(self, data: Any) -> list[str]:
        if isinstance(data, list):
            return [str(item).strip() for item in data if item]
        return []

    # Keywords don't attach to chunks - they're collected separately


__all__ = ["KeywordModule"]
