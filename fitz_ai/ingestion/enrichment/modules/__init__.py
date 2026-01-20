# fitz_ai/ingestion/enrichment/modules/__init__.py
"""
Enrichment modules - pluggable components for the enrichment bus.

All enrichment modules live here:
- modules/base.py - EnrichmentModule ABC
- modules/chunk/ - Chunk-level modules (summary, keywords, entities)
"""

from fitz_ai.ingestion.enrichment.modules.base import ChatClient, EnrichmentModule
from fitz_ai.ingestion.enrichment.modules.chunk import (
    EntityModule,
    KeywordModule,
    SummaryModule,
)

__all__ = [
    "EnrichmentModule",
    "ChatClient",
    "SummaryModule",
    "KeywordModule",
    "EntityModule",
]
