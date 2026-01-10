# fitz_ai/ingestion/enrichment/chunk/__init__.py
"""
Chunk-level enrichment bus.

Provides unified per-chunk enrichment with pluggable modules.
"""

from fitz_ai.ingestion.enrichment.chunk.enricher import (
    ChunkEnricher,
    ChunkEnrichmentResult,
    EnrichmentBatchResult,
    EnrichmentModule,
    EntityModule,
    KeywordModule,
    SummaryModule,
    create_default_enricher,
)

__all__ = [
    "ChunkEnricher",
    "ChunkEnrichmentResult",
    "EnrichmentBatchResult",
    "EnrichmentModule",
    "EntityModule",
    "KeywordModule",
    "SummaryModule",
    "create_default_enricher",
]
