# fitz_ai/ingestion/enrichment/modules/chunk/__init__.py
"""
Chunk-level enrichment modules.

Each module extracts specific metadata from chunks during the enrichment bus pass.
"""

from fitz_ai.ingestion.enrichment.modules.chunk.entities import EntityModule
from fitz_ai.ingestion.enrichment.modules.chunk.keywords import KeywordModule
from fitz_ai.ingestion.enrichment.modules.chunk.summary import SummaryModule

__all__ = [
    "SummaryModule",
    "KeywordModule",
    "EntityModule",
]
