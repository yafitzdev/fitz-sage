# fitz_ai/retrieval/__init__.py
"""
Retrieval intelligence infrastructure.

This package contains query-time retrieval components:
- sparse: BM25/TF-IDF sparse index for hybrid search
- expansion: Query expansion with synonyms and acronyms
- temporal: Temporal query detection and handling
- aggregation: Aggregation query detection (list all, count, enumerate)
- entity_graph: Entity-based related chunk discovery
- vocabulary: Keyword vocabulary for exact matching
"""

from fitz_ai.retrieval.aggregation import (
    AggregationDetector,
    AggregationIntent,
    AggregationResult,
)
from fitz_ai.retrieval.entity_graph import EntityGraphStore
from fitz_ai.retrieval.vocabulary import (
    Keyword,
    KeywordDetector,
    KeywordMatcher,
    VocabularyStore,
    create_matcher_from_store,
    generate_variations,
)

__all__ = [
    # Aggregation
    "AggregationDetector",
    "AggregationIntent",
    "AggregationResult",
    # Entity graph
    "EntityGraphStore",
    # Vocabulary
    "Keyword",
    "KeywordDetector",
    "KeywordMatcher",
    "VocabularyStore",
    "create_matcher_from_store",
    "generate_variations",
]
