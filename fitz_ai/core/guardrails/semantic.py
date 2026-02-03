# fitz_ai/core/guardrails/semantic.py
"""
Semantic Matcher - Language-agnostic concept detection using embeddings.

This module provides semantic matching capabilities for guardrails,
replacing brittle regex patterns with embedding-based similarity.

The key insight: multilingual embedding models map semantically similar
concepts to nearby vectors regardless of language. "because" (English),
"parce que" (French), "因为" (Chinese) all cluster together.

Usage:
    from fitz_ai.core.guardrails.semantic import SemanticMatcher

    matcher = SemanticMatcher(embedder)
    if matcher.has_causal_language(chunk.content):
        ...
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

from fitz_ai.core.chunk import Chunk

# Type alias for embedder function
EmbedderFunc = Callable[[str], list[float]]


# =============================================================================
# Concept Definitions
# =============================================================================
# These are "anchor phrases" that define concepts. The embedder will map
# semantically similar phrases in ANY language to nearby vectors.

CAUSAL_CONCEPTS: tuple[str, ...] = (
    "because of this",
    "this was caused by",
    "the reason is",
    "this led to",
    "as a result of",
    "due to",
    "this happened because",
    "the cause was",
    "consequently",
    "therefore this occurred",
)

ASSERTION_CONCEPTS: tuple[str, ...] = (
    "this is definitely",
    "it was confirmed that",
    "the answer is",
    "this states that",
    "according to this",
    "it is known that",
    "the fact is",
    "this proves that",
)

CAUSAL_QUERY_CONCEPTS: tuple[str, ...] = (
    "why did this happen",
    "what caused this",
    "what is the reason",
    "explain why",
    "how did this occur",
    "what led to this",
    # Prediction queries that need causal evidence
    "what will be the impact",
    "what will happen next",
    "predict the outcome",
    "what are the consequences",
    # Preference/choice queries that need causal reasoning
    "why do people prefer",
    "why is this better",
    "what makes this different",
)

FACT_QUERY_CONCEPTS: tuple[str, ...] = (
    "what is the answer",
    "which one is correct",
    "who is responsible",
    "where is this located",
    "when did this happen",
    "what is the value",
)

RESOLUTION_QUERY_CONCEPTS: tuple[str, ...] = (
    "which one is authoritative",
    "which source should I trust",
    "how to resolve this conflict",
    "which is the correct version",
    "reconcile these differences",
    "why do these disagree",
)


# =============================================================================
# Vector Math
# =============================================================================


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Returns value in [-1, 1] where 1 means identical direction.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(f"Vector dimension mismatch: {len(vec_a)} vs {len(vec_b)}")

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def mean_vector(vectors: list[list[float]]) -> list[float]:
    """Compute element-wise mean of vectors."""
    if not vectors:
        raise ValueError("Cannot compute mean of empty vector list")

    dim = len(vectors[0])
    result = [0.0] * dim

    for vec in vectors:
        for i, val in enumerate(vec):
            result[i] += val

    n = len(vectors)
    return [v / n for v in result]


# =============================================================================
# Semantic Matcher
# =============================================================================


@dataclass
class SemanticMatcher:
    """
    Language-agnostic semantic concept detection using embeddings.

    This class replaces regex-based pattern matching with embedding
    similarity, enabling robust detection across any language.

    Concept vectors are lazily computed and cached for efficiency.

    Attributes:
        embedder: Function that converts text to embedding vector
        causal_threshold: Similarity threshold for causal language detection
        assertion_threshold: Similarity threshold for assertion detection
        query_threshold: Similarity threshold for query type classification
        relevance_threshold: Similarity threshold for query-context relevance
    """

    embedder: EmbedderFunc
    causal_threshold: float = 0.65
    assertion_threshold: float = 0.60
    query_threshold: float = 0.60  # Balance causal detection vs false positives
    relevance_threshold: float = 0.62  # Balanced - between 0.55 and 0.65

    # Internal caches (not part of dataclass comparison)
    _concept_cache: dict[str, list[list[float]]] = field(
        default_factory=dict, repr=False, compare=False
    )
    _centroid_cache: dict[str, list[float]] = field(default_factory=dict, repr=False, compare=False)

    # -------------------------------------------------------------------------
    # Caching
    # -------------------------------------------------------------------------

    def _get_concept_vectors(self, key: str, concepts: tuple[str, ...]) -> list[list[float]]:
        """Get cached concept vectors or compute and cache them."""
        if key not in self._concept_cache:
            self._concept_cache[key] = [self.embedder(c) for c in concepts]
        return self._concept_cache[key]

    def _get_centroid(self, key: str, concepts: tuple[str, ...]) -> list[float]:
        """Get cached centroid vector or compute and cache it."""
        if key not in self._centroid_cache:
            vectors = self._get_concept_vectors(key, concepts)
            self._centroid_cache[key] = mean_vector(vectors)
        return self._centroid_cache[key]

    def _embed_text(self, text: str) -> list[float]:
        """Embed text using the configured embedder."""
        return self.embedder(text)

    # -------------------------------------------------------------------------
    # Similarity Computation
    # -------------------------------------------------------------------------

    def max_similarity_to_concepts(
        self,
        text: str,
        concept_key: str,
        concepts: tuple[str, ...],
    ) -> float:
        """
        Find maximum cosine similarity between text and any concept.

        This is more sensitive than centroid comparison - useful when
        any single concept match is meaningful.
        """
        text_vec = self._embed_text(text)
        concept_vecs = self._get_concept_vectors(concept_key, concepts)

        return max(cosine_similarity(text_vec, cv) for cv in concept_vecs)

    def similarity_to_centroid(
        self,
        text: str,
        concept_key: str,
        concepts: tuple[str, ...],
    ) -> float:
        """
        Compute similarity between text and concept centroid.

        Centroid comparison is faster (single comparison) and less
        sensitive to individual concept variations.
        """
        text_vec = self._embed_text(text)
        centroid = self._get_centroid(concept_key, concepts)

        return cosine_similarity(text_vec, centroid)

    # -------------------------------------------------------------------------
    # Query Type Detection
    # -------------------------------------------------------------------------

    def is_causal_query(self, query: str) -> bool:
        """
        Detect if query asks for causal explanation.

        Works across languages - "why did this happen", "pourquoi",
        "为什么" all detected as causal queries.
        """
        similarity = self.max_similarity_to_concepts(query, "causal_query", CAUSAL_QUERY_CONCEPTS)
        return similarity >= self.query_threshold

    def is_fact_query(self, query: str) -> bool:
        """
        Detect if query asks for factual information.

        Works across languages for who/what/where/when type questions.
        """
        similarity = self.max_similarity_to_concepts(query, "fact_query", FACT_QUERY_CONCEPTS)
        return similarity >= self.query_threshold

    def is_resolution_query(self, query: str) -> bool:
        """
        Detect if query explicitly asks for conflict resolution.

        Queries like "Which source should I trust?" should allow
        decisive answers even when conflicts exist.
        """
        similarity = self.max_similarity_to_concepts(
            query, "resolution_query", RESOLUTION_QUERY_CONCEPTS
        )
        return similarity >= self.query_threshold

    # -------------------------------------------------------------------------
    # Evidence Detection
    # -------------------------------------------------------------------------

    def has_causal_language(self, text: str) -> bool:
        """
        Check if text contains causal language.

        Detects "because", "due to", etc. in any language.
        """
        similarity = self.max_similarity_to_concepts(text, "causal", CAUSAL_CONCEPTS)
        return similarity >= self.causal_threshold

    def has_assertion(self, text: str) -> bool:
        """
        Check if text contains definitive assertions.

        Detects "is", "was confirmed", etc. in any language.
        """
        similarity = self.max_similarity_to_concepts(text, "assertion", ASSERTION_CONCEPTS)
        return similarity >= self.assertion_threshold

    def count_causal_chunks(self, chunks: Sequence[Chunk]) -> int:
        """Count chunks containing causal language."""
        return sum(1 for chunk in chunks if self.has_causal_language(chunk.content))

    def count_assertion_chunks(self, chunks: Sequence[Chunk]) -> int:
        """Count chunks containing assertions."""
        return sum(1 for chunk in chunks if self.has_assertion(chunk.content))

    # -------------------------------------------------------------------------
    # Query-Context Relevance
    # -------------------------------------------------------------------------

    def is_relevant_to_query(self, query: str, text: str) -> bool:
        """
        Check if text is semantically relevant to the query.

        This is the critical check that prevents the constraint system
        from treating irrelevant context as "evidence". A scientific paper
        about myelodysplasia should not count as evidence for a query
        about Q4 2024 revenue.

        Uses direct embedding similarity - if the query and text are
        about completely different topics, similarity will be low.
        """
        query_vec = self._embed_text(query)
        text_vec = self._embed_text(text)
        similarity = cosine_similarity(query_vec, text_vec)
        return similarity >= self.relevance_threshold

    def chunk_relevance_score(self, query: str, chunk: Chunk) -> float:
        """
        Get the relevance score between query and chunk.

        Returns similarity score in [0, 1] range.
        """
        query_vec = self._embed_text(query)
        chunk_vec = self._embed_text(chunk.content)
        return cosine_similarity(query_vec, chunk_vec)

    def count_relevant_chunks(self, query: str, chunks: Sequence[Chunk]) -> int:
        """Count chunks that are semantically relevant to the query."""
        return sum(1 for chunk in chunks if self.is_relevant_to_query(query, chunk.content))

    def get_relevant_chunks(self, query: str, chunks: Sequence[Chunk]) -> list[Chunk]:
        """Filter chunks to only those relevant to the query."""
        return [chunk for chunk in chunks if self.is_relevant_to_query(query, chunk.content)]


__all__ = [
    "SemanticMatcher",
    "EmbedderFunc",
    "cosine_similarity",
    "mean_vector",
    "CAUSAL_CONCEPTS",
    "ASSERTION_CONCEPTS",
    "CAUSAL_QUERY_CONCEPTS",
    "FACT_QUERY_CONCEPTS",
    "RESOLUTION_QUERY_CONCEPTS",
]
