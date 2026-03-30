# fitz_sage/governance/constraints/semantic.py
"""
Semantic Matcher - Language-agnostic concept detection using embeddings.

This module provides semantic matching capabilities for guardrails,
replacing brittle regex patterns with embedding-based similarity.

The key insight: multilingual embedding models map semantically similar
concepts to nearby vectors regardless of language. "because" (English),
"parce que" (French), "因为" (Chinese) all cluster together.

Usage:
    from fitz_sage.governance.constraints.semantic import SemanticMatcher

    matcher = SemanticMatcher(embedder)
    if matcher.has_causal_language(chunk.content):
        ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Sequence

if TYPE_CHECKING:
    from fitz_sage.governance.constraints.aspect_classifier import QueryAspect

from fitz_sage.core.math import cosine_similarity, mean_vector
from fitz_sage.governance.protocol import EvidenceItem

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

PREDICTIVE_QUERY_CONCEPTS: tuple[str, ...] = (
    "what will happen in the future",
    "what is the forecast for next year",
    "predict the future outcome of this",
    "what will the results be going forward",
    "what are the projections for this",
)

OPINION_QUERY_CONCEPTS: tuple[str, ...] = (
    "should I choose this option",
    "what is the best approach to take",
    "recommend a course of action for this",
    "is this worth doing or investing in",
    "which is better for my situation",
)

SPECULATIVE_QUERY_CONCEPTS: tuple[str, ...] = (
    "will this succeed or fail",
    "what are the chances of success here",
    "will this become widely adopted",
    "is this likely to be approved",
    "how likely is this outcome to occur",
)

PREDICTIVE_EVIDENCE_CONCEPTS: tuple[str, ...] = (
    "the forecast shows growth of",
    "projected to reach by next year",
    "expected to increase significantly",
    "analysts estimate future value at",
    "anticipated to grow in coming years",
)

HEDGE_EVIDENCE_CONCEPTS: tuple[str, ...] = (
    "this may indicate preliminary findings",
    "limited evidence suggests this could",
    "more research is needed to confirm",
    "results are inconclusive at this stage",
    "it remains unclear whether this applies",
    "early trials show tentative results",
    "this is potentially associated with",
)


# =============================================================================
# Aspect Classification Concepts
# =============================================================================
# Anchor phrases representing each query aspect intent.
# Used to classify what a query is asking for (e.g. cause vs definition vs pricing).

ASPECT_QUERY_CONCEPTS: dict[str, tuple[str, ...]] = {
    "cause": (
        "why did this happen",
        "what caused this",
        "what is the root cause",
        "what led to this",
        "explain the reason for",
    ),
    "effect": (
        "what are the effects of this",
        "what are the consequences",
        "what happened as a result",
        "what is the impact of this",
        "what did this lead to",
    ),
    "symptom": (
        "what are the symptoms",
        "signs of this condition",
        "how does it manifest",
        "what are the indicators",
        "what does it look like",
    ),
    "treatment": (
        "how do I fix this",
        "what is the solution",
        "how do I resolve this problem",
        "how to cure or treat this",
        "what intervention is recommended",
    ),
    "definition": (
        "what is this concept",
        "define this term",
        "what does this mean",
        "explain what this is",
        "what is the meaning of",
    ),
    "process": (
        "how does this work",
        "how is this done step by step",
        "what is the mechanism",
        "how is it made or produced",
        "walk me through the procedure",
    ),
    "application": (
        "what is this used for",
        "what are the use cases",
        "how is it applied in practice",
        "when would you use this",
        "what problems does it solve",
    ),
    "pricing": (
        "how much does this cost",
        "what is the price",
        "what are the fees",
        "what does it cost to use",
        "what is the subscription fee",
    ),
    "comparison": (
        "compare these two options",
        "what is the difference between",
        "which one is better",
        "how do they differ",
        "contrast these alternatives",
    ),
    "timeline": (
        "when did this happen",
        "history of this",
        "what is the chronological sequence",
        "when was this created or founded",
        "in what order did these events occur",
    ),
    "proof": (
        "prove this theorem",
        "show the mathematical proof",
        "provide evidence for this claim",
        "how is this verified or validated",
        "what is the formal derivation",
    ),
}

# Anchor phrases for detecting what a chunk of content is ABOUT.
# Used to determine whether a retrieved chunk addresses the query aspect.
ASPECT_CHUNK_CONCEPTS: dict[str, tuple[str, ...]] = {
    "cause": (
        "this was caused by",
        "the reason for this is",
        "due to this factor",
        "this led to the outcome",
        "the contributing factor was",
    ),
    "effect": (
        "as a result of this",
        "the consequences were",
        "this resulted in",
        "the outcome of this was",
        "the downstream effect was",
    ),
    "symptom": (
        "symptoms include the following",
        "signs of this condition are",
        "patients commonly experience",
        "observable indicators of this",
        "manifestations include",
    ),
    "treatment": (
        "the treatment involves",
        "the recommended solution is",
        "this can be fixed by",
        "the intervention is",
        "to resolve this issue",
    ),
    "definition": (
        "is defined as",
        "this term refers to",
        "by this we mean",
        "is a type of",
        "is the concept of",
    ),
    "process": (
        "the mechanism works by",
        "the steps involved are",
        "is manufactured through",
        "the workflow consists of",
        "the procedure is as follows",
    ),
    "application": (
        "is commonly used for",
        "typical use cases include",
        "can be applied to",
        "is deployed in contexts where",
        "practical applications include",
    ),
    "pricing": (
        "the price is",
        "costs per month",
        "the fee for this is",
        "is priced at",
        "the subscription costs",
    ),
    "comparison": (
        "compared to other options",
        "unlike the alternative",
        "in contrast to",
        "outperforms in this regard",
        "differs from the other by",
    ),
    "timeline": (
        "was founded in the year",
        "the history of this shows",
        "chronologically this occurred",
        "dates back to",
        "was released or launched in",
    ),
    "proof": (
        "is mathematically proven by",
        "the formal proof shows",
        "evidence demonstrates that",
        "it can be verified that",
        "by induction or contradiction",
    ),
}


# =============================================================================
# Info Type Detection Concepts
# =============================================================================
# Anchor phrases for identifying what specific information type a query requests.
# Used by SpecificInfoTypeConstraint to detect missing info in context.

INFO_TYPE_CONCEPTS: dict[str, tuple[str, ...]] = {
    "pricing": (
        "how much does this cost",
        "what is the price of this",
        "what are the fees for this",
        "what does it cost to subscribe",
        "is this service free or paid",
    ),
    "quantity": (
        "how many of these are there",
        "what is the total count",
        "how many instances exist",
        "number of items in this",
        "total quantity available",
    ),
    "temporal": (
        "when is the deadline for this",
        "what is the due date",
        "when will this be released",
        "what is the expiration date",
        "when does this expire or end",
    ),
    "specification": (
        "what is the maximum capacity",
        "minimum technical requirements",
        "what are the system limits",
        "upper bound or lower bound for",
        "what are the technical specs",
    ),
    "measurement": (
        "what is the recommended dosage",
        "dimensions of this object",
        "proper dose amount to take",
        "physical size or weight of",
        "how large or heavy is this",
    ),
    "warranty": (
        "what does the warranty cover",
        "what is the guarantee period",
        "how long is the warranty",
        "what is covered under warranty",
        "warranty terms and conditions",
    ),
    "rate": (
        "what is the success rate",
        "percentage of cases affected",
        "average salary or compensation",
        "rate at which this occurs",
        "what percentage qualifies",
    ),
    "decision": (
        "should we proceed with this",
        "is this worth the investment",
        "recommend a course of action",
        "should I choose this option",
        "advise on whether to do this",
    ),
}


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
    chunk_aspect_threshold: float = 0.55  # Permissive — chunks have varied language
    info_type_threshold: float = 0.63  # Conservative — only fire on clear intent
    hedge_threshold: float = 0.60  # Threshold for hedging language in evidence

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

    def is_predictive_query(self, query: str) -> bool:
        """Detect if query asks about future outcomes, forecasts, or projections."""
        similarity = self.max_similarity_to_concepts(
            query, "predictive_query", PREDICTIVE_QUERY_CONCEPTS
        )
        return similarity >= self.query_threshold

    def is_opinion_query(self, query: str) -> bool:
        """Detect if query asks for recommendations, judgments, or opinions."""
        similarity = self.max_similarity_to_concepts(query, "opinion_query", OPINION_QUERY_CONCEPTS)
        return similarity >= self.query_threshold

    def is_speculative_query(self, query: str) -> bool:
        """Detect if query requires speculation beyond available facts."""
        similarity = self.max_similarity_to_concepts(
            query, "speculative_query", SPECULATIVE_QUERY_CONCEPTS
        )
        return similarity >= self.query_threshold

    def is_uncertainty_query(self, query: str) -> tuple[bool, str]:
        """Check if query requires epistemic qualification.

        Returns (is_uncertainty, query_type) where query_type is one of
        'causal', 'predictive', 'opinion', 'speculative', 'none'.
        """
        if self.is_causal_query(query):
            return True, "causal"
        if self.is_predictive_query(query):
            return True, "predictive"
        if self.is_opinion_query(query):
            return True, "opinion"
        if self.is_speculative_query(query):
            return True, "speculative"
        return False, "none"

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

    def count_causal_chunks(self, chunks: Sequence[EvidenceItem]) -> int:
        """Count chunks containing causal language."""
        return sum(1 for chunk in chunks if self.has_causal_language(chunk.content))

    def count_assertion_chunks(self, chunks: Sequence[EvidenceItem]) -> int:
        """Count chunks containing assertions."""
        return sum(1 for chunk in chunks if self.has_assertion(chunk.content))

    def has_predictive_language(self, text: str) -> bool:
        """Check if text contains forward-looking / predictive language."""
        similarity = self.max_similarity_to_concepts(
            text, "predictive_evidence", PREDICTIVE_EVIDENCE_CONCEPTS
        )
        return similarity >= self.causal_threshold

    def has_hedged_language(self, text: str) -> bool:
        """Check if text contains uncertainty / hedging language."""
        similarity = self.max_similarity_to_concepts(
            text, "hedge_evidence", HEDGE_EVIDENCE_CONCEPTS
        )
        return similarity >= self.hedge_threshold

    def classify_evidence_character(self, text: str) -> str:
        """Classify evidence as 'hedged', 'assertive', or 'mixed'.

        Returns:
            'hedged'    — uncertainty / tentative language dominates
            'mixed'     — both hedging and assertive language present
            'assertive' — firm claims, established facts, or neutral factual statements
        """
        is_hedged = self.has_hedged_language(text)
        is_assertive = self.has_assertion(text)
        if is_hedged and not is_assertive:
            return "hedged"
        if is_hedged and is_assertive:
            return "mixed"
        return "assertive"

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

    def chunk_relevance_score(self, query: str, chunk: EvidenceItem) -> float:
        """
        Get the relevance score between query and chunk.

        Returns similarity score in [0, 1] range.
        """
        query_vec = self._embed_text(query)
        chunk_vec = self._embed_text(chunk.content)
        return cosine_similarity(query_vec, chunk_vec)

    def count_relevant_chunks(self, query: str, chunks: Sequence[EvidenceItem]) -> int:
        """Count chunks that are semantically relevant to the query."""
        return sum(1 for chunk in chunks if self.is_relevant_to_query(query, chunk.content))

    def get_relevant_chunks(self, query: str, chunks: Sequence[EvidenceItem]) -> list[EvidenceItem]:
        """Filter chunks to only those relevant to the query."""
        return [chunk for chunk in chunks if self.is_relevant_to_query(query, chunk.content)]

    # -------------------------------------------------------------------------
    # Aspect Classification (replaces AspectClassifier regex)
    # -------------------------------------------------------------------------

    def classify_query_aspect(self, query: str) -> "QueryAspect":
        """
        Classify query into aspect category using embedding similarity.

        Computes similarity to each aspect's centroid and returns the
        best-matching aspect above query_threshold, or GENERAL.
        """
        from fitz_sage.governance.constraints.aspect_classifier import QueryAspect

        best_aspect = QueryAspect.GENERAL
        best_score = self.query_threshold

        for aspect_val, concepts in ASPECT_QUERY_CONCEPTS.items():
            key = f"aspect_query_{aspect_val}"
            score = self.similarity_to_centroid(query, key, concepts)
            if score > best_score:
                best_score = score
                best_aspect = QueryAspect(aspect_val)

        return best_aspect

    def classify_chunk_aspects(self, text: str) -> "list[QueryAspect]":
        """
        Extract all aspects present in chunk content using embedding similarity.

        Uses a lower threshold than query classification — chunks have more
        varied language, so permissive matching reduces false negatives.
        Returns [GENERAL] when no specific aspect is detected.
        """
        from fitz_sage.governance.constraints.aspect_classifier import QueryAspect

        aspects = []
        for aspect_val, concepts in ASPECT_CHUNK_CONCEPTS.items():
            key = f"aspect_chunk_{aspect_val}"
            score = self.similarity_to_centroid(text, key, concepts)
            if score >= self.chunk_aspect_threshold:
                aspects.append(QueryAspect(aspect_val))

        return aspects if aspects else [QueryAspect.GENERAL]

    # -------------------------------------------------------------------------
    # Info Type Detection (replaces SpecificInfoTypeConstraint regex)
    # -------------------------------------------------------------------------

    def identify_info_type(self, query: str) -> str | None:
        """
        Identify what specific information type the query requests.

        Conservative: only returns a type when similarity clearly exceeds
        info_type_threshold. Returns None for generic questions.
        """
        best_type = None
        best_score = self.info_type_threshold

        for info_type, concepts in INFO_TYPE_CONCEPTS.items():
            key = f"info_type_{info_type}"
            score = self.similarity_to_centroid(query, key, concepts)
            if score > best_score:
                best_score = score
                best_type = info_type

        return best_type


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
    "PREDICTIVE_QUERY_CONCEPTS",
    "OPINION_QUERY_CONCEPTS",
    "SPECULATIVE_QUERY_CONCEPTS",
    "PREDICTIVE_EVIDENCE_CONCEPTS",
    "HEDGE_EVIDENCE_CONCEPTS",
    "ASPECT_QUERY_CONCEPTS",
    "ASPECT_CHUNK_CONCEPTS",
    "INFO_TYPE_CONCEPTS",
]
