# fitz_ai/core/guardrails/feature_extractor.py
"""
Feature Extractor for Governance Classifier.

Extracts a flat feature dict from query + chunks + constraint results for
training/inference with a tabular governance classifier. Features come from
three tiers:

- Tier 1: Signals already computed inside constraints (surfaced via metadata)
- Tier 2: Cheap computation on existing data (no LLM, no I/O)
- Tier 3: DetectionSummary from retrieval (threaded via StageContext)

The classifier uses these features to predict one of 4 governance modes:
abstain, disputed, qualified, confident.
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import TYPE_CHECKING, Any, Sequence

from fitz_ai.core.chunk import Chunk

if TYPE_CHECKING:
    from fitz_ai.core.guardrails.base import ConstraintResult


def extract_features(
    query: str,
    chunks: Sequence[Chunk],
    constraint_results: dict[str, "ConstraintResult"],
    detection_summary: Any | None = None,
) -> dict[str, Any]:
    """
    Extract a flat feature dict for the governance classifier.

    Args:
        query: The user's question
        chunks: Retrieved chunks
        constraint_results: Map of constraint_name -> ConstraintResult
        detection_summary: Optional DetectionSummary from retrieval

    Returns:
        Flat dict of feature_name -> value (numeric, bool, or string)
    """
    features: dict[str, Any] = {}

    # === Tier 1: Constraint metadata (surfaced by constraint plugins) ===
    _extract_constraint_features(features, constraint_results)

    # === Tier 2: Cheap computation on query + chunks ===
    _extract_query_features(features, query)
    _extract_chunk_features(features, query, chunks)

    # === Tier 3: DetectionSummary from retrieval ===
    _extract_detection_features(features, detection_summary)

    return features


def _extract_constraint_features(
    features: dict[str, Any],
    constraint_results: dict[str, "ConstraintResult"],
) -> None:
    """Extract features from constraint result metadata."""
    # Aggregate constraint signals
    num_denials = sum(
        1 for r in constraint_results.values() if not r.allow_decisive_answer
    )
    features["num_constraints_fired"] = num_denials

    signals = [r.signal for r in constraint_results.values() if r.signal]
    features["has_abstain_signal"] = "abstain" in signals
    features["has_disputed_signal"] = "disputed" in signals
    features["has_qualified_signal"] = "qualified" in signals

    # IE constraint features
    ie = constraint_results.get("insufficient_evidence")
    if ie:
        m = ie.metadata
        features["ie_fired"] = not ie.allow_decisive_answer
        features["ie_signal"] = ie.signal
        features["ie_max_similarity"] = m.get("max_similarity")
        features["ie_entity_match_found"] = m.get("ie_entity_match_found")
        features["ie_primary_match_found"] = m.get("ie_primary_match_found")
        features["ie_critical_match_found"] = m.get("ie_critical_match_found")
        features["ie_query_aspect"] = m.get("ie_query_aspect")
        features["ie_summary_overlap"] = m.get("ie_summary_overlap")
        features["ie_has_matching_aspect"] = m.get("ie_has_matching_aspect")
        features["ie_has_conflicting_aspect"] = m.get("ie_has_conflicting_aspect")
    else:
        features["ie_fired"] = None

    # ConflictAware features
    ca = constraint_results.get("conflict_aware")
    if ca:
        m = ca.metadata
        features["ca_fired"] = not ca.allow_decisive_answer
        features["ca_signal"] = ca.signal
        features["ca_numerical_variance_detected"] = m.get("ca_numerical_variance_detected")
        features["ca_is_uncertainty_query"] = m.get("ca_is_uncertainty_query")
        features["ca_skipped_hedged_pairs"] = m.get("ca_skipped_hedged_pairs")
        features["ca_pairs_checked"] = m.get("ca_pairs_checked")
        features["ca_first_evidence_char"] = m.get("ca_first_evidence_char")
        features["ca_relevance_filtered_count"] = m.get("ca_relevance_filtered_count")
        features["ca_evidence_characters"] = m.get("evidence_characters")
    else:
        features["ca_fired"] = None

    # CausalAttribution features
    caa = constraint_results.get("causal_attribution")
    if caa:
        m = caa.metadata
        features["caa_fired"] = not caa.allow_decisive_answer
        features["caa_query_type"] = m.get("caa_query_type")
        features["caa_has_causal_evidence"] = m.get("caa_has_causal_evidence")
        features["caa_has_predictive_evidence"] = m.get("caa_has_predictive_evidence")
    else:
        features["caa_fired"] = None

    # SpecificInfoType features
    sit = constraint_results.get("specific_info_type")
    if sit:
        m = sit.metadata
        features["sit_fired"] = not sit.allow_decisive_answer
        features["sit_entity_mismatch"] = m.get("sit_entity_mismatch")
        features["sit_info_type_requested"] = m.get("sit_info_type_requested")
        features["sit_has_specific_info"] = m.get("sit_has_specific_info")
    else:
        features["sit_fired"] = None

    # AnswerVerification features
    av = constraint_results.get("answer_verification")
    if av:
        m = av.metadata
        features["av_fired"] = not av.allow_decisive_answer
        features["av_jury_votes_no"] = m.get("jury_votes")
    else:
        features["av_fired"] = None


def _extract_query_features(features: dict[str, Any], query: str) -> None:
    """Extract cheap features from the query text."""
    features["query_word_count"] = len(query.split())


def _extract_chunk_features(
    features: dict[str, Any], query: str, chunks: Sequence[Chunk]
) -> None:
    """Extract cheap features from chunks and their relationship to query."""
    features["num_chunks"] = len(chunks)

    if not chunks:
        features["num_unique_sources"] = 0
        features["mean_vector_score"] = None
        features["std_vector_score"] = None
        features["score_spread"] = None
        features["vocab_overlap_ratio"] = 0.0
        features["dominant_content_type"] = None
        return

    # Source diversity
    doc_ids = set()
    for c in chunks:
        doc_id = getattr(c, "doc_id", None) or c.metadata.get("doc_id") or c.metadata.get("source_file")
        if doc_id:
            doc_ids.add(doc_id)
    features["num_unique_sources"] = len(doc_ids) if doc_ids else len(chunks)

    # Vector score distribution
    scores = []
    for c in chunks:
        score = c.metadata.get("vector_score") or c.metadata.get("score")
        if score is not None:
            try:
                scores.append(float(score))
            except (TypeError, ValueError):
                pass

    if scores:
        features["mean_vector_score"] = statistics.mean(scores)
        features["std_vector_score"] = statistics.stdev(scores) if len(scores) > 1 else 0.0
        features["score_spread"] = max(scores) - min(scores)
    else:
        features["mean_vector_score"] = None
        features["std_vector_score"] = None
        features["score_spread"] = None

    # Vocabulary overlap between query and chunks
    query_words = set(query.lower().split())
    # Remove common stop words
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                  "being", "have", "has", "had", "do", "does", "did", "will",
                  "would", "could", "should", "may", "might", "can", "shall",
                  "in", "on", "at", "to", "for", "of", "with", "by", "from",
                  "and", "or", "but", "not", "no", "if", "then", "than",
                  "that", "this", "these", "those", "what", "which", "who",
                  "how", "when", "where", "why", "it", "its"}
    query_content_words = query_words - stop_words
    if query_content_words:
        chunk_text = " ".join(c.content.lower() for c in chunks)
        chunk_words = set(chunk_text.split())
        overlap = query_content_words & chunk_words
        features["vocab_overlap_ratio"] = len(overlap) / len(query_content_words)
    else:
        features["vocab_overlap_ratio"] = 0.0

    # Dominant content type from enrichment metadata
    content_types = [c.metadata.get("content_type") for c in chunks if c.metadata.get("content_type")]
    if content_types:
        features["dominant_content_type"] = Counter(content_types).most_common(1)[0][0]
    else:
        features["dominant_content_type"] = None


def _extract_detection_features(
    features: dict[str, Any], detection_summary: Any | None
) -> None:
    """Extract features from DetectionSummary (Tier 3)."""
    if detection_summary is None:
        features["detection_temporal"] = None
        features["detection_aggregation"] = None
        features["detection_comparison"] = None
        features["detection_boost_recency"] = None
        features["detection_boost_authority"] = None
        features["detection_needs_rewriting"] = None
        return

    features["detection_temporal"] = getattr(detection_summary, "has_temporal_intent", None)
    features["detection_aggregation"] = getattr(detection_summary, "has_aggregation_intent", None)
    features["detection_comparison"] = getattr(detection_summary, "has_comparison_intent", None)
    features["detection_boost_recency"] = getattr(detection_summary, "boost_recency", None)
    features["detection_boost_authority"] = getattr(detection_summary, "boost_authority", None)
    features["detection_needs_rewriting"] = getattr(detection_summary, "needs_rewriting", None)


__all__ = ["extract_features"]
