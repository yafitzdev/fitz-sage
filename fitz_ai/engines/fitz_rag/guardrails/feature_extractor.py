# fitz_ai/engines/fitz_rag/guardrails/feature_extractor.py
"""
Feature Extractor for Governance Classifier.

Extracts a flat feature dict from query + chunks + constraint results for
training/inference with a tabular governance classifier. Features come from
three tiers:

- Tier 1: Signals already computed inside constraints (surfaced via metadata)
- Tier 2: Cheap computation on existing data (no LLM, no I/O)
- Tier 3: DetectionSummary from retrieval (threaded via StageContext)

The classifier uses these features to predict one of 3 governance modes:
abstain, disputed, trustworthy.
"""

from __future__ import annotations

import re
import statistics
from typing import TYPE_CHECKING, Any, Sequence

from fitz_ai.core.chunk import Chunk

if TYPE_CHECKING:
    from fitz_ai.engines.fitz_rag.guardrails.base import ConstraintResult


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
    num_denials = sum(1 for r in constraint_results.values() if not r.allow_decisive_answer)
    features["num_constraints_fired"] = num_denials

    signals = [r.signal for r in constraint_results.values() if r.signal]
    features["has_qualified_signal"] = "qualified" in signals

    # IE constraint features
    ie = constraint_results.get("insufficient_evidence")
    if ie:
        features["ie_fired"] = not ie.allow_decisive_answer
        features["ie_signal"] = ie.signal
    else:
        features["ie_fired"] = None

    # ConflictAware features
    ca = constraint_results.get("conflict_aware")
    if ca:
        m = ca.metadata
        features["ca_fired"] = not ca.allow_decisive_answer
        features["ca_signal"] = ca.signal
        features["ca_numerical_variance_detected"] = m.get("ca_numerical_variance_detected")
        features["ca_skipped_hedged_pairs"] = m.get("ca_skipped_hedged_pairs")
        features["ca_pairs_checked"] = m.get("ca_pairs_checked")
        features["ca_first_evidence_char"] = m.get("ca_first_evidence_char")
        features["ca_evidence_characters"] = m.get("ca_evidence_characters")
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
        features["av_jury_votes_no"] = av.metadata.get("jury_votes")
    else:
        features["av_jury_votes_no"] = None


def _extract_query_features(features: dict[str, Any], query: str) -> None:
    """Extract cheap features from the query text."""
    features["query_word_count"] = len(query.split())


def _extract_chunk_features(features: dict[str, Any], query: str, chunks: Sequence[Chunk]) -> None:
    """Extract cheap features from chunks and their relationship to query."""
    features["num_chunks"] = len(chunks)

    if not chunks:
        features["num_unique_sources"] = 0
        features["mean_vector_score"] = None
        features["score_spread"] = None
        features["vocab_overlap_ratio"] = 0.0
        # Inter-chunk defaults (including ctx_* features)
        for k in (
            "max_pairwise_overlap",
            "min_pairwise_overlap",
            "chunk_length_cv",
            "assertion_density",
            "number_density",
            "ctx_length_mean",
            "ctx_length_std",
            "ctx_total_chars",
            "ctx_contradiction_count",
            "ctx_negation_count",
            "ctx_number_count",
            "ctx_number_variance",
            "ctx_max_pairwise_sim",
            "ctx_mean_pairwise_sim",
            "ctx_min_pairwise_sim",
        ):
            features[k] = 0.0
        features["year_count"] = 0
        features["has_distinct_years"] = False
        return

    # Source diversity
    doc_ids = set()
    for c in chunks:
        doc_id = (
            getattr(c, "doc_id", None) or c.metadata.get("doc_id") or c.metadata.get("source_file")
        )
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
        features["score_spread"] = max(scores) - min(scores)
    else:
        features["mean_vector_score"] = None
        features["score_spread"] = None

    # Vocabulary overlap between query and chunks
    query_words = set(query.lower().split())
    query_content_words = query_words - _STOP_WORDS
    if query_content_words:
        chunk_text = " ".join(c.content.lower() for c in chunks)
        chunk_words = set(chunk_text.split())
        overlap = query_content_words & chunk_words
        features["vocab_overlap_ratio"] = len(overlap) / len(query_content_words)
    else:
        features["vocab_overlap_ratio"] = 0.0

    # Inter-chunk text features (Tier 2b — deterministic, no LLM)
    _extract_interchunk_features(features, chunks)


_STOP_WORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "can",
    "shall",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "and",
    "or",
    "but",
    "not",
    "no",
    "if",
    "then",
    "than",
    "that",
    "this",
    "these",
    "those",
    "what",
    "which",
    "who",
    "how",
    "when",
    "where",
    "why",
    "it",
    "its",
}

_HEDGE_WORDS = {
    "may",
    "might",
    "could",
    "possibly",
    "perhaps",
    "likely",
    "unlikely",
    "sometimes",
    "often",
    "typically",
    "generally",
    "usually",
    "probably",
    "approximately",
    "roughly",
    "about",
    "around",
    "estimated",
    "suggests",
    "appears",
    "seems",
    "potentially",
    "tends",
}

_ASSERTION_WORDS = {
    "always",
    "never",
    "must",
    "certainly",
    "definitely",
    "clearly",
    "obviously",
    "undoubtedly",
    "absolutely",
    "exactly",
    "precisely",
    "proven",
    "confirmed",
    "established",
    "demonstrates",
    "proves",
    "invariably",
    "unquestionably",
}

_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?(?:%|st|nd|rd|th)?\b")
_PLAIN_NUMBER_RE = re.compile(r"\b\d+\.?\d*\b")
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")

_CONTRADICTION_MARKERS = [
    "however",
    "but",
    "although",
    "contrary",
    "disagree",
    "whereas",
    "nevertheless",
    "conversely",
    "despite",
    "in contrast",
    "on the other hand",
    "contradicts",
    "inconsistent",
    "conflicts with",
    "differs from",
]
_NEGATION_WORDS = {
    "not",
    "no",
    "never",
    "neither",
    "nor",
    "none",
    "nothing",
    "hardly",
    "barely",
    "scarcely",
    "doesn't",
    "don't",
    "isn't",
    "wasn't",
    "weren't",
    "won't",
    "can't",
    "couldn't",
    "shouldn't",
}


def _extract_interchunk_features(features: dict[str, Any], chunks: Sequence[Chunk]) -> None:
    """Extract inter-chunk text relationship features (deterministic, no LLM).

    Includes features previously only available at training time via
    train_classifier.py compute_context_features(). All features here are
    available at both training and inference time.
    """
    _defaults = {
        "max_pairwise_overlap": 0.0,
        "min_pairwise_overlap": 0.0,
        "chunk_length_cv": 0.0,
        "assertion_density": 0.0,
        "number_density": 0.0,
        "ctx_length_mean": 0.0,
        "ctx_length_std": 0.0,
        "ctx_total_chars": 0.0,
        "ctx_contradiction_count": 0.0,
        "ctx_negation_count": 0.0,
        "ctx_number_count": 0.0,
        "ctx_number_variance": 0.0,
        "ctx_max_pairwise_sim": 0.0,
        "ctx_mean_pairwise_sim": 0.0,
        "ctx_min_pairwise_sim": 0.0,
        "year_count": 0,
        "has_distinct_years": False,
    }
    if len(chunks) < 2:
        features.update(_defaults)
        return

    # Precompute per-chunk data
    texts = [c.content for c in chunks]
    chunk_word_sets = []
    chunk_lengths = []
    char_lengths = []
    total_hedge = 0
    total_assert = 0
    total_numbers = 0
    all_text_lower = ""
    all_numbers: list[float] = []
    all_years: set[str] = set()

    for c in chunks:
        text = c.content
        text_lower = text.lower()
        all_text_lower += " " + text_lower

        words = text_lower.split()
        content_words = set(words) - _STOP_WORDS
        chunk_word_sets.append(content_words)
        chunk_lengths.append(len(words))
        char_lengths.append(len(text))

        word_set = set(words)
        total_hedge += len(word_set & _HEDGE_WORDS)
        total_assert += len(word_set & _ASSERTION_WORDS)
        total_numbers += len(_NUMBER_RE.findall(text))

        for m in _PLAIN_NUMBER_RE.finditer(text):
            try:
                all_numbers.append(float(m.group(0)))
            except ValueError:
                pass
        for ym in _YEAR_RE.finditer(text):
            all_years.add(ym.group(0))

    # --- Pairwise Jaccard overlap ---
    overlaps = []
    for i in range(len(chunk_word_sets)):
        for j in range(i + 1, len(chunk_word_sets)):
            a, b = chunk_word_sets[i], chunk_word_sets[j]
            union = a | b
            overlaps.append(len(a & b) / len(union) if union else 0.0)

    features["max_pairwise_overlap"] = max(overlaps) if overlaps else 0.0
    features["min_pairwise_overlap"] = min(overlaps) if overlaps else 0.0

    # --- Chunk length coefficient of variation ---
    mean_len = statistics.mean(chunk_lengths)
    if mean_len > 0 and len(chunk_lengths) > 1:
        features["chunk_length_cv"] = statistics.stdev(chunk_lengths) / mean_len
    else:
        features["chunk_length_cv"] = 0.0

    # --- Assertion vs hedge density ---
    epistemic_total = total_assert + total_hedge
    if epistemic_total > 0:
        features["assertion_density"] = (total_assert - total_hedge) / epistemic_total
    else:
        features["assertion_density"] = 0.0

    # --- Number density (numbers per chunk) ---
    features["number_density"] = total_numbers / len(chunks)

    # --- Context length stats (ported from train_classifier.py) ---
    features["ctx_length_mean"] = statistics.mean(char_lengths)
    features["ctx_length_std"] = statistics.pstdev(char_lengths)
    features["ctx_total_chars"] = sum(char_lengths)

    # --- Contradiction and negation markers ---
    features["ctx_contradiction_count"] = float(
        sum(1 for m in _CONTRADICTION_MARKERS if m in all_text_lower)
    )
    all_words = all_text_lower.split()
    features["ctx_negation_count"] = float(sum(1 for w in all_words if w in _NEGATION_WORDS))

    # --- Numerical content ---
    features["ctx_number_count"] = float(len(all_numbers))
    if len(all_numbers) > 1:
        mean_n = sum(all_numbers) / len(all_numbers)
        features["ctx_number_variance"] = float(
            sum((x - mean_n) ** 2 for x in all_numbers) / len(all_numbers)
        )
    else:
        features["ctx_number_variance"] = 0.0

    # --- TF-IDF pairwise similarity ---
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as _cos_sim

        tfidf = TfidfVectorizer(max_features=500, stop_words="english")
        matrix = tfidf.fit_transform(texts)
        sim_matrix = _cos_sim(matrix)
        n = sim_matrix.shape[0]
        pairwise_sims = [float(sim_matrix[i, j]) for i in range(n) for j in range(i + 1, n)]
        features["ctx_max_pairwise_sim"] = max(pairwise_sims)
        features["ctx_mean_pairwise_sim"] = sum(pairwise_sims) / len(pairwise_sims)
        features["ctx_min_pairwise_sim"] = min(pairwise_sims)
    except Exception:
        features["ctx_max_pairwise_sim"] = 0.0
        features["ctx_mean_pairwise_sim"] = 0.0
        features["ctx_min_pairwise_sim"] = 0.0

    # --- Temporal features ---
    features["year_count"] = len(all_years)
    features["has_distinct_years"] = len(all_years) > 1


def _extract_detection_features(features: dict[str, Any], detection_summary: Any | None) -> None:
    """Extract features from DetectionSummary (Tier 3)."""
    if detection_summary is None:
        features["detection_temporal"] = None
        features["detection_comparison"] = None
        return

    features["detection_temporal"] = getattr(detection_summary, "has_temporal_intent", None)
    features["detection_comparison"] = getattr(detection_summary, "has_comparison_intent", None)


__all__ = ["extract_features"]
