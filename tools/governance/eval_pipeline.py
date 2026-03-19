# tools/governance/eval_pipeline.py
"""
Full pipeline eval for governance classifier.

Unlike extract_features.py (which uses synthetic inline contexts with no embeddings),
this script adds REAL embedding computation and detection analysis to get realistic
Tier 2/3 features. This answers: "Does the classifier generalize to production data?"

Three-way comparison: expected_mode vs governor vs classifier.

Usage:
    python -m tools.governance.eval_pipeline
    python -m tools.governance.eval_pipeline --limit 10
    python -m tools.governance.eval_pipeline --embedding ollama --chat ollama
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from fitz_ai.config import load_engine_config
from fitz_ai.core.chunk import Chunk
from fitz_ai.governance import AnswerGovernor
from fitz_ai.governance.constraints.base import ConstraintResult
from fitz_ai.governance.constraints.feature_extractor import extract_features
from fitz_ai.governance.constraints.plugins.answer_verification import (
    AnswerVerificationConstraint,
)
from fitz_ai.governance.constraints.plugins.causal_attribution import (
    CausalAttributionConstraint,
)
from fitz_ai.governance.constraints.plugins.conflict_aware import ConflictAwareConstraint
from fitz_ai.governance.constraints.plugins.insufficient_evidence import (
    InsufficientEvidenceConstraint,
)
from fitz_ai.governance.constraints.plugins.specific_info_type import (
    SpecificInfoTypeConstraint,
)
from fitz_ai.llm import get_chat_factory, get_embedder
from fitz_ai.retrieval.detection.registry import DetectionOrchestrator

# Paths
_DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent / "fitz-gov" / "data" / "tier1_core"
)
_DEFAULT_MODEL = Path(__file__).resolve().parent / "data" / "model_v1.joblib"
_DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "eval_results.csv"

# Feature type sets (copied from extract_features.py / train_classifier.py for preprocessing)
_BOOL_FEATURES = {
    "ie_fired",
    "ie_entity_match_found",
    "ie_primary_match_found",
    "ie_critical_match_found",
    "ie_has_matching_aspect",
    "ie_has_conflicting_aspect",
    "ie_summary_overlap",
    "ca_fired",
    "ca_numerical_variance_detected",
    "ca_is_uncertainty_query",
    "caa_fired",
    "caa_has_causal_evidence",
    "caa_has_predictive_evidence",
    "sit_fired",
    "sit_entity_mismatch",
    "sit_has_specific_info",
    "av_fired",
    "has_abstain_signal",
    "has_disputed_signal",
    "has_qualified_signal",
    "query_has_comparison_words",
    "has_distinct_years",
    "detection_temporal",
    "detection_comparison",
    "detection_aggregation",
    "detection_boost_recency",
    "detection_boost_authority",
    "detection_needs_rewriting",
}

_NUMERIC_FEATURES = {
    "ie_max_similarity",
    "ca_pairs_checked",
    "ca_relevance_filtered_count",
    "av_jury_votes_no",
    "num_constraints_fired",
    "query_word_count",
    "num_chunks",
    "num_unique_sources",
    "mean_vector_score",
    "max_vector_score",
    "min_vector_score",
    "std_vector_score",
    "score_spread",
    "vocab_overlap_ratio",
    "max_pairwise_overlap",
    "min_pairwise_overlap",
    "chunk_length_cv",
    "assertion_density",
    "hedge_density",
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
    "year_count",
}

_STRING_FEATURES = {
    "ie_signal",
    "ie_query_aspect",
    "ie_detection_reason",
    "ca_signal",
    "ca_first_evidence_char",
    "ca_evidence_characters",
    "caa_query_type",
    "sit_info_type_requested",
    "query_question_type",
}

_META_COLS = {
    "case_id",
    "expected_mode",
    "governor_predicted",
    "classifier_predicted",
    "difficulty",
    "subcategory",
}


# ---------------------------------------------------------------------------
# Data loading (reused from extract_features.py)
# ---------------------------------------------------------------------------


def load_cases(data_dir: Path) -> list[dict[str, Any]]:
    """Load all cases from tier1_core JSON files."""
    import json

    cases = []
    json_files = sorted(data_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {data_dir}")
    for path in json_files:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for case in data["cases"]:
            cases.append(case)
    return cases


def case_to_chunks(case: dict[str, Any]) -> list[Chunk]:
    """Convert a case's inline contexts to Chunk objects."""
    case_id = case["id"]
    chunks = []
    for i, ctx in enumerate(case.get("contexts", [])):
        chunks.append(
            Chunk(
                id=f"{case_id}_ctx_{i}",
                doc_id=case_id,
                content=ctx,
                chunk_index=i,
                metadata={"source_file": f"case_{case_id}"},
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# Embedding enrichment
# ---------------------------------------------------------------------------


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def enrich_chunks_with_embeddings(
    query: str,
    chunks: list[Chunk],
    embedder,
) -> None:
    """Compute embeddings and set vector_score on each chunk (in-place)."""
    try:
        texts = [query] + [c.content for c in chunks]
        embeddings = embedder.embed_batch(texts)
        query_emb = embeddings[0]
        for i, chunk in enumerate(chunks):
            chunk.metadata["vector_score"] = cosine_similarity(query_emb, embeddings[i + 1])
    except Exception as e:
        print(f"  WARNING: Embedding failed: {e}", file=sys.stderr)
        for chunk in chunks:
            chunk.metadata["vector_score"] = 0.0


# ---------------------------------------------------------------------------
# Constraint running
# ---------------------------------------------------------------------------


def make_constraints(chat, chat_balanced=None, embedder=None) -> list:
    """Create a fresh set of constraints."""
    return [
        InsufficientEvidenceConstraint(chat=chat, embedder=embedder),
        CausalAttributionConstraint(),
        SpecificInfoTypeConstraint(),
        ConflictAwareConstraint(chat=chat),
        AnswerVerificationConstraint(chat=chat, chat_balanced=chat_balanced),
    ]


def run_constraints_individually(
    query: str,
    chunks: Sequence[Chunk],
    constraints: list,
) -> dict[str, ConstraintResult]:
    """Run each constraint independently (no staged short-circuit)."""
    results: dict[str, ConstraintResult] = {}
    for constraint in constraints:
        try:
            result = constraint.apply(query, chunks)
            metadata = dict(result.metadata)
            metadata["constraint_name"] = constraint.name
            result = ConstraintResult(
                allow_decisive_answer=result.allow_decisive_answer,
                reason=result.reason,
                signal=result.signal,
                metadata=metadata,
            )
            results[constraint.name] = result
        except Exception as e:
            print(f"  WARNING: Constraint '{constraint.name}' raised: {e}", file=sys.stderr)
    return results


def fill_defaults(features: dict[str, Any]) -> dict[str, Any]:
    """Replace None values with typed defaults."""
    for key, value in features.items():
        if value is None:
            if key in _BOOL_FEATURES:
                features[key] = False
            elif key in _NUMERIC_FEATURES:
                features[key] = 0
            elif key in _STRING_FEATURES:
                features[key] = "none"
    return features


# ---------------------------------------------------------------------------
# Classifier inference
# ---------------------------------------------------------------------------


class GovernanceClassifier:
    """Wrapper for trained classifier model inference.

    Supports three formats:
    - 4-class: single model with {"model", "encoders", "feature_names", "labels"}
    - twostage: two binary classifiers with {"stage1_model", "stage2_model", ...}
    - calibrated: twostage with predict_proba thresholds (matches production GovernanceDecider)
    """

    def __init__(self, model_path: Path):
        artifact = joblib.load(model_path)
        self.encoders = artifact["encoders"]
        self.feature_names = artifact["feature_names"]
        self.labels = artifact["labels"]
        self._mode = artifact.get("mode", "4class")
        self._s1_threshold = artifact.get("stage1_threshold", None)
        self._s2_threshold = artifact.get("stage2_threshold", None)

        if self._mode == "twostage":
            self._stage1 = artifact["stage1_model"]
            self._stage2 = artifact["stage2_model"]
            s1_name = artifact.get("stage1_name", "unknown")
            s2_name = artifact.get("stage2_name", "unknown")
            acc = artifact.get("combined_accuracy", 0)
            calibrated = self._s1_threshold is not None
            print(
                f"Loaded twostage classifier: {s1_name}/{s2_name} "
                f"({len(self.feature_names)} features, {acc:.1%} acc"
                f"{f', calibrated s1={self._s1_threshold:.2f} s2={self._s2_threshold:.2f}' if calibrated else ', raw'})"
            )
        else:
            self.model = artifact["model"]
            self.model_name = artifact.get("model_name", "unknown")
            print(f"Loaded classifier: {self.model_name} ({len(self.feature_names)} features)")

    def _build_row(self, features: dict[str, Any]) -> pd.DataFrame:
        """Build feature DataFrame from feature dict."""
        row = {}
        for name in self.feature_names:
            val = features.get(name, 0)
            if name in self.encoders:
                val = str(val) if val is not None else "none"
                le = self.encoders[name]
                if val in le.classes_:
                    val = le.transform([val])[0]
                else:
                    val = 0  # Unknown category
            elif isinstance(val, bool) or val in ("True", "False"):
                val = 1 if val in (True, "True") else 0
            elif val is None:
                val = 0
            else:
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    val = 0
            row[name] = val
        return pd.DataFrame(
            [[row[name] for name in self.feature_names]], columns=self.feature_names
        )

    def predict(self, features: dict[str, Any]) -> str:
        """Predict governance mode from feature dict.

        Uses predict_proba with calibrated thresholds when available
        (matches production GovernanceDecider behavior).
        """
        X = self._build_row(features)

        if self._mode == "twostage":
            if self._s1_threshold is not None:
                return self._predict_calibrated(X)
            return self._predict_raw(X)
        else:
            pred_idx = self.model.predict(X)[0]
            return self.labels[pred_idx] if isinstance(pred_idx, int) else str(pred_idx)

    def _predict_calibrated(self, X: pd.DataFrame) -> str:
        """Two-stage prediction with calibrated probability thresholds."""
        # Stage 1: answerable vs abstain
        s1_probas = self._stage1.predict_proba(X)
        s1_classes = list(self._stage1.classes_)
        answerable_idx = s1_classes.index("answerable")
        p_answerable = float(s1_probas[0, answerable_idx])

        if p_answerable < self._s1_threshold:
            return "abstain"

        # Stage 2: trustworthy vs disputed
        s2_probas = self._stage2.predict_proba(X)
        s2_classes = list(self._stage2.classes_)
        trustworthy_idx = s2_classes.index("trustworthy")
        p_trustworthy = float(s2_probas[0, trustworthy_idx])

        if p_trustworthy >= self._s2_threshold:
            return "trustworthy"
        return "disputed"

    def _predict_raw(self, X: pd.DataFrame) -> str:
        """Two-stage prediction with raw class predictions (no thresholds)."""
        stage1_pred = self._stage1.predict(X)[0]
        if str(stage1_pred) == "abstain":
            return "abstain"
        stage2_pred = self._stage2.predict(X)[0]
        return str(stage2_pred)


# ---------------------------------------------------------------------------
# Per-case processing
# ---------------------------------------------------------------------------

_thread_local = threading.local()


def process_case(
    case: dict[str, Any],
    chat,
    embedder,
    detection_orchestrator: DetectionOrchestrator,
    classifier: GovernanceClassifier,
    chat_balanced=None,
) -> dict[str, Any] | None:
    """Process a single case with full feature enrichment."""
    # Thread-local constraints
    if not hasattr(_thread_local, "constraints"):
        _thread_local.constraints = make_constraints(
            chat, chat_balanced=chat_balanced, embedder=embedder
        )

    query = case["query"]
    chunks = case_to_chunks(case)

    # Tier 2 enrichment: compute real embeddings → vector_score
    enrich_chunks_with_embeddings(query, chunks, embedder)

    # Tier 3 enrichment: run detection orchestrator → DetectionSummary
    try:
        detection_summary = detection_orchestrator.detect_for_retrieval(query)
    except Exception as e:
        print(f"  WARNING: Detection failed for {case['id']}: {e}", file=sys.stderr)
        detection_summary = None

    # Tier 1: run constraints individually
    result_map = run_constraints_individually(query, chunks, _thread_local.constraints)

    # Extract features with ALL tiers
    features = extract_features(query, chunks, result_map, detection_summary)
    features = fill_defaults(features)

    # Governor prediction
    governor = AnswerGovernor()
    decision = governor.decide(list(result_map.values()))
    governor_predicted = decision.mode.value

    # Classifier prediction
    classifier_predicted = classifier.predict(features)

    return {
        "case_id": case["id"],
        "expected_mode": case["expected_mode"],
        "governor_predicted": governor_predicted,
        "classifier_predicted": classifier_predicted,
        "difficulty": case.get("difficulty", "unknown"),
        "subcategory": case.get("subcategory", "unknown"),
        **features,
    }


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


def _collapse_3class(label: str) -> str:
    """Collapse confident/qualified → trustworthy for two-stage comparison."""
    return "trustworthy" if label in ("confident", "qualified") else label


def print_evaluation(rows: list[dict[str, Any]], twostage: bool = False) -> None:
    """Print comprehensive evaluation comparing governor vs classifier vs expected."""
    n = len(rows)

    expected = [r["expected_mode"] for r in rows]
    governor = [r["governor_predicted"] for r in rows]
    classifier = [r["classifier_predicted"] for r in rows]

    # Collapse to 3-class if twostage
    if twostage:
        expected = [_collapse_3class(e) for e in expected]
        governor = [_collapse_3class(g) for g in governor]
        modes = ["abstain", "disputed", "trustworthy"]
    else:
        modes = ["abstain", "disputed", "qualified", "confident"]

    gov_correct = sum(1 for e, g in zip(expected, governor) if e == g)
    cls_correct = sum(1 for e, c in zip(expected, classifier) if e == c)
    agreement = sum(1 for g, c in zip(governor, classifier) if g == c)

    print("\n" + "=" * 70)
    print(f"FULL PIPELINE EVALUATION RESULTS ({'3-class twostage' if twostage else '4-class'})")
    print("=" * 70)

    print(f"\nTotal cases: {n}")
    print(f"Governor accuracy:    {gov_correct}/{n} ({100 * gov_correct / n:.1f}%)")
    print(f"Classifier accuracy:  {cls_correct}/{n} ({100 * cls_correct / n:.1f}%)")
    print(f"Agreement rate:       {agreement}/{n} ({100 * agreement / n:.1f}%)")
    print(
        f"Delta:                {cls_correct - gov_correct:+d} ({100 * (cls_correct - gov_correct) / n:+.1f}pp)"
    )

    # Per-class breakdown
    print("\n--- Per-class accuracy ---")
    print(f"{'Mode':14s} {'Count':>6s} {'Governor':>10s} {'Classifier':>12s} {'Delta':>8s}")
    print("-" * 52)
    for mode in modes:
        mode_rows = [(e, g, c) for e, g, c in zip(expected, governor, classifier) if e == mode]
        if not mode_rows:
            continue
        cnt = len(mode_rows)
        gov_ok = sum(1 for e, g, c in mode_rows if e == g)
        cls_ok = sum(1 for e, g, c in mode_rows if e == c)
        delta = cls_ok - gov_ok
        print(
            f"{mode:14s} {cnt:6d} {gov_ok:4d} ({100 * gov_ok / cnt:5.1f}%) "
            f"{cls_ok:4d} ({100 * cls_ok / cnt:5.1f}%) {delta:+4d}"
        )

    # Confusion matrices
    col_width = 12
    for name, preds in [("Governor", governor), ("Classifier", classifier)]:
        print(f"\n--- {name} confusion matrix ---")
        header = f"{'predicted ->':>20s}" + "".join(f"{m:>{col_width}s}" for m in modes)
        print(header)
        print("-" * (20 + col_width * len(modes)))
        for actual_mode in modes:
            counts = Counter(p for e, p in zip(expected, preds) if e == actual_mode)
            row_vals = [counts.get(m, 0) for m in modes]
            label = f"actual {actual_mode}"
            print(f"{label:>20s}" + "".join(f"{v:>{col_width}d}" for v in row_vals))

    # Disagreement analysis
    disagree = [
        (r["case_id"], r["expected_mode"], r["governor_predicted"], r["classifier_predicted"])
        for r in rows
        if r["governor_predicted"] != r["classifier_predicted"]
    ]
    print(f"\n--- Disagreements: {len(disagree)}/{n} ({100 * len(disagree) / n:.1f}%) ---")

    # Who's right when they disagree?
    gov_right = sum(1 for _, e, g, c in disagree if g == e)
    cls_right = sum(1 for _, e, g, c in disagree if c == e)
    neither = len(disagree) - gov_right - cls_right
    print(f"  Governor right: {gov_right}  Classifier right: {cls_right}  Neither: {neither}")

    # Feature distribution (Tier 2/3 check)
    vec_scores = [r.get("mean_vector_score", 0) for r in rows]
    det_temporal = sum(1 for r in rows if r.get("detection_temporal"))
    det_aggregation = sum(1 for r in rows if r.get("detection_aggregation"))
    det_comparison = sum(1 for r in rows if r.get("detection_comparison"))
    print("\n--- Feature distribution (Tier 2/3 check) ---")
    print(
        f"  mean_vector_score: mean={np.mean(vec_scores):.3f}, std={np.std(vec_scores):.3f}, "
        f"non-zero={sum(1 for v in vec_scores if v > 0)}/{n}"
    )
    print(f"  detection_temporal: {det_temporal}/{n}")
    print(f"  detection_aggregation: {det_aggregation}/{n}")
    print(f"  detection_comparison: {det_comparison}/{n}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Full pipeline eval for governance classifier")
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA_DIR)
    parser.add_argument("--model", type=Path, default=_DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0, help="Process only first N cases (0=all)")
    parser.add_argument("--chat", type=str, default=None, help="Override chat provider")
    parser.add_argument("--embedding", type=str, default=None, help="Override embedding provider")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent workers")
    args = parser.parse_args()

    # Load config
    print("Loading fitz-ai config...")
    cfg = load_engine_config("fitz_krag")

    # Chat client
    chat_spec = args.chat or cfg.chat_smart
    chat_factory = get_chat_factory({
        "fast": cfg.chat_fast,
        "balanced": cfg.chat_balanced,
        "smart": cfg.chat_smart,
    })
    chat = chat_factory("fast")
    chat_balanced = chat_factory("balanced")
    print(f"Chat provider: {chat_spec}")

    # Embedding client
    embed_spec = args.embedding or cfg.embedding
    embedder = get_embedder(embed_spec)
    print(f"Embedding provider: {embed_spec}")

    # Detection orchestrator (uses chat_factory for LLM classification)
    detection_orchestrator = DetectionOrchestrator(chat_factory=chat_factory)
    print("Detection orchestrator ready")

    # Load classifier
    classifier = GovernanceClassifier(args.model)

    # Load cases
    print(f"Loading cases from {args.data_dir}...")
    cases = load_cases(args.data_dir)
    if args.limit > 0:
        cases = cases[: args.limit]
    print(f"Loaded {len(cases)} cases")

    # Process cases
    all_rows: list[dict[str, Any]] = []
    errors = 0
    start_time = time.time()

    if args.workers <= 1:
        # Single-threaded (simpler for debugging)
        for case in tqdm(cases, desc="Evaluating"):
            try:
                row = process_case(
                    case,
                    chat,
                    embedder,
                    detection_orchestrator,
                    classifier,
                    chat_balanced=chat_balanced,
                )
                if row:
                    all_rows.append(row)
            except Exception as e:
                errors += 1
                print(f"\n  ERROR on case {case['id']}: {e}", file=sys.stderr)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_case,
                    case,
                    chat,
                    embedder,
                    detection_orchestrator,
                    classifier,
                    chat_balanced=chat_balanced,
                ): case["id"]
                for case in cases
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
                case_id = futures[future]
                try:
                    row = future.result()
                    if row:
                        all_rows.append(row)
                except Exception as e:
                    errors += 1
                    print(f"\n  ERROR on case {case_id}: {e}", file=sys.stderr)

    elapsed = time.time() - start_time
    print(f"\nProcessed {len(all_rows)} cases in {elapsed:.1f}s ({errors} errors)")

    if not all_rows:
        print("No rows to evaluate!", file=sys.stderr)
        sys.exit(1)

    # Sort by case_id
    all_rows.sort(key=lambda r: r["case_id"])

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(all_rows[0].keys())
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Saved {len(all_rows)} rows to {args.output}")

    # Evaluation
    print_evaluation(all_rows, twostage=classifier._mode == "twostage")


if __name__ == "__main__":
    main()
