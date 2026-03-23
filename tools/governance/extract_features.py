# tools/governance/extract_features.py
"""
Feature extraction for governance classifier training.

Loads labeled cases from fitz-gov, runs each constraint individually
(bypassing staged short-circuit to get all features), extracts features
via the feature_extractor, and saves to CSV.

Usage:
    python -m tools.governance.extract_features --embedding ollama --chat ollama
    python -m tools.governance.extract_features --chat cohere --embedding cohere --workers 3
    python -m tools.governance.extract_features --limit 10  # Quick test
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

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

# Default path to fitz-gov tier1 data (sibling repo)
_DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent / "fitz-gov" / "data" / "tier1_core"
)
_OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "features.csv"

# Typed defaults for None feature values
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
    "av_citation_found",
    "av_contradicting_citations",
    "has_abstain_signal",
    "has_disputed_signal",
    "has_qualified_signal",
    "has_any_denial",
    "query_has_comparison_words",
    "has_distinct_years",
    "detection_temporal",
    "detection_aggregation",
    "detection_comparison",
    "detection_boost_recency",
    "detection_boost_authority",
    "detection_needs_rewriting",
    "has_cross_chunk_divergence",
    "has_within_chunk_divergence",
}

_NUMERIC_FEATURES = {
    "ie_max_similarity",
    "ca_pairs_checked",
    "ca_relevance_filtered_count",
    "av_citation_quality",
    "av_citations_count",
    "num_constraints_fired",
    "num_strong_denials",
    "query_word_count",
    "num_chunks",
    "num_unique_sources",
    "mean_vector_score",
    "max_vector_score",
    "min_vector_score",
    "std_vector_score",
    "score_spread",
    "vocab_overlap_ratio",
    "hedge_density",
    "year_count",
    "cross_chunk_num_conflicts",
    "cross_chunk_max_divergence",
    "within_chunk_num_conflicts",
    "within_chunk_max_divergence",
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
    "dominant_content_type",
}


def load_cases(data_dir: Path) -> list[dict[str, Any]]:
    """Load all cases from tier1_core JSON files."""
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


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
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
    """Compute embeddings and set vector_score on each chunk (in-place).

    Simulates the vector_score that production retrieval provides from pgvector.
    """
    try:
        texts = [query] + [c.content for c in chunks]
        embeddings = embedder.embed_batch(texts)
        query_emb = embeddings[0]
        for i, chunk in enumerate(chunks):
            chunk.metadata["vector_score"] = _cosine_similarity(query_emb, embeddings[i + 1])
    except Exception as e:
        print(f"  WARNING: Embedding failed: {e}", file=sys.stderr)
        for chunk in chunks:
            chunk.metadata["vector_score"] = 0.0


def make_constraints(chat, chat_balanced=None, embedder=None) -> list:
    """Create a fresh set of constraints (one set per worker thread).

    Mirrors create_default_constraints() but adds adaptive=True for conflict detection.
    """
    return [
        InsufficientEvidenceConstraint(chat=chat, embedder=embedder),
        CausalAttributionConstraint(embedder=embedder),
        SpecificInfoTypeConstraint(embedder=embedder),
        ConflictAwareConstraint(chat=chat, adaptive=True, embedder=embedder),
        AnswerVerificationConstraint(chat=chat),
    ]


def run_constraints_individually(
    query: str,
    chunks: Sequence[Chunk],
    constraints: list,
) -> dict[str, ConstraintResult]:
    """
    Run each constraint independently (no staged short-circuit).

    Returns map of constraint_name -> ConstraintResult with injected metadata.
    """
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


def get_governor_prediction(result_map: dict[str, ConstraintResult]) -> str:
    """Run AnswerGovernor.decide() on constraint results to get baseline prediction."""
    governor = AnswerGovernor()
    decision = governor.decide(list(result_map.values()))
    return decision.mode.value


# Thread-local storage for constraint instances (one set per thread)
_thread_local = threading.local()


def process_case(
    case: dict[str, Any],
    chat,
    chat_balanced=None,
    embedder=None,
    detection_orchestrator: DetectionOrchestrator | None = None,
) -> dict[str, Any] | None:
    """Process a single case with full feature enrichment.

    Matches production behavior: computes vector_score (Tier 2) and
    detection summary (Tier 3) so training features match inference.
    """
    # Each thread gets its own constraint instances to avoid shared state
    if not hasattr(_thread_local, "constraints"):
        _thread_local.constraints = make_constraints(
            chat, chat_balanced=chat_balanced, embedder=embedder
        )

    case_id = case["id"]
    query = case["query"]
    chunks = case_to_chunks(case)

    # Tier 2: compute vector_score (matches production retrieval behavior)
    if embedder is not None:
        enrich_chunks_with_embeddings(query, chunks, embedder)

    # Tier 3: detection summary
    detection_summary = None
    if detection_orchestrator is not None:
        try:
            detection_summary = detection_orchestrator.detect_for_retrieval(query)
        except Exception as e:
            print(f"  WARNING: Detection failed for {case_id}: {e}", file=sys.stderr)

    # Tier 1: run constraints individually
    result_map = run_constraints_individually(query, chunks, _thread_local.constraints)
    features = extract_features(query, chunks, result_map, detection_summary)
    features = fill_defaults(features)
    governor_predicted = get_governor_prediction(result_map)

    return {
        "case_id": case_id,
        "expected_mode": case["expected_mode"],
        "governor_predicted": governor_predicted,
        "difficulty": case.get("difficulty", "unknown"),
        "subcategory": case.get("subcategory", "unknown"),
        **features,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract governance classifier features from fitz-gov cases"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=_DEFAULT_DATA_DIR, help="Path to tier1_core JSON files"
    )
    parser.add_argument("--output", type=Path, default=_OUTPUT_PATH, help="Output CSV path")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N cases (0 = all)")
    parser.add_argument(
        "--chat", type=str, default=None, help="Override chat provider (e.g. 'cohere', 'anthropic')"
    )
    parser.add_argument(
        "--embedding", type=str, default=None, help="Embedding provider (e.g. 'ollama', 'cohere')"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent workers (default: 1)",
    )
    args = parser.parse_args()

    # Load config and create chat client
    print("Loading fitz-ai config...")
    cfg = load_engine_config("fitz_krag")

    # Chat provider: --chat overrides ALL tiers (fast/balanced/smart)
    if args.chat:
        chat_spec = args.chat
        tier_specs = {"fast": chat_spec, "balanced": chat_spec, "smart": chat_spec}
    else:
        chat_spec = cfg.chat_smart
        tier_specs = {
            "fast": cfg.chat_fast,
            "balanced": cfg.chat_balanced,
            "smart": cfg.chat_smart,
        }
    chat_factory = get_chat_factory(tier_specs)
    chat = chat_factory("fast")
    chat_balanced = chat_factory("balanced")
    model_name = getattr(chat, "_model", "unknown")
    balanced_model = getattr(chat_balanced, "_model", "unknown")
    print(f"Using chat provider: {chat_spec} (fast: {model_name}, balanced: {balanced_model})")

    # Embedding client (Tier 2: vector_score)
    embedder = None
    embed_spec = args.embedding or cfg.embedding
    if embed_spec:
        embedder = get_embedder(embed_spec)
        print(f"Embedding provider: {embed_spec}")
    else:
        print("WARNING: No embedding provider — vector_score will be 0 (use --embedding)")

    # Detection orchestrator (Tier 3)
    detection_orchestrator = DetectionOrchestrator(chat_factory=chat_factory)
    print("Detection orchestrator ready")
    print(f"Concurrency: {args.workers} workers")

    # Load cases
    print(f"Loading cases from {args.data_dir}...")
    cases = load_cases(args.data_dir)
    if args.limit > 0:
        cases = cases[: args.limit]
    print(f"Loaded {len(cases)} cases")

    # SQLite checkpoint store — each row committed individually, crash-safe.
    # CSV is only written as final export after all cases complete.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    db_path = args.output.with_suffix(".db")
    db = sqlite3.connect(str(db_path), check_same_thread=False)
    db.execute("CREATE TABLE IF NOT EXISTS features (case_id TEXT PRIMARY KEY, data TEXT)")
    db.commit()

    completed_ids: set[str] = set()
    for (cid,) in db.execute("SELECT case_id FROM features").fetchall():
        completed_ids.add(cid)
    if completed_ids:
        print(f"Resuming: {len(completed_ids)} cases already completed, skipping them")

    remaining_cases = [c for c in cases if c["id"] not in completed_ids]
    print(f"Processing {len(remaining_cases)} remaining cases ({len(completed_ids)} skipped)")

    errors = 0
    written = 0
    start_time = time.time()
    write_lock = threading.Lock()

    def _write_row(row: dict[str, Any]) -> None:
        nonlocal written
        case_id = row["case_id"]
        with write_lock:
            db.execute(
                "INSERT OR REPLACE INTO features (case_id, data) VALUES (?, ?)",
                (case_id, json.dumps(row)),
            )
            db.commit()
            written += 1

    if args.workers <= 1:
        for case in tqdm(remaining_cases, desc="Extracting features"):
            try:
                row = process_case(case, chat, chat_balanced, embedder, detection_orchestrator)
                if row:
                    _write_row(row)
            except Exception as e:
                errors += 1
                print(f"\n  ERROR on case {case['id']}: {e}", file=sys.stderr)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_case, case, chat, chat_balanced, embedder, detection_orchestrator
                ): case["id"]
                for case in remaining_cases
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Extracting features"
            ):
                case_id = futures[future]
                try:
                    row = future.result()
                    if row:
                        _write_row(row)
                except Exception as e:
                    errors += 1
                    print(f"\n  ERROR on case {case_id}: {e}", file=sys.stderr)

    elapsed = time.time() - start_time
    total_in_db = db.execute("SELECT COUNT(*) FROM features").fetchone()[0]
    print(f"\nProcessed {written} cases in {elapsed:.1f}s ({errors} errors)")
    print(f"Throughput: {written/elapsed:.1f} cases/sec")
    print(f"Total rows in DB: {total_in_db}")

    if total_in_db == 0:
        print("No rows written!", file=sys.stderr)
        db.close()
        sys.exit(1)

    # Export SQLite → CSV (sorted, deterministic)
    all_rows = []
    for (data,) in db.execute("SELECT data FROM features ORDER BY case_id"):
        all_rows.append(json.loads(data))
    db.close()

    fieldnames = list(all_rows[0].keys())
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Saved {len(all_rows)} rows x {len(fieldnames)} columns to {args.output}")

    # Quick summary
    from collections import Counter

    expected_dist = Counter(r["expected_mode"] for r in all_rows)
    governor_dist = Counter(r["governor_predicted"] for r in all_rows)
    governor_correct = sum(1 for r in all_rows if r["expected_mode"] == r["governor_predicted"])

    print(f"\nExpected mode distribution: {dict(expected_dist)}")
    print(f"Governor prediction distribution: {dict(governor_dist)}")
    print(
        f"Governor baseline accuracy: {governor_correct}/{len(all_rows)} ({100*governor_correct/len(all_rows):.1f}%)"
    )


if __name__ == "__main__":
    main()
