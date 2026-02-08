# tools/governance/extract_features.py
"""
Feature extraction for governance classifier training.

Loads labeled cases from fitz-gov, runs each constraint individually
(bypassing staged short-circuit to get all features), extracts features
via the feature_extractor, and saves to CSV.

Usage:
    python -m tools.governance.extract_features
    python -m tools.governance.extract_features --chat cohere --workers 100
    python -m tools.governance.extract_features --limit 10  # Quick test
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

from tqdm import tqdm

from fitz_ai.config import load_engine_config
from fitz_ai.core.chunk import Chunk
from fitz_ai.core.governance import AnswerGovernor
from fitz_ai.core.guardrails.base import ConstraintResult
from fitz_ai.core.guardrails.feature_extractor import extract_features
from fitz_ai.core.guardrails.plugins.answer_verification import AnswerVerificationConstraint
from fitz_ai.core.guardrails.plugins.causal_attribution import CausalAttributionConstraint
from fitz_ai.core.guardrails.plugins.conflict_aware import ConflictAwareConstraint
from fitz_ai.core.guardrails.plugins.insufficient_evidence import InsufficientEvidenceConstraint
from fitz_ai.core.guardrails.plugins.specific_info_type import SpecificInfoTypeConstraint
from fitz_ai.llm import get_chat_factory

# Default path to fitz-gov tier1 data (sibling repo)
_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "fitz-gov" / "data" / "tier1_core"
_OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "features.csv"

# Typed defaults for None feature values
_BOOL_FEATURES = {
    "ie_fired", "ie_entity_match_found", "ie_primary_match_found",
    "ie_critical_match_found", "ie_has_matching_aspect", "ie_has_conflicting_aspect",
    "ca_fired", "ca_numerical_variance_detected", "ca_is_uncertainty_query",
    "caa_fired", "caa_has_causal_evidence", "caa_has_predictive_evidence",
    "sit_fired", "sit_entity_mismatch", "sit_has_specific_info",
    "av_fired",
    "has_abstain_signal", "has_disputed_signal", "has_qualified_signal",
    "detection_temporal", "detection_aggregation", "detection_comparison",
    "detection_boost_recency", "detection_boost_authority", "detection_needs_rewriting",
}

_NUMERIC_FEATURES = {
    "ie_max_similarity", "ie_summary_overlap",
    "ca_skipped_hedged_pairs", "ca_pairs_checked",
    "ca_relevance_filtered_count",
    "av_jury_votes_no",
    "num_constraints_fired",
    "query_word_count", "num_chunks", "num_unique_sources",
    "mean_vector_score", "std_vector_score", "score_spread",
    "vocab_overlap_ratio",
}

_STRING_FEATURES = {
    "ie_signal", "ie_query_aspect", "ca_signal",
    "ca_first_evidence_char", "ca_evidence_characters",
    "caa_query_type", "sit_info_type_requested",
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


def make_constraints(chat) -> list:
    """Create a fresh set of constraints (one set per worker thread)."""
    return [
        InsufficientEvidenceConstraint(chat=chat),
        CausalAttributionConstraint(),
        SpecificInfoTypeConstraint(),
        ConflictAwareConstraint(chat=chat),
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


def process_case(case: dict[str, Any], chat) -> dict[str, Any] | None:
    """Process a single case. Thread-safe via thread-local constraints."""
    # Each thread gets its own constraint instances to avoid shared state
    if not hasattr(_thread_local, "constraints"):
        _thread_local.constraints = make_constraints(chat)

    case_id = case["id"]
    query = case["query"]
    chunks = case_to_chunks(case)

    result_map = run_constraints_individually(query, chunks, _thread_local.constraints)
    features = extract_features(query, chunks, result_map, detection_summary=None)
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
    parser = argparse.ArgumentParser(description="Extract governance classifier features from fitz-gov cases")
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA_DIR, help="Path to tier1_core JSON files")
    parser.add_argument("--output", type=Path, default=_OUTPUT_PATH, help="Output CSV path")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N cases (0 = all)")
    parser.add_argument("--chat", type=str, default=None, help="Override chat provider (e.g. 'cohere', 'anthropic')")
    parser.add_argument("--workers", type=int, default=3, help="Concurrent workers (default: 3, each makes ~7 LLM calls)")
    args = parser.parse_args()

    # Load config and create chat client
    print("Loading fitz-ai config...")
    cfg = load_engine_config("fitz_rag")

    chat_spec = args.chat or cfg.chat
    chat_config = {k: v for k, v in cfg.chat_kwargs.model_dump().items() if v is not None}
    chat_factory = get_chat_factory(chat_spec, config=chat_config)
    chat = chat_factory("fast")
    model_name = getattr(chat, "_model", "unknown")
    print(f"Using chat provider: {chat_spec} (model: {model_name})")
    print(f"Concurrency: {args.workers} workers")

    # Load cases
    print(f"Loading cases from {args.data_dir}...")
    cases = load_cases(args.data_dir)
    if args.limit > 0:
        cases = cases[: args.limit]
    print(f"Loaded {len(cases)} cases")

    # Process cases concurrently
    all_rows: list[dict[str, Any]] = []
    errors = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_case, case, chat): case["id"] for case in cases}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting features"):
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
    print(f"Throughput: {len(all_rows)/elapsed:.1f} cases/sec")

    if not all_rows:
        print("No rows to write!", file=sys.stderr)
        sys.exit(1)

    # Sort by case_id for deterministic output
    all_rows.sort(key=lambda r: r["case_id"])

    # Write CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
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
    print(f"Governor baseline accuracy: {governor_correct}/{len(all_rows)} ({100*governor_correct/len(all_rows):.1f}%)")


if __name__ == "__main__":
    main()
