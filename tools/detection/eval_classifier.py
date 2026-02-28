# tools/detection/eval_classifier.py
"""
Benchmark evaluation for DetectionClassifier.

Loads all tier1_core JSON files from fitz-gov, runs DetectionClassifier.predict()
on every case, and computes per-label precision / recall / F1 against
detection_labels ground truth.

Cases without detection_labels are treated as all-negative.

Usage:
    python -m tools.detection.eval_classifier [--fitzgov-dir PATH]

Exit codes:
    0  — all label recalls >= 0.70
    1  — one or more label recalls < 0.70 (soft guard)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent
_DATA_DIR = (_SCRIPT_DIR / "../../../../fitz-gov/data/tier1_core").resolve()

_LABELS = ["temporal", "comparison", "aggregation", "freshness"]
_RECALL_GUARD = 0.70


def load_cases(data_dir: Path) -> list[dict]:
    """Load all cases from tier1_core JSON files."""
    files = sorted(data_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {data_dir}")

    cases = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for case in payload.get("cases", []):
            cases.append(
                {
                    "id": case.get("id", ""),
                    "query": case.get("query", ""),
                    "reasoning_type": case.get("reasoning_type", ""),
                    "query_type": case.get("query_type", ""),
                    "detection_labels": case.get("detection_labels", []),
                }
            )
    return cases


def _binary_metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return recall, precision, f1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fitzgov-dir", type=Path, default=_DATA_DIR)
    args = parser.parse_args()

    data_dir = args.fitzgov_dir.resolve()
    cases = load_cases(data_dir)
    print(f"Loaded {len(cases)} cases from {data_dir}")

    # Import here to ensure the installed package is used
    from fitz_ai.retrieval.detection.classifier import DetectionClassifier
    from fitz_ai.retrieval.detection.protocol import DetectionCategory

    clf = DetectionClassifier()
    if not clf.available:
        print("ERROR: DetectionClassifier model not available (missing artifact).", file=sys.stderr)
        sys.exit(1)

    _label_to_category = {
        "temporal": DetectionCategory.TEMPORAL,
        "comparison": DetectionCategory.COMPARISON,
        "aggregation": DetectionCategory.AGGREGATION,
        "freshness": DetectionCategory.FRESHNESS,
    }

    # Accumulate confusion matrix per label
    counters: dict[str, dict[str, int]] = {
        label: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for label in _LABELS
    }
    none_count = 0

    for case in cases:
        query = case["query"]
        # Ground truth: detection_labels for aggregation/freshness;
        # reasoning_type/query_type for temporal/comparison (same as training)
        detection_labels = set(case["detection_labels"])
        gt_labels = set(detection_labels)
        if case["reasoning_type"] == "temporal" or case["query_type"] == "when":
            gt_labels.add("temporal")
        if case["reasoning_type"] == "comparative" or case["query_type"] == "compare":
            gt_labels.add("comparison")

        result = clf.predict(query)
        if result is None:
            none_count += 1
            # Treat as all-positive (fail-open): no false negatives, but may have false positives
            result = set(_label_to_category.values())

        for label in _LABELS:
            category = _label_to_category[label]
            is_gt = label in gt_labels
            is_pred = category in result

            if is_gt and is_pred:
                counters[label]["tp"] += 1
            elif not is_gt and is_pred:
                counters[label]["fp"] += 1
            elif is_gt and not is_pred:
                counters[label]["fn"] += 1
            else:
                counters[label]["tn"] += 1

    print(f"\nFail-open (None) predictions: {none_count}")
    print("\n--- Per-label metrics ---")
    print(f"{'label':<14} {'recall':>8} {'precision':>10} {'f1':>8} {'tp':>6} {'fp':>6} {'fn':>6}")
    print("-" * 60)

    any_below_guard = False
    for label in _LABELS:
        c = counters[label]
        recall, precision, f1 = _binary_metrics(c["tp"], c["fp"], c["fn"])
        flag = " !" if recall < _RECALL_GUARD else ""
        print(
            f"{label:<14} {recall:>8.3f} {precision:>10.3f} {f1:>8.3f}"
            f" {c['tp']:>6} {c['fp']:>6} {c['fn']:>6}{flag}"
        )
        if recall < _RECALL_GUARD:
            any_below_guard = True

    if any_below_guard:
        print(
            f"\nWARNING: One or more labels below recall guard ({_RECALL_GUARD:.0%})",
            file=sys.stderr,
        )
        sys.exit(1)
    else:
        print(f"\nAll labels at or above recall guard ({_RECALL_GUARD:.0%}).")


if __name__ == "__main__":
    main()
