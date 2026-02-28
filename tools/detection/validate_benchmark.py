# tools/detection/validate_benchmark.py
"""
Validate fitz-gov tier1_core benchmark JSON files.

Checks:
- Files are JSON-parseable
- Every case has required fields: id, query, reasoning_type
- Counts cases with detection_labels vs. without
- Prints label distribution per file

Usage:
    python -m tools.detection.validate_benchmark [--fitzgov-dir PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent
_DATA_DIR = (_SCRIPT_DIR / "../../../../fitz-gov/data/tier1_core").resolve()

_REQUIRED_FIELDS = ("id", "query", "reasoning_type")


def validate_file(path: Path) -> tuple[bool, dict]:
    """Validate a single JSON file. Returns (ok, stats)."""
    stats: dict = {
        "file": path.name,
        "total": 0,
        "missing_fields": [],
        "labeled": 0,
        "label_counts": {},
        "errors": [],
    }

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        stats["errors"].append(f"JSON parse error: {exc}")
        return False, stats

    cases = payload.get("cases", [])
    stats["total"] = len(cases)

    for case in cases:
        case_id = case.get("id", "<no-id>")

        # Check required fields
        for field in _REQUIRED_FIELDS:
            if field not in case or not case[field]:
                stats["missing_fields"].append(f"{case_id}.{field}")

        # Count detection_labels
        labels = case.get("detection_labels", [])
        if labels:
            stats["labeled"] += 1
            for label in labels:
                stats["label_counts"][label] = stats["label_counts"].get(label, 0) + 1

    ok = not stats["errors"] and not stats["missing_fields"]
    return ok, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fitzgov-dir", type=Path, default=_DATA_DIR)
    args = parser.parse_args()

    data_dir = args.fitzgov_dir.resolve()
    files = sorted(data_dir.glob("*.json"))
    if not files:
        print(f"ERROR: No JSON files found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    all_ok = True
    grand_total = 0
    grand_labeled = 0
    grand_counts: dict[str, int] = {}

    for path in files:
        ok, stats = validate_file(path)
        all_ok = all_ok and ok

        print(f"\n{'OK' if ok else 'FAIL'}  {stats['file']}")
        print(f"  Cases: {stats['total']}  |  Labeled: {stats['labeled']}")

        if stats["label_counts"]:
            for label, count in sorted(stats["label_counts"].items()):
                print(f"    {label}: {count}")
                grand_counts[label] = grand_counts.get(label, 0) + count

        if stats["missing_fields"]:
            print(f"  MISSING FIELDS ({len(stats['missing_fields'])}):")
            for entry in stats["missing_fields"][:10]:
                print(f"    {entry}")
            if len(stats["missing_fields"]) > 10:
                print(f"    ... and {len(stats['missing_fields']) - 10} more")

        if stats["errors"]:
            for err in stats["errors"]:
                print(f"  ERROR: {err}")

        grand_total += stats["total"]
        grand_labeled += stats["labeled"]

    print(f"\n{'=' * 50}")
    print(f"TOTAL  {grand_total} cases  |  {grand_labeled} labeled")
    for label, count in sorted(grand_counts.items()):
        print(f"  {label}: {count}")

    if not all_ok:
        print("\nVALIDATION FAILED", file=sys.stderr)
        sys.exit(1)
    else:
        print("\nAll files valid.")


if __name__ == "__main__":
    main()
