# tools/retrieval_eval/eval.py
"""
Evaluate retrieval quality against ground truth.

Uses fitz-ai's own ChatFactory — no fitz-graveyard dependency.

Usage:
    python tools/retrieval_eval/eval.py --source-dir .
    python tools/retrieval_eval/eval.py --source-dir . --category retrieval
    python tools/retrieval_eval/eval.py --source-dir . --ids 1,2,3 -v

Or run eval_run.py from PyCharm's Run button.
"""

import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    stream=sys.stderr,
)

GROUND_TRUTH = Path(__file__).parent / "ground_truth.json"
RESULTS_DIR = Path(__file__).parent / "results"


def run_retrieval(source_dir: str, query: str, provider: str, model: str, base_url: str) -> list[str]:
    """Run retrieval pipeline and return selected file paths."""
    from fitz_ai.code import CodeRetriever
    from fitz_ai.llm.factory import get_chat_factory

    # Provider spec format: "lmstudio/model-name" or just "lmstudio"
    spec = f"{provider}/{model}" if model else provider
    config = {}
    if base_url:
        config["base_url"] = base_url

    chat_factory = get_chat_factory(spec, config)

    retriever = CodeRetriever(
        source_dir=source_dir,
        chat_factory=chat_factory,
        llm_tier="smart",
    )

    results = retriever.retrieve(query)
    return [r.file_path for r in results]


def score(retrieved: list[str], critical: list[str], relevant: list[str]) -> dict:
    """Score retrieval against ground truth."""
    retrieved_set = set(retrieved)
    critical_set = set(critical)
    relevant_set = set(relevant)
    all_gt = critical_set | relevant_set

    critical_found = critical_set & retrieved_set
    all_found = all_gt & retrieved_set

    return {
        "critical_recall": round(len(critical_found) / len(critical_set), 2) if critical_set else 1.0,
        "critical_found": sorted(critical_found),
        "critical_missed": sorted(critical_set - retrieved_set),
        "total_recall": round(len(all_found) / len(all_gt), 2) if all_gt else 1.0,
        "relevant_found": sorted(relevant_set & retrieved_set),
        "relevant_missed": sorted(relevant_set - retrieved_set),
        "precision": round(len(all_found) / len(retrieved_set), 2) if retrieved_set else 0.0,
        "retrieved_count": len(retrieved),
    }


def run_eval(
    source_dir: str,
    provider: str = "lmstudio",
    model: str = "qwen3.5-4b",
    base_url: str = "http://localhost:1234/v1",
    category: str | None = None,
    ids: str | None = None,
    verbose: bool = True,
):
    """Run full evaluation."""
    queries = json.loads(GROUND_TRUTH.read_text())

    if category:
        queries = [q for q in queries if q["category"] == category]
    if ids:
        id_set = set(int(i) for i in ids.split(","))
        queries = [q for q in queries if q["id"] in id_set]

    if not queries:
        print("No queries match filters.")
        return

    print(f"Running {len(queries)} retrieval evaluations...\n")

    results = []
    for q in queries:
        t0 = time.monotonic()
        try:
            retrieved = run_retrieval(source_dir, q["query"], provider, model, base_url)
        except Exception as e:
            print(f"  [{q['id']:2d}] FAILED: {e}")
            results.append({"id": q["id"], "error": str(e)})
            continue
        elapsed = time.monotonic() - t0

        s = score(retrieved, q["critical_files"], q.get("relevant_files", []))
        s["id"] = q["id"]
        s["query"] = q["query"][:60]
        s["category"] = q["category"]
        s["elapsed_s"] = round(elapsed, 1)
        results.append(s)

        status = "PASS" if s["critical_recall"] == 1.0 else "MISS"
        print(
            f"  [{q['id']:2d}] {status} "
            f"crit={s['critical_recall']:.0%} "
            f"total={s['total_recall']:.0%} "
            f"({s['elapsed_s']}s) "
            f"{q['query'][:50]}"
        )
        if verbose and s["critical_missed"]:
            for m in s["critical_missed"]:
                print(f"       MISSED: {m}")

    # Summary
    valid = [r for r in results if "error" not in r]
    if not valid:
        print("\nNo successful runs.")
        return

    avg_critical = sum(r["critical_recall"] for r in valid) / len(valid)
    avg_total = sum(r["total_recall"] for r in valid) / len(valid)
    perfect = sum(1 for r in valid if r["critical_recall"] == 1.0)
    avg_time = sum(r["elapsed_s"] for r in valid) / len(valid)

    print(f"\n{'='*60}")
    print(f"RESULTS ({len(valid)} queries)")
    print(f"{'='*60}")
    print(f"Critical recall:  {avg_critical:.0%} avg ({perfect}/{len(valid)} perfect)")
    print(f"Total recall:     {avg_total:.0%} avg")
    print(f"Avg time:         {avg_time:.1f}s per query")

    # By category
    categories = sorted(set(r["category"] for r in valid))
    if len(categories) > 1:
        print(f"\nBy category:")
        for cat in categories:
            cat_results = [r for r in valid if r["category"] == cat]
            cat_crit = sum(r["critical_recall"] for r in cat_results) / len(cat_results)
            cat_perfect = sum(1 for r in cat_results if r["critical_recall"] == 1.0)
            print(f"  {cat:20s} crit={cat_crit:.0%} ({cat_perfect}/{len(cat_results)} perfect)")

    # Most-missed files
    miss_count: dict[str, int] = {}
    for r in valid:
        for f in r.get("critical_missed", []):
            miss_count[f] = miss_count.get(f, 0) + 1
    if miss_count:
        print(f"\nMost-missed critical files:")
        for f, count in sorted(miss_count.items(), key=lambda x: -x[1])[:10]:
            print(f"  {count}x {f}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"eval_{ts}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out}")
