# tools/retrieval_eval/table_eval.py
"""
Evaluate table row retrieval quality against ground truth.

Tests the full TABLE pipeline in isolation:
  Analysis + Detection + Embedding (parallel)
  → Retrieval router (table search: keyword + semantic)
  → Content reading
  → Table handler (LLM SQL generation + PostgreSQL execution)

Skips: synthesis, governance, cloud cache.

Uses fitz-ai's own engine with config from ~/.fitz/config.yaml
(ollama/qwen3.5 + nomic-embed-text by default).

Usage:
    python -m tools.retrieval_eval.table_eval
    python -m tools.retrieval_eval.table_eval --category lookup
    python -m tools.retrieval_eval.table_eval --ids 1,2,3 -v
    python -m tools.retrieval_eval.table_eval --skip-ingest

Scoring:
  - table_found:   did retrieval find the correct table?
  - value_recall:  % of critical values found in SQL results
  - column_recall: % of expected columns used in SQL results
  - precision:     % of results from correct table
"""

import json
import logging
import re
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    stream=sys.stderr,
)

GROUND_TRUTH = Path(__file__).parent / "table_ground_truth.json"
CORPUS_DIR = Path(__file__).parent / "table_corpus"
RESULTS_DIR = Path(__file__).parent / "results"
COLLECTION = "table_eval"


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------


def _normalize(value: str) -> str:
    """Normalize a value for fuzzy matching."""
    return value.strip().lower().replace(",", "").replace("$", "")


def _value_in_content(value: str, content: str) -> bool:
    """Check if a value appears in content (fuzzy: handles rounding, prefixes)."""
    norm_val = _normalize(value)
    norm_content = _normalize(content)

    # Exact substring
    if norm_val in norm_content:
        return True

    # Numeric fuzzy: "3044" should match "3044.22" or "3,044.22"
    try:
        num = float(norm_val)
        # Check if any number in content is close (within 5% for aggregations)
        numbers = re.findall(r"[\d]+(?:\.[\d]+)?", norm_content)
        for n_str in numbers:
            try:
                n = float(n_str)
                if n == 0:
                    continue
                if abs(n - num) / max(abs(n), abs(num)) < 0.05:
                    return True
            except ValueError:
                continue
    except ValueError:
        pass

    return False


def _result_matches_table(meta: dict, content: str, table_file: str) -> bool:
    """Check if a result is from the expected table file."""
    table_stem = Path(table_file).stem.lower()
    file_path = str(meta.get("file_path", "")).lower()
    location = str(meta.get("location", "")).lower()
    name = str(meta.get("name", "")).lower()

    return (
        table_stem in file_path
        or table_stem in location
        or table_stem in name
        or table_stem.replace("_", " ") in name
    )


def _extract_sql_columns(content: str) -> list[str]:
    """Extract column names from SQL results in content."""
    # Look for markdown table headers: | col1 | col2 |
    match = re.search(r"\|(.+)\|", content)
    if match:
        cols = [c.strip().lower() for c in match.group(1).split("|") if c.strip()]
        # Filter out separator rows
        cols = [c for c in cols if not re.match(r"^-+$", c)]
        return cols
    return []


def score_query(read_results, query_gt: dict) -> dict:
    """Score table retrieval results for a single query."""
    table_file = query_gt["table_file"]
    critical_values = query_gt.get("critical_values", [])
    relevant_values = query_gt.get("relevant_values", [])
    expected_columns = query_gt.get("expected_columns", [])

    # Build items from read results
    items = []
    for r in read_results:
        meta = dict(r.metadata)
        meta.update({k: v for k, v in r.address.metadata.items() if k != "context_type"})
        meta["file_path"] = r.file_path
        meta["location"] = r.address.location
        items.append(
            {
                "meta": meta,
                "content": r.content or "",
                "kind": r.address.kind.value,
            }
        )

    # Check if we found the right table
    table_items = [
        it for it in items if _result_matches_table(it["meta"], it["content"], table_file)
    ]
    table_found = len(table_items) > 0

    # Check if any result has SQL execution results (table handler ran)
    sql_items = [it for it in table_items if it["meta"].get("sql_executed")]
    has_sql = len(sql_items) > 0

    # Combine all content from matching table results
    all_content = " ".join(it["content"] for it in table_items)

    # Score critical values
    critical_found = []
    critical_missed = []
    for val in critical_values:
        if _value_in_content(val, all_content):
            critical_found.append(val)
        else:
            critical_missed.append(val)

    # Score relevant values
    relevant_found = []
    for val in relevant_values:
        if _value_in_content(val, all_content):
            relevant_found.append(val)

    # Score columns
    result_columns = _extract_sql_columns(all_content)
    columns_found = []
    for col in expected_columns:
        col_lower = col.lower()
        if any(col_lower in rc or rc in col_lower for rc in result_columns):
            columns_found.append(col)

    # Calculate metrics
    value_recall = len(critical_found) / len(critical_values) if critical_values else 1.0
    total_values = critical_values + relevant_values
    total_found = critical_found + relevant_found
    total_recall = len(total_found) / len(total_values) if total_values else 1.0
    column_recall = len(columns_found) / len(expected_columns) if expected_columns else 1.0

    # Precision: fraction of results that are from the right table
    precision = len(table_items) / len(items) if items else 0.0

    return {
        "table_found": table_found,
        "has_sql": has_sql,
        "value_recall": round(value_recall, 2),
        "critical_found": critical_found,
        "critical_missed": critical_missed,
        "total_recall": round(total_recall, 2),
        "column_recall": round(column_recall, 2),
        "precision": round(precision, 2),
        "result_count": len(items),
        "table_result_count": len(table_items),
        "sql_executed": sql_items[0]["meta"].get("sql_executed", "") if sql_items else "",
    }


# ---------------------------------------------------------------------------
# Ingestion helpers
# ---------------------------------------------------------------------------


def _clean_collection(collection: str) -> None:
    """Delete stale manifest + parsed cache so point() re-indexes from scratch."""
    import shutil

    from fitz_ai.core.paths import FitzPaths

    col_dir = FitzPaths.workspace() / "collections" / collection
    if col_dir.exists():
        shutil.rmtree(col_dir)
        print(f"Cleaned collection directory: {col_dir}")


def _wait_for_indexing(engine, timeout: float = 300) -> None:
    """Wait for background indexing to complete."""
    worker = getattr(engine, "_bg_worker", None)
    if not worker:
        return
    thread = getattr(worker, "_thread", None)
    if not thread or not thread.is_alive():
        return

    print("Waiting for background indexing (table summaries + embeddings)...")
    t0 = time.monotonic()
    thread.join(timeout=timeout)
    elapsed = time.monotonic() - t0

    if thread.is_alive():
        print(f"  WARNING: Indexing still running after {timeout:.0f}s timeout")
    else:
        print(f"  Indexing complete in {elapsed:.1f}s")


def ingest_corpus(engine, corpus_dir: Path, collection: str) -> bool:
    """Ingest table corpus into a collection. Returns True on success."""
    csv_files = [f for f in corpus_dir.iterdir() if f.suffix.lower() in (".csv", ".tsv")]
    if not csv_files:
        print(f"No CSV files found in {corpus_dir}")
        print("Add CSV files to tools/retrieval_eval/table_corpus/")
        return False

    _clean_collection(collection)

    engine.load(collection)

    print(f"Ingesting {len(csv_files)} tables into collection '{collection}'...")
    for f in csv_files:
        rows = sum(1 for _ in open(f, encoding="utf-8")) - 1  # minus header
        print(f"  {f.name}: {rows} rows")

    t0 = time.monotonic()
    engine.point(source=corpus_dir, collection=collection)
    elapsed = time.monotonic() - t0
    print(f"Manifest built in {elapsed:.1f}s")

    _wait_for_indexing(engine)
    return True


# ---------------------------------------------------------------------------
# Isolated retrieval — runs retrieval + table handler, skips synthesis
# ---------------------------------------------------------------------------


def run_table_retrieval(engine, query_text: str, top_k: int = 10):
    """Run table search + SQL execution in isolation.

    Executes:
      1. Embed query (for semantic table search)
      2. Table search strategy (keyword + semantic hybrid)
      3. Content reading (table schemas)
      4. Table handler (LLM SQL generation + PostgreSQL execution)

    Skips: analysis, detection, rewriting, HyDE, agentic search, synthesis.
    This is intentional — table queries don't benefit from document-oriented
    intelligence (temporal detection, comparison analysis, etc.).
    """
    sanitized = re.sub(r"<[^>]+>", "", query_text).strip()[:500]

    # 1. Table search — hybrid (keyword + semantic).
    # Keyword search alone misses queries without table/column words (e.g., "who earns").
    # Semantic search on table summaries catches these.
    table_strategy = engine._retrieval_router._table_strategy
    if not table_strategy:
        return []

    try:
        query_vector = engine._embedder.embed(sanitized, task_type="query")
    except Exception:
        query_vector = None

    addresses = table_strategy.retrieve(sanitized, limit=top_k, query_vector=query_vector)

    if not addresses:
        return []

    # 3. Read content (table schemas)
    read_results = engine._reader.read(addresses, min(len(addresses), top_k))
    if not read_results:
        return []

    # 4. Table handler: LLM column selection + SQL generation + execution
    augmented = engine._table_handler.process(sanitized, read_results)
    return augmented


# ---------------------------------------------------------------------------
# Main eval
# ---------------------------------------------------------------------------


def run_eval(
    corpus_dir: str | Path = CORPUS_DIR,
    collection: str = COLLECTION,
    category: str | None = None,
    ids: str | None = None,
    top_k: int = 10,
    verbose: bool = True,
    skip_ingest: bool = False,
):
    """Run full table retrieval evaluation."""
    from fitz_ai.runtime import create_engine

    corpus_dir = Path(corpus_dir)
    if not corpus_dir.exists():
        print(f"Corpus directory not found: {corpus_dir}")
        return

    queries = json.loads(GROUND_TRUTH.read_text())

    if category:
        queries = [q for q in queries if q.get("category") == category]
    if ids:
        id_set = set(int(i) for i in ids.split(","))
        queries = [q for q in queries if q["id"] in id_set]

    if not queries:
        print("No queries match filters.")
        return

    # Create engine
    engine = create_engine("fitz_krag")

    # Wait for the background warmup thread to finish loading LLM models.
    # Without this, the warmup loads chat_smart (9b) while we try to use
    # chat_fast (0.6b/2b), causing model thrashing on ollama.
    import threading

    for t in threading.enumerate():
        if t.name != "MainThread" and t.daemon and t.is_alive():
            t.join(timeout=180)

    if not skip_ingest:
        if not ingest_corpus(engine, corpus_dir, collection):
            return
    else:
        print(f"Skipping ingestion, loading collection '{collection}'...")
        engine.load(collection)

    print(f"\nRunning {len(queries)} table retrieval evaluations...\n")

    results = []
    for q in queries:
        t0 = time.monotonic()
        try:
            read_results = run_table_retrieval(engine, q["query"], top_k)
            elapsed = time.monotonic() - t0

            scores = score_query(read_results, q)
            scores["id"] = q["id"]
            scores["query"] = q["query"][:60]
            scores["category"] = q.get("category", "")
            scores["time"] = round(elapsed, 1)

            is_pass = scores["value_recall"] == 1.0 and scores["table_found"]
            status = "PASS" if is_pass else "MISS"

            tbl = "TBL" if scores["table_found"] else "---"
            sql = "SQL" if scores["has_sql"] else "---"
            label = (
                f"{tbl} {sql} val={scores['value_recall']:.0%} "
                f"col={scores['column_recall']:.0%}"
            )

            print(f"  [{q['id']:2d}] {status} {label} ({elapsed:.1f}s) {q['query'][:50]}")

            if not is_pass and verbose:
                if not scores["table_found"]:
                    print(f"       TABLE NOT FOUND (got {scores['result_count']} results)")
                for missed in scores["critical_missed"]:
                    print(f"       MISSED VALUE: {missed}")
                if scores["sql_executed"]:
                    print(f"       SQL: {scores['sql_executed'][:80]}")

            results.append(scores)
        except Exception as e:
            import traceback

            elapsed = time.monotonic() - t0
            print(f"  [{q['id']:2d}] FAILED ({elapsed:.1f}s): {e}")
            if verbose:
                traceback.print_exc()
            results.append({"id": q["id"], "error": str(e)})

    # Summary
    scored = [r for r in results if "error" not in r]
    if not scored:
        print("\nAll queries failed.")
        return

    tables_found = sum(1 for r in scored if r["table_found"])
    has_sql = sum(1 for r in scored if r["has_sql"])
    avg_val = sum(r["value_recall"] for r in scored) / len(scored)
    avg_total = sum(r["total_recall"] for r in scored) / len(scored)
    avg_col = sum(r["column_recall"] for r in scored) / len(scored)
    avg_prec = sum(r["precision"] for r in scored) / len(scored)
    perfect = sum(1 for r in scored if r["value_recall"] == 1.0 and r["table_found"])
    avg_time = sum(r["time"] for r in scored) / len(scored)
    total_time = sum(r["time"] for r in scored)

    print(f"\n{'=' * 60}")
    print(f"RESULTS ({len(scored)} queries, {total_time:.0f}s total)")
    print(f"{'=' * 60}")
    print(f"Table found:     {tables_found}/{len(scored)}")
    print(f"SQL executed:    {has_sql}/{len(scored)}")
    print(f"Value recall:    {avg_val:.0%} avg ({perfect}/{len(scored)} perfect)")
    print(f"Total recall:    {avg_total:.0%} avg")
    print(f"Column recall:   {avg_col:.0%} avg")
    print(f"Precision:       {avg_prec:.0%} avg")
    print(f"Avg time:        {avg_time:.1f}s per query")

    # By category
    categories = sorted(set(r.get("category", "uncategorized") for r in scored))
    if len(categories) > 1:
        print("\nBy category:")
        for cat in categories:
            cat_results = [r for r in scored if r.get("category") == cat]
            if cat_results:
                cat_val = sum(r["value_recall"] for r in cat_results) / len(cat_results)
                cat_perfect = sum(
                    1 for r in cat_results if r["value_recall"] == 1.0 and r["table_found"]
                )
                cat_time = sum(r["time"] for r in cat_results) / len(cat_results)
                print(
                    f"  {cat:15s} val={cat_val:.0%} "
                    f"({cat_perfect}/{len(cat_results)} perfect) "
                    f"avg={cat_time:.1f}s"
                )

    # Most-missed values
    all_missed = []
    for r in scored:
        for val in r.get("critical_missed", []):
            all_missed.append(f"{val} (q{r['id']})")
    if all_missed:
        print(f"\nMissed critical values ({len(all_missed)}):")
        for val in all_missed[:15]:
            print(f"  {val}")

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"table_eval_{ts}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Table row retrieval eval")
    parser.add_argument("--corpus-dir", default=str(CORPUS_DIR))
    parser.add_argument("--collection", default=COLLECTION)
    parser.add_argument("--category", default=None)
    parser.add_argument("--ids", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("-v", "--verbose", action="store_true", default=True)
    parser.add_argument("--skip-ingest", action="store_true")
    args = parser.parse_args()

    run_eval(
        corpus_dir=args.corpus_dir,
        collection=args.collection,
        category=args.category,
        ids=args.ids,
        top_k=args.top_k,
        verbose=args.verbose,
        skip_ingest=args.skip_ingest,
    )
