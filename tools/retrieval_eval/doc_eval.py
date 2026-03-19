# tools/retrieval_eval/doc_eval.py
"""
Evaluate document retrieval quality against ground truth.

Ingests test corpus, runs queries, scores retrieved provenance
against expected sections/tables/pages.

Scoring:
  - critical_recall: % of critical sections found in provenance
  - relevant_recall: % of all expected sections found
  - keyword_hit:     % of expected keywords present in retrieved excerpts
  - precision:       % of retrieved provenance that matches any expected section

A provenance item "matches" an expected section if:
  1. Document filename matches (substring)
  2. AND either:
     a. Heading matches (case-insensitive substring), OR
     b. Page number is within +/-1 of expected
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

GROUND_TRUTH = Path(__file__).parent / "doc_ground_truth.json"
CORPUS_DIR = Path(__file__).parent / "test_corpus"
RESULTS_DIR = Path(__file__).parent / "results"
COLLECTION = "doc_eval"


def _provenance_matches_section(prov_meta: dict, prov_excerpt: str, section: dict) -> bool:
    """Check if a provenance item matches an expected section."""
    # Heading match (fuzzy substring)
    heading = section.get("heading", "")
    prov_title = str(prov_meta.get("title", ""))
    heading_match = heading and heading.lower() in prov_title.lower()

    # Page match (within +/-1)
    expected_page = section.get("page")
    prov_page = prov_meta.get("page_number") or prov_meta.get("page_start")
    page_match = (
        expected_page is not None
        and prov_page is not None
        and abs(int(prov_page) - int(expected_page)) <= 1
    )

    return heading_match or page_match


def _check_keywords(excerpts: list[str], keywords: list[str]) -> float:
    """Check what fraction of keywords appear in any excerpt."""
    if not keywords:
        return 1.0
    combined = " ".join(excerpts).lower()
    found = sum(1 for kw in keywords if kw.lower() in combined)
    return found / len(keywords)


def _provenance_matches_document(prov, document: str) -> bool:
    """Check if provenance is from the expected document."""
    source = str(getattr(prov, "source_id", "") or "")
    meta = getattr(prov, "metadata", {}) or {}
    title = str(meta.get("title", ""))
    file_path = str(meta.get("file_path", "") or meta.get("source_file", ""))

    doc_lower = document.lower()
    return (
        doc_lower in source.lower()
        or doc_lower in title.lower()
        or doc_lower in file_path.lower()
    )


def score_query(provenance_list, query_gt: dict) -> dict:
    """Score retrieval results for a single query."""
    document = query_gt["document"]
    critical = query_gt.get("critical_sections", [])
    relevant = query_gt.get("relevant_sections", [])
    all_expected = critical + relevant

    # Filter provenance to matching document
    doc_provs = []
    for prov in provenance_list:
        if _provenance_matches_document(prov, document):
            doc_provs.append(prov)

    # Extract metadata and excerpts
    prov_metas = []
    prov_excerpts = []
    for prov in doc_provs:
        meta = getattr(prov, "metadata", {}) or {}
        excerpt = getattr(prov, "excerpt", "") or ""
        prov_metas.append(meta)
        prov_excerpts.append(excerpt)

    # Score critical sections
    critical_found = []
    critical_missed = []
    for section in critical:
        found = any(
            _provenance_matches_section(meta, exc, section)
            for meta, exc in zip(prov_metas, prov_excerpts)
        )
        if found:
            critical_found.append(section.get("heading", f"page {section.get('page')}"))
        else:
            critical_missed.append(section.get("heading", f"page {section.get('page')}"))

    # Score all sections (critical + relevant)
    all_found = []
    for section in all_expected:
        found = any(
            _provenance_matches_section(meta, exc, section)
            for meta, exc in zip(prov_metas, prov_excerpts)
        )
        if found:
            all_found.append(section.get("heading", f"page {section.get('page')}"))

    # Keyword check across all critical sections
    all_keywords = []
    for section in critical:
        all_keywords.extend(section.get("keywords", []))
    keyword_hit = _check_keywords(prov_excerpts, all_keywords)

    # Precision: how many provenance items matched any expected section
    matched_provs = 0
    for meta, exc in zip(prov_metas, prov_excerpts):
        if any(_provenance_matches_section(meta, exc, s) for s in all_expected):
            matched_provs += 1

    critical_recall = len(critical_found) / len(critical) if critical else 1.0
    total_recall = len(all_found) / len(all_expected) if all_expected else 1.0
    precision = matched_provs / len(doc_provs) if doc_provs else 0.0

    return {
        "critical_recall": round(critical_recall, 2),
        "critical_found": critical_found,
        "critical_missed": critical_missed,
        "total_recall": round(total_recall, 2),
        "keyword_hit": round(keyword_hit, 2),
        "precision": round(precision, 2),
        "doc_provenance_count": len(doc_provs),
        "total_provenance_count": len(provenance_list),
    }


def _ensure_lmstudio_models(
    chat_model: str = "qwen3-coder-30b-a3b-instruct",
    embed_model: str = "text-embedding-nomic-embed-text-v1.5@q8_0",
    ctx: int = 32768,
) -> None:
    """Ensure both chat and embedding models are loaded in LM Studio."""
    import shutil
    import subprocess

    lms = shutil.which("lms")
    if not lms:
        return
    try:
        result = subprocess.run(
            [lms, "ps"], capture_output=True, text=True, timeout=10,
            encoding="utf-8", errors="replace",
        )
        loaded = result.stdout + result.stderr
    except Exception:
        loaded = ""

    for model, extra_args in [
        (chat_model, ["-c", str(ctx), "--parallel", "1"]),
        (embed_model, []),
    ]:
        if model in loaded:
            continue
        print(f"Loading {model} in LM Studio...")
        subprocess.run(
            [lms, "load", model, "-y"] + extra_args,
            capture_output=True, text=True, timeout=300,
            encoding="utf-8", errors="replace",
        )
        print(f"  {model} loaded.")


def ingest_corpus(engine, corpus_dir: Path, collection: str) -> dict:
    """Ingest test corpus into a collection. Returns ingestion stats."""
    doc_files = [
        f for f in corpus_dir.iterdir()
        if f.suffix.lower() in (".pdf", ".docx", ".pptx", ".md", ".txt", ".csv", ".xlsx")
    ]
    if not doc_files:
        print(f"No documents found in {corpus_dir}")
        print("Add test documents to tools/retrieval_eval/test_corpus/")
        return {}

    print(f"Ingesting {len(doc_files)} documents into collection '{collection}'...")
    t0 = time.monotonic()
    result = engine.point(source=corpus_dir, collection=collection)
    elapsed = time.monotonic() - t0
    print(f"Ingestion complete in {elapsed:.1f}s")
    if isinstance(result, dict):
        for k, v in result.items():
            if isinstance(v, (int, float)) and v > 0:
                print(f"  {k}: {v}")
    return result or {}


def run_query(engine, query_text: str, top_k: int = 10):
    """Run a query and return provenance list."""
    from fitz_ai.core import Query
    query = Query(text=query_text)
    answer = engine.answer(query)
    return answer.provenance if answer else []


def run_eval(
    corpus_dir: str | Path = CORPUS_DIR,
    collection: str = COLLECTION,
    category: str | None = None,
    ids: str | None = None,
    top_k: int = 10,
    verbose: bool = True,
    skip_ingest: bool = False,
):
    """Run full document retrieval evaluation."""
    from fitz_ai.runtime import create_engine

    corpus_dir = Path(corpus_dir)
    if not corpus_dir.exists():
        print(f"Corpus directory not found: {corpus_dir}")
        return

    queries = json.loads(GROUND_TRUTH.read_text())
    if not queries or (len(queries) == 1 and "EXAMPLE" in queries[0].get("query", "")):
        print("Ground truth contains only the example query.")
        print("Add test documents to test_corpus/ and update doc_ground_truth.json")
        return

    if category:
        queries = [q for q in queries if q.get("category") == category]
    if ids:
        id_set = set(int(i) for i in ids.split(","))
        queries = [q for q in queries if q["id"] in id_set]

    if not queries:
        print("No queries match filters.")
        return

    # Pre-load LM Studio models (chat + embedding)
    _ensure_lmstudio_models()

    # Create engine and ingest
    engine = create_engine("fitz_krag")
    if not skip_ingest:
        stats = ingest_corpus(engine, corpus_dir, collection)
        if not stats:
            return
    else:
        print(f"Skipping ingestion, loading collection '{collection}'...")
        engine.load(collection)

    print(f"\nRunning {len(queries)} document retrieval evaluations...\n")

    results = []
    for q in queries:
        t0 = time.monotonic()
        try:
            provenance = run_query(engine, q["query"], top_k)
            elapsed = time.monotonic() - t0

            # Debug: dump first provenance metadata for first query
            if q["id"] == 1 and provenance:
                print(f"  DEBUG: {len(provenance)} provenance items")
                for i, prov in enumerate(provenance[:3]):
                    print(f"  DEBUG prov[{i}]:")
                    print(f"    source_id: {getattr(prov, 'source_id', None)}")
                    print(f"    excerpt:   {(getattr(prov, 'excerpt', '') or '')[:100]}")
                    meta = getattr(prov, 'metadata', {}) or {}
                    for k, v in meta.items():
                        print(f"    meta.{k}: {v}")

            scores = score_query(provenance, q)
            scores["id"] = q["id"]
            scores["time"] = round(elapsed, 1)

            is_pass = scores["critical_recall"] == 1.0
            status = "PASS" if is_pass else "MISS"
            label = f"crit={scores['critical_recall']:.0%} total={scores['total_recall']:.0%} kw={scores['keyword_hit']:.0%}"

            query_preview = q["query"][:60]
            print(f"  [{q['id']:2d}] {status} {label} ({elapsed:.1f}s) {query_preview}")

            if not is_pass and verbose:
                for missed in scores["critical_missed"]:
                    print(f"       MISSED: {missed}")

            results.append(scores)
        except Exception as e:
            print(f"  [{q['id']:2d}] FAILED: {e}")
            results.append({"id": q["id"], "error": str(e)})

    # Summary
    scored = [r for r in results if "error" not in r]
    if not scored:
        print("\nAll queries failed.")
        return

    avg_crit = sum(r["critical_recall"] for r in scored) / len(scored)
    avg_total = sum(r["total_recall"] for r in scored) / len(scored)
    avg_kw = sum(r["keyword_hit"] for r in scored) / len(scored)
    avg_prec = sum(r["precision"] for r in scored) / len(scored)
    perfect = sum(1 for r in scored if r["critical_recall"] == 1.0)
    avg_time = sum(r["time"] for r in scored) / len(scored)

    print(f"\n{'=' * 60}")
    print(f"RESULTS ({len(queries)} queries)")
    print(f"{'=' * 60}")
    print(f"Critical recall:  {avg_crit:.0%} avg ({perfect}/{len(scored)} perfect)")
    print(f"Total recall:     {avg_total:.0%} avg")
    print(f"Keyword hit:      {avg_kw:.0%} avg")
    print(f"Precision:        {avg_prec:.0%} avg")
    print(f"Avg time:         {avg_time:.1f}s per query")

    # By category
    categories = sorted(set(q.get("category", "uncategorized") for q in queries))
    if len(categories) > 1:
        print(f"\nBy category:")
        for cat in categories:
            cat_results = [r for r, q in zip(results, queries)
                          if q.get("category") == cat and "error" not in r]
            if cat_results:
                cat_crit = sum(r["critical_recall"] for r in cat_results) / len(cat_results)
                cat_perfect = sum(1 for r in cat_results if r["critical_recall"] == 1.0)
                print(f"  {cat:25s} crit={cat_crit:.0%} ({cat_perfect}/{len(cat_results)} perfect)")

    # Most-missed sections
    all_missed = []
    for r in scored:
        all_missed.extend(r.get("critical_missed", []))
    if all_missed:
        from collections import Counter
        missed_counts = Counter(all_missed).most_common(10)
        print(f"\nMost-missed critical sections:")
        for section, count in missed_counts:
            print(f"  {count}x {section}")

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"doc_eval_{ts}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")
