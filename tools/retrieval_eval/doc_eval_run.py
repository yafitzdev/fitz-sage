# tools/retrieval_eval/doc_eval_run.py
"""Run document retrieval eval with PyCharm's Run button. Edit settings below."""

# --- SETTINGS ---
CORPUS_DIR = "C:/Users/yanfi/PycharmProjects/fitz-ai/tools/retrieval_eval/test_corpus"
COLLECTION = "doc_eval"
CATEGORY = None  # e.g. "section-lookup", "table-lookup", None for all
IDS = None  # e.g. "1,2,3", None for all
VERBOSE = True
SKIP_INGEST = False  # True to skip re-ingestion (use existing collection)
# -----------------

if __name__ == "__main__":
    from tools.retrieval_eval.doc_eval import run_eval

    run_eval(
        corpus_dir=CORPUS_DIR,
        collection=COLLECTION,
        category=CATEGORY,
        ids=IDS,
        verbose=VERBOSE,
        skip_ingest=SKIP_INGEST,
    )
