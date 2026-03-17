# tools/retrieval_eval/eval_run.py
"""Run retrieval eval with PyCharm's Run button. Edit settings below."""

# --- SETTINGS ---
SOURCE_DIR = "C:/Users/yanfi/PycharmProjects/fitz-ai"
PROVIDER = "lmstudio"         # "lmstudio", "openai", "ollama", "cohere"
MODEL = "qwen3.5-4b"          # model name for the scan call
BASE_URL = "http://localhost:1234/v1"  # LM Studio default
CATEGORY = None               # e.g. "retrieval", "ingestion", None for all
IDS = None                    # e.g. "1,2,3", None for all
VERBOSE = True
# -----------------

if __name__ == "__main__":
    from tools.retrieval_eval.eval import run_eval

    run_eval(
        source_dir=SOURCE_DIR,
        provider=PROVIDER,
        model=MODEL,
        base_url=BASE_URL,
        category=CATEGORY,
        ids=IDS,
        verbose=VERBOSE,
    )
