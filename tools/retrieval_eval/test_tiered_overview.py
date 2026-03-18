# tools/retrieval_eval/test_tiered_overview.py
"""
A/B test: full raw_summaries (with seed source inline) vs
no-seeds (structural index + signatures only, model uses read_file).

10 calls each, temperature=0.
"""
import re
import sys
import time
from pathlib import Path

import httpx

# ── Config ──────────────────────────────────────────────────────────────
SOURCE_DIR = Path("C:/Users/yanfi/PycharmProjects/fitz-ai")
MODEL = "qwen3-coder-30b-a3b-instruct"
BASE_URL = "http://localhost:1234/v1"
QUERY = "Add token usage tracking so I can see how many LLM tokens each query uses"
NUM_FILES = 30
NUM_SEEDS = 5
RUNS = 10


def build_indexes():
    """Build old (with seeds) and new (no seeds) raw_summaries."""
    sys.path.insert(0, str(SOURCE_DIR))
    sys.path.insert(0, "C:/Users/yanfi/PycharmProjects/fitz-graveyard")
    from fitz_graveyard.planning.agent.indexer import (
        build_structural_index,
        extract_interface_signatures,
        extract_library_signatures,
    )

    from fitz_ai.code.indexer import build_file_list

    all_files = build_file_list(SOURCE_DIR, 2000)
    py_files = [f for f in all_files if f.endswith(".py")][:NUM_FILES]
    seed_files = py_files[:NUM_SEEDS]

    full_index = build_structural_index(SOURCE_DIR, py_files, max_file_bytes=100_000)
    sigs = extract_interface_signatures(str(SOURCE_DIR), py_files, 100_000) or ""
    lib_sigs = extract_library_signatures(str(SOURCE_DIR), py_files, all_files, 100_000) or ""

    # Read seed file contents
    seed_blocks = []
    for path in seed_files:
        try:
            content = (SOURCE_DIR / path).read_text(encoding="utf-8", errors="replace")
        except Exception:
            content = "(could not read)"
        seed_blocks.append(f"### {path}\n```\n{content}\n```")

    # OLD: full index + signatures + seed source inline
    old_parts = []
    if sigs:
        old_parts.append(f"--- INTERFACE SIGNATURES (auto-extracted, ground truth) ---\n{sigs}")
    if lib_sigs:
        old_parts.append(
            f"--- LIBRARY API REFERENCE (installed packages, ground truth) ---\n{lib_sigs}"
        )
    old_parts.append(f"--- STRUCTURAL OVERVIEW (all selected files) ---\n{full_index}")
    old_parts.append(
        f"--- SEED FILES ({len(seed_files)}/{len(py_files)} — "
        f"use read_file/read_files for the rest) ---\n\n" + "\n\n".join(seed_blocks)
    )
    old_raw = "\n\n".join(old_parts)

    # NEW: full index + signatures, NO seed source
    new_parts = []
    if sigs:
        new_parts.append(f"--- INTERFACE SIGNATURES (auto-extracted, ground truth) ---\n{sigs}")
    if lib_sigs:
        new_parts.append(
            f"--- LIBRARY API REFERENCE (installed packages, ground truth) ---\n{lib_sigs}"
        )
    new_parts.append(f"--- STRUCTURAL OVERVIEW (all selected files) ---\n{full_index}")
    new_parts.append(f"--- {len(py_files)} files available via read_file(path) ---")
    new_raw = "\n\n".join(new_parts)

    return old_raw, new_raw


def build_reasoning_prompt(krag_context: str) -> list[dict]:
    prompt_template = (
        Path("C:/Users/yanfi/PycharmProjects/fitz-graveyard")
        / "fitz_graveyard/planning/prompts/architecture_design.txt"
    ).read_text()
    prompt = prompt_template.format(
        context=f"Task: {QUERY}",
        krag_context=krag_context,
        binding_constraints="(none)",
    )
    return [
        {"role": "system", "content": "You are a senior software architect."},
        {"role": "user", "content": prompt},
    ]


def call_llm(messages: list[dict]) -> tuple[str, float]:
    client = httpx.Client(
        base_url=BASE_URL,
        timeout=httpx.Timeout(600.0, connect=5.0),
    )
    t0 = time.monotonic()
    resp = client.post(
        "/chat/completions",
        json={
            "model": MODEL,
            "messages": messages,
            "max_tokens": 4096,
            "temperature": 0,
        },
    )
    elapsed = time.monotonic() - t0
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"] if data.get("choices") else ""
    return content, elapsed


def extract_signals(text: str) -> dict:
    text_lower = text.lower()
    return {
        "length": len(text),
        "mentions_openai_provider": "openai" in text_lower and "provider" in text_lower,
        "mentions_answer": "answer.py" in text_lower or "answer_mode" in text_lower,
        "mentions_engine": "engine.py" in text_lower or "core/engine" in text_lower,
        "mentions_middleware": "middleware" in text_lower,
        "mentions_decorator": "decorator" in text_lower,
        "mentions_token": "token" in text_lower,
        "mentions_usage": "usage" in text_lower,
        "mentions_callback": "callback" in text_lower,
        "mentions_wrapper": "wrapper" in text_lower,
        "num_approaches": len(re.findall(r"approach\s+[A-Z0-9]", text, re.IGNORECASE)),
        "has_adrs": "adr" in text_lower or "decision record" in text_lower,
        "has_components": "component" in text_lower,
        "has_data_model": "data model" in text_lower or "data_model" in text_lower,
        "file_refs": sorted(set(re.findall(r"fitz_ai/[\w/]+\.py", text))),
    }


def summarize_runs(label: str, results: list[dict]):
    print(f"\n{'=' * 60}")
    print(f"  {label} -- {len(results)} runs")
    print(f"{'=' * 60}")

    times = [r["time"] for r in results]
    lengths = [r["signals"]["length"] for r in results]
    print(f"  Time:   {min(times):.1f}s - {max(times):.1f}s (avg {sum(times)/len(times):.1f}s)")
    print(
        f"  Length: {min(lengths):,} - {max(lengths):,} chars (avg {sum(lengths)//len(lengths):,})"
    )

    bool_keys = [k for k in results[0]["signals"] if isinstance(results[0]["signals"][k], bool)]
    print(f"\n  Signal consistency (across {len(results)} runs):")
    for key in sorted(bool_keys):
        vals = [r["signals"][key] for r in results]
        true_count = sum(vals)
        print(f"    {key:30s} {true_count}/{len(results)}")

    approaches = [r["signals"]["num_approaches"] for r in results]
    print(f"\n  Approaches: {approaches}")

    all_files = [set(r["signals"]["file_refs"]) for r in results]
    union = sorted(set.union(*all_files)) if all_files else []
    intersection = sorted(set.intersection(*all_files)) if all_files else []
    print(f"\n  Files referenced (union):        {len(union)}")
    for f in union:
        count = sum(1 for s in all_files if f in s)
        print(f"    {count}/{len(results)} {f}")
    print(f"  Files referenced (intersection): {len(intersection)}")


def main():
    print("Building indexes...")
    old_raw, new_raw = build_indexes()
    print(f"Old raw_summaries (with seeds): {len(old_raw):,} chars (~{len(old_raw)//4:,} tok)")
    print(f"New raw_summaries (no seeds):   {len(new_raw):,} chars (~{len(new_raw)//4:,} tok)")
    print(
        f"Savings: {len(old_raw) - len(new_raw):,} chars (~{(len(old_raw)-len(new_raw))//4:,} tok)\n"
    )

    msgs_old = build_reasoning_prompt(old_raw)
    msgs_new = build_reasoning_prompt(new_raw)
    print(f"Old prompt total: {sum(len(m['content']) for m in msgs_old):,} chars")
    print(f"New prompt total: {sum(len(m['content']) for m in msgs_new):,} chars\n")

    old_results = []
    new_results = []

    for i in range(RUNS):
        print(f"Run {i+1}/{RUNS}: old...", end=" ", flush=True)
        out, t = call_llm(msgs_old)
        old_results.append(
            {"run": i + 1, "time": t, "output": out, "signals": extract_signals(out)}
        )
        print(f"{t:.1f}s ({len(out):,} chars)", end=" | ", flush=True)

        print("new...", end=" ", flush=True)
        out, t = call_llm(msgs_new)
        new_results.append(
            {"run": i + 1, "time": t, "output": out, "signals": extract_signals(out)}
        )
        print(f"{t:.1f}s ({len(out):,} chars)")

    summarize_runs("OLD (full index + seeds inline)", old_results)
    summarize_runs("NEW (full index, no seeds)", new_results)

    out_dir = SOURCE_DIR / "tools" / "retrieval_eval" / "results"
    out_dir.mkdir(exist_ok=True)
    for label, results in [("old_seeds", old_results), ("new_noseeds", new_results)]:
        for r in results:
            p = out_dir / f"seeds_ab_{label}_run{r['run']:02d}.txt"
            p.write_text(r["output"], encoding="utf-8")
    print(f"\nAll outputs saved to {out_dir}/seeds_ab_*.txt")


if __name__ == "__main__":
    main()
