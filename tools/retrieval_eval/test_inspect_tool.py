# tools/retrieval_eval/test_inspect_tool.py
"""
A/B test: full structural index in prompt vs inspect_files() tool.

OLD: Full structural index (classes, functions, imports) for all 60 files in prompt.
NEW: One-liner manifest (path + docstring) in prompt + inspect_files tool.

10 runs each, temperature=0.
"""
import json
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
NUM_FILES = 60
RUNS = 10


def _build_data():
    """Build full index, manifest, and per-file structural entries."""
    sys.path.insert(0, str(SOURCE_DIR))
    sys.path.insert(0, "C:/Users/yanfi/PycharmProjects/fitz-graveyard")
    from fitz_ai.code.indexer import build_file_list
    from fitz_graveyard.planning.agent.indexer import (
        build_structural_index,
        extract_interface_signatures,
        extract_library_signatures,
    )

    all_files = build_file_list(SOURCE_DIR, 2000)
    py_files = [f for f in all_files if f.endswith(".py")][:NUM_FILES]

    full_index = build_structural_index(SOURCE_DIR, py_files, max_file_bytes=100_000)
    sigs = extract_interface_signatures(str(SOURCE_DIR), py_files, 100_000) or ""
    lib_sigs = extract_library_signatures(str(SOURCE_DIR), py_files, all_files, 100_000) or ""

    # Parse per-file entries from full index
    file_entries: dict[str, str] = {}
    for path in py_files:
        marker = f"## {path}\n"
        idx = full_index.find(marker)
        if idx >= 0:
            entry_start = idx + len(marker)
            entry_end = full_index.find("\n## ", entry_start)
            entry = full_index[entry_start:entry_end].strip() if entry_end > 0 else full_index[entry_start:].strip()
            file_entries[path] = entry

    # Build one-liner manifest
    manifest_lines = []
    for path in py_files:
        entry = file_entries.get(path, "")
        lines = entry.split("\n") if entry else []
        doc_line = next((l.strip() for l in lines if l.strip().startswith("doc:")), "")
        manifest_lines.append(f"  {path} — {doc_line}" if doc_line else f"  {path}")
    manifest = "\n".join(manifest_lines)

    return {
        "full_index": full_index,
        "sigs": sigs,
        "lib_sigs": lib_sigs,
        "manifest": manifest,
        "file_entries": file_entries,
        "py_files": py_files,
    }


def _build_old_prompt(data: dict) -> list[dict]:
    """Full structural index in prompt (current approach)."""
    prompt_template = (
        Path("C:/Users/yanfi/PycharmProjects/fitz-graveyard")
        / "fitz_graveyard/planning/prompts/architecture_design.txt"
    ).read_text()

    parts = []
    if data["sigs"]:
        parts.append(f"--- INTERFACE SIGNATURES ---\n{data['sigs']}")
    if data["lib_sigs"]:
        parts.append(f"--- LIBRARY API REFERENCE ---\n{data['lib_sigs']}")
    parts.append(f"--- STRUCTURAL OVERVIEW (all selected files) ---\n{data['full_index']}")
    parts.append(f"--- {len(data['py_files'])} files available via read_file(path) ---")
    krag = "\n\n".join(parts)

    prompt = prompt_template.format(
        context=f"Task: {QUERY}",
        krag_context=krag,
        binding_constraints="(none)",
    )
    return [
        {"role": "system", "content": "You are a senior software architect."},
        {"role": "user", "content": prompt},
    ]


def _build_new_prompt(data: dict) -> list[dict]:
    """One-liner manifest + inspect_files tool."""
    prompt_template = (
        Path("C:/Users/yanfi/PycharmProjects/fitz-graveyard")
        / "fitz_graveyard/planning/prompts/architecture_design.txt"
    ).read_text()

    parts = []
    if data["sigs"]:
        parts.append(f"--- INTERFACE SIGNATURES ---\n{data['sigs']}")
    if data["lib_sigs"]:
        parts.append(f"--- LIBRARY API REFERENCE ---\n{data['lib_sigs']}")
    parts.append(
        f"--- FILE MANIFEST ({len(data['py_files'])} files) ---\n"
        f"{data['manifest']}"
    )
    parts.append(TOOL_INSTRUCTIONS)
    krag = "\n\n".join(parts)

    prompt = prompt_template.format(
        context=f"Task: {QUERY}",
        krag_context=krag,
        binding_constraints="(none)",
    )
    return [
        {"role": "system", "content": "You are a senior software architect."},
        {"role": "user", "content": prompt},
    ]


TOOL_INSTRUCTIONS = """
You have access to a tool to inspect file details before reasoning:

<tool>
inspect_files: Returns structural detail (classes, methods, imports) for requested files.
Usage: To inspect files, respond with ONLY a JSON block like this:
```json
{"tool": "inspect_files", "paths": ["fitz_ai/path/to/file.py", "fitz_ai/other/file.py"]}
```
You will receive the structural details, then continue your analysis.
You may call this tool ONCE before giving your final answer.
</tool>
"""


def call_llm_plain(messages: list[dict]) -> tuple[str, float]:
    """Single chat completion, no tools."""
    client = httpx.Client(base_url=BASE_URL, timeout=httpx.Timeout(600.0, connect=5.0))
    t0 = time.monotonic()
    resp = client.post("/chat/completions", json={
        "model": MODEL, "messages": messages,
        "max_tokens": 4096, "temperature": 0,
    })
    elapsed = time.monotonic() - t0
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"] if data.get("choices") else "", elapsed


def call_llm_with_tools(messages: list[dict], file_entries: dict, max_rounds: int = 1) -> tuple[str, float, int, int]:
    """Chat completion with manual inspect_files tool loop.

    Returns (final_text, elapsed, tool_calls_made, files_inspected).
    """
    client = httpx.Client(base_url=BASE_URL, timeout=httpx.Timeout(600.0, connect=5.0))
    msgs = list(messages)
    total_tool_calls = 0
    total_files_inspected = 0
    t0 = time.monotonic()

    for _round in range(max_rounds + 1):
        resp = client.post("/chat/completions", json={
            "model": MODEL, "messages": msgs,
            "max_tokens": 4096, "temperature": 0,
        })
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"] if data.get("choices") else ""

        # Check if the model wants to call inspect_files
        tool_match = re.search(r'```json\s*\n?\s*\{[^}]*"tool"\s*:\s*"inspect_files"[^}]*\}', content, re.DOTALL)
        if not tool_match or _round == max_rounds:
            elapsed = time.monotonic() - t0
            return content, elapsed, total_tool_calls, total_files_inspected

        # Parse the tool call
        total_tool_calls += 1
        try:
            json_str = re.search(r'\{[^}]*"tool"[^}]*\}', tool_match.group(), re.DOTALL).group()
            call_data = json.loads(json_str)
            paths = call_data.get("paths", [])
        except (json.JSONDecodeError, AttributeError):
            paths = []

        # Build tool response
        result_parts = []
        for p in paths:
            entry = file_entries.get(p, "")
            if entry:
                result_parts.append(f"## {p}\n{entry}")
                total_files_inspected += 1
            else:
                result_parts.append(f"## {p}\n(not found in index)")

        tool_response = "\n\n".join(result_parts) if result_parts else "(no valid paths)"

        msgs.append({"role": "assistant", "content": content})
        msgs.append({"role": "user", "content": f"<tool_response>\n{tool_response}\n</tool_response>\n\nNow continue with your architectural analysis."})

    elapsed = time.monotonic() - t0
    return content, elapsed, total_tool_calls, total_files_inspected


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
    print(f"  Length: {min(lengths):,} - {max(lengths):,} chars (avg {sum(lengths)//len(lengths):,})")

    if "tool_calls" in results[0]:
        tcs = [r["tool_calls"] for r in results]
        inspected = [r["files_inspected"] for r in results]
        print(f"  Tool calls: {tcs}")
        print(f"  Files inspected: {inspected}")

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
    print("Building data...")
    data = _build_data()

    msgs_old = _build_old_prompt(data)
    msgs_new = _build_new_prompt(data)
    old_chars = sum(len(m["content"]) for m in msgs_old)
    new_chars = sum(len(m["content"]) for m in msgs_new)
    print(f"Old prompt (full index): {old_chars:,} chars (~{old_chars//4:,} tok)")
    print(f"New prompt (manifest):   {new_chars:,} chars (~{new_chars//4:,} tok)")
    print(f"Savings: {old_chars - new_chars:,} chars (~{(old_chars-new_chars)//4:,} tok)\n")

    old_results = []
    new_results = []

    for i in range(RUNS):
        print(f"Run {i+1}/{RUNS}: old...", end=" ", flush=True)
        out, t = call_llm_plain(msgs_old)
        old_results.append({"run": i+1, "time": t, "output": out, "signals": extract_signals(out)})
        print(f"{t:.1f}s ({len(out):,} chars)", end=" | ", flush=True)

        print("new (tool)...", end=" ", flush=True)
        out, t, tc, fi = call_llm_with_tools(msgs_new, data["file_entries"])
        new_results.append({
            "run": i+1, "time": t, "output": out, "signals": extract_signals(out),
            "tool_calls": tc, "files_inspected": fi,
        })
        print(f"{t:.1f}s ({len(out):,} chars, {tc} calls, {fi} files)")

    summarize_runs("OLD (full structural index in prompt)", old_results)
    summarize_runs("NEW (manifest + inspect_files tool)", new_results)

    out_dir = SOURCE_DIR / "tools" / "retrieval_eval" / "results"
    out_dir.mkdir(exist_ok=True)
    for label, results in [("old_fullindex", old_results), ("new_inspect", new_results)]:
        for r in results:
            p = out_dir / f"inspect_ab_{label}_run{r['run']:02d}.txt"
            p.write_text(r["output"], encoding="utf-8")
    print(f"\nAll outputs saved to {out_dir}/inspect_ab_*.txt")


if __name__ == "__main__":
    main()
