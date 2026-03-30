# Context Optimization for Retrieval Pipeline

## Background

The planning pipeline sends `raw_summaries` to the LLM reasoning call. This contains:

- **Interface signatures** (auto-extracted, ground truth)
- **Library API reference**
- **Structural index** (classes, functions, imports for each file)
- **Seed files** (5 files with full source code inline)

The goal: fit more files into the context window (especially 32K) to improve retrieval recall.

## Baseline: 30B Model Retrieval Eval

- **Model:** qwen3-coder-30b-a3b-instruct (MoE, 30B total, 3B active)
- 40-query ground truth benchmark on fitz-sage codebase
- **Critical recall:** 89% avg (28/40 perfect)
- **Total recall:** 74% avg
- **Avg time:** 18.1s per query
- Compared to qwen3.5-4b: 78% critical recall, 17/40 perfect, 22s avg
- 30B is both better AND faster (MoE generates faster)

## Experiment 1: Tiered Overview (One-Line Pool Manifest)

**Hypothesis:** Replace full structural index for non-seed files with one-line docstring manifest. Save tokens while preserving reasoning quality.

**Results (10 runs each, temperature=0):**

- Old prompt: 12,437 chars | New prompt: 7,901 chars (37% smaller)
- Savings: ~1,125 tokens on prompt
- REGRESSION: `mentions_openai_provider` dropped 10/10 → 0/10
- REGRESSION: `mentions_callback` dropped 10/10 → 0/10
- REGRESSION: `mentions_wrapper` dropped 10/10 → 1/10
- New version proposed different (not necessarily worse) architecture but lost awareness of provider layer

**Conclusion:** One-line docstring insufficient — model can't reason about file connections without seeing class/function names.

## Experiment 2: Tiered Overview (Two-Line Pool Manifest)

Added classes/functions line alongside docstring for pool files.

**Results (10 runs each, temperature=0):**

- Old prompt: 12,437 chars | New prompt: 9,737 chars (22% smaller)
- Savings: ~660 tokens on prompt
- Recovered `core/token_tracking.py` reference (0/10 → 9/10)
- Still lost `mentions_openai_provider` (0/10)
- REGRESSION: Model hallucinated 12 approaches instead of 6

**Conclusion:** Better than one-line but still loses import-chain awareness. The `imports:` line in structural index is what tells the model how files connect.

## Experiment 3: No Seeds (Full Index, No Inline Source)

**Hypothesis:** Remove seed file source code from prompt entirely. Model can use `read_file()` tool during reasoning if it needs source.

**Results (10 runs each, temperature=0):**

- Old prompt: 37,634 chars (~9.4K tok) | New prompt: 17,895 chars (~4.5K tok)
- Savings: ~4,934 tokens (52% reduction!)
- NO regressions on any signal
- New version MORE consistent: 10/10 identical outputs vs 9/10
- Faster: 25.6s avg vs 33.9s (25% faster)
- File intersection: 4/4 (new) vs 3/5 (old) — higher agreement
- `mentions_wrapper`: 1/10 (old) → 10/10 (new) — actually IMPROVED

**Conclusion:** Seed source code was noise. 5K tokens of raw code that the model mostly ignored. Structural index + signatures provide everything needed for architectural reasoning.

## Key Finding

The structural index (with full classes/functions/imports) is **load-bearing** — trimming it loses signal. But seed file source code inline is **not** — removing it saves ~5K tokens with zero quality loss.

## Token Budget (No Seeds)

| Files | raw_summaries | Peak (+ 1K prompt + 16K output) | 32K headroom |
|-------|---------------|----------------------------------|--------------|
| 30    | 3,204 tok     | 20,204 tok                       | 12.5K OK     |
| 50    | 5,802 tok     | 22,802 tok                       | 10K OK       |
| 100   | 10,860 tok    | 27,860 tok                       | 4.9K OK      |
| 150   | 18,671 tok    | 35,671 tok                       | OVERFLOW     |

Note: 4.9K headroom at 100 files is tight when tool use is involved. Each `read_file()` round-trip costs ~500-2K tokens. 5 rounds = ~5K tokens = overflow risk.

**Safe limits:** 50-60 files at 32K context, 100 files at 65K context.

## Proposed: Dynamic Structural Index via Tool (Not Yet Implemented)

Instead of baking the full structural index into the prompt, split it:

1. Put one-liner manifest (100 paths + docstrings) in prompt (~2K tokens)
2. Expose `inspect_files(paths: list[str])` tool that returns full structural detail (classes, methods, imports) for requested files
3. Model sees all 100 paths, picks 10-15 it cares about, calls `inspect_files()` once, then reasons

**Benefits:**

- Prompt drops from ~11K to ~3K tokens for raw_summaries
- 100+ files at 32K context becomes feasible
- Model only pays for structural detail it actually uses
- One extra round-trip (~1K tokens for the tool call)

**Risks:**

- Model might not inspect enough files (under-exploration)
- Extra round-trip adds latency (~5-10s)
- More complex implementation in `_reason_with_tools`

## Experiment 4: inspect_files Tool (Implemented)

A/B tested with 60 files, 10 runs each, temperature=0:

| Metric | Old (full index in prompt) | New (manifest + inspect_files) |
|--------|---------------------------|-------------------------------|
| Prompt size | ~8,262 tok | ~4,356 tok |
| Avg time | 32.1s | **19.3s** (40% faster) |
| Tool calls | — | 1 call, 4 files (every run) |
| Output consistency | 9/10 identical | **10/10 identical** |
| mentions_decorator | 9/10 | **10/10** |
| mentions_wrapper | 0/10 | **10/10** |
| mentions_middleware | 10/10 | 10/10 |
| mentions_token | 10/10 | 10/10 |
| Approaches | 8 | 8 |

Key findings:
- Model reliably calls inspect_files (10/10 runs, always 4 files)
- Zero quality regression, actually improved on `mentions_wrapper`
- 40% faster due to smaller prompt (less prefill)
- Saves ~4K tokens of prompt space for tool-use headroom

**Status:** Implemented in gatherer + base stage.

## Implementation Summary

Changes made:
1. **Gatherer** (`gatherer.py`): Builds one-liner manifest (path + docstring) instead of full structural index in `raw_summaries`. Stores per-file structural entries in `file_index_entries` dict.
2. **Orchestrator** (`orchestrator.py`): Passes `file_index_entries` as `_file_index_entries` in `prior_outputs`.
3. **Base stage** (`base.py`): Adds `inspect_files(paths)` tool alongside `read_file`/`read_files`. Updated tool hint to suggest inspect-first workflow.
4. **Config** (`schema.py`): Default `max_seed_files` bumped from 30 to 60.

Token budget with manifest approach (no seeds, no full index in prompt):
| Files | raw_summaries | Peak (+ prompt + 16K output + 5K tools) | Fits 32K? |
|-------|--------------|------------------------------------------|-----------|
| 60    | ~2.5K tok    | ~24.5K tok                               | YES (8K headroom) |
| 100   | ~4K tok      | ~26K tok                                 | YES (7K headroom) |
| 150   | ~6K tok      | ~28K tok                                 | YES (5K headroom) |
