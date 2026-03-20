# docs/roadmap/query-intelligence-pipeline.md
# Query Intelligence Pipeline — Rewrite-First with Batched Classification

## Problem

The retrieval pipeline makes 3 independent LLM calls per query (analysis, detection, rewriting), all using the "fast" tier. On local providers like ollama, these serialize due to single-model execution, causing 60-90s latency. Batching all 3 into 1 call cuts latency to ~19s but degrades the 2B model's output quality (95% → 80% recall).

Additionally, analysis and detection currently classify the *original* query, not the rewritten one. Classifying the cleaned-up query produces more accurate results.

## Solution: Rewrite-First, Then Batched Classification

```
Step 1: Rewrite (1 LLM call)
  original query → rewritten query

Step 2: Batch(Analysis + Detection) + Embedding (parallel)
  rewritten query → { analysis, detection } + query vector
```

**Why this ordering:**
- Rewriting removes noise, resolves pronouns, simplifies phrasing
- Analysis ("is this code or docs?") is more accurate on the cleaned query
- Detection ("is this temporal/comparison?") is more accurate on the cleaned query
- 2 LLM calls instead of 3 on local (saves 1 model swap = ~10-15s)
- The 2B model handles a 2-task classification prompt well (analysis + detection are both "classify this query" tasks)
- Rewriting keeps its own focused prompt (it's a generative task, not classification)

**For cloud/API users:** rewrite runs first (~1s), then batch + embed in parallel (~1s). Total ~2s. Fewer API calls = lower cost.

## Phase 1: Rewrite-First Pipeline — Done

Reorder the dispatch so rewriting runs first, then analysis + detection batch on the rewritten query.

| Task | Status |
|------|--------|
| Reorder dispatch: rewrite → batch(analysis + detection) | **Done** |
| Batch prompt: analysis + detection only (not rewriting) | **Done** |
| Unify HyDE ownership to router (was duplicated on 3 objects) | **Done** |
| Extract shared parsers (parse_analysis_dict, distribute_to_modules, parse_rewrite_dict) | **Done** |
| Add gate_categories() to DetectionOrchestrator | **Done** |

### Benchmark Results (20 queries, 3 PDFs, ollama/qwen3.5:2b)

| Metric | Before (3 individual calls) | After (rewrite-first) |
|--------|------|------|
| Critical recall | 95% (19/20) | 88% (17/20) |
| Avg query time | 30.9s | 22.4s |
| Total time | 617s | 448s |

Trade-off: 7% recall loss for 28% speed gain. The recall gap is model-dependent — stronger models (4B+, API) should close it since the architecture is sound.

## Phase 2: Extended Classification Signals — Proposed

Piggyback additional signals onto the batched analysis + detection call at near-zero marginal cost. These are **advisory signals** — used to boost/adjust existing retrieval, not as hard gates. The pipeline must work correctly without them (graceful degradation for weaker models).

### Current retrieval trigger mechanisms (fragmented)

Retrieval features are currently triggered by three independent mechanisms that don't share a unified model:

1. **Analysis type → static weight map** (`query_analyzer.py`)
   - Hard-coded dict: CODE → code=0.75/section=0.10, DOCUMENTATION → code=0.10/section=0.75, etc.
   - Gates: HyDE skipped if code+high-conf, multi-query skipped if code/data or conf≥0.8

2. **Detection flags → per-feature triggers** (`detection/modules/`)
   - Each module independently sets flags: temporal→sub-queries, aggregation→fetch_multiplier, comparison→entity-queries, freshness→recency boost

3. **Hard-coded config** (no query awareness)
   - BM25/semantic weights (0.6/0.4), top_k, reranking on/off, context budget

### Proposed: unified retrieval profile

Extended classification signals feed a `RetrievalProfile` that tunes the pipeline per-query. Signals are advisory — missing signals fall back to current defaults.

| Signal | Source | Tunes | Fallback |
|--------|--------|-------|----------|
| **specificity** (broad/moderate/narrow) | Batch LLM | top_k: broad=30, narrow=10 | Config default |
| **answer_type** (factual/procedural/comparative/exploratory) | Batch LLM | Context budget + assembly strategy | Full context |
| **domain** (general/technical/legal/financial/medical) | Batch LLM | Vocabulary matching boost | No boost |
| **multi_hop** (true/false) | Batch LLM | Trigger multi-hop proactively | Only on first-pass failure |
| **primary_type** | Batch LLM (existing) | Strategy weights | GENERAL weights |
| **confidence** | Batch LLM (existing) | HyDE/multi-query/agentic gates | 0.5 |
| **temporal/comparison/aggregation/freshness** | Batch LLM (existing) | Detection-specific routing | Not detected |

### Extended batch prompt

```json
{
  "analysis": {
    "primary_type": "code|documentation|general|cross|data",
    "confidence": 0.0-1.0,
    "entities": [],
    "refined_query": "cleaned query",
    "specificity": "broad|moderate|narrow",
    "answer_type": "factual|procedural|comparative|exploratory",
    "domain": "general|technical|legal|financial|medical",
    "multi_hop": false
  },
  "detection": {
    "temporal": { ... },
    "aggregation": { ... },
    "comparison": { ... },
    "freshness": { ... },
    "rewriter": { ... }
  }
}
```

New fields are optional — missing fields use current defaults. This means the feature degrades gracefully with weaker models.

## Files

| File | Role |
|------|------|
| `engines/fitz_krag/query_batcher.py` | Batched LLM call (analysis + detection) |
| `engines/fitz_krag/query_analyzer.py` | `parse_analysis_dict()` shared parser |
| `engines/fitz_krag/engine.py` | Dispatch logic (rewrite → batch → retrieve) |
| `retrieval/detection/llm_classifier.py` | `distribute_to_modules()` shared parser |
| `retrieval/detection/registry.py` | `gate_categories()` for ML/semantic gating |
| `retrieval/rewriter/rewriter.py` | `parse_rewrite_dict()` shared parser |
