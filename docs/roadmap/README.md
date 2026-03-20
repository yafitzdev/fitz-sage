# docs/roadmap/README.md
# fitz-ai Roadmap

---

## Next Major: Corpus Intelligence (v0.11.0)

**[Corpus Intelligence — Self-Aware RAG with Actionable Quality Signals](./corpus-intelligence.md)**

Surface fitz-ai's hidden intelligence to developers. Actionable ABSTAIN (explains gaps, suggests documents to add), confidence scores, answer explanations, and corpus health reports. Zero new LLM calls — exposes signals already computed by governance and entity graph.

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Actionable ABSTAIN (gap analysis in answer text) | **Done** |
| 2 | Confidence score on every Answer | Proposed |
| 3 | Answer explanation from constraint metadata | Proposed |
| 4 | Corpus health report (`fitz_ai.health()`) | Proposed |
| 5 | ABSTAIN-driven ingestion suggestions | Proposed |

---

## Next: Query Intelligence Pipeline

**[Rewrite-First with Batched Classification](./query-intelligence-pipeline.md)**

Reorder query preprocessing: rewrite first, then batch analysis + detection on the cleaned query. Reduces local LLM calls from 3 to 2, improves classification accuracy. Phase 2 adds extended signals (specificity, domain, multi-hop) to replace hard-coded retrieval gates.

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Rewrite-first pipeline + batched analysis/detection | In Progress |
| 2 | Extended classification signals (specificity, domain, multi-hop) | Proposed |

---

## Future: KRAG Agent

**[KRAG Agent — Retrieval-as-Tools with Epistemic Self-Verification](./krag-agent.md)**

Transform the pipeline into an autonomous agent using retrieval strategies as composable tools. Better suited if fitz-ai expands beyond library into research/investigation use cases.

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Tool definitions & agent core loop | Proposed |
| 2 | Self-verification tool (governance as in-loop primitive) | Proposed |
| 3 | Cross-collection federation | Proposed |
| 4 | Reasoning trace & observability | Proposed |
| 5 | Auto mode & confidence routing (pipeline vs agent) | Proposed |

---

## Ingestion Quality (post-v0.9.0)

PDF parsing is solved (pypdfium2 fast path + Docling OCR fallback with caching).
Remaining domains:

| # | Problem | File | Impact | Effort |
|---|---------|------|--------|--------|
| 1 | [DOCX/PPTX have no structure-aware chunking](./01-docx-pptx-chunking.md) | `ingestion/chunking/` | High | Medium |
| 2 | [Non-Python code silently loses all symbols](./02-tree-sitter-failures.md) | `progressive/worker.py` | High | Low |
| 3 | [CSV rows invisible to semantic search](./03-table-vector-gap.md) | `chunking/plugins/table.py` | Medium | Medium |
| 4 | [Python syntax errors = zero extraction](./04-python-syntax-fallback.md) | `strategies/python_code.py` | Medium | Low |

Each file is self-contained: problem, evidence, proposed fix, affected files.
