# Documentation Fixes Plan

Tracking document for documentation accuracy improvements identified on 2026-01-31.

---

## Priority 1: Critical Accuracy Fixes

### Version Consistency
- [x] **CLAUDE.md** - Updated version from 0.6.2 to 0.7.0
- [x] **fitz_ai/__init__.py** - Updated version from 0.5.2 to 0.7.0 (was out of sync with pyproject.toml)
- [x] **README.md** - Badge already correct at 0.7.0

### False Claims
- [x] **docs/ENGINES.md** (Line 17) - Changed "Qdrant, FAISS" to "pgvector"

### Wrong File Paths
- [x] **docs/ENRICHMENT.md** (Line 313) - Fixed path to `fitz_ai/ingestion/enrichment/bus.py`

---

## Priority 2: Feature Doc Accuracy

### Enum Mismatch
- [x] **docs/features/temporal-queries.md** - Updated intent enum table to match code:
  - COMPARISON, TREND, POINT_IN_TIME, RANGE, SEQUENCE

### Detection Mechanism Clarification
- [x] **docs/features/freshness-authority.md** - Clarified that detection is LLM-based (via FreshnessModule)

---

## Priority 3: Broken Links

### Archive References
- [ ] **roadmap/archive/rag-gaps.md** - 8 broken relative paths (SKIPPED - orphaned content is intentional)

### Missing File References
- [x] **docs/features/tabular-data-routing.md** - Removed broken link to `./multi-table-joins.md`
- [x] **docs/CUSTOM_ENGINES.md** - Link to `../CONTRIBUTING.md` is correct (file exists at root)
- [ ] ~~docs/limitations/llm-sql-generation.md~~ - File was deleted by user

---

## Priority 4: Missing Feature Documentation

- [x] **docs/features/entity-graph.md** - Created documentation for entity graph feature
- [x] **docs/features/sparse-search.md** - Created documentation for BM25/full-text search

### Cross-Linking
- [x] **README.md** - Added entity-graph and sparse-search to feature table
- [x] **docs/features/hybrid-search.md** - Added Related Features section linking to sparse-search
- [x] **docs/ENRICHMENT.md** - Added link from Entity Module to entity-graph feature

---

## Out of Scope (Intentional)

The following are intentionally not linked from README:
- `docs/blog/` - Internal blog drafts
- `docs/roadmap/` - Internal planning
- `docs/limitations/` - Reduced to 1 file (2 obsolete files removed)

---

## Progress Log

| Date | Item | Status |
|------|------|--------|
| 2026-01-31 | Plan created | Done |
| 2026-01-31 | Version sync (CLAUDE.md, __init__.py) | Done |
| 2026-01-31 | ENGINES.md vector DB claim | Done |
| 2026-01-31 | ENRICHMENT.md file path | Done |
| 2026-01-31 | temporal-queries.md enum | Done |
| 2026-01-31 | freshness-authority.md detection | Done |
| 2026-01-31 | tabular-data-routing.md broken link | Done |
| 2026-01-31 | Created entity-graph.md | Done |
| 2026-01-31 | Created sparse-search.md | Done |
| 2026-01-31 | Added to README feature table | Done |
| 2026-01-31 | Cross-linked related docs | Done |

---

## Additional Feature Documentation

- [x] **docs/features/enterprise-gateway.md** - Created documentation for enterprise LLM gateway support

### Cross-Linking
- [x] **README.md** - Added enterprise gateway to "Other Features at a Glance"
- [x] **docs/CONFIG.md** - Added enterprise to available plugins, linked to feature doc

---

## Reranking Feature Documentation

- [x] **docs/features/reranking.md** - Created documentation for cross-encoder reranking

### Cross-Linking
- [x] **README.md** - Added reranking to feature table
- [x] **docs/CONFIG.md** - Added link to reranking feature doc
- [x] **docs/features/hybrid-search.md** - Added reranking to Related Features
- [x] **docs/features/sparse-search.md** - Added reranking to Related Features
- [x] **docs/features/multi-hop-reasoning.md** - Added reranking to Related Features
- [x] **docs/features/unified-storage.md** - Added reranking to Related Features

---

## Reranking Baked-In Refactor (2026-02-01)

Removed `dense_rerank` plugin - reranking is now baked into `dense` plugin and auto-enabled when rerank provider is configured.

### Code Changes
- [x] **fitz_ai/engines/fitz_krag/retrieval/plugins/dense.yaml** - Added rerank/threshold steps with `enabled_if: reranker`
- [x] **fitz_ai/engines/fitz_krag/retrieval/plugins/dense_rerank.yaml** - Deleted
- [x] **fitz_ai/engines/fitz_krag/config/schema.py** - Updated retrieval_plugin description
- [x] **fitz_ai/engines/fitz_krag/config/default.yaml** - Updated comment
- [x] **fitz_ai/engines/fitz_krag/retrieval/loader.py** - Updated docstring
- [x] **fitz_ai/engines/fitz_krag/retrieval/registry.py** - Updated docstring
- [x] **tests/unit/test_retrieval_yaml_plugins.py** - Updated tests to use `dense` with reranker

### Documentation Updates
- [x] **CLAUDE.md** - Updated feature control section, added reranking to baked-in list
- [x] **docs/FEATURE_CONTROL.md** - Rewrote reranking section for provider-presence pattern
- [x] **docs/features/reranking.md** - Updated to reflect baked-in behavior
- [x] **docs/CONFIG.md** - Removed dense_rerank from plugins table
- [x] **docs/ARCHITECTURE.md** - Updated feature control examples
- [x] **docs/config_examples.md** - Updated examples
- [x] **docs/ENGINES.md** - Updated config example
- [x] **docs/CLI.md** - Updated config example
- [x] **docs/PLUGINS.md** - Updated reranking control section
- [x] **docs/features/multi-query-rag.md** - Updated plugin reference

---

## Summary

**Completed**: 11 fixes + 4 new feature docs + cross-linking + reranking baked-in refactor
**Remaining**: None (archive broken links skipped intentionally)
