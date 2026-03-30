## Code Generation Rules (MANDATORY)

1. **Never alter unrelated code** - Touch only code directly related to the requested change
2. **File path comment required** - First line: `# fitz_sage/engines/fitz_krag/engine.py`
3. **No legacy code** - No backwards compatibility, no deprecated code, no shims. Delete completely when removing
4. **Tests follow architecture** - Fix tests to match new architecture, never compromise code quality for tests
5. **Always use .venv for pip** - `.venv/Scripts/pip install <package>` (Windows) or `.venv/bin/pip install <package>` (Unix)

## Project Overview

**fitz-sage** - Local-first, modular RAG knowledge engine platform with epistemic honesty and full provenance.

```bash
pip install -e ".[dev]"   # Install for development
pytest                    # Run tests
black . && isort .        # Format code
python -m tools.contract_map --fail-on-errors  # Check architecture
```

## Architecture

```
fitz_sage/
├── core/          # Paradigm-agnostic (Query, Answer, Provenance, Constraints)
├── engines/fitz_krag/   # KRAG: retrieval/, generation/, pipeline/
├── retrieval/     # SHARED intelligence (detection, sparse, entity_graph, hyde, rewriter)
├── ingestion/     # Parser → Chunking → Enrichment
├── llm/           # Chat, Embedding, Rerank, Vision providers (Python, protocol-based)
├── storage/       # PostgreSQL connection manager
├── vector_db/     # Vector DB abstraction + pgvector plugin
├── cloud/         # Encrypted cache API (AES-256-GCM, org_key never leaves local)
├── tabular/       # CSV/table query with SQL generation
├── runtime/       # Multi-engine orchestration
├── cli/           # 14 commands; api/; sdk/
```

**Layer dependencies** (enforced by `contract_map`):
- `core/` ← no imports from engines/, ingestion/
- `engines/` ← core/, llm/, vector_db/, storage/, retrieval/
- `retrieval/`, `llm/`, `ingestion/` ← core/ only
- `vector_db/` ← core/, storage/
- `runtime/`, `cli/` ← everything

**Engine protocol**: all engines implement `answer(query: Query) -> Answer` from `core/engine.py`.

## Code Style

- **Black** (line-length 100), **isort** (black profile)
- **Type hints** required for public APIs; **Docstrings**: Google style
- `snake_case` modules/functions, `PascalCase` classes, `UPPER_SNAKE` constants

## Developer Tools (use these, not find/grep/cat)

| Task | Tool |
|------|------|
| Find files | `fd <pattern>` |
| Search text | `rg <pattern>` |
| Search code structure | `ast-grep --pattern '<code>'` |
| Symbol definitions | `ctags -R` then `grep "Symbol" tags` |
| Codebase stats | `tokei` / `tokei <dir>` |
| JSON/YAML | `jq '.<key>' file.json` / `yq '.<key>' file.yaml` |
| CSV | `xsv headers`, `xsv count`, `xsv stats` |
| Git diffs | `git diff \| delta` |

## Testing

```bash
pytest                      # All tests
pytest -m "not slow"        # Skip slow
pytest --cov=fitz_sage        # With coverage
pytest tests/unit/          # Unit only
```

**Markers**: `slow`, `integration`, `e2e`, `performance`, `security`, `scalability`, `chaos`

## Design Principles

1. **Explicit over clever** - No magic, config-driven
2. **Honest over helpful** - "I don't know" > hallucination
3. **No `enabled` flags** - Provider presence IS the feature toggle
4. **All retrieval intelligence is baked in** - not configured

## Feature Control

Config declares WHAT provider; plugin choice determines IF feature is used:
- `rerank: cohere` → reranking enabled; `rerank: null` → disabled
- `parser: "docling_vision"` → uses VLM from `vision:` config; `parser: "docling"` → no VLM

## Retrieval Intelligence (automatic, not configured)

Temporal, query expansion, hybrid BM25+dense (RRF), multi-query, comparison handling, keyword filtering, entity graph, freshness boosting, aggregation detection, multi-hop, HyDE, reranking.

**Core files**:
- `engines/fitz_krag/retrieval/steps/vector_search.py` - orchestrates all intelligence
- `retrieval/detection/registry.py` - `DetectionOrchestrator` (ML gate → LLM modules)
- `retrieval/detection/classifier.py` - `DetectionClassifier` (ML + keyword gating, skips unnecessary LLM calls)
- `retrieval/detection/modules/` - temporal, aggregation, comparison, freshness, rewriter
- `retrieval/detection/detectors/expansion.py` - dict-based synonym/acronym expansion (not LLM)

**Detection gating**: `DetectionClassifier` (ML model + keyword regex) predicts which categories need LLM → orchestrator runs only flagged modules. Fail-open: if classifier unavailable, all LLM modules run.

## Extensibility (DO NOT create parallel implementations)

| Need | Extend | NOT |
|------|--------|-----|
| Detect query patterns | `retrieval/detection/modules/` (new `DetectionModule`) | Inline regex in VectorSearchStep |
| Chunk metadata at ingestion | `ingestion/enrichment/modules/chunk/` (new `EnrichmentModule`) | Post-hoc at query time |
| Add synonyms/acronyms | `detection/detectors/expansion.py` dicts | New expander class |

**Detection module pattern** (LLM-based): implement `category`, `json_key`, `prompt_fragment()`, `parse_result()` → add to `DEFAULT_MODULES` in `modules/__init__.py`. All modules combine into one LLM call. ML gating in `classifier.py` decides which modules actually run.

**Enrichment module pattern**: implement `name`, `json_key`, `prompt_instruction()`, `parse_result()`, `apply_to_chunk()` → register in `enrichment/bus.py::create_default_enricher()`. Batched (15/call), one LLM call per batch.

**Enrichment pipeline**: `ChunkEnricher` (summary, keywords, entities, content_type) → `HierarchyEnricher` (L1 group summaries, L2 corpus chunk) → `ArtifactRegistry` (navigation, interfaces, etc.)

## Key Files

| Purpose | Path |
|---------|------|
| Engine protocol | `core/engine.py` |
| KRAG engine | `engines/fitz_krag/engine.py` |
| Config schema | `engines/fitz_krag/config/schema.py` |
| Config loader | `config/loader.py` |
| Detection orchestrator | `retrieval/detection/registry.py` |
| Detection classifier | `retrieval/detection/classifier.py` |
| Semantic matcher | `governance/constraints/semantic.py` |
| Enrichment bus | `ingestion/enrichment/bus.py` |
| Cloud client | `cloud/client.py`, `cloud/crypto.py` |
| Multi-hop | `engines/fitz_krag/retrieval/multihop/` |
| Guardrails | `engines/fitz_krag/guardrails/plugins/` |
| Parser routing | `ingestion/parser/router.py` |

## Configuration System

Layered merge: package defaults → `~/.fitz/config/<engine>.yaml` (user overrides).

Sections: `chat`, `embedding`, `rerank`, `vision`, `vector_db`, `vector_db_kwargs`, `retrieval`, `chunking`, `cloud`.

Storage: `local` (embedded PostgreSQL via fitz-pgserver, zero config) or `external` (provide `connection_string`).
