# Feature Control Architecture

This document explains how optional features (VLM for figure description, reranking) are controlled in Fitz.

---

## Design Philosophy

Fitz uses a **provider-presence pattern** for optional features:

- **Config declares WHICH** provider/model to use
- **Provider presence determines IF** the feature is used
- **No `enabled: true/false` flags** - setting a provider enables the feature

This keeps the config declarative and avoids boolean flags that can get out of sync.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  CONFIG (.fitz/config.yaml — auto-created on first run)         │
│  Declares WHICH provider/model to use                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  vision: cohere                  rerank: cohere/rerank-v3.5    │
│  parser: docling_vision          collection: default            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PROVIDER PRESENCE determines IF the feature is used            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  VLM (controlled by parser plugin):                             │
│    ┌──────────────────────┐    ┌──────────────────┐             │
│    │  parser: docling     │    │ parser:          │             │
│    ├──────────────────────┤    │ docling_vision   │             │
│    │ No VLM               │    ├──────────────────┤             │
│    │ Figures → "[Figure]" │    │ Uses VLM from    │             │
│    └──────────────────────┘    │ vision: config   │             │
│                                └──────────────────┘             │
│                                                                 │
│  Reranking (controlled by provider presence):                   │
│    ┌────────────────────┐    ┌────────────────────────┐         │
│    │   rerank: null     │    │ rerank:                │         │
│    ├────────────────────┤    │ cohere/rerank-v3.5     │         │
│    │ No reranking       │    ├────────────────────────┤         │
│    │ Pure vector search │    │ Reranking auto-        │         │
│    └────────────────────┘    │ enabled (baked)        │         │
│                              └────────────────────────┘         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## VLM (Vision Language Model) Control

VLM is used to describe figures and images in PDFs during ingestion.

### How it works:

1. Set a vision provider (cohere, openai, anthropic, ollama) in `.fitz/config.yaml`
2. Config saves the provider in `vision:` section
3. **Parser plugin** determines if VLM is actually used:
   - `parser: docling` → Figures become `[Figure]` placeholder
   - `parser: docling_vision` → Figures get VLM-generated descriptions

### Config example:

```yaml
# Parser choice enables VLM
parser: docling_vision  # ← Uses VLM
# parser: docling       # ← No VLM
# parser: glm_ocr       # ← Fast default, no VLM

# Vision provider (used only if parser: docling_vision)
vision: cohere          # or openai, anthropic, ollama
```

### Key files:

| File | Purpose |
|------|---------|
| `fitz_ai/ingestion/parser/router.py` | Routes files to parsers based on config |
| `fitz_ai/ingestion/parser/plugins/docling.py` | Standard parser (no VLM) |
| `fitz_ai/ingestion/parser/plugins/docling_vision.py` | VLM-enabled parser |
| `fitz_ai/cli/commands/ingest.py` | Reads `chunking.default.parser` config |

---

## Reranking Control

Reranking improves retrieval quality by re-scoring chunks with a cross-encoder model.

### How it works:

1. Set a rerank provider (typically cohere) in `.fitz/config.yaml`
2. Config saves the provider in `rerank:` section
3. **Reranking is automatically enabled** when a rerank provider is configured
4. No separate plugin choice needed - it's baked into the `dense` retrieval pipeline

### Config example:

```yaml
# Rerank provider presence enables reranking
rerank: cohere/rerank-v3.5      # ← Reranking enabled
# rerank: null                  # ← No reranking (default)
```

### Key files:

| File | Purpose |
|------|---------|
| `fitz_ai/engines/fitz_krag/retrieval/plugins/dense.yaml` | Retrieval pipeline (rerank steps have `enabled_if: reranker`) |
| `fitz_ai/engines/fitz_krag/retrieval/loader.py` | Skips rerank steps when no reranker provided |
| `fitz_ai/llm/providers/cohere.py` | Cohere rerank implementation |

---

## Why This Pattern?

### Advantages:

1. **No boolean flags to sync** - Provider presence itself is the toggle
2. **Baked-in intelligence** - Reranking joins other automatic features like hybrid search
3. **Simpler config** - One retrieval plugin, not two
4. **Explicit** - Reading the config tells you exactly what will happen

### Comparison with alternatives:

```yaml
# ❌ OLD: Plugin choice was the toggle
retrieval:
  plugin_name: dense_rerank     # Had to choose plugin

# ✅ NEW: Provider presence is the toggle
rerank: cohere/rerank-v3.5      # This alone enables reranking
```

---

## Adding New Optional Features

Follow this pattern for any new optional feature:

1. **For ingestion-time features** (like VLM): Create two parser plugins
2. **For query-time features** (like reranking): Use `enabled_if` in pipeline steps

Example for a hypothetical "summarizer" feature at query time:

```yaml
# In retrieval plugin YAML:
steps:
  - type: summarize
    enabled_if: summarizer      # Only runs if summarizer dependency provided

# In config:
summarizer: cohere              # Presence enables the feature
```

---

## Quick Reference

| Feature | Config Key | Enable | Disable |
|---------|-----------|--------|---------|
| VLM | `vision:` + `parser:` | `parser: docling_vision` | `parser: docling` or `parser: glm_ocr` |
| Rerank | `rerank:` | `rerank: cohere/rerank-v3.5` | `rerank: null` (or omit) |

---

## See Also

- [Reranking Feature](features/retrieval/reranking.md) - Detailed reranking documentation
- [PLUGINS.md](PLUGINS.md) - Plugin development guide
- [CLI.md](CLI.md) - CLI reference
