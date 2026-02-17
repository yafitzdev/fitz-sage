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
│                         fitz init                               │
│              (configures providers and models)                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  CONFIG (.fitz/config/fitz_krag.yaml)                           │
│  Declares WHICH provider/model to use                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  vision:                         rerank:                        │
│    plugin_name: cohere             plugin_name: cohere          │
│    kwargs: {}                      kwargs:                      │
│                                      model: rerank-v3.5         │
│                                                                 │
│  chunking:                       retrieval:                     │
│    default:                        plugin_name: dense           │
│      parser: docling_vision        collection: default          │
│      plugin_name: recursive        top_k: 5                     │
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
│    │     docling          │    │  docling_vision  │             │
│    ├──────────────────────┤    ├──────────────────┤             │
│    │ No VLM               │    │ Uses VLM from    │             │
│    │ Figures → "[Figure]" │    │ vision: config   │             │
│    └──────────────────────┘    └──────────────────┘             │
│                                                                 │
│  Reranking (controlled by provider presence):                   │
│    ┌────────────────────┐    ┌──────────────────┐               │
│    │   rerank: null     │    │  rerank: cohere  │               │
│    ├────────────────────┤    ├──────────────────┤               │
│    │ No reranking       │    │ Reranking auto-  │               │
│    │ Pure vector search │    │ enabled (baked)  │               │
│    └────────────────────┘    └──────────────────┘               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## VLM (Vision Language Model) Control

VLM is used to describe figures and images in PDFs during ingestion.

### How it works:

1. **`fitz init`** prompts for a vision provider (cohere, openai, anthropic, ollama)
2. Config saves the provider in `vision:` section
3. **Parser plugin** determines if VLM is actually used:
   - `parser: docling` → Figures become `[Figure]` placeholder
   - `parser: docling_vision` → Figures get VLM-generated descriptions

### Config example:

```yaml
# Parser choice enables VLM
chunking:
  default:
    parser: docling_vision  # ← Uses VLM
    # parser: docling       # ← No VLM
    plugin_name: recursive
    kwargs:
      chunk_size: 1000
      chunk_overlap: 200

# Vision provider (used only if parser: docling_vision)
vision:
  plugin_name: cohere
  kwargs: {}
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

1. **`fitz init`** prompts for a rerank provider (typically cohere)
2. Config saves the provider in `rerank:` section
3. **Reranking is automatically enabled** when a rerank provider is configured
4. No separate plugin choice needed - it's baked into the `dense` retrieval pipeline

### Config example:

```yaml
# Rerank provider presence enables reranking
rerank: cohere                  # ← Reranking enabled
# rerank: null                  # ← No reranking (default)

# Retrieval pipeline (reranking auto-injected when provider configured)
retrieval:
  plugin_name: dense            # Single plugin - handles both cases
  collection: default
  top_k: 5
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
rerank: cohere                  # This alone enables reranking
retrieval:
  plugin_name: dense            # Single plugin
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

| Feature | Config Section | Enable | Disable |
|---------|---------------|--------|---------|
| VLM | `vision:` + `chunking.default.parser` | `parser: docling_vision` | `parser: docling` |
| Rerank | `rerank:` | `rerank: cohere` | `rerank: null` (or omit) |

---

## See Also

- [Reranking Feature](features/retrieval/reranking.md) - Detailed reranking documentation
- [PLUGINS.md](PLUGINS.md) - Plugin development guide
- [CLI.md](CLI.md) - CLI reference
