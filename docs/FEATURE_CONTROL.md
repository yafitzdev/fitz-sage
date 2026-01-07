# Feature Control Architecture

This document explains how optional features (VLM for figure description, reranking) are controlled in Fitz.

---

## Design Philosophy

Fitz uses a **plugin-based feature control** pattern:

- **Config declares WHICH** provider/model to use
- **Plugin choice determines IF** the feature is used
- **No `enabled: true/false` flags** - the plugin name IS the toggle

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
│  CONFIG (.fitz/config/fitz_rag.yaml)                            │
│  Declares WHICH provider/model to use                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  vision:                         rerank:                        │
│    plugin_name: cohere             plugin_name: cohere          │
│    kwargs: {}                      kwargs:                      │
│                                      model: rerank-v3.5         │
│                                                                 │
│  chunking:                       retrieval:                     │
│    default:                        plugin_name: dense_rerank    │
│      parser: docling_vision        collection: default          │
│      plugin_name: recursive        top_k: 5                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PLUGIN determines IF the feature is used                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Parser Plugin (controls VLM usage):                            │
│    ┌──────────────────────┐    ┌──────────────────┐             │
│    │     docling          │    │  docling_vision  │             │
│    ├──────────────────────┤    ├──────────────────┤             │
│    │ No VLM               │    │ Uses VLM from    │             │
│    │ Figures → "[Figure]" │    │ vision: config   │             │
│    └──────────────────────┘    └──────────────────┘             │
│                                                                 │
│  Retrieval Plugin (controls reranking):                         │
│    ┌────────────────────┐    ┌──────────────────┐               │
│    │      dense         │    │   dense_rerank   │               │
│    ├────────────────────┤    ├──────────────────┤               │
│    │ No reranking       │    │ Uses reranker    │               │
│    │ Pure vector search │    │ from rerank: cfg │               │
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
3. **Retrieval plugin** determines if reranking is actually used:
   - `retrieval.plugin_name: dense` → Pure vector search
   - `retrieval.plugin_name: dense_rerank` → Vector search + reranking

### Config example:

```yaml
# Retrieval plugin choice enables reranking
retrieval:
  plugin_name: dense_rerank  # ← Uses reranking
  # plugin_name: dense       # ← No reranking
  collection: default
  top_k: 5

# Rerank provider (used only if retrieval uses reranking)
rerank:
  plugin_name: cohere
  kwargs:
    model: rerank-v3.5
```

### Key files:

| File | Purpose |
|------|---------|
| `fitz_ai/retrieval/plugins/dense.yaml` | Pure dense retrieval |
| `fitz_ai/retrieval/plugins/dense_rerank.yaml` | Dense + reranking |
| `fitz_ai/engines/fitz_rag/engine.py` | Loads retrieval plugin |

---

## Why This Pattern?

### Advantages:

1. **No boolean flags to sync** - The plugin name itself is the toggle
2. **Clear separation** - Config says "what", plugin says "if"
3. **Extensibility** - Add new parser/retrieval variants without config changes
4. **Explicit** - Reading the config tells you exactly what will happen

### Comparison with alternatives:

```yaml
# ❌ BAD: Boolean flags (can get out of sync)
vision:
  enabled: true          # ← Easy to forget to toggle
  plugin_name: cohere

# ✅ GOOD: Plugin choice is the toggle
chunking:
  default:
    parser: docling_vision  # ← This IS the toggle
```

---

## Adding New Optional Features

Follow this pattern for any new optional feature:

1. **Create two plugins** - one with the feature, one without
2. **Add provider config section** - for the underlying service
3. **Let plugin choice control usage** - no enabled flags

Example for a hypothetical "summarizer" feature:

```yaml
# Summarization provider config
summarization:
  plugin_name: cohere
  kwargs:
    model: command-r

# Chunker plugin controls usage
chunking:
  default:
    plugin_name: recursive            # ← No summarization
    # plugin_name: recursive_summary  # ← With summarization
```

---

## Quick Reference

| Feature | Config Section | Plugin Location | Enable | Disable |
|---------|---------------|-----------------|--------|---------|
| VLM | `vision:` | `chunking.default.parser` | `docling_vision` | `docling` |
| Rerank | `rerank:` | `retrieval.plugin_name` | `dense_rerank` | `dense` |

---

## See Also

- [PLUGINS.md](PLUGINS.md) - Plugin development guide
- [CLI.md](CLI.md) - CLI reference
- `fitz_ai/cli/commands/init.py` - Init wizard implementation
