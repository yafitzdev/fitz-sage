# Configuration Reference

All configuration lives in `.fitz/config.yaml` (auto-created on first run).

---

## Example Config

```yaml
# LLM providers — format: provider/model
chat_fast: ollama/qwen3.5:2b        # Lightweight tasks (classify, detect, rewrite)
chat_balanced: ollama/qwen3.5:4b    # SQL generation, enrichment
chat_smart: ollama/qwen3.5:9b       # Answer synthesis

embedding: ollama/nomic-embed-text   # Text-to-vector
rerank: null                         # Reranker (cohere/rerank-v3.5 or null)
vision: null                         # VLM for PDF figures (cohere, openai, ollama)

collection: default                  # Active collection name
```

---

## LLM Tiers

| Key | Purpose | Used for |
|-----|---------|----------|
| `chat_fast` | Cheapest/fastest model | Query rewriting, classification, detection, guardrails |
| `chat_balanced` | Middle tier | SQL generation, table queries, enrichment |
| `chat_smart` | Best quality model | Answer synthesis |

**Local optimization:** When using `ollama`, the engine maps all tiers to `chat_balanced` automatically to avoid VRAM model swapping. Users control which model via `chat_balanced`.

**Cloud providers:** All three tiers are used as configured (no swap cost with API calls).

### Provider format

`provider/model` — examples:

```yaml
# Ollama (local)
chat_fast: ollama/qwen3.5:2b
embedding: ollama/nomic-embed-text

# Cohere (cloud)
chat_smart: cohere/command-a-03-2025
embedding: cohere/embed-v4.0
rerank: cohere/rerank-v3.5

# OpenAI (cloud)
chat_smart: openai/gpt-4o
embedding: openai/text-embedding-3-small
```

---

## Feature Control

Features are controlled by provider presence — no `enabled` flags:

| Feature | Enabled when | Disabled when |
|---------|-------------|---------------|
| Reranking | `rerank: cohere/rerank-v3.5` | `rerank: null` |
| VLM parsing | `parser: docling_vision` + `vision:` set | `parser: docling` or `parser: glm_ocr` |
| Enrichment | Chat provider available | No chat provider |

---

## Storage

```yaml
# Local (default) — embedded PostgreSQL, zero config
vector_db: pgvector

# External PostgreSQL
vector_db: pgvector
vector_db_kwargs:
  mode: external
  connection_string: postgresql://user:pass@host:5432/dbname
```

Data lives in `.fitz/pgdata/`. Collections are managed via `fitz collections`.

---

## Parser

```yaml
parser: glm_ocr    # Hybrid: pdfplumber + GLM-OCR for scanned pages
# parser: docling  # Docling (requires pip install fitz-ai[docs])
```

| Parser | Speed | Scanned pages | Install |
|--------|-------|---------------|---------|
| `glm_ocr` | Fast (28s/100pg) | GLM-OCR via ollama | Base install |
| `docling` | Slow (21min/100pg) | No | `pip install fitz-ai[docs]` |
| `docling_vision` | Slow | VLM figure descriptions | `pip install fitz-ai[docs]` + `vision:` |

---

## Environment Variables

| Provider | Variable |
|----------|----------|
| Cohere | `COHERE_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` |

Ollama requires no API key — just `ollama serve`.

---

## See Also

- [Feature Control](FEATURE_CONTROL.md)
- [CLI](CLI.md)
