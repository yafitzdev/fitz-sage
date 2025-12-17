# README.md
# Fitz v0.3.0 - Multi-Engine Knowledge Platform

**Fitz** is a modular knowledge engine platform that supports multiple retrieval-generation paradigms. Build RAG applications today, switch to CLaRa tomorrow, or create your own custom engine.

[![Tests](https://img.shields.io/github/actions/workflow/status/yafitzdev/fitz/ci.yml?label=tests)](https://github.com/yafitzdev/fitz/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Fitz** is a modular knowledge engine platform that supports multiple retrieval-generation paradigms. Build RAG applications today, experiment with CLaRa's compression-native approach, or create your own custom engine.

---

## What's New in v0.3.0

- ✅ **Multi-Engine Architecture**: Pluggable engine system with standardized interfaces
- ✅ **CLaRa Engine**: Apple's Continuous Latent Reasoning with 16x-128x document compression
- ✅ **Universal Runtime**: `run(query, engine="clara")` - switch engines with one parameter
- ✅ **Custom Engine Support**: Implement `KnowledgeEngine` protocol and register
- ✅ **Forward Compatible**: Your code won't break when new engines are added

This enables:

- ✅ **Clean abstractions**: Query → Engine → Answer (paradigm-agnostic)
- ✅ **Multiple paradigms**: Classic RAG today, CLaRa for research, custom engines anytime
- ✅ **Better organization**: Engines are self-contained, infrastructure is shared

---

## Quick Start

### Installation

```bash
pip install fitz

# With CLaRa support (requires GPU with 16GB+ VRAM)
pip install fitz[clara]
```

### Basic Usage

```python
from fitz.engines.classic_rag import run_classic_rag

# Simple query
answer = run_classic_rag("What is quantum computing?")
print(answer.text)

# View sources
for source in answer.provenance:
    print(f"Source: {source.source_id}")
    print(f"Excerpt: {source.excerpt}")
```

### Advanced Usage

```python
from fitz.core import Query, Constraints
from fitz.engines.classic_rag import create_classic_rag_engine

# Create reusable engine
engine = create_classic_rag_engine("config.yaml")

# Query with constraints
query = Query(
    text="Explain quantum entanglement",
    constraints=Constraints(
        max_sources=5,
        filters={"topic": "physics"}
    )
)

answer = engine.answer(query)
```

---

## Available Engines

### Classic RAG (Production Ready)

Traditional Retrieval-Augmented Generation - the default engine for production use.

```python
from fitz.engines.classic_rag import run_classic_rag

answer = run_classic_rag("What is machine learning?")
```

**Best for**: Production deployments, general knowledge bases, fine-grained retrieval control.

### CLaRa (Experimental)

Apple's Continuous Latent Reasoning - compression-native RAG with 16x-128x document compression.

```python
from fitz.engines.clara import run_clara, create_clara_engine

# Create engine and add documents
engine = create_clara_engine()
engine.add_documents(["Document 1...", "Document 2..."])

# Query
answer = engine.answer(Query(text="What patterns emerge?"))
```

**Best for**: Research, large document collections, multi-hop reasoning queries.

> ⚠️ **Hardware Requirements for CLaRa**:
> - **GPU**: 16GB+ VRAM (RTX 4090, A100, etc.)
> - **RAM**: 32GB+ for CPU fallback
> - **Disk**: ~14GB for model download
> 
> The CLaRa engine is fully implemented and tested, but the underlying 7B parameter model requires significant compute resources. For development/testing without a powerful GPU, use the Classic RAG engine.

---

## Architecture

```
fitz/
├── core/                  # Paradigm-agnostic contracts
│   ├── engine.py          # KnowledgeEngine protocol
│   ├── query.py           # Query type
│   └── answer.py          # Answer type
├── engines/
│   ├── classic_rag/       # Traditional RAG (production ready)
│   │   ├── engine.py      # ClassicRagEngine
│   │   ├── runtime.py     # run_classic_rag()
│   │   ├── pipeline/      # RAG pipeline logic
│   │   ├── retrieval/     # Retrieval plugins
│   │   └── generation/    # Generation logic
│   └── clara/             # CLaRa engine (experimental)
│       ├── engine.py      # ClaraEngine
│       ├── runtime.py     # run_clara()
│       └── config/        # CLaRa configuration
├── runtime/               # Multi-engine orchestration
├── llm/                   # Shared LLM service
├── vector_db/             # Shared vector DB service
└── ingest/                # Shared ingestion
```

**Key principle:** Query → Engine → Answer

All engines implement the same `KnowledgeEngine` protocol:
```python
def answer(self, query: Query) -> Answer:
    ...
```

---

## CLI

### Query Command

```bash
# Basic query
fitz-pipeline query "What is quantum computing?"

# With constraints
fitz-pipeline query "Explain X" --max-sources 5

# With filters
fitz-pipeline query "What is Y?" --filters '{"topic": "physics"}'

# With preset
fitz-pipeline query "Analyze this" --preset local
```

### Config Command

```bash
# Show resolved configuration
fitz-pipeline config show

# Show with custom config
fitz-pipeline config show --config my_config.yaml
```

---

## Configuration

### Classic RAG Config

```yaml
llm:
  plugin_name: openai
  kwargs:
    model: gpt-4

embedding:
  plugin_name: openai
  kwargs:
    model: text-embedding-3-small

vector_db:
  plugin_name: qdrant
  kwargs:
    url: http://localhost:6333

retriever:
  plugin_name: dense
  collection: my_knowledge
  top_k: 5

rgs:
  max_chunks: 10
  enable_citations: true
  strict_grounding: true
```

### CLaRa Config

```yaml
clara:
  model:
    model_name_or_path: "apple/CLaRa-7B-Instruct"
    variant: "instruct"
    device: "cuda"
    torch_dtype: "bfloat16"
  
  compression:
    compression_rate: 16
    doc_max_length: 2048
  
  retrieval:
    top_k: 5
  
  generation:
    max_new_tokens: 256
```

### Local Development (No API Keys)

```yaml
llm:
  plugin_name: local
  kwargs:
    model: llama3.2

embedding:
  plugin_name: local
  kwargs:
    model: nomic-embed-text

vector_db:
  plugin_name: local-faiss
```

---

## Custom Engines

Create your own engine by implementing the `KnowledgeEngine` protocol:

```python
from fitz.core import Query, Answer, Provenance
from fitz.runtime import EngineRegistry

class MyCustomEngine:
    def answer(self, query: Query) -> Answer:
        # Your logic here
        return Answer(
            text="Generated answer",
            provenance=[Provenance(source_id="src_1", excerpt="...")],
            metadata={"engine": "my_custom"}
        )

# Register
registry = EngineRegistry.get_global()
registry.register("my_custom", lambda c: MyCustomEngine())

# Use
from fitz import run
answer = run("Question?", engine="my_custom")
```

---

## Migration Guide (v0.2.x → v0.3.0)

### Old Code (v0.2.x)
```python
from fitz.pipeline.pipeline.engine import RAGPipeline
from fitz.pipeline.config.loader import load_config

config = load_config()
pipeline = RAGPipeline.from_config(config)
result = pipeline.run("What is X?")

print(result.answer)
for source in result.sources:
    print(source.chunk_id)
```

### New Code (v0.3.0)
```python
from fitz.engines.classic_rag import run_classic_rag

answer = run_classic_rag("What is X?")

print(answer.text)
for source in answer.provenance:
    print(source.source_id)
```

### Breaking Changes

1. **Import paths changed**: `fitz.pipeline.*` → `fitz.engines.classic_rag.*`
2. **Answer format changed**: `RGSAnswer.answer` → `Answer.text`
3. **Source format changed**: `sources` → `provenance`
4. **Entry point changed**: `RAGPipeline.run()` → `run_classic_rag()`

---

## Examples

See `examples/` directory:
- `quickstart.py` - Basic usage
- `full_pipeline.py` - Complete example with all features
- `ingestion_example.py` - Document ingestion

---

## Contributing

The engine architecture makes it easier to contribute:
- Add new engines in `fitz/engines/<name>/`
- Share infrastructure (`llm/`, `vector_db/`, etc.)
- Don't touch core contracts unless absolutely necessary

---

## Support

- Documentation: https://fitz.readthedocs.io
- Issues: https://github.com/yafitzdev/fitz/issues
- Discussions: https://github.com/yafitzdev/fitz/discussions