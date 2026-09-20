<!-- docs/MANAGED_MODELS.md -->
# Managed Models

fitz-sage retrieval works without an API key or an external inference server.
Its default query expansion is derived from the indexed collection and stored
in SQLite, so it does not download or run a generative model.

## Local Models

| Job | Model | Runtime | Why it exists |
|---|---|---|---|
| Reranking | `Alibaba-NLP/gte-reranker-modernbert-base` | raw `onnxruntime`, CPU | Cross-encoder precision over broad recall candidates. |
| Governance | `yafitzdev/pyrrho-v2-nano-g1` at revision `948f0500b74871cfaec7689a01d4eab0dd516e1b` | raw `onnxruntime`, CPU | Accepted immutable Pyrrho default; custom local or commit-pinned models are supported. |

Both models load pre-built ONNX graphs through plain ONNX Runtime. They do not
require `optimum`, `llama.cpp`, GGUF, or an OpenAI-compatible server.

## Model-Free Semantic Expansion

During foreground indexing, fitz-sage extracts terms and evidence into the
collection database. The index records:

- abbreviation definitions such as `Service Level Agreement (SLA)`;
- aliases and renamed terms expressed in source text;
- error codes and nearby product, component, and operation phrases;
- code/config identifier spellings and human-readable forms;
- repeated phrases and sentence-level co-occurrence relationships.

Queries match these forms and traverse supported relations and clusters. Every
suggestion in the run trace includes its relation, score, and supporting
document count. Reindexing a file replaces its evidence, and deleting a file
retracts it. The aggregate graph is rebuilt lazily when the corpus changes.

## Download Behavior

| Model | Trigger |
|---|---|
| ONNX reranker | First retrieval pass that has enough candidates to rerank. |
| Pyrrho v2 | First query-plan or evidence-decision call. |

After the first download, subsequent runs reuse the cached snapshots. SQLite
semantic expansion is ready as soon as `point()` finishes and needs no model
download.

## Local Smoke Check

Run the standard local CPU path against a tiny generated corpus:

```bash
python tools/smoke_local_retrieval.py
```

The script indexes the corpus, reports semantic-index statistics, and executes
queries through the term graph, reranker, and Pyrrho. It is a runtime smoke
check, not a retrieval-quality benchmark.

## Offline and Air-Gapped Use

For disconnected deployments, warm the remaining managed models on a connected
machine, then copy the cache to the target machine:

```bash
python -c "from fitz_sage.llm.providers.onnx_reranker import OnnxReranker; OnnxReranker().rerank('warmup', ['one', 'two'])"
python -c "from fitz_sage.integrations.pyrrho import create_pyrrho; create_pyrrho('pyrrho').decide('warmup', [{'source_id': 'warmup', 'text': 'warmup evidence'}])"
```

The Hugging Face cache contains the Pyrrho snapshot, reranker ONNX file,
tokenizers, and configs. On the target machine, point `HF_HOME` at the copied
cache and set `HF_HUB_OFFLINE=1`.

## Optional Chat Work

Query intelligence, entity extraction, hierarchy summaries, and answer
synthesis can use a configured chat provider. These features are optional and
separate from standard query expansion. Without a chat tier, `point()` marks
background enrichment not applicable and the complete retrieval path remains
available.
