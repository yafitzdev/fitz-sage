# KRAG — Knowledge Routing Augmented Generation

## The Problem with Traditional RAG

Traditional RAG follows a linear pipeline: **chunk documents → embed chunks → retrieve by similarity → generate answer**. This works for simple factual lookups but breaks down in predictable ways:

1. **Chunks are dumb boundaries.** A 512-token window doesn't know where a function ends or a section begins. You get half a class definition in one chunk and the other half in the next.
2. **No structural awareness.** Chunks don't know that `file_a.py` imports `helper()` from `file_b.py`. There's no concept of "what calls this?" or "what depends on that?"
3. **Content types are flattened.** Code, prose, tables, and figures all become identical text blobs. A SQL table chunked into text fragments loses its queryable structure entirely.
4. **Retrieval is one-shot.** One embedding, one similarity search, one result set. If the answer spans multiple sources or requires traversing relationships, traditional RAG gives you random fragments.

## The Problem with Agentic RAG

Agentic RAG addresses some of traditional RAG's limitations by wrapping an LLM agent around the retrieval loop. The agent can rewrite queries, decide which tools to call, and iterate until it finds a good answer. But it introduces its own problems:

1. **Latency.** Every agent "hop" is an LLM call. A 3-hop retrieval means 3 round-trips to the LLM before you even start generating. For real-time applications, this is often unacceptable.
2. **Cost.** Each reasoning step consumes tokens. Agent-driven retrieval can 5-10x the token cost of a single query.
3. **Unpredictable behavior.** Agent decisions are non-deterministic. The same query might take 1 hop or 5 hops depending on the LLM's reasoning. This makes latency, cost, and quality hard to predict or guarantee.
4. **Complexity without structure.** The agent compensates for missing structure with reasoning. It doesn't *know* that `AuthService` depends on `TokenValidator` — it has to *figure it out* by searching, reading, and reasoning across chunks. This is expensive work that could be a simple graph lookup if the structure existed.

## The Problem with GraphRAG

GraphRAG (Microsoft) takes a different angle: build a knowledge graph from documents using LLM-based entity and relationship extraction, then use community detection to create hierarchical summaries. It's strong for global queries ("what are the main themes?") but has its own trade-offs:

1. **Ingestion cost.** Every document goes through LLM entity/relationship extraction. For a large codebase or document corpus, this means thousands of LLM calls just to build the graph. Ingestion can be 10-100x more expensive than traditional RAG.
2. **Hallucinated edges.** The knowledge graph is LLM-generated. The model decides what entities exist and how they relate. It can invent relationships that don't exist or miss ones that do. There's no ground truth — the graph is as reliable as the LLM's reading comprehension.
3. **No native code understanding.** GraphRAG extracts entities like "AuthService" and "TokenValidator" as text labels, but doesn't know that one is a class and the other is a function, or that they're connected by an import statement. The structural relationship is guessed, not parsed.
4. **Community detection overhead.** Building hierarchical communities (Leiden algorithm) adds complexity and another layer of approximation. The communities are statistical clusters, not semantic groupings.

## KRAG: Structure-First Retrieval

KRAG takes a different approach: **extract structure at ingestion time, use it at query time.** Instead of compensating for missing knowledge with agent reasoning, KRAG builds the structural knowledge into the index.

```
Traditional RAG:
  Document → [chunk] [chunk] [chunk] → embed → similarity search → answer

Agentic RAG:
  Document → [chunk] [chunk] [chunk] → embed → agent(search → reason → re-search → reason) → answer

GraphRAG:
  Document → LLM extracts entities/relations → knowledge graph → community detection → hierarchical summaries → answer

KRAG:
  Document → [symbols] [sections] [tables] → embed with structure → routed search → expand via graph → answer
```

### Core Ideas

**1. Addresses, not chunks**

KRAG doesn't store text fragments. It stores *addresses* — pointers to specific, meaningful units of content:

- **Symbols**: A function, class, or method — extracted by tree-sitter with its qualified name (`module.ClassName.method`), line range, references to other symbols, and import relationships.
- **Sections**: A heading and its content — extracted with parent/child hierarchy, level, and title.
- **Tables**: A structured dataset — stored as a native PostgreSQL table with auto-detected schema.

Each address has a summary, a vector embedding, and structural metadata. You never search raw text — you search the structured index.

**2. Import graphs, not text search**

When KRAG ingests a Python codebase, it doesn't just extract symbols — it builds an import graph. Every file's imports are tracked: what it imports, where from, and what imports it. This means:

- "What depends on AuthService?" → graph traversal, not text search
- "What would break if I change this function?" → reverse dependency lookup
- "Show me the callers of validate_token()" → follow references across files

Traditional RAG would need to search for the text "validate_token" and hope it appears near import statements. Agentic RAG would need the agent to reason about it over multiple hops. KRAG just walks the graph.

**3. Content-type routing**

Different content types need different retrieval strategies. KRAG routes queries to the right strategy automatically:

| Content | Strategy | Why |
|---------|----------|-----|
| Code | Symbol search (name + BM25 + vector) | Functions have names, types, and summaries — use all three signals |
| Documents | Section search (hierarchy + BM25 + vector) | Sections have titles, parents, and children — use the tree |
| Tables | SQL generation from natural language | Tables are structured data — query them with SQL, not similarity |
| Fallback | Chunk search | When nothing else fits, fall back to traditional retrieval |

Traditional RAG runs the same similarity search regardless of whether you're asking about a function, a policy document, or a spreadsheet.

**4. Structural expansion**

After finding relevant addresses, KRAG expands context using structural knowledge:

- **Same-file references**: If symbol A references symbols B and C in the same file, include them.
- **Import-based expansion**: If the query is about `engine.py`, include key symbols from files it imports.
- **Entity linking**: If two symbols share a named entity (e.g., both mention "AuthService"), link them.
- **Section hierarchy**: If a deeply nested section matches, include its parent for context.

This isn't agent reasoning — it's deterministic graph traversal with zero LLM calls.

## Comparison

| Dimension | Traditional RAG | Agentic RAG | GraphRAG | KRAG |
|-----------|----------------|-------------|----------|------|
| **Retrieval unit** | Fixed-size text chunk | Fixed-size text chunk | Entity/community node | Symbols, sections, tables |
| **Structure awareness** | None | Reasoned per-query (LLM) | LLM-extracted graph | Deterministic (parsed AST + imports) |
| **Cross-file dependencies** | Text search | Agent discovers via multi-hop | Entity co-occurrence | Import graph traversal |
| **Content-type handling** | All treated as text | Agent decides tools | All treated as text | Routed by content type |
| **Ingestion cost** | Low (chunk + embed) | Low (chunk + embed) | Very high (LLM per doc for entity extraction) | Medium (parse + summarize + embed) |
| **Query latency** | Fast (1 search) | Slow (N LLM calls) | Fast (graph lookup) | Fast (1 search + graph expansion) |
| **Cost per query** | Low | High (agent reasoning) | Low | Low (no extra LLM for retrieval) |
| **Predictability** | Deterministic | Non-deterministic | Deterministic | Deterministic |
| **Graph accuracy** | N/A | N/A | Probabilistic (LLM-generated) | Exact (parsed from source) |
| **Code understanding** | Line-split fragments | Agent reads and reasons | Entity labels only | Parsed AST with qualified names |
| **Table queries** | Chunked text | Agent generates SQL | Chunked text | Native SQL execution |
| **Global queries** | Poor (no overview) | Agent synthesizes | Strong (community summaries) | Good (hierarchical summaries) |
| **Best for** | Simple document Q&A | Complex multi-source research | Corpus-level themes | Code + document + data retrieval |

## Where KRAG Uses Agent-Style Techniques

KRAG isn't anti-agent. It incorporates agent-inspired techniques where they add value — but makes them deterministic or bounded:

- **Multi-hop reasoning**: Iterative retrieval for complex questions, but with a fixed hop limit and deterministic bridge extraction — not open-ended agent reasoning.
- **HyDE**: Generates hypothetical documents for abstract queries, but as a single bounded step — not an agent loop.
- **Query rewriting**: Resolves pronouns and context via LLM, but as a single pre-processing step.
- **Detection-based routing**: Classifies query intent (temporal, comparison, aggregation) via one LLM call, then uses deterministic strategies — not agent deliberation.

The pattern: use the LLM for *classification* and *generation*, but use *structure* for retrieval and expansion.

## Architecture

```
                    ┌───────────────────────────────────────────┐
                    │             INGESTION                     │
                    │                                           │
  Code files ──────►  tree-sitter ──► SymbolStore (+ imports)   │
  Documents ───────►  parser ──────► SectionStore (+ hierarchy) │
  CSV/tables ──────►  schema detect ► TableStore (+ SQL)        │
                    │                                           │
                    │  All units get: summary, embedding,       │
                    │  keywords, entities, hierarchy summary    │
                    └───────────────────────────────────────────┘
                                      │
                                      ▼
                    ┌───────────────────────────────────────────┐
                    │             QUERY TIME                    │
                    │                                           │
  Query ──► Rewrite ──► Analyze ──► Detect ──► Route            │
                    │                             │             │
                    │              ┌──────────────┤             │
                    │              ▼              ▼             │
                    │         CodeSearch    SectionSearch       │
                    │        (name+BM25     (BM25+vector        │
                    │         +vector)       +hierarchy)        │
                    │              │              │             │
                    │              ▼              ▼             │
                    │           Merge + Rerank + Expand         │
                    │           (import graph, entities,        │
                    │            same-file refs, neighbors)     │
                    │              │                            │
                    │              ▼                            │
                    │         Read + Assemble + Generate        │
                    └───────────────────────────────────────────┘
```

## Files

| Component | Path |
|-----------|------|
| Engine | `fitz_ai/engines/fitz_krag/engine.py` |
| Config | `fitz_ai/engines/fitz_krag/config/schema.py` |
| Query analyzer | `fitz_ai/engines/fitz_krag/query_analyzer.py` |
| Retrieval router | `fitz_ai/engines/fitz_krag/retrieval/router.py` |
| Code search | `fitz_ai/engines/fitz_krag/retrieval/strategies/code_search.py` |
| Section search | `fitz_ai/engines/fitz_krag/retrieval/strategies/section_search.py` |
| Table handler | `fitz_ai/engines/fitz_krag/retrieval/table_handler.py` |
| Context expander | `fitz_ai/engines/fitz_krag/retrieval/expander.py` |
| Reranker | `fitz_ai/engines/fitz_krag/retrieval/reranker.py` |
| Multi-hop | `fitz_ai/engines/fitz_krag/retrieval/multihop.py` |
| Symbol store | `fitz_ai/engines/fitz_krag/ingestion/symbol_store.py` |
| Section store | `fitz_ai/engines/fitz_krag/ingestion/section_store.py` |
| Table store | `fitz_ai/engines/fitz_krag/ingestion/table_store.py` |
| Import graph | `fitz_ai/engines/fitz_krag/ingestion/import_graph_store.py` |
| Ingestion pipeline | `fitz_ai/engines/fitz_krag/ingestion/pipeline.py` |

## Related Features

- [**Hybrid Search**](hybrid-search.md) — Dense + sparse fusion used within KRAG's symbol and section search
- [**Multi-Hop Reasoning**](multi-hop-reasoning.md) — Iterative retrieval for complex questions
- [**HyDE**](hyde.md) — Hypothetical document generation for abstract queries
- [**Entity Graph**](entity-graph.md) — Entity-based linking across retrieval units
- [**Hierarchical RAG**](hierarchical-rag.md) — L1/L2 summaries for corpus-level context
- [**Contextual Embeddings**](contextual-embeddings.md) — Summary-prefixed embeddings for disambiguation
