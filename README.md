

<div align="center">

# fitz-ai

### Intelligent, honest knowledge retrieval in 5 minutes. No infrastructure. No boilerplate.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/fitz-ai.svg)](https://pypi.org/project/fitz-ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.8.1-green.svg)](CHANGELOG.md)
[![Coverage](https://img.shields.io/badge/coverage-99%25-brightgreen)](https://github.com/yafitzdev/fitz-ai)


[Quick Start](#quick-start) • [Installation](#installation) • [Documentation](docs/) • [GitHub](https://github.com/yafitzdev/fitz-ai)

</div>

<br />

---

```bash
pip install fitz-ai

fitz quickstart ./docs "What is our refund policy?"
```

That's it. Your documents are now searchable with AI.


![fitz-ai quickstart demo](https://raw.githubusercontent.com/yafitzdev/fitz-ai/main/docs/assets/quickstart_demo.gif)

<br>

<details>

<summary><strong>Python SDK</strong> → <a href="docs/SDK.md">Full SDK Reference</a></summary>

<br>

```python
import fitz_ai

fitz_ai.ingest("./docs")
answer = fitz_ai.query("What is our refund policy?")
```

</details>

<br>

<details>

<summary><strong>REST API</strong> → <a href="docs/API.md">Full API Reference</a></summary>

<br>

```bash
pip install fitz-ai[api]

fitz serve  # http://localhost:8000/docs for interactive API
```

</details>

---

### About 🧑‍🌾

  Solo project by Yan Fitzner ([LinkedIn](https://www.linkedin.com/in/yan-fitzner/), [GitHub](https://github.com/yafitzdev)).

  - ~50k lines of Python
  - 1500+ tests, 99% coverage
  - Zero LangChain/LlamaIndex dependencies — built from scratch

![fitz-ai honest_rag](https://raw.githubusercontent.com/yafitzdev/fitz-ai/main/docs/assets/honest_rag.jpg)

---

<details>

<summary><strong>📦 What is RAG?</strong></summary>

<br>

RAG is how ChatGPT's "file search," Notion AI, and enterprise knowledge tools actually work under the hood.
Instead of sending all your documents to an AI, RAG:

1. [X] **Indexes your documents once** — Splits them into chunks, converts to vectors, stores in a database
2. [X] **Retrieves only what's relevant** — When you ask a question, finds the 5-10 most relevant chunks
3. [X] **Sends just those chunks to the LLM** — The AI answers based on focused, relevant context

Traditional approach:
```
  [All 10,000 documents] → LLM → Answer
  ❌ Impossible (too large)
  ❌ Expensive (if possible)
  ❌ Unfocused
```
RAG approach:
```
  Question → [Search index] → [5 relevant chunks] → LLM → Answer
  ✅ Works at any scale
  ✅ Costs pennies per query
  ✅ Focused context = better answers
```

</details>

---

<details>

<summary><strong>📦 Why Can't I Just Send My Documents to ChatGPT directly?</strong></summary>

<br>

You can—but you'll hit walls fast.

**Context window limits 🚨** 
> GPT-4 accepts ~128k tokens. That's roughly 300 pages. Your company wiki, codebase, or document archive is likely 10x-100x larger. You physically cannot paste it all.

**Cost explosion 💥**
> Even if you could fit everything, you'd pay for every token on every query. Sending 100k tokens costs ~\$1-3 per question. Ask 50 questions a day? That's $50-150 daily—for one user.

**No selective retrieval ❌**
> When you paste documents, the model reads everything equally. It can't focus on what's relevant. Ask about refund policies and it's also processing your hiring guidelines, engineering specs, and meeting notes—wasting context and degrading answers.

**No persistence 💢**
> Every conversation starts fresh. You re-upload, re-paste, re-explain. There's no knowledge base that accumulates and improves.

</details>

---

### Why Fitz?

**Super fast setup 🐆**
> Point at a folder. Ask a question. Get an answer with sources. Even for tables! Everything else is handled by Fitz.

**Honest answers ✅** → [Governance Benchmark](docs/features/governance-benchmarking.md)
> Most RAG tools confidently answer even when the answer isn't in your documents. Ask "What was our Q4 revenue?" when your docs only cover Q1-Q3, and typical RAG hallucinates a number. Fitz says: *"I cannot find Q4 revenue figures in the provided documents."
> 
> → Fitz detects disputes at **89.7% recall** on [fitz-gov](https://github.com/yafitzdev/fitz-gov), a 1,100+ case benchmark for epistemic honesty.

**Queries that actually work 📊**
> Standard RAG fails silently on real queries. Fitz has built-in intelligence: hierarchical summaries for "What are the trends?", exact keyword matching for "Find TC-1000", multi-query decomposition for complex questions, address-based code retrieval with import graph traversal, and SQL execution for tabular data. No configuration—it just works.

**Tabular data that is actually searchable 📈** → [Unified Storage](docs/features/unified-storage.md)
> CSV and table data is a nightmare in most RAG systems—chunked arbitrarily, structure lost, queries fail. Fitz stores tables natively in PostgreSQL alongside your vectors—same database, no sync issues. Auto-detects schema and runs real SQL. Ask "What's the average price by region?" and get an actual computed answer, not fragmented rows.

**Other Features at a Glance 🃏**
>1. [x] **Fully local execution possible.** Embedded PostgreSQL + Ollama, no API keys required to start.
>2. [x] **Plugin-based architecture.** Swap LLMs, rerankers, and retrieval pipelines via YAML config.
>3. [x] **[KRAG (Knowledge Routing Augmented Generation)](docs/features/krag.md).** Address-based retrieval.
>4. [X] **Incremental ingestion.** Only reprocesses changed files, even with new chunking settings.
>5. [x] **Full provenance.** Every answer traces back to the exact source symbol, section, or document.
>6. [x] **Data privacy**: No telemetry, no cloud, no external calls except to the LLM provider you configure.
>7. [x] **[Enterprise gateway support](docs/features/enterprise-gateway.md).** OAuth2 M2M, custom CA certs, mTLS, and corporate proxy/gateway integration.

####

> [!TIP]
> Any questions left? Try fitz on itself:
> 
> ```bash
> fitz quickstart ./fitz_ai "How does the ingestion pipeline work?"
> ```
>
> The codebase speaks for itself.

---

### What You Can Search

You feed Fitz documents — code files, PDFs, markdown, CSVs. FitzKRAG extracts structured retrieval units from them, each with its own storage and search strategy.

<br>

| Retrieval Unit              | Extracted From | How It Works |
|-----------------------------|----------------|-------------|
| **Symbols 🖌️**             | Code files | Tree-sitter parses functions, classes, and methods into addressable units with qualified names, references, and import graphs. Cross-file dependencies are graph traversals, not text searches. |
| **Sections 📑**             | Documents (PDF, markdown, text) | Headings and paragraphs are extracted with parent/child hierarchy. Deeply nested sections include parent context; top-level headings include child summaries. |
| **Tables 📅**               | CSV files or tables within documents | Native PostgreSQL storage with auto-detected schema. Real SQL execution from natural language — not chunked text. |
| **Images 🖼️**              | Figures and diagrams within documents | VLM-powered figure extraction and visual understanding. *(Coming soon)* |
| **Chunks 🧩**               | Any content as fallback | Traditional chunk-based retrieval when structured extraction doesn't apply. Automatic fallback — no configuration needed. |

<br>

> [!NOTE]
> All retrieval units share the same retrieval intelligence (temporal handling, comparison queries, multi-hop reasoning, etc.) and the same enrichment pipeline (summaries, keywords, entities, hierarchical summaries).

---

### Retrieval Intelligence

Most RAG implementations are naive vector search—they fail silently on real-world queries. Fitz has **built-in intelligence** that handles edge cases automatically:

<br>

| Feature | Query | Naive RAG Problem | Fitz Solution |
|---------|-------|-------------------|------------------|
| [**epistemic-honesty**](docs/features/epistemic-honesty.md) | "What was our Q4 revenue?" | ❌ Hallucinated number — Info doesn't exist, but LLM won't admit it | ✅ "I don't know" |
| [**keyword-vocabulary**](docs/features/keyword-vocabulary.md) | "Find TC_1000" | ❌ Wrong test case — Embeddings see TC_1000 ≈ TC_2000 (semantically similar) | ✅ Exact keyword matching |
| [**hybrid-search**](docs/features/hybrid-search.md) | "X100 battery specs" | ❌ Returns Y200 docs — Semantic search misses exact model numbers | ✅ Hybrid search (dense + sparse) |
| [**sparse-search**](docs/features/sparse-search.md) | "error code E_AUTH_401" | ❌ No exact match — Embeddings miss precise error codes | ✅ PostgreSQL full-text search |
| [**multi-hop**](docs/features/multi-hop-reasoning.md) | "Who wrote the paper cited by the 2023 review?" | ❌ Returns the review only — Single-step search can't traverse references | ✅ Iterative retrieval |
| [**hierarchical-rag**](docs/features/hierarchical-rag.md) | "What are the design principles?" | ❌ Random fragments — Answer is spread across docs; no single chunk contains it | ✅ Hierarchical summaries |
| [**multi-query**](docs/features/multi-query-rag.md) | *[User pastes 500-char test report]* "What failed and why?" | ❌ Vaguely related chunks — Long input → averaged embedding → matches nothing specifically | ✅ Multi-query decomposition |
| [**comparison-queries**](docs/features/comparison-queries.md) | "Compare React vs Vue performance" | ❌ Incomplete comparison — Only retrieves one entity, missing the other | ✅ Multi-entity retrieval |
| [**entity-graph**](docs/features/entity-graph.md) | "What else mentions AuthService?" | ❌ Isolated chunks — No awareness of shared entities across docs | ✅ Entity-based linking across sources |
| [**temporal-queries**](docs/features/temporal-queries.md) | "What changed between Q1 and Q2?" | ❌ Random chunks — No awareness of time periods in query | ✅ Temporal query handling |
| [**aggregation-queries**](docs/features/aggregation-queries.md) | "List all the test cases that failed" | ❌ Partial list — No mechanism for comprehensive retrieval | ✅ Aggregation query handling |
| [**freshness-authority**](docs/features/freshness-authority.md) | "What does the official spec say?" | ❌ Returns notes — Can't distinguish authoritative vs informal sources | ✅ Freshness/authority boosting |
| [**query-expansion**](docs/features/query-expansion.md) | "How do I fetch the db config?" | ❌ No matches — User says "fetch", docs say "retrieve"; "db" vs "database" | ✅ Query expansion |
| [**query-rewriting**](docs/features/query-rewriting.md) | "Tell me more about it" *(after discussing TechCorp)* | ❌ Lost context — Pronouns like "it" reference nothing, retrieval fails | ✅ Conversational context resolution |
| [**hyde**](docs/features/hyde.md) | "What's TechCorp's approach to sustainability?" | ❌ Poor recall — Abstract queries don't embed close to concrete documents | ✅ Hypothetical document generation |
| [**contextual-embeddings**](docs/features/contextual-embeddings.md) | "When does it expire?" | ❌ Ambiguous chunk — "It expires in 24h" embedded without context; "it" = ? | ✅ Summary-prefixed symbol/section embeddings |
| [**reranking**](docs/features/reranking.md) | "What's the battery warranty?" | ❌ Imprecise ranking — Vector similarity ≠ true relevance; best answer buried | ✅ Cross-encoder precision |

<br>

> [!IMPORTANT]
> These features are **always on**—no configuration needed. Fitz automatically detects when to use each capability.

---

### Governance — Know What You Don't Know

[Feature docs](docs/features/governance-benchmarking.md) • [Benchmark results](docs/evaluation/fitz-gov-3.0-results.md) • [Classifier experiments](docs/evaluation/classifier/NOTEPAD.md)

Most RAG systems hallucinate confidently. Fitz **measures and enforces** epistemic honesty using a two-stage ML classifier trained on 1,100+ labeled cases from [fitz-gov](https://github.com/yafitzdev/fitz-gov), a benchmark for epistemic honesty.

<br>

```
  Query + Retrieved Context
            │
            ▼
  ┌─────────────────────┐
  │ 5 Constraints       │     Contradiction detection, evidence sufficiency,
  │ (epistemic sensors) │     causal attribution, answer verification, specific info type
  └──────────┬──────────┘
             │ 51 features extracted
             ▼
  ┌─────────────────────┐
  │ Stage 1: RF         │     Can the evidence answer this query?
  │ Answerability       ├───► NO ──► ABSTAIN
  └──────────┬──────────┘
             │ YES
             ▼
  ┌─────────────────────┐     Do the sources conflict?
  │ Stage 2: ET         ├───► YES ──► DISPUTED
  │ Conflict Detection  │
  └──────────┬──────────┘
             │                Consistent evidence found
             └──────────────► NO ──► TRUSTWORTHY          
              
```

<br>

| Decision | Meaning                              | Recall |
|----------|--------------------------------------|--------|
| **ABSTAIN** | Evidence doesn't answer the question | **81.2%** |
| **DISPUTED** | Sources contradict each other        | **89.7%** |
| **TRUSTWORTHY** | Consistent, sufficient evidence      | **70.6%** |

<br>

> [!NOTE]
> Governance asks "given three relevant documents that partially contradict each other, should you flag a dispute, hedge the answer, or trust the consensus?" That's a judgment call even humans disagree on. 92% of our test cases are rated "hard."

<strong>The system fails safe 🛡️</strong>
> The safety-first threshold is tuned so that when the classifier is wrong, it over-hedges ("disputed" instead of "trustworthy") — annoying but harmless. Over-confidence ("trustworthy" instead of "disputed") is the rarest error mode: only 3 cases in 1,100+.

<strong>These scores are a floor, not a ceiling 👣</strong>
> All benchmarks were measured using `qwen2.5:3b` — a 3B parameter local model. The governance constraints run on the fast-tier LLM to keep latency low. Stronger models produce better constraint signals, which feed better features into the classifier. Upgrading your chat provider should improve governance accuracy for free.

<strong>Zero extra latency ⏱️</strong>
> The constraints already run as part of the pipeline. The ML classifier just replaces hand-coded rules with a local sklearn model — inference takes microseconds, no additional API calls.

---

<details>

<summary><strong>📦 Plugin Generator</strong> → <a href="docs/PLUGINS.md">Plugin Development Guide</a></summary>

<br>

#### Generate plugins with AI 🤖

>Fitz can generate fully working plugins from natural language descriptions. Describe what you want, and fitz creates, validates, and saves the plugin automatically.
>
>```bash
>fitz plugin
>? Plugin type: constraint
>? Description: flags answers that cite a single source when multiple exist
>
>Generating...
>✓ Syntax valid
>✓ Schema valid
>✓ Plugin loads correctly
>✓ Functional test passed
>
>Created: ~/.fitz/plugins/constraints/single_source_warning.py
>```
>
>The generated plugin is immediately usable—no manual editing required.

<br>

#### Supported plugin types

>| Type | Format | Description |
>|------|--------|-------------|
>| `llm-chat` | YAML | Connect to a chat LLM provider |
>| `llm-embedding` | YAML | Connect to an embedding provider |
>| `llm-rerank` | YAML | Connect to a reranking provider |
>| `retrieval` | YAML | Define a retrieval strategy |
>| `constraint` | Python | Epistemic safety guardrail |
>| `reader` | Python | Custom file format reader |
>| `chunker` | Python | Custom chunking logic (fallback strategy) |

<br>

#### How it works

>1. **Prompt building**: Fitz loads existing plugin examples and schema definitions
>2. **Generation**: Your configured LLM generates the plugin code
>3. **Multi-level validation**: Syntax → Schema → Integration → Functional tests
>4. **Auto-retry**: If validation fails, fitz feeds the error back and retries (up to 3 attempts)
>5. **Save**: Working plugins are saved to `~/.fitz/plugins/`
>
>Generated plugins are auto-discovered by fitz on next run—no registration needed.

<br>

#### Example: Custom constraint

>```bash
>fitz plugin
>? Plugin type: constraint
>? Description: detects when sources use different time frames
>
># Creates ~/.fitz/plugins/constraints/temporal_mismatch.py
>```
>
>Generated plugins are auto-discovered — no registration needed.

</details>

---

<details>

<summary><strong>📦 Quick Start</strong></summary>

<br>

#### CLI
>
>```bash
>pip install fitz-ai
>
>fitz quickstart ./docs "Your question here"
>```
>
>Fitz auto-detects your LLM provider:
>1. **Ollama running?** → Uses it automatically (fully local)
>2. **`COHERE_API_KEY` or `OPENAI_API_KEY` set?** → Uses it automatically
>3. **First time?** → Guides you through free Cohere signup (2 minutes)
>
>After first run, it's completely zero-friction.

<br>

#### Python SDK
>
>```python
>import fitz_ai
>
>fitz_ai.ingest("./docs")
>answer = fitz_ai.query("Your question here")
>
>print(answer.text)
>for source in answer.provenance:
>    print(f"  - {source.source_id}: {source.excerpt[:50]}...")
>```
>
>The SDK provides:
>- Module-level functions matching CLI (`ingest`, `query`)
>- Auto-config creation (no setup required)
>- Full provenance tracking
>- Same honest retrieval as the CLI
>
>For advanced use (multiple collections), use the `fitz` class directly:
>```python
>from fitz_ai import fitz
>
>physics = fitz(collection="physics")
>physics.ingest("./physics_papers")
>answer = physics.query("Explain entanglement")
>```

<br>

#### Fully Local (Ollama)
>
>```bash
>pip install fitz-ai[local]
>
>ollama pull llama3.2
>ollama pull nomic-embed-text
>
>fitz quickstart ./docs "Your question here"
>```
>
>Fitz auto-detects Ollama when running. No API keys needed—no data leaves your machine.

</details>

---

<details>

<summary><strong>📦 Real-World Usage</strong></summary>

<br>

Fitz is a foundation. It handles document ingestion and grounded retrieval—you build whatever sits on top: chatbots, dashboards, alerts, or automation.

<br>

<strong>Chatbot Backend 🤖</strong>

> Connect fitz to Slack, Discord, Teams, or your own UI. One function call returns an answer with sources—no hallucinations, full provenance. You handle the conversation flow; fitz handles the knowledge.
>
> *Example:* A SaaS company plugs fitz into their support bot. Tier-1 questions like "How do I reset my password?" get instant answers. Their support team focuses on edge cases while fitz deflects 60% of incoming tickets.

<br>

<strong>Internal Knowledge Base 📖</strong>

> Point fitz at your company's wiki, policies, and runbooks. Employees ask natural language questions instead of hunting through folders or pinging colleagues on Slack.
>
> *Example:* A 200-person startup ingests their Notion workspace and compliance docs. New hires find answers to "How do I request PTO?" on day one—no more waiting for someone in HR to respond.

<br>

<strong>Continuous Intelligence & Alerting (Watchdog) 🐶</strong>

> Pair fitz with cron, Airflow, or Lambda. Ingest data on a schedule, run queries automatically, trigger alerts when conditions match. Fitz provides the retrieval primitive; you wire the automation.
>
> *Example:* A security team ingests SIEM logs nightly. Every morning, a scheduled job asks "Were there failed logins from unusual locations?" If fitz finds evidence, an alert fires to the on-call channel before anyone checks email.

<br>

<strong>Web Knowledge Base 🌎</strong>

> Scrape the web with Scrapy, BeautifulSoup, or Playwright. Save to disk, ingest with fitz. The web becomes a queryable knowledge base.
>
> *Example:* A football analytics hobbyist scrapes Premier League match reports. After ingesting, they ask "How did Arsenal perform against top 6 teams?" or "What tactics did Liverpool use in away games?"—insights that would take hours to compile manually.

<br>

<strong>Codebase Search 🐍</strong>

> FitzKRAG uses address-based retrieval for code: tree-sitter parses your codebase into symbols (functions, classes, methods) with qualified names, references, and import graphs. No chunking—each symbol is a precise, addressable unit. Cross-file dependencies are tracked, so "what calls this function?" is a graph traversal, not a text search.
>
> *Example:* A team inherits a legacy Django monolith—200k lines, sparse docs. They ingest the codebase and ask "Where is user authentication handled?" or "What depends on the billing module?" FitzKRAG returns specific functions with their callers and dependencies. New developers onboard in days instead of weeks.

</details>

---

<details>

<summary><strong>📦 Architecture</strong> → <a href="docs/ARCHITECTURE.md">Full Architecture Guide</a></summary>

<br>

```
┌───────────────────────────────────────────────────────────────┐
│                         fitz-ai                               │
├───────────────────────────────────────────────────────────────┤
│  User Interfaces                                              │
│  CLI: quickstart | init | ingest | query | chat | serve       │
│  SDK: fitz_ai.fitz() → ingest() → ask()                       │
│  API: /query | /chat | /ingest | /collections | /health       │
├───────────────────────────────────────────────────────────────┤
│  Engines                                                      │
│  ┌────────────┐  ┌────────────┐                               │
│  │  FitzKRAG  │  │  Custom... │  (extensible registry)        │
│  └────────────┘  └────────────┘                               │
├───────────────────────────────────────────────────────────────┤
│  LLM Plugins (YAML-defined)                                   │
│  ┌────────┐ ┌───────────┐ ┌────────┐                          │
│  │  Chat  │ │ Embedding │ │ Rerank │                          │
│  └────────┘ └───────────┘ └────────┘                          │
│  openai, cohere, anthropic, ollama, azure...                  │
├───────────────────────────────────────────────────────────────┤
│  Storage (PostgreSQL + pgvector)                              │
│  vectors | metadata | tables | keywords | full-text search    │
├───────────────────────────────────────────────────────────────┤
│  Retrieval (address-based, baked-in intelligence)             │
│  symbols | sections | tables | import graphs | reranking      │
├───────────────────────────────────────────────────────────────┤
│  Enrichment (baked in)                                        │
│  summaries | keywords | entities | hierarchical summaries     │
├───────────────────────────────────────────────────────────────┤
│  Constraints (epistemic safety)                               │
│  ConflictAware | InsufficientEvidence | CausalAttribution     │
└───────────────────────────────────────────────────────────────┘
```

</details>

---

<details>

<summary><strong>📦 CLI Reference</strong> → <a href="docs/CLI.md">Full CLI Guide</a></summary>

<br>

```bash
fitz quickstart [PATH] [QUESTION]    # Zero-config RAG (start here)
fitz init                            # Interactive setup wizard
fitz ingest                          # Interactive ingestion
fitz query                           # Single question with sources
fitz chat                            # Multi-turn conversation with your knowledge base
fitz collections                     # List and delete knowledge collections
fitz keywords                        # Manage keyword vocabulary for exact matching
fitz plugin                          # Generate plugins with AI
fitz serve                           # Start REST API server
fitz config                          # View/edit configuration
fitz doctor                          # System diagnostics
```

</details>

---

<details>

<summary><strong>📦 Python SDK Reference</strong> → <a href="docs/SDK.md">Full SDK Guide</a></summary>

<br>

**Simple usage (module-level, matches CLI):**
```python
import fitz_ai

fitz_ai.ingest("./docs")
answer = fitz_ai.query("What is the refund policy?")
print(answer.text)
```

<br>

**Advanced usage (multiple collections):**
```python
from fitz_ai import fitz

# Create separate instances for different collections
physics = fitz(collection="physics")
physics.ingest("./physics_papers")

legal = fitz(collection="legal")
legal.ingest("./contracts")

# Query each collection
physics_answer = physics.query("Explain entanglement")
legal_answer = legal.query("What are the payment terms?")
```

<br>

**Working with answers:**
```python
answer = fitz_ai.query("What is the refund policy?")

print(answer.text)
print(answer.mode)  # CONFIDENT, QUALIFIED, DISPUTED, or ABSTAIN

for source in answer.provenance:
    print(f"Source: {source.source_id}")
    print(f"Excerpt: {source.excerpt}")
```

</details>

---

<details>

<summary><strong>📦 REST API Reference</strong> → <a href="docs/API.md">Full API Guide</a></summary>

<br>

**Start the server:**
```bash
pip install fitz-ai[api]

fitz serve                    # localhost:8000
fitz serve -p 3000            # custom port
fitz serve --host 0.0.0.0     # all interfaces
```

**Interactive docs:** Visit `http://localhost:8000/docs` for Swagger UI.

<br>

**Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/query` | Query knowledge base |
| POST | `/chat` | Multi-turn chat (stateless) |
| POST | `/ingest` | Ingest documents from path |
| GET | `/collections` | List all collections |
| GET | `/collections/{name}` | Get collection stats |
| DELETE | `/collections/{name}` | Delete a collection |
| GET | `/health` | Health check |

<br>

**Example request:**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?", "collection": "default"}'
```

</details>

---

### License

MIT

---

### Links

- [GitHub](https://github.com/yafitzdev/fitz-ai)
- [PyPI](https://pypi.org/project/fitz-ai/)
- [Changelog](CHANGELOG.md)

**Documentation:**
- [CLI Reference](docs/CLI.md)
- [Python SDK](docs/SDK.md)
- [REST API](docs/API.md)
- [Configuration Guide](docs/CONFIG.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Unified Storage (PostgreSQL + pgvector)](docs/features/unified-storage.md)
- [Ingestion Pipeline](docs/INGESTION.md)
- [Enrichment (Hierarchies, Entities)](docs/ENRICHMENT.md)
- [Epistemic Constraints](docs/CONSTRAINTS.md)
- [Governance Benchmarking (fitz-gov)](docs/features/governance-benchmarking.md)
- [Plugin Development](docs/PLUGINS.md)
- [Feature Control](docs/FEATURE_CONTROL.md)
- [KRAG — Knowledge Routing Augmented Generation](docs/features/krag.md)
- [Custom Engines](docs/CUSTOM_ENGINES.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
