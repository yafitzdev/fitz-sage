# fitz-ai

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/fitz-ai.svg)](https://pypi.org/project/fitz-ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.5.0-green.svg)](CHANGELOG.md)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)](https://github.com/yafitzdev/fitz-ai)

---

**Intelligent, honest RAG in 5 minutes. No infrastructure. No boilerplate.**

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

  - ~65k lines of Python
  - 750+ tests, 100% coverage
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
> Point at a folder. Ask a question. Get an answer with sources. Everything else is handled by Fitz.

**Honest answers ✅**
> Most RAG tools confidently answer even when the answer isn't in your documents. Ask "What was our Q4 revenue?" when your docs only cover Q1-Q3, and typical RAG hallucinates a number. Fitz says: *"I cannot find Q4 revenue figures in the provided documents."*

**Swap engines, keep everything else ⚙️**
> RAG is evolving fast—GraphRAG, HyDE, ColBERT, whatever's next. Fitz lets you switch engines in one line. Your ingested data stays. Your queries stay. No migration, no re-ingestion, no new API to learn. Frameworks lock you in; Fitz lets you move.

**Queries that actually work 📊**
> Standard RAG fails silently on real queries. Fitz has built-in intelligence: hierarchical summaries for "What are the trends?", exact keyword matching for "Find TC-1001", multi-query decomposition for complex questions, and AST-aware chunking for code. No configuration—it just works.

**Other Features at a Glance 🃏**
>
>1. [x] **Local execution possible.** FAISS and Ollama support, no API keys required to start.
>2. [x] **Plugin-based architecture.** Swap LLMs, vector databases, rerankers, and retrieval pipelines via YAML config.
>3. [x] **Multiple engines.** Supports FitzRAG, GraphRAG and CLaRa out of the box—swap engines in one line.
>4. [X] **Incremental ingestion.** Only reprocesses changed files, even with new chunking settings.
>5. [x] **Full provenance.** Every answer traces back to the exact chunk and document.
>6. [x] **Data privacy**: No telemetry, no cloud, no external calls except to the LLM provider you configure.

####

Any questions left? Try fitz on itself:

```bash
fitz quickstart ./fitz_ai "How does the chunking pipeline work?"
```

The codebase speaks for itself.

---

### Retrieval Intelligence

Most RAG implementations are naive vector search—they fail silently on real-world queries. Fitz has **built-in intelligence** that handles edge cases automatically:

| Query | Problem | Naive RAG | FitzRAG                     |
|-------|---------|-----------|-----------------------------|
| "What was our Q4 revenue?" | Answer not in docs | ❌ Hallucinated answer | ✅ "I don't know"            |
| "What are the design principles?" | Global/analytical query | ❌ Random chunks | ✅ Hierarchical summaries    |
| "Find TC_CAN_001" | Exact identifier | ❌ Returns TC_CAN_002 | ✅ Keyword matching          |
| "Summarize failures and root causes" | Complex multi-part | ❌ Query dilution | ✅ Multi-query decomposition |
| "How does auth module work?" | Code structure | ❌ Split functions | ✅ AST-aware chunking        |

These features are **always on**—no configuration needed. Fitz automatically detects when to use each capability.

<details>

<summary><strong>Multi-Query Decomposition</strong></summary>

<br>

>**The problem ☔️**
>
>Long, complex queries dilute into weak embeddings. Ask "Summarize the test failures, their root causes, and recommended fixes" and vector search returns chunks vaguely related to tests—missing failures, causes, or fixes entirely.
>
>**The solution ☀️**
>
>Fitz automatically detects long queries (>300 chars) and decomposes them:
>```
>Original: "Summarize the test failures, their root causes, and recommended fixes"
>     ↓
>Decomposed:
>  → "test failures"
>  → "root causes of failures"
>  → "recommended fixes"
>     ↓
>3 focused searches → deduplicated results → complete answer
>```
>
>**Always on.** Short queries run as single searches (no overhead). Long queries automatically expand. No configuration needed.

</details>

<details>

<summary><strong>Keyword Vocabulary (Exact Match)</strong></summary>

<br>

>**The problem ☔️**
>
>Semantic search struggles with identifiers. Ask "What happened with TC-1001?" and embeddings return TC-1002, TC-1003, or unrelated test cases—because they're "semantically similar."
>
>**The solution ☀️**
>
>Fitz auto-detects identifiers during ingestion and builds a vocabulary:
>- **Test cases**: TC-1001, testcase_42
>- **Tickets**: JIRA-4521, BUG-789
>- **Versions**: v2.0.1, 1.0.0-beta
>- **Code**: `AuthService`, `handle_login()`
>
>At query time, keywords pre-filter chunks before semantic search:
>```
>Q: "What happened with TC-1001?"
>→ Chunks filtered to only those containing TC-1001
>→ Semantic search runs on filtered set
>→ Result: Only TC-1001 content, never TC-1002
>```
>
>**Variation matching** handles format differences automatically:
>```
>TC-1001 → tc-1001, TC_1001, tc 1001
>JIRA-123 → jira-123, JIRA123, jira 123
>```

</details>

<details>

<summary><strong>Hierarchical RAG</strong></summary>

<br>

>**The problem ☔️**
>
>Standard RAG can't answer analytical queries. Ask "What are the trends?" and it returns random chunks instead of aggregated insights.
>
>**The solution ☀️**
>
>Fitz generates multi-level summaries during ingestion:
>- **Level 0**: Original chunks
>- **Level 1**: Group summaries (per source file)
>- **Level 2**: Corpus summary (all documents)
>
>```
>Q: "What are the overall trends?"
>→ Returns L2 corpus summary + L1 group summaries
>
>Q: "What did users say about the async tutorial?"
>→ Returns L0 individual chunks from that file
>```
>
>Query routing is automatic—summaries match analytical queries via embedding similarity.

</details>

<details>

<summary><strong>Code-Aware Chunking</strong></summary>

<br>

>**The problem ☔️**
>
>Naive chunking splits code mid-function, breaking syntax and losing context. A 50-line class becomes 3 fragments that don't make sense alone.
>
>**The solution ☀️**
>
>Fitz uses AST-aware chunking for code:
>
>| Language | Strategy |
>|----------|----------|
>| **Python** | Classes, functions, methods as units. Large classes split by method. Imports preserved. |
>| **Markdown** | Header-aware splits. Code blocks kept intact. YAML frontmatter extracted. |
>| **PDF** | Section detection (1.1, 2.3.1, roman numerals). Keywords like "Abstract", "Conclusion". |
>
>```python
># Naive chunking:
>def authenticate(user):     # ← chunk 1 ends here
>    if not user.token:      # ← chunk 2 starts here (broken)
>        raise AuthError()
>
># Fitz chunking:
>def authenticate(user):     # ← entire function = 1 chunk
>    if not user.token:
>        raise AuthError()
>    return validate(user.token)
>```
>
>Docstrings, decorators, and type hints stay attached to their functions.

</details>

<details>

<summary><strong>Epistemic Honesty</strong></summary>

<br>

>**The problem ☔️**
>
>Most RAG systems confidently answer even when the answer isn't in the documents. Ask "What was our Q4 revenue?" when docs only cover Q1-Q3, and they hallucinate a number.
>
>**The solution ☀️**
>
>Fitz has built-in epistemic guardrails that detect uncertainty:
>
>```
>Q: "What was our Q4 revenue?"
>A: "I cannot find Q4 revenue figures in the provided documents.
>    The available financial data covers Q1-Q3 only."
>
>   Mode: ABSTAIN
>```
>
>Three constraint plugins run automatically:
>
>| Constraint | What it catches |
>|------------|-----------------|
>| **ConflictAware** | Sources disagree → surfaces the conflict |
>| **InsufficientEvidence** | No supporting evidence → refuses to guess |
>| **CausalAttribution** | Correlation ≠ causation → blocks hallucinated "why" |
>
>Every answer includes a **mode** indicating confidence:
>- `CONFIDENT` — Strong evidence supports the answer
>- `QUALIFIED` — Answer given with noted limitations
>- `DISPUTED` — Sources conflict, both views presented
>- `ABSTAIN` — Insufficient evidence, refuses to answer

</details>

<details>

<summary><strong>Roadmap</strong></summary>

<br>

>| Feature | Status | Description |
>|---------|--------|-------------|
>| Hierarchical RAG | ✅ Done | Multi-level summaries for analytical queries |
>| Keyword Vocabulary | ✅ Done | Exact matching for identifiers |
>| Multi-Query Decomposition | ✅ Done | Automatic expansion for complex queries |
>| Code-Aware Chunking | ✅ Done | AST-aware splitting for Python, Markdown, PDF |
>| Epistemic Honesty | ✅ Done | "I don't know" when evidence is insufficient |
>| Comparison Queries | 🔜 Next | Multi-entity retrieval ("A vs B") |
>| Tabular Data Routing | 📋 Planned | Route table queries to structured search |
>| Multi-Hop Reasoning | 📋 Planned | Chain retrieval across related entities |

</details>

---

<details>

<summary><strong>📦 Fitz vs LangChain vs LlamaIndex</strong></summary>

<br>

#### Fitz opts for a deliberately narrower approach.
>
>LangChain and LlamaIndex are powerful **LLM application frameworks** designed to help developers build complex, end-to-end AI systems. 
>Fitz provides a **minimal, replaceable RAG engine** with strong epistemic guarantees — without locking users into a framework, ecosystem, or long-term architectural commitment.
>
>Fitz is not a competitor in scope.  
>It is an infrastructure primitive.

<br>

#### Core philosophical differences ⚖️
>
>| Dimension | Fitz | LangChain | LlamaIndex |
>|--------|------|-----------|------------|
>| Primary role | **RAG engine** | LLM application framework | LLM data framework |
>| User commitment | **No framework lock-in** | High | High |
>| Engine coupling | **Swappable in one line** | Deep | Deep |
>| Design goal | Correctness & honesty | Flexibility | Data integration |
>| Long-term risk | Low | Migration-heavy | Migration-heavy |

<br>

#### Epistemic behavior (truth over fluency) 🎯
>
>| Aspect | Fitz | LangChain / LlamaIndex |
>|-----|------|------------------------|
>| “I don’t know” | **First-class behavior** | Not guaranteed |
>| Hallucination handling | Designed-in | Usually prompt-level |
>| Confidence signaling | Explicit | Implicit |
>
>Fitz treats uncertainty as a **feature**, not a failure.  
>If the system cannot support an answer with retrieved evidence, it says so.

<br>

#### Transparency & provenance 🔎
>
>| Capability | Fitz | LangChain / LlamaIndex |
>|---------|------|------------------------|
>| Source attribution | **Mandatory** | Optional |
>| Retrieval trace | **Explicit & structured** | Often opaque |
>| Debuggability | Built-in | Tool-dependent |
>
>Every answer in Fitz is fully auditable down to the retrieval step.

<br>

#### Scope & complexity 🪐
>
>| Aspect | Fitz | LangChain / LlamaIndex |
>|-----|------|------------------------|
>| Chains / agents | ❎ | ✔ |
>| Prompt graphs | ❎ | ✔ |
>| UI abstractions | ❎ | Often |
>| Cognitive overhead | **Very low** | High |
>
>Fitz intentionally does less — so it can be trusted more.

<br>

#### Use Fitz if you want:
>
>- A replaceable RAG engine, not a framework marriage
>- Strong epistemic guarantees (“I don’t know” is valid output)
>- Full provenance for every answer
>- A transparent, extensible plugin architecture
>- A future-proof ingestion pipeline that survives engine changes

</details>

---
<details>

<summary><strong>📦 Features</strong></summary>

<br>

#### Swappable RAG Engines 🔄

>Your data stays. Your queries stay. Only the engine changes.
>
>```
>        ┌─────────────────────────────────────┐
>        │           Your Query                │
>        │   "What are the payment terms?"     │
>        └──────────────────┬──────────────────┘
>                           │
>                           ▼
>        ┌─────────────────────────────────────┐
>        │       engine="..."                  │
>        │  ┌─────────┐ ┌───────┐ ┌─────────┐  │
>        │  │ fitz    │ │ clara │ │ graph   │  │
>        │  │  _rag   │ │       │ │  _rag   │  │
>        │  └────┬────┘ └───┬───┘ └────┬────┘  │
>        │       └──────────┼──────────┘       │
>        └──────────────────┼──────────────────┘
>                           │
>                           ▼
>        ┌─────────────────────────────────────┐
>        │       Your Ingested Knowledge       │
>        │      (unchanged across engines)     │
>        └─────────────────────────────────────┘
>```
>
>```python
>answer = run("What are the payment terms?", engine="fitz_rag")
>answer = run("What are the payment terms?", engine="clara")
>answer = run("What are the payment terms?", engine="graph_rag")  # future
>```
>
>No migration. No re-ingestion. No new API to learn.

<br>

#### Full Provenance 🗂️

>Every answer traces back to its source:
>
>```
>Answer: The refund policy allows returns within 30 days...
>
>Sources:
>  [1] policies/refund.md [chunk 3] (score: 0.92)
>  [2] faq/payments.md [chunk 1] (score: 0.87)
>```

<br>

#### Incremental Ingestion ⚡ → [Ingestion Guide](docs/INGESTION.md)

>Fitz tracks file hashes and only re-ingests what changed:
>
>```
>$ fitz ingest ./src
>
>Scanning... 847 files
>  → 12 new files
>  → 3 modified files
>  → 832 unchanged (skipped)
>
>Ingesting 15 files...
>```
>
>Re-running ingestion on a large codebase takes seconds, not minutes. Changed your chunking config? Fitz detects that too and re-processes affected files.

</details>

---

<details>

<summary><strong>📦 Plugin Generator</strong> → <a href="docs/PLUGINS.md">Plugin Development Guide</a></summary>

<br>

#### Generate plugins with AI 🤖

>Fitz can generate fully working plugins from natural language descriptions. Describe what you want, and fitz creates, validates, and saves the plugin automatically.
>
>```bash
>fitz plugin
>? Plugin type: chunker
>? Description: sentence-based chunker that splits on periods
>
>Generating...
>✓ Syntax valid
>✓ Schema valid
>✓ Plugin loads correctly
>✓ Functional test passed
>
>Created: ~/.fitz/plugins/chunking/sentence_chunker.py
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
>| `vector-db` | YAML | Connect to a vector database |
>| `retrieval` | YAML | Define a retrieval strategy |
>| `chunker` | Python | Custom document chunking logic |
>| `reader` | Python | Custom file format reader |
>| `constraint` | Python | Epistemic safety guardrail |

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

#### Example: Custom chunker

>```bash
>fitz plugin
>? Plugin type: chunker
>? Description: splits text by paragraphs, keeping code blocks intact
>
># Creates ~/.fitz/plugins/chunking/paragraph_chunker.py
>```
>
>```python
># Generated plugin is immediately usable
>fitz ingest ./docs --chunker paragraph_chunker
>```

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
>- Same honest RAG as the CLI
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

> Fitz includes built-in AST-aware chunking for code bases. Functions, classes, and modules become individual searchable units with docstrings and imports preserved. Ask questions in natural language; get answers pointing to specific code.
>
> *Example:* A team inherits a legacy Django monolith—200k lines, sparse docs. They ingest the codebase and ask "Where is user authentication handled?" or "What API endpoints modify the billing table?" New developers onboard in days instead of weeks.

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
│  ┌───────────┐  ┌───────────┐  ┌────────────┐                 │
│  │  FitzRAG  │  │   CLaRa   │  │  GraphRAG  │  (pluggable)    │
│  └───────────┘  └───────────┘  └────────────┘                 │
├───────────────────────────────────────────────────────────────┤
│  Plugin System (all YAML-defined)                             │
│  ┌────────┐ ┌───────────┐ ┌────────┐ ┌──────────┐             │
│  │  Chat  │ │ Embedding │ │ Rerank │ │ VectorDB │             │
│  └────────┘ └───────────┘ └────────┘ └──────────┘             │
│  openai, cohere, anthropic, ollama, azure...                  │
├───────────────────────────────────────────────────────────────┤
│  Retrieval Pipelines (plugin choice controls features)        │
│  dense (no rerank) | dense_rerank (with rerank)               │
├───────────────────────────────────────────────────────────────┤
│  Enrichment (opt-in)                                          │
│  entities | entity links | semantic clusters | hierarchical   │
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

**Example requests:**

```bash
# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?", "collection": "default"}'

# Ingest
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"source": "./docs", "collection": "mydata"}'

# Chat (stateless - client manages history)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What about returns?",
    "history": [
      {"role": "user", "content": "What is the refund policy?"},
      {"role": "assistant", "content": "The refund policy allows..."}
    ],
    "collection": "default"
  }'
```

</details>

---

<details>

<summary><strong>📦 Beyond RAG</strong></summary>

<br>

> **RAG is a method. Knowledge access is a strategy.**

Fitz is not a RAG framework. It's a knowledge platform that *currently* uses RAG as its primary engine.

```python
from fitz_ai import run

# Fitz RAG - fast, reliable vector search
answer = run("What are the payment terms?", engine="fitz_rag")

# CLaRa - compressed RAG, 16x smaller context
answer = run("What are the payment terms?", engine="clara")

# GraphRAG - knowledge graph with entity extraction and community summaries
answer = run("What are the payment terms?", engine="graphrag")
```

The engine is an implementation detail. Your ingested knowledge, your queries, your workflow—all stay the same. When a better retrieval paradigm emerges, swap one line, not your entire codebase.

</details>

---

<details>

<summary><strong>📦 Philosophy</strong></summary>

<br>

**Principles:**
- **Explicit over clever**: No magic. Read the config, know what happens.
- **Answers over architecture**: Optimize for time-to-insight, not flexibility.
- **Honest over helpful**: Better to say "I don't know" than hallucinate.
- **Files over frameworks**: YAML plugins over class hierarchies.

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
- [Ingestion Pipeline](docs/INGESTION.md)
- [Enrichment (Hierarchies, Entities)](docs/ENRICHMENT.md)
- [Epistemic Constraints](docs/CONSTRAINTS.md)
- [Plugin Development](docs/PLUGINS.md)
- [Feature Control](docs/FEATURE_CONTROL.md)
- [Custom Engines](docs/CUSTOM_ENGINES.md)
- [Engine Comparison](docs/ENGINES.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
