# fitz-ai ✨

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/fitz-ai.svg)](https://pypi.org/project/fitz-ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.3.6-green.svg)](CHANGELOG.md)

**Setup RAG in 5 minutes. No infrastructure. No boilerplate.**

```bash
pip install fitz-ai

fitz quickstart ./docs "What is our refund policy?"
```

That's it. Your documents are now searchable with AI.

---

## Why Fitz? ☀️

- **Point at a folder. Ask a question. Get an answer with sources.**
- **Says "I don't know" when the answer isn't there.** No hallucinations, no confident nonsense.
- **Smart chunking out of the box.** AST-aware for Python, section-based for PDFs, heading-aware for Markdown.
- **Runs locally.** Ollama support, no API keys required to start.
- **One config file when you need control.** Zero config when you don't.
- **Full provenance.** Every answer traces back to the exact chunk and document.

Two commands to your first answer:

```bash
pip install fitz-ai
fitz quickstart ./docs "What is our refund policy?"
```

That's the whole tutorial.

---

## Features

### 1. Zero-Config Start, Full Control Later

```bash
# Day 1: Just works
fitz quickstart ./contracts "What are the payment terms?"

# Day 30: Full customization when you need it
fitz init                    # Configure everything
fitz ingest ./docs           # Fine-tune chunking
fitz query "..." --retrieval dense_rerank
```

Most RAG frameworks force you to understand the entire stack before you can ask a single question. Fitz inverts this: start with answers, learn the internals only if you need them.

### 2. Epistemic Honesty Built-In

When your documents don't contain the answer, fitz says so:

```
Q: "What was our Q4 revenue?"
A: "I cannot find Q4 revenue figures in the provided documents. 
    The available financial data covers Q1-Q3 only."
    
    Mode: ABSTAIN
```

Three constraint plugins run automatically:
- 🧊 **ConflictAwareConstraint**: Detects contradictions across sources
- 🎲 **InsufficientEvidenceConstraint**: Blocks confident answers without evidence  
- 📦 **CausalAttributionConstraint**: Prevents hallucinated cause-effect claims

No prompt engineering required. No "be careful" system messages. Just honest answers.

### 3. Full Provenance

Every answer includes sources - which documents, which chunks, what scores:

```
Answer: The refund policy allows returns within 30 days of purchase...

Sources:
  [1] policies/refund.md [chunk 3] (vec=0.847, rerank=0.92)
  [2] faq/payments.md [chunk 1] (vec=0.812, rerank=0.87)
```

No black boxes. Audit every answer back to its source.

### 4. YAML-Defined Everything

```yaml
# Add a new LLM provider: just drop a YAML file
# fitz_ai/llm/chat/my_provider.yaml
plugin_name: "my_provider"
plugin_type: "chat"
endpoint:
  path: "/v1/chat/completions"
  method: "POST"
defaults:
  model: "my-model"
  temperature: 0.2
```

LangChain requires you to subclass `BaseChatModel` and implement 47 methods. Fitz plugins are YAML files. Adding OpenAI, Cohere, Anthropic, Ollama, or your custom endpoint is the same: describe the API, done.

---

## Quick Start 🚀

```bash
pip install fitz-ai

fitz quickstart ./docs "Your question here"
```

That's it. Fitz will prompt you for anything it needs.

- **Don't have docs yet?** Point fitz at its own source:

    ```bash
    fitz quickstart ./fitz_ai "How does the chunking pipeline work?"
    ```
    
    Learn the platform by querying it. The codebase is the documentation.


- Want to go fully local with Ollama? No problem:

    ```bash
    pip install fitz-ai
    
    ollama pull llama3.2
    ollama pull nomic-embed-text
    
    fitz quickstart ./docs "Your question here"
    ```

    No data leaves your machine. No API costs. Same interface.


- **Data privacy**: Fitz runs entirely on your infrastructure. No telemetry, no cloud, no external calls except to the LLM provider you configure.


- **Production deployments**: For Qdrant, Docker, and API serving, see the [Deployment Guide](docs/deployment.md).

---

## Real-World Usage 💼

Fitz is a foundation. It handles document ingestion and grounded retrieval—you build whatever sits on top: chatbots, dashboards, alerts, or automation.

### Chatbot Backend

Connect fitz to Slack, Discord, Teams, or your own UI. One function call returns an answer with sources—no hallucinations, full provenance. You handle the conversation flow; fitz handles the knowledge.

*Example:* A SaaS company plugs fitz into their support bot. Tier-1 questions like "How do I reset my password?" get instant answers. Their support team focuses on edge cases while fitz deflects 60% of incoming tickets.

### Internal Knowledge Base

Point fitz at your wiki, policies, and runbooks. Employees ask natural language questions instead of hunting through folders or pinging colleagues on Slack.

*Example:* A 200-person startup ingests their Notion workspace and compliance docs. New hires find answers to "How do I request PTO?" on day one—no more waiting for someone in HR to respond.

### Continuous Intelligence & Alerting

Pair fitz with cron, Airflow, or Lambda. Ingest data on a schedule, run queries automatically, trigger alerts when conditions match. Fitz provides the retrieval primitive; you wire the automation.

*Example:* A security team ingests SIEM logs nightly. Every morning, a scheduled job asks "Were there failed logins from unusual locations?" If fitz finds evidence, an alert fires to the on-call channel before anyone checks email.

### Web Knowledge Base

Scrape the web with Scrapy, BeautifulSoup, or Playwright. Save to disk, ingest with fitz. The web becomes a queryable knowledge base.

*Example:* A football analytics hobbyist scrapes Premier League match reports. After ingesting, they ask "How did Arsenal perform against top 6 teams?" or "What tactics did Liverpool use in away games?"—insights that would take hours to compile manually.

### Codebase Search

Fitz includes built-in AST-aware chunking for Python. Functions, classes, and modules become individual searchable units with docstrings and imports preserved. Ask questions in natural language; get answers pointing to specific code.

*Example:* A team inherits a legacy Django monolith—200k lines, sparse docs. They ingest the codebase and ask "Where is user authentication handled?" or "What API endpoints modify the billing table?" New developers onboard in days instead of weeks.

---

## Architecture 🏛

```
┌─────────────────────────────────────────────────────────────┐
│                         fitz-ai                             │
├─────────────────────────────────────────────────────────────┤
│  CLI Layer                                                  │
│  quickstart | init | ingest | query | config | doctor       │
├─────────────────────────────────────────────────────────────┤
│  Engines                                                    │
│  ┌─────────────┐  ┌─────────────┐                           │
│  │ Classic RAG │  │   CLaRa     │  (pluggable)              │
│  └─────────────┘  └─────────────┘                           │
├─────────────────────────────────────────────────────────────┤
│  Plugin System (all YAML-defined)                           │
│  ┌────────┐ ┌───────────┐ ┌────────┐ ┌─────────┐            │
│  │  LLM   │ │ Embedding │ │ Rerank │ │VectorDB │            │
│  └────────┘ └───────────┘ └────────┘ └─────────┘            │
│  openai, cohere, anthropic, ollama, azure...                │
├─────────────────────────────────────────────────────────────┤
│  Retrieval Pipelines (YAML-composed)                        │
│  dense.yaml | dense_rerank.yaml | custom...                 │
├─────────────────────────────────────────────────────────────┤
│  Constraints (epistemic safety)                             │
│  ConflictAware | InsufficientEvidence | CausalAttribution   │
└─────────────────────────────────────────────────────────────┘
```

---

## CLI Reference

```bash
fitz quickstart [PATH] [QUESTION]  # Zero-config RAG (start here)
fitz init                          # Interactive setup wizard
fitz ingest [PATH]                 # Ingest documents
fitz query [QUESTION]              # Query knowledge base
fitz collections                   # List and delete knowledge collections
fitz config                        # View/edit configuration
fitz doctor                        # System diagnostics
```

---

## Comparison ⚖️

| | fitz-ai | LangChain | LlamaIndex |
|--|---------|-----------|------------|
| Time to first answer | 5 min | 30+ min | 20+ min |
| Config required to start | None | Yes | Yes |
| Knows when to say "I don't know" | Built-in | DIY | DIY |
| Source citations | Automatic | Manual setup | Manual setup |
| Add new LLM provider | Drop a YAML | Subclass + 200 LOC | Subclass + 150 LOC |
| Swap retrieval paradigm | Change 1 line | Rewrite pipeline | Rewrite pipeline |

**Choose fitz-ai if**: You want to query your documents with AI, not build an AI platform.

**Choose LangChain if**: You're building complex agent workflows with tool use, memory, and multi-step reasoning.

**Choose LlamaIndex if**: You need deep customization of retrieval strategies across heterogeneous data sources.

---

## Beyond RAG 🔮

> **RAG is a method. Knowledge access is a strategy.**

Fitz is not a RAG framework. It's a knowledge platform that *currently* uses RAG as its primary engine.

```python
from fitz_ai import run

# Today: Classic RAG
answer = run("What are the payment terms?", engine="classic_rag")

# Also available: CLaRa (compressed RAG, 16x smaller context)
answer = run("What are the payment terms?", engine="clara")

# Tomorrow: GraphRAG, HyDE, or whatever comes next
answer = run("What are the payment terms?", engine="graph_rag")
```

The engine is an implementation detail. Your ingested knowledge, your queries, your workflow—all stay the same. When a better retrieval paradigm emerges, swap one line, not your entire codebase.

---

## Philosophy 📍

**Principles:**
- **Explicit over clever**: No magic. Read the config, know what happens.
- **Answers over architecture**: Optimize for time-to-insight, not flexibility.
- **Honest over helpful**: Better to say "I don't know" than hallucinate.
- **Files over frameworks**: YAML plugins over class hierarchies.

---

## License

MIT

---

## About

Solo project by [Yan Fitzner](https://github.com/yafitzdev). ~15k lines of Python. 400+ tests. Built from scratch—no LangChain or LlamaIndex under the hood.

---

## Links

- [GitHub](https://github.com/yafitzdev/fitz-ai)
- [PyPI](https://pypi.org/project/fitz-ai/)
- [Changelog](CHANGELOG.md)
- [CLI Documentation](docs/CLI.md)