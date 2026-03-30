# docs/roadmap/krag-agent.md
# KRAG Agent — Retrieval-as-Tools with Epistemic Self-Verification

**Status:** Proposed
**Target:** v0.11.0
**Impact:** Transformational — turns fitz-sage from best RAG pipeline into first autonomous knowledge agent with epistemic integrity

---

## Problem

fitz-sage has the most sophisticated retrieval intelligence stack of any RAG system: typed retrieval units (symbols, sections, tables), ML-gated detection, multi-strategy dispatch, multi-hop reasoning, governance cascade. But it's all hardcoded into a **single-shot pipeline**. The LLM gets one chance with whatever the pipeline retrieved.

Every component is a *tool* waiting to be used by an agent. The pipeline is an agent that doesn't know it's an agent.

```
CURRENT:  Query → [fixed pipeline] → Answer
PROPOSED: Query → [LLM agent with retrieval tools] → Evidence-verified Answer
```

---

## Why This, Above Everything Else

| Criterion | Score | Reasoning |
|-----------|-------|-----------|
| **Innovation** | Highest | First agentic RAG with typed retrieval tools + epistemic self-verification |
| **Accretive** | Highest | Every existing feature becomes more valuable as a tool |
| **Useful** | Highest | Solves #1 RAG problem (wrong context) through iterative refinement |
| **Compelling** | Highest | Demo writes itself — visible reasoning over code + docs + data |

### What makes this different from generic "agentic RAG"

Every other agentic RAG system gives the LLM generic "search" tools over flat chunks. KRAG Agent is unique because:

1. **Tools return typed results** — the agent sees *symbols*, *sections*, and *table rows*, not opaque chunks. It reasons: "I found the function but need the doc that describes its contract."
2. **The agent can self-verify** — `verify(claim, sources)` checks reasoning against the governance cascade *before* answering. Epistemic honesty becomes deliberate, not post-hoc.
3. **Cross-type reasoning is natural** — "The code does X (symbol), the docs say Y (section), the config table shows Z (table row)" emerges from tool composition.
4. **Detection system becomes the planner** — QueryAnalyzer + DetectionOrchestrator outputs become the agent's initial plan, not hardcoded strategy weights.

---

## Architecture

### Tools (wrapping existing components)

| Tool | Wraps | Returns |
|------|-------|---------|
| `search_code(query)` | CodeSearchStrategy | Typed SYMBOL results with qualified names |
| `search_docs(query)` | SectionSearchStrategy | Typed SECTION results with hierarchy |
| `query_table(question)` | TableSearchStrategy | SQL-executed TABLE results with schema |
| `expand(address)` | CodeExpander | Import graph, class context, entity links |
| `verify(claim, sources)` | Governance constraints | TRUSTWORTHY / DISPUTED / ABSTAIN per claim |
| `related(entity)` | EntityGraphStore | Cross-document entity traversal |
| `read(path, lines)` | FileStore | Raw file access for targeted reading |
| `search_collection(query, collection)` | Cross-collection | Federated search (new capability for free) |

### Agent Loop

```
1. PLAN    — Analyze query via QueryAnalyzer + DetectionOrchestrator
             Agent receives analysis as structured context, decides tools
2. RETRIEVE — Execute chosen tools (parallel where independent)
3. REFLECT  — Evaluate evidence sufficiency, identify gaps
4. ITERATE  — If gaps: formulate follow-up queries, invoke more tools
             (max_steps configurable, default 6)
5. SYNTHESIZE — Generate answer from accumulated evidence
6. VALIDATE  — Self-check via verify() before responding
7. RESPOND   — Return Answer with full provenance chain + reasoning trace
```

### Dual Mode

```yaml
# Config: fitz_krag.yaml
mode: "pipeline"    # Default — fast, deterministic, current behavior
mode: "agent"       # Thorough — LLM-driven retrieval with self-verification
mode: "auto"        # Smart — pipeline for high-confidence, agent for low-confidence
```

`auto` mode uses QueryAnalyzer confidence: ≥0.85 → pipeline (fast path), <0.85 → agent (thorough path). The existing `FitzKragEngine` stays untouched as the fast path.

---

## Phases

### Phase 1: Tool Definitions & Agent Core

**Goal:** Wrap existing retrieval strategies as callable tools, build the agent loop.

**Files to create:**
- `engines/fitz_krag/agent/tools.py` — Tool definitions wrapping strategies
- `engines/fitz_krag/agent/loop.py` — Agent loop with plan/retrieve/reflect/synthesize
- `engines/fitz_krag/agent/planner.py` — Initial plan from QueryAnalyzer + Detection

**Files to modify:**
- `engines/fitz_krag/engine.py` — Add `mode` config, route to agent or pipeline
- `engines/fitz_krag/config/schema.py` — Add `mode` field

**Key decisions:**
- Tool-call format: Use structured JSON tool calls (compatible with all major LLM providers)
- Agent prompt: System prompt that describes available tools and when to use each
- Max steps: Default 6, configurable. Hard ceiling at 10 to prevent runaway loops.
- Evidence accumulation: List of `(tool_name, query, results)` tuples passed to synthesizer

**Constraints:**
- Must work with local models (Ollama/LM Studio 7B+). Tool-call prompts must be simple enough for small models.
- No new dependencies. Uses existing `ChatProvider` protocol for agent LLM calls.

### Phase 2: Self-Verification Tool

**Goal:** Let the agent verify claims against evidence mid-loop, before final answer.

**Files to create:**
- `engines/fitz_krag/agent/verifier.py` — Wraps governance constraints as a callable tool

**How it works:**
- Agent calls `verify(claim="JWT tokens rotated every 24h", sources=["auth_handler.py:42"])`
- Verifier runs the claim through existing governance pipeline (CascadeClassifier + constraint plugins)
- Returns `{mode: "trustworthy", confidence: 0.94}` or `{mode: "disputed", reason: "spec says AES-256 but code uses AES-128"}`
- Agent can decide to investigate disputes or abstain if evidence insufficient

**Key insight:** This reuses the governance system not as a post-hoc gate but as an in-loop reasoning primitive. No new ML models needed.

### Phase 3: Cross-Collection Federation

**Goal:** Agent can search across multiple collections as part of its reasoning.

**Files to create:**
- `engines/fitz_krag/agent/federation.py` — Cross-collection tool wrapper

**How it works:**
- Agent has a `search_collection(query, collection_name)` tool
- Can compare information across knowledge bases
- Example: "Does our internal API match the public documentation?" → search `code` collection + search `docs` collection → compare

**This is trivial once the agent loop exists** — it's just another tool that instantiates a strategy against a different collection.

### Phase 4: Reasoning Trace & Observability

**Goal:** Capture the agent's reasoning as a navigable trace.

**Files to create:**
- `engines/fitz_krag/agent/trace.py` — Structured trace capture

**Trace format:**
```python
@dataclass
class ReasoningTrace:
    steps: list[TraceStep]  # Each tool call + result + agent reflection
    total_tools_used: int
    total_llm_calls: int
    elapsed_ms: int

@dataclass
class TraceStep:
    tool: str
    query: str
    result_summary: str
    agent_reflection: str  # "Found code but need docs" etc.
    evidence_added: list[str]
```

**Exposed via:**
- `Answer.metadata["trace"]` — programmatic access
- `fitz query --trace` — CLI flag for readable trace output
- `GET /query?trace=true` — API parameter

### Phase 5: Auto Mode & Confidence Routing

**Goal:** Automatically choose pipeline vs. agent based on query complexity.

**Files to modify:**
- `engines/fitz_krag/engine.py` — Routing logic
- `engines/fitz_krag/query_analyzer.py` — Emit routing signal

**Routing heuristics:**
- High confidence (≥0.85) + single intent → pipeline (fast, ~2s)
- Low confidence (<0.85) or multi-type or comparison → agent (thorough, ~10-20s)
- Explicit `mode: "agent"` in config → always agent
- `--agent` CLI flag for per-query override

---

## Killer Demo

```
$ fitz query "Is our auth middleware compliant with the security requirements?" \
    --source ./project --agent --trace

━━━ KRAG Agent ━━━

Step 1: PLAN
  Query analysis: CROSS-TYPE (code + documentation), confidence: 0.62
  Plan: search code for auth implementation → search docs for security reqs → cross-reference

Step 2: search_code("authentication middleware")
  → auth_handler.authenticate (symbol, auth_handler.py:15-40)
  → jwt_validator.validate (symbol, jwt_validator.py:3-25)

Step 3: search_docs("security requirements")
  → security_spec.md §4.1: Token rotation
  → security_spec.md §4.2: Encryption standards
  → security_spec.md §4.3: Session management

Step 4: expand("jwt_validator.validate")
  → imports: session_store.encrypt_token
  → class context: JWTValidator.__init__ configures AES mode

Step 5: REFLECT
  "I have code + requirements. Need to verify compliance claim by claim."

Step 6: query_table("security compliance checklist")
  → compliance.csv: 12 items, 3 auth-related (rows 1, 4, 7)

Step 7: verify("JWT tokens rotated every 24h", [auth_handler.py:42, security_spec.md §4.1])
  → TRUSTWORTHY (confidence: 0.94)

Step 8: verify("Session tokens encrypted with AES-256", [session_store.py:15, security_spec.md §4.2])
  → DISPUTED — code uses AES-128, spec requires AES-256

Step 9: SYNTHESIZE

━━━ Answer (mode: DISPUTED) ━━━

The authentication middleware is **partially compliant** with security requirements.

**Compliant:**
- JWT token rotation every 24 hours [S1: auth_handler.py:42] matches requirement [S2: security_spec.md §4.1]
- Session timeout enforcement [S3: middleware.py:8-12] meets §4.3

**Non-compliant:**
- Session token encryption uses AES-128 [S4: session_store.py:15] but security spec §4.2
  requires AES-256 [S5: security_spec.md §4.2]

Sources: [S1-S5 with file:line provenance]

━━━ Trace: 8 tools, 4 LLM calls, 14.2s ━━━
```

---

## What This Subsumes

Features that become unnecessary or are absorbed by the agent:

| Existing Feature | Agent Equivalent |
|-----------------|------------------|
| Multi-hop (fixed 2-hop cycle) | Agent iterates as many times as needed |
| Hardcoded strategy weights | Agent decides based on analysis |
| HyDE gating heuristics | Agent invokes HyDE when it judges results are too abstract |
| Multi-query decomposition | Agent decomposes naturally via multiple tool calls |
| Post-hoc governance | In-loop self-verification via `verify()` tool |

The pipeline mode remains for fast, simple queries. Agent mode handles everything the pipeline can't.

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Local model can't do tool-use well | Simple JSON tool format; test with Llama 3.1 8B+ which handles tools. Fallback: pipeline mode. |
| Latency (multiple LLM calls) | Auto mode routes simple queries to pipeline. Agent reserved for complex. Max_steps ceiling. |
| Cost (API models) | Detection gating still applies within agent. Most tools are DB queries, not LLM calls. |
| Regression on simple queries | Pipeline mode unchanged. Agent is additive, not replacement. |
| Runaway agent loops | Hard max_steps (default 6, ceiling 10). Reflect step checks sufficiency. |

---

## Success Criteria

1. Agent mode produces higher-quality answers than pipeline on multi-type queries (code + docs + data)
2. Agent mode produces correct DISPUTED/ABSTAIN signals via self-verification ≥90% of cases where pipeline governance also catches them
3. Latency for agent mode ≤20s on local models (Ollama qwen2.5:14b or equivalent)
4. Pipeline mode performance unchanged (zero regression)
5. Auto mode correctly routes ≥85% of queries (simple → pipeline, complex → agent)

---

## Alternatives Considered

| Alternative | Why Not Primary |
|-------------|----------------|
| Evidence Graphs (structured claim extraction) | Adds 2+ LLM calls to every query; agent does this naturally when needed |
| Self-Calibrating RAG (governance → tuning loop) | Research project, months to validate; agent is shippable |
| Retrieval Debugger / Trace | Developer tool, not product feature; agent traces ARE the debugger (Phase 4) |
| Knowledge Graph Construction | Expensive at ingestion, rigid schema; agent navigates existing entity graph dynamically |
| Cross-Collection Federation | Becomes trivial once agent exists — just another tool (Phase 3) |
