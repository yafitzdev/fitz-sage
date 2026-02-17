# Fitz-Planner MCP Server

## What This Is

An MCP server that adds overnight architectural planning capabilities to Claude Code, using local LLMs (Qwen Coder Next via Ollama) and KRAG-based context retrieval from fitz-ai. Developers queue planning jobs from their Claude Code chat, local models generate comprehensive architectural plans overnight, and results are ready to implement in the morning. Built as a separate, publishable package for any Claude Code user.

## Core Value

Shift expensive AI planning work from paid cloud APIs to free local LLMs running overnight, so developers are prepared for the post-subsidy AI era without sacrificing plan quality.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] MCP server exposes tools (create_plan, check_status, get_plan, list_plans) via stdio protocol
- [ ] Planning jobs run as async background tasks inside the MCP server process
- [ ] SQLite job queue persists jobs across MCP server restarts with auto-resume
- [ ] Ollama integration with Qwen Coder Next (80B primary, 32B fallback, auto-fallback on OOM)
- [ ] KRAG context retrieval from fitz-ai (on by default, opt-out for non-fitz-ai users)
- [ ] Multi-query KRAG strategy: architecture decisions, code symbols, past experiments, integration points
- [ ] Full architectural plan output: ADRs, schema designs, implementation roadmap, risk assessment
- [ ] Per-section confidence scoring with flagging for low-confidence sections
- [ ] Optional Anthropic API review of flagged sections (explicit opt-in per request, shows cost before charging)
- [ ] YAML configuration for models, thresholds, KRAG settings, output preferences
- [ ] Plan output as structured markdown with metadata
- [ ] Installable via pip, publishable to PyPI
- [ ] Cross-platform: Windows, macOS, Linux

### Out of Scope

- Separate background daemon — MCP server handles jobs internally via async tasks
- Web UI — Claude Code chat is the interface
- Real-time streaming of plan generation progress to Claude Code — status polling is sufficient
- IDE plugins — MCP protocol is the integration point
- Cloud-hosted planning — local-first is the entire point
- GSD artifact generation (PROJECT.md/ROADMAP.md format) — full architectural plans only for v1
- Multi-user / team features — single developer tool

## Context

- **Parent project**: fitz-ai is a local-first modular RAG knowledge engine with KRAG (retrieval intelligence). Fitz-planner uses it as the context retrieval layer.
- **MCP protocol**: Model Context Protocol lets tools expose capabilities to Claude Code via stdio. The MCP server is a long-lived process started by Claude Code.
- **Cost motivation**: Planning consumes ~36% of AI coding costs. At €100-300/month, that's €36-108/month on planning alone. Local LLMs reduce this to electricity cost.
- **Post-subsidy preparation**: Current AI pricing is subsidized. When subsidies end, having local planning infrastructure means cost insulation.
- **Hardware trajectory**: Developer currently has 16GB VRAM + 16GB RAM (fits 32B Q4). Upgrading to 96GB RAM (fits 80B Q4 with CPU offload). Auto-fallback handles both scenarios.
- **Qwen Coder Next**: Selected for strong research/reasoning capabilities. 80B Q4_K_M is primary model (~45GB), 32B Q4 is fallback (~18GB).

## Constraints

- **Separate repo**: Standalone package, depends on fitz-ai as optional dependency
- **Python 3.10+**: Match fitz-ai compatibility
- **Local-first**: No cloud dependency for core planning (API review is explicitly optional)
- **MCP protocol**: Must conform to MCP server specification for Claude Code compatibility
- **Ollama dependency**: Requires Ollama installed locally for LLM inference
- **Long-running jobs**: Planning takes 2-8 hours depending on model/hardware — architecture must handle this gracefully within MCP server lifecycle

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Separate repo (not module in fitz-ai) | Publishable for non-fitz-ai users, clean dependency boundary | — Pending |
| MCP server with internal async jobs (no daemon) | Simpler architecture, matches Claude Code model, auto-resumes on restart | — Pending |
| KRAG on by default, opt-out | Showcases fitz-ai, best quality path, still works without it | — Pending |
| Qwen Coder Next 80B/32B via Ollama | Best local reasoning quality, fits target hardware, overnight speed irrelevant | — Pending |
| SQLite for job queue | Zero dependencies, persistent, good enough for 1-10 jobs/day | — Pending |
| API review as explicit opt-in per request | User controls costs, shows price before charging, privacy-preserving default | — Pending |
| Full architectural plan output (not GSD artifacts) | Richer output, usable by any developer, not tied to GSD workflow | — Pending |

---
*Last updated: 2026-02-17 after initialization*
