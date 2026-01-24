# Freshness & Authority Boosting

## Problem

Standard RAG treats all documents equally. This fails when:

- "What's the latest status on feature X?" - Can't distinguish old vs new docs
- "What does the official spec say?" - Can't distinguish spec vs notes

## Solution: Intent-Triggered Freshness

Detect query intent from keywords and boost documents accordingly:

- **Recency keywords** ("latest", "recent", "current") → Boost newer documents
- **Authority keywords** ("official", "spec", "authoritative") → Boost authoritative sources

## How It Works

```
Query comes in
    │
    ├─ Contains recency keywords? ("latest", "recent", "current", "new", "updated")
    │       │
    │       ▼
    │   Boost by modified_at (exponential decay, half-life 90 days)
    │
    ├─ Contains authority keywords? ("official", "spec", "authoritative", "canonical")
    │       │
    │       ▼
    │   Boost by source_type (spec > design > document > notes)
    │
    └─ No intent keywords? → Pass through unchanged
```

## Key Design Decisions

1. **Always-on, intent-triggered** - Baked into every pipeline. Only activates when query signals intent.

2. **Metadata captured at ingestion** - File timestamps (`modified_at`) and source type (`source_type`) inferred from paths.

3. **Additive score adjustment** - Doesn't override relevance, just adds recency/authority boost.

4. **Graceful degradation** - If metadata missing, chunks pass through unchanged.

## Source Type Inference

During ingestion, source authority is inferred from file paths:

| Path Pattern | Source Type | Authority Score |
|--------------|-------------|-----------------|
| `/spec/`, `/specs/`, `/requirements/` | spec | 1.0 |
| `/design/`, `/architecture/`, `/adr/` | design | 0.8 |
| (default) | document | 0.6 |
| `/notes/`, `/drafts/`, `/scratch/` | notes | 0.4 |

## Example

**Query:** "What does the official spec say about battery warranty?"

**Before freshness boost:**
1. notes/meeting_notes.md (score: 0.85) - "discussed battery warranty..."
2. spec/requirements.md (score: 0.82) - "Battery warranty: 8 years..."
3. products.md (score: 0.80) - "8-year warranty included"

**After freshness boost:** (authority keyword "official" detected)
1. spec/requirements.md (score: 0.97) - boosted by +0.15 (spec authority)
2. notes/meeting_notes.md (score: 0.91) - boosted by +0.06 (notes authority)
3. products.md (score: 0.89) - boosted by +0.09 (document authority)

## Configuration

No configuration required. Feature is baked into the retrieval pipeline.

Default weights:
- `recency_weight`: 0.15 (max boost for very recent docs)
- `authority_weight`: 0.15 (max boost for spec docs)
- `recency_half_life_days`: 90 (days until recency score halves)

## Intent Keywords

**Recency triggers:**
- "latest", "recent", "current", "new", "updated", "newest", "now", "today"

**Authority triggers:**
- "official", "spec", "specification", "requirement", "authoritative", "canonical", "standard", "definitive"

## Implementation

- **Detection module:** `fitz_ai/retrieval/detection/modules/freshness.py`
- **Metadata ingestion:** `fitz_ai/ingestion/source/plugins/filesystem.py`
- **Orchestrator:** `fitz_ai/retrieval/detection/registry.py`

Detection is now LLM-based via the unified `DetectionOrchestrator`. The `FreshnessModule` determines `boost_recency` and `boost_authority` flags from query intent.

## Benefits

| Without Freshness | With Freshness |
|-------------------|----------------|
| Old docs rank equally | Recent docs boosted when asked |
| Notes = Specs | Specs prioritized when "official" |
| User filters manually | Intent-based automatic boosting |
