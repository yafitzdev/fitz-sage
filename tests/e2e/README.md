# E2E Retrieval Tests

Comprehensive end-to-end tests for retrieval intelligence features.

## Quick Start

```bash
# Run all tests (takes ~30 minutes with gpt-4o)
pytest tests/e2e/

# Run specific feature
pytest tests/e2e/ -k TestTableQueries
pytest tests/e2e/ -k TestMultiHop

# Run first N scenarios for quick feedback
pytest tests/e2e/ -k "E01 or E02 or E03 or E04 or E05"
```

## Speed Optimization

### 1. Use Faster LLM (Recommended)

Edit `fitz.yaml` to use gpt-4o-mini:

```yaml
chat:
  plugin_name: openai
  kwargs:
    model: gpt-4o-mini  # 10x faster, 20x cheaper than gpt-4o
```

**Impact:** 30 min → 3-5 min

### 2. Run Test Classes in Parallel

Since all tests are in one file, you can manually run classes in parallel:

```bash
# Terminal 1
pytest tests/e2e/ -k TestTableQueries &

# Terminal 2
pytest tests/e2e/ -k TestMultiHop &

# Terminal 3
pytest tests/e2e/ -k TestBasicRetrieval &

# Wait for all
wait
```

**Impact:** Linear speedup based on number of terminals

### 3. Run Subset for Fast Feedback

```bash
# Critical features only (10 scenarios, ~2 min)
pytest tests/e2e/ -k "E13 or E20 or E35 or E54 or E63 or E67 or E71 or E75 or E01 or E05"

# One per feature type (~15 scenarios, ~5 min)
pytest tests/e2e/ -k "E01 or E05 or E08 or E10 or E13 or E16 or E20 or E35 or E54 or E63 or E67 or E71 or E75"
```

## Test Classes

| Class | Feature | Scenarios |
|-------|---------|-----------|
| `TestMultiHop` | Multi-hop reasoning | 4 |
| `TestEntityGraph` | Entity graph expansion | 3 |
| `TestComparison` | Comparison queries | 7 |
| `TestConflictAware` | Conflict detection | 2 |
| `TestInsufficientEvidence` | Epistemic honesty | 4 |
| `TestCausalAttribution` | Causal reasoning | 2 |
| `TestTableQueries` | CSV/table data | 8 |
| `TestCodeSearch` | Code-aware retrieval | 7 |
| `TestLongDocument` | Long document handling | 5 |
| `TestBasicRetrieval` | Basic retrieval | 12 |
| `TestFreshness` | Freshness/authority | 4 |
| `TestHybridSearch` | Dense + sparse | 4 |
| `TestQueryExpansion` | Synonym generation | 4 |
| `TestTemporal` | Temporal queries | 4 |
| `TestAggregation` | Aggregation queries | 5 |

## Notes

- Tests share one ingested collection per run (module scope)
- Setup includes entity graphs, vocabulary, tables
- Cleanup happens automatically after tests complete
- Each run uses a unique collection name to avoid conflicts
