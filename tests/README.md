# tests/README.md
# Fitz-AI Test Suite

Comprehensive testing strategy for production-ready RAG.

## Test Categories

| Category | Purpose | Marker | Run Command |
|----------|---------|--------|-------------|
| **Unit** | Test individual functions/classes | - | `pytest tests/unit/` |
| **E2E** | Correctness across full pipeline | `e2e` | `pytest -m e2e` |
| **Performance** | Latency, memory, throughput | `performance` | `pytest -m performance` |
| **Scalability** | Large corpus, concurrent load | `scalability` | `pytest -m scalability` |
| **Security** | Prompt injection, data leakage | `security` | `pytest -m security` |
| **Chaos** | Failure modes, recovery | `chaos` | `pytest -m chaos` |
| **Load** | Concurrent users (Locust) | - | `locust -f tests/load/locustfile.py` |

## Quick Start

```bash
# Run all fast tests (unit + e2e)
pytest tests/unit/ tests/e2e/ -v

# Run specific categories
pytest -m e2e              # Correctness (122 scenarios)
pytest -m security         # Security tests
pytest -m performance      # Benchmarks
pytest -m chaos            # Failure handling

# Run everything except slow tests
pytest -m "not slow and not scalability"

# Run with coverage
pytest --cov=fitz_ai --cov-report=html
```

## Test Structure

```
tests/
├── unit/                 # Fast, isolated unit tests (~80 files)
│   ├── test_*.py         # Component/function tests
│   └── tabular/          # Tabular data processing tests
├── e2e/                  # End-to-end correctness tests
│   ├── fixtures/         # Test documents (md, csv, py, etc.)
│   ├── scenarios.py      # 122 test scenarios
│   ├── runner.py         # E2E test runner with tiered execution
│   └── e2e_config.yaml   # Test-specific LLM/embedding config
├── performance/          # Latency and throughput benchmarks
│   ├── test_latency.py   # Query latency (p50/p95/p99)
│   └── conftest.py       # Performance measurement fixtures
├── load/                 # Concurrent load testing
│   ├── locustfile.py     # Locust user simulation
│   └── test_scalability.py  # Corpus size and concurrent queries
├── security/             # Security and privacy tests
│   ├── test_prompt_injection.py  # Injection attacks
│   ├── test_data_leakage.py      # PII and access control
│   └── test_input_validation.py  # Malformed input handling
└── chaos/                # Reliability and failure tests
    └── test_failure_modes.py     # LLM/DB failures, recovery
```

## E2E Test Scenarios (122 tests)

| Feature | Count | What It Tests |
|---------|-------|---------------|
| MULTI_HOP | 6 | Multi-step reasoning chains |
| ENTITY_GRAPH | 6 | Entity relationship expansion |
| COMPARISON | 7 | Product/entity comparisons |
| MULTI_QUERY | 6 | Complex multi-part queries |
| KEYWORD_EXACT | 6 | Exact term matching |
| CONFLICT_AWARE | 6 | Contradictory source detection |
| INSUFFICIENT_EVIDENCE | 6 | "I don't know" responses |
| CAUSAL_ATTRIBUTION | 6 | Cause-effect claims |
| TABLE_SCHEMA | 6 | CSV/table structure |
| TABLE_QUERY | 8 | SQL-like queries |
| CODE_SEARCH | 8 | Code comprehension |
| LONG_DOC | 6 | Long document retrieval |
| BASIC_RETRIEVAL | 13 | Simple fact lookup |
| DEDUP | 6 | Cross-document deduplication |
| FRESHNESS | 6 | Authoritative source preference |
| HYBRID_SEARCH | 6 | Dense + sparse retrieval |
| QUERY_EXPANSION | 6 | Synonym/acronym expansion |
| TEMPORAL | 6 | Time-based queries |
| AGGREGATION | 6 | List/count queries |

## Running Load Tests

```bash
# Install load testing dependencies
pip install -e ".[loadtest]"

# Run with Locust (headless, 10 users, 60 seconds)
cd tests/load
locust -f locustfile.py --headless -u 10 -r 2 -t 60s

# Or with web UI
locust -f locustfile.py
# Open http://localhost:8089
```

## Performance Thresholds

Default thresholds (configurable in `tests/performance/conftest.py`):

| Metric | Threshold | Description |
|--------|-----------|-------------|
| query_p95_ms | 5000 | 95th percentile query latency |
| query_p99_ms | 10000 | 99th percentile query latency |
| retrieval_p95_ms | 500 | Retrieval-only (no LLM) |
| ingestion_mb_per_doc | 50 | Memory per document |

## Security Test Coverage

- **Prompt Injection**: Direct attacks, roleplay, encoding bypasses
- **Data Leakage**: PII fabrication, cross-collection access
- **Input Validation**: Unicode, special chars, length limits
- **Output Sanitization**: Raw dump prevention, source attribution

## Adding New Tests

1. **E2E scenario**: Add to `tests/e2e/scenarios.py`
2. **Security test**: Add to appropriate file in `tests/security/`
3. **Performance benchmark**: Add to `tests/performance/test_latency.py`
4. **Failure mode**: Add to `tests/chaos/test_failure_modes.py`

## CI/CD Integration

```yaml
# Example GitHub Actions workflow
jobs:
  test:
    steps:
      - run: pytest tests/unit/ -v
      - run: pytest -m e2e --tb=short
      - run: pytest -m security
      - run: pytest -m "performance and not slow"
```
