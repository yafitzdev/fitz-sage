# tests/performance/conftest.py
"""
Performance test fixtures.

These tests measure latency, memory usage, and throughput
under normal (single-user) conditions.

Note: e2e fixtures (e2e_runner) are imported via tests/conftest.py
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import Callable

import psutil
import pytest


def pytest_collection_modifyitems(items):
    """Add tier4 and performance markers to all tests in this directory."""
    for item in items:
        if "/performance/" in str(item.fspath) or "\\performance\\" in str(item.fspath):
            item.add_marker(pytest.mark.tier4)
            item.add_marker(pytest.mark.performance)


@dataclass
class PerfMetrics:
    """Container for performance measurements."""

    latencies_ms: list[float] = field(default_factory=list)
    memory_mb: list[float] = field(default_factory=list)

    @property
    def p50(self) -> float:
        if not self.latencies_ms:
            return 0
        sorted_lat = sorted(self.latencies_ms)
        return sorted_lat[len(sorted_lat) // 2]

    @property
    def p95(self) -> float:
        if not self.latencies_ms:
            return 0
        sorted_lat = sorted(self.latencies_ms)
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    @property
    def p99(self) -> float:
        if not self.latencies_ms:
            return 0
        sorted_lat = sorted(self.latencies_ms)
        idx = int(len(sorted_lat) * 0.99)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    @property
    def avg_memory_mb(self) -> float:
        return sum(self.memory_mb) / len(self.memory_mb) if self.memory_mb else 0

    @property
    def peak_memory_mb(self) -> float:
        return max(self.memory_mb) if self.memory_mb else 0


@pytest.fixture
def measure_perf() -> Callable:
    """Factory fixture for measuring performance."""

    def _measure(func: Callable, iterations: int = 10, warmup: int = 2) -> PerfMetrics:
        metrics = PerfMetrics()
        process = psutil.Process()

        # Warmup runs (not measured)
        for _ in range(warmup):
            func()
            gc.collect()

        # Measured runs
        for _ in range(iterations):
            gc.collect()
            mem_before = process.memory_info().rss / 1024 / 1024

            start = time.perf_counter()
            func()
            elapsed_ms = (time.perf_counter() - start) * 1000

            mem_after = process.memory_info().rss / 1024 / 1024

            metrics.latencies_ms.append(elapsed_ms)
            metrics.memory_mb.append(mem_after - mem_before)

        return metrics

    return _measure


# Performance thresholds (adjust based on your requirements)
# Note: These thresholds account for:
# - Cloud LLM latency (Cohere API: ~2-5s per call)
# - Cloud embedding latency (~500ms per call, multiple calls for query expansion)
# - Network variance in CI environments
#
# Retrieval pipeline work includes:
# - Query rewriting (skipped for simple queries via heuristics)
# - Synonym/acronym expansion (creates 2-4 query variations)
# - Embedding each variation (cloud API calls - main latency source)
# - Hybrid search (dense + sparse with RRF fusion)
# - Entity graph expansion
# - Keyword filtering
#
# Optimization opportunity: Use batch embeddings to reduce API round-trips
PERF_THRESHOLDS = {
    "query_p95_ms": 15000,  # 15 seconds max for p95 (includes LLM generation)
    "query_p99_ms": 20000,  # 20 seconds max for p99 (multi-hop or complex queries)
    "ingestion_mb_per_doc": 50,  # Max 50MB memory per document
    "retrieval_p95_ms": 8000,  # 8s for retrieval (embedding API calls for query expansion)
}
