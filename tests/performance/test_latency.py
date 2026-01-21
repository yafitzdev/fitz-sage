# tests/performance/test_latency.py
"""
Latency and throughput benchmarks.

Run with: pytest tests/performance/ -v -s --tb=short
"""

from __future__ import annotations

import pytest

from .conftest import PERF_THRESHOLDS

pytestmark = pytest.mark.performance


class TestQueryLatency:
    """Query latency benchmarks."""

    @pytest.fixture(autouse=True)
    def setup_pipeline(self, e2e_runner):
        """Use e2e runner's pre-ingested data."""
        self.runner = e2e_runner

    def test_simple_query_latency(self, measure_perf):
        """Simple factual query should be fast."""

        def query():
            return self.runner.pipeline.run("Where is TechCorp headquartered?")

        metrics = measure_perf(query, iterations=5, warmup=1)

        print("\nSimple Query Latency:")
        print(f"  p50: {metrics.p50:.0f}ms")
        print(f"  p95: {metrics.p95:.0f}ms")
        print(f"  p99: {metrics.p99:.0f}ms")

        assert (
            metrics.p95 < PERF_THRESHOLDS["query_p95_ms"]
        ), f"p95 latency {metrics.p95:.0f}ms exceeds threshold {PERF_THRESHOLDS['query_p95_ms']}ms"

    def test_complex_query_latency(self, measure_perf):
        """Multi-hop query latency."""

        def query():
            return self.runner.pipeline.run(
                "What does Sarah Chen's company's main competitor manufacture?"
            )

        metrics = measure_perf(query, iterations=5, warmup=1)

        print("\nComplex Query Latency:")
        print(f"  p50: {metrics.p50:.0f}ms")
        print(f"  p95: {metrics.p95:.0f}ms")
        print(f"  p99: {metrics.p99:.0f}ms")

        assert metrics.p99 < PERF_THRESHOLDS["query_p99_ms"]

    def test_long_query_latency(self, measure_perf):
        """Long multi-part query latency."""

        def query():
            return self.runner.pipeline.run(
                "I need a comprehensive analysis of TechCorp's product lineup "
                "including all vehicle models with their prices, ranges, battery "
                "capacities, and key distinguishing features. Also include "
                "information about the target market segment for each model."
            )

        metrics = measure_perf(query, iterations=3, warmup=1)

        print("\nLong Query Latency:")
        print(f"  p50: {metrics.p50:.0f}ms")
        print(f"  p95: {metrics.p95:.0f}ms")


class TestRetrievalLatency:
    """Retrieval-only latency (no LLM generation)."""

    @pytest.fixture(autouse=True)
    def setup_pipeline(self, e2e_runner):
        self.runner = e2e_runner

    def test_retrieval_only_latency(self, measure_perf):
        """Pure retrieval without LLM should be very fast."""

        def retrieve():
            # Access retrieval step directly
            return self.runner.pipeline.retrieval.retrieve("TechCorp electric vehicles")

        metrics = measure_perf(retrieve, iterations=10, warmup=2)

        print("\nRetrieval-Only Latency:")
        print(f"  p50: {metrics.p50:.0f}ms")
        print(f"  p95: {metrics.p95:.0f}ms")

        assert metrics.p95 < PERF_THRESHOLDS["retrieval_p95_ms"], (
            f"Retrieval p95 {metrics.p95:.0f}ms exceeds threshold "
            f"{PERF_THRESHOLDS['retrieval_p95_ms']}ms"
        )


class TestMemoryUsage:
    """Memory usage benchmarks."""

    @pytest.fixture(autouse=True)
    def setup_pipeline(self, e2e_runner):
        self.runner = e2e_runner

    def test_query_memory_stability(self, measure_perf):
        """Memory should not grow unbounded across queries."""

        def query():
            return self.runner.pipeline.run("What is TechCorp?")

        metrics = measure_perf(query, iterations=20, warmup=2)

        print("\nMemory Usage Across 20 Queries:")
        print(f"  Avg delta: {metrics.avg_memory_mb:.1f}MB")
        print(f"  Peak delta: {metrics.peak_memory_mb:.1f}MB")

        # Memory should not grow significantly per query
        # (some variance is normal due to GC timing)
        assert (
            metrics.peak_memory_mb < 100
        ), f"Memory grew {metrics.peak_memory_mb:.1f}MB - potential leak"
