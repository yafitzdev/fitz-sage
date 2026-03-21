# tests/load/test_scalability.py
"""
Scalability tests — concurrent queries and stability under load.

Tests here verify the harness handles concurrent access correctly and
doesn't degrade. They do NOT assert absolute throughput (hardware-dependent).

Run with: pytest tests/load/test_scalability.py -v -s --tb=short -m scalability
"""

from __future__ import annotations

import concurrent.futures
import time

import pytest

from fitz_ai.core import Query

pytestmark = pytest.mark.scalability


class TestConcurrentQueries:
    """Test behavior under concurrent query load."""

    @pytest.fixture(autouse=True)
    def setup_pipeline(self, krag_e2e_runner):
        self.runner = krag_e2e_runner

    def test_concurrent_queries_all_succeed(self):
        """Multiple concurrent queries should not crash or deadlock."""
        queries = [
            "Where is TechCorp headquartered?",
            "What is the price of Model Y200?",
            "Who is the CEO of TechCorp?",
            "Compare Model X100 vs Model Y200",
            "What employees work in Engineering?",
        ]

        def run_query(query: str) -> tuple[str, float, bool]:
            start = time.perf_counter()
            try:
                answer = self.runner.engine.answer(Query(text=query))
                elapsed = time.perf_counter() - start
                return (query, elapsed, answer is not None)
            except Exception:
                elapsed = time.perf_counter() - start
                return (query, elapsed, False)

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(run_query, q) for q in queries]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        print("\nConcurrent Query Results:")
        successes = 0
        for query, elapsed, success in results:
            status = "PASS" if success else "FAIL"
            print(f"  [{status}] {query[:40]}... ({elapsed:.2f}s)")
            if success:
                successes += 1

        assert successes == len(
            queries
        ), f"Only {successes}/{len(queries)} concurrent queries succeeded"

    def test_sequential_throughput_consistent(self):
        """Throughput should not degrade over sequential queries.

        Measures whether the 5th query is significantly slower than
        the 1st — catches resource leaks, connection pool exhaustion, etc.
        """
        query = "Where is TechCorp headquartered?"
        times = []

        for _ in range(5):
            start = time.perf_counter()
            self.runner.engine.answer(Query(text=query))
            times.append(time.perf_counter() - start)

        first = times[0]
        last = times[-1]
        ratio = last / first if first > 0 else float("inf")

        print(f"\nSequential Consistency (5 queries):")
        print(f"  First: {first:.1f}s, Last: {last:.1f}s, Ratio: {ratio:.2f}x")
        for i, t in enumerate(times):
            print(f"  Query {i + 1}: {t:.1f}s")

        # Last query should not be more than 2x slower than first
        # (allows for GC pauses but catches degradation)
        assert ratio < 2.0, (
            f"Throughput degraded: query 5 is {ratio:.1f}x slower than query 1 "
            f"({last:.1f}s vs {first:.1f}s)"
        )
