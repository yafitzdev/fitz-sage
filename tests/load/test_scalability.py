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

    def test_concurrent_queries_dont_crash(self):
        """Two concurrent queries should not crash or deadlock.

        Only 2 concurrent — local ollama serializes LLM calls, so more
        just queues up and triggers timeouts. This tests thread safety
        of the harness (shared DB connections, stores, etc.), not throughput.
        """
        queries = [
            "Where is TechCorp headquartered?",
            "What is the price of Model Y200?",
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

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(run_query, q) for q in queries]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        print("\nConcurrent Query Results:")
        for query, elapsed, success in results:
            status = "PASS" if success else "FAIL"
            print(f"  [{status}] {query[:40]}... ({elapsed:.2f}s)")

        successes = sum(1 for _, _, s in results if s)
        assert successes >= 1, "Both concurrent queries failed — possible deadlock"

    def test_sequential_throughput_consistent(self):
        """Throughput should not degrade over sequential queries.

        Measures whether the 3rd query is significantly slower than
        the 1st — catches resource leaks, connection pool exhaustion, etc.
        """
        query = "Where is TechCorp headquartered?"
        times = []

        for _ in range(3):
            start = time.perf_counter()
            self.runner.engine.answer(Query(text=query))
            times.append(time.perf_counter() - start)

        first = times[0]
        last = times[-1]
        ratio = last / first if first > 0 else float("inf")

        print(f"\nSequential Consistency (3 queries):")
        for i, t in enumerate(times):
            print(f"  Query {i + 1}: {t:.1f}s")
        print(f"  Ratio last/first: {ratio:.2f}x")

        assert ratio < 2.0, (
            f"Throughput degraded: query 3 is {ratio:.1f}x slower than query 1 "
            f"({last:.1f}s vs {first:.1f}s)"
        )
