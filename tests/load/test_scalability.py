# tests/load/test_scalability.py
"""
Scalability tests — stability under sequential load.

Verifies the harness doesn't degrade across repeated queries.
Does NOT test concurrency (local ollama serializes LLM calls).

Run with: pytest tests/load/test_scalability.py -v -s --tb=short -m scalability
"""

from __future__ import annotations

import time

import pytest

from fitz_sage.core import Query

pytestmark = pytest.mark.scalability


class TestSequentialStability:
    """Test that repeated queries don't degrade."""

    @pytest.fixture(autouse=True)
    def setup_pipeline(self, krag_e2e_runner):
        self.runner = krag_e2e_runner

    def test_no_throughput_degradation(self):
        """3rd query should not be significantly slower than 1st.

        Catches resource leaks, connection pool exhaustion, memory pressure.
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

        print("\nSequential Consistency (3 queries):")
        for i, t in enumerate(times):
            print(f"  Query {i + 1}: {t:.1f}s")
        print(f"  Ratio last/first: {ratio:.2f}x")

        assert ratio < 2.0, (
            f"Throughput degraded: query 3 is {ratio:.1f}x slower than query 1 "
            f"({last:.1f}s vs {first:.1f}s)"
        )
