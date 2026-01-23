# tests/load/test_scalability.py
"""
Scalability tests for corpus size and concurrent queries.

Run with: pytest tests/load/test_scalability.py -v -s --tb=short -m scalability
"""

from __future__ import annotations

import concurrent.futures
import gc
import tempfile
import time
from pathlib import Path

import psutil
import pytest

pytestmark = pytest.mark.scalability


class TestCorpusScalability:
    """Test behavior with varying corpus sizes."""

    def generate_documents(self, count: int, tmp_dir: Path) -> list[Path]:
        """Generate synthetic documents for testing."""
        docs = []
        for i in range(count):
            doc_path = tmp_dir / f"doc_{i:05d}.md"
            content = f"""# Document {i}

## Section A
This is synthetic document number {i} for scalability testing.
It contains information about topic_{i % 100} and category_{i % 10}.

## Section B
The quick brown fox jumps over the lazy dog. This sentence contains
every letter of the alphabet and is used for testing purposes.

## Data
- ID: DOC-{i:05d}
- Category: cat_{i % 10}
- Topic: topic_{i % 100}
- Value: {i * 1.5:.2f}
"""
            doc_path.write_text(content)
            docs.append(doc_path)
        return docs

    @pytest.mark.parametrize("doc_count", [100, 500, 1000])
    def test_ingestion_scalability(self, doc_count):
        """Test ingestion time scales reasonably with corpus size."""
        pytest.skip("Ingestion scalability test requires full ingestion pipeline setup")
        # Note: This test would need to set up embedder, vector DB, chunking, etc.
        # Skipping for now as it requires significant infrastructure

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            _docs = self.generate_documents(doc_count, tmp_path)

            gc.collect()
            mem_before = psutil.Process().memory_info().rss / 1024 / 1024
            start = time.perf_counter()

            # Ingest to a test collection
            _collection_name = f"scale_test_{doc_count}"
            # Would need to call run_diff_ingest with full setup
            pass

            elapsed = time.perf_counter() - start
            mem_after = psutil.Process().memory_info().rss / 1024 / 1024

            docs_per_sec = doc_count / elapsed
            mem_per_doc = (mem_after - mem_before) / doc_count

            print(f"\nIngestion Scalability ({doc_count} docs):")
            print(f"  Time: {elapsed:.1f}s ({docs_per_sec:.1f} docs/sec)")
            print(f"  Memory: {mem_after - mem_before:.1f}MB ({mem_per_doc:.2f}MB/doc)")

            # Assertions
            assert docs_per_sec > 1, f"Ingestion too slow: {docs_per_sec:.2f} docs/sec"
            assert mem_per_doc < 10, f"Memory per doc too high: {mem_per_doc:.1f}MB"


class TestConcurrentQueries:
    """Test behavior under concurrent query load."""

    @pytest.fixture(autouse=True)
    def setup_pipeline(self, e2e_runner):
        self.runner = e2e_runner

    def test_concurrent_queries(self):
        """Multiple concurrent queries should not fail."""
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
                result = self.runner.pipeline.run(query)
                elapsed = time.perf_counter() - start
                return (query, elapsed, result is not None)
            except Exception:
                elapsed = time.perf_counter() - start
                return (query, elapsed, False)

        # Run queries concurrently
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

    @pytest.mark.parametrize("num_queries", [10, 25, 50])
    def test_sequential_throughput(self, num_queries):
        """Measure queries per second throughput."""
        query = "Where is TechCorp headquartered?"

        start = time.perf_counter()
        for _ in range(num_queries):
            self.runner.pipeline.run(query)
        elapsed = time.perf_counter() - start

        qps = num_queries / elapsed

        print(f"\nSequential Throughput ({num_queries} queries):")
        print(f"  Total time: {elapsed:.1f}s")
        print(f"  Throughput: {qps:.2f} queries/sec")

        # Throughput depends on LLM latency:
        # - Cloud LLM (Cohere): ~10-15s per query -> 0.07-0.1 qps
        # - Local LLM (Ollama): ~5-10s per query -> 0.1-0.2 qps
        # Minimum threshold accounts for cloud LLM with network variance
        assert qps > 0.05, f"Throughput too low: {qps:.2f} qps"
