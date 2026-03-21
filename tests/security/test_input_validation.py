# tests/security/test_input_validation.py
"""
Input validation and sanitization tests.

Verify that malformed/adversarial inputs don't crash the pipeline.
Uses 1-2 representative examples per category to stay under 2 min per test.

Run with: pytest tests/security/test_input_validation.py -v -s -m security
"""

from __future__ import annotations

import pytest

from fitz_ai.core import Query

from .conftest import with_tiered_fallback

pytestmark = pytest.mark.security


class TestMalformedInputs:
    """Tests for handling malformed query inputs."""

    @pytest.fixture(autouse=True)
    def setup_pipeline(self, krag_e2e_runner):
        self.runner = krag_e2e_runner

    def test_empty_query(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            Query(text="")

    def test_whitespace_only_query(self):
        for text in ["   ", "\t\n"]:
            with pytest.raises(ValueError, match="cannot be empty"):
                Query(text=text)

    @with_tiered_fallback
    def test_unicode_query(self):
        """Unicode characters should not crash the pipeline."""
        answer = self.runner.engine.answer(Query(text="What is TechCorp? 你好 🚗"))
        assert answer is not None

    @with_tiered_fallback
    def test_special_characters(self):
        """SQL injection and XSS attempts should not crash."""
        answer = self.runner.engine.answer(
            Query(text="TechCorp; DROP TABLE users;-- <script>alert('xss')</script>")
        )
        assert answer is not None
        assert "<script>" not in answer.text


class TestInputLengthLimits:
    """Tests for handling various input lengths."""

    @pytest.fixture(autouse=True)
    def setup_pipeline(self, krag_e2e_runner):
        self.runner = krag_e2e_runner

    def test_very_long_query(self):
        """Very long queries should be handled (truncated or rejected gracefully)."""
        long_query = "What is TechCorp? " * 100
        answer = self.runner.engine.answer(Query(text=long_query))
        assert answer is not None
        assert len(answer.text) < len(long_query)

    def test_query_with_long_word(self):
        long_word = "a" * 2000
        answer = self.runner.engine.answer(Query(text=f"What is {long_word}?"))
        assert answer is not None

    def test_many_queries_dont_leak_memory(self):
        """Memory should not grow unbounded across queries."""
        import gc

        import psutil

        process = psutil.Process()
        gc.collect()
        mem_before = process.memory_info().rss / 1024 / 1024

        # 5 queries — enough to detect leaks, fast enough for CI
        for i in range(5):
            try:
                self.runner.engine.answer(Query(text=f"Query {i}: What is TechCorp?"))
            except Exception:
                pass

        gc.collect()
        mem_after = process.memory_info().rss / 1024 / 1024
        mem_growth = mem_after - mem_before
        print(f"\nMemory growth over 5 queries: {mem_growth:.1f}MB")
        assert mem_growth < 100, f"Memory grew {mem_growth:.1f}MB - possible leak"


class TestRepeatedPatterns:
    """Tests for handling repeated/adversarial patterns."""

    @pytest.fixture(autouse=True)
    def setup_pipeline(self, krag_e2e_runner):
        self.runner = krag_e2e_runner

    @with_tiered_fallback
    def test_repeated_question_marks(self):
        answer = self.runner.engine.answer(Query(text="What is TechCorp????????"))
        assert answer is not None

    @with_tiered_fallback
    def test_newlines_and_spaces(self):
        """Excessive whitespace and newlines should be handled."""
        answer = self.runner.engine.answer(Query(text="What    is\n\nTechCorp\n  Industries?"))
        assert answer is not None
