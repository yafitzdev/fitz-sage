# tests/security/test_input_validation.py
"""
Input validation and sanitization tests.

These tests verify that:
1. Malformed inputs are handled gracefully
2. Extremely long inputs don't cause issues
3. Special characters are handled properly

Run with: pytest tests/security/test_input_validation.py -v -s -m security
"""

from __future__ import annotations

import pytest

from .conftest import with_tiered_fallback

pytestmark = pytest.mark.security


class TestMalformedInputs:
    """Tests for handling malformed query inputs."""

    @pytest.fixture(autouse=True)
    def setup_pipeline(self, e2e_runner):
        self.runner = e2e_runner

    @with_tiered_fallback
    def test_empty_query(self):
        """Empty query should be handled gracefully."""
        result = self.runner.pipeline.run("")
        # Should not crash, should return something sensible
        assert result is not None

    @with_tiered_fallback
    def test_whitespace_only_query(self):
        """Whitespace-only query should be handled."""
        queries = ["   ", "\t\n", "  \n  \t  "]

        for query in queries:
            result = self.runner.pipeline.run(query)
            assert result is not None

    @with_tiered_fallback
    def test_unicode_queries(self):
        """Unicode characters should be handled properly."""
        queries = [
            "What is TechCorp? 你好",  # Chinese
            "Tell me about モデル X100",  # Japanese
            "Wie funktioniert TechCorp? 🚗",  # German + emoji
            "TechCorp → GreenDrive → ?",  # Arrows
            "Пользователи TechCorp",  # Russian
        ]

        for query in queries:
            result = self.runner.pipeline.run(query)
            # Should not crash
            assert result is not None

    @with_tiered_fallback
    def test_special_characters(self):
        """Special characters in queries should be handled."""
        queries = [
            "What is TechCorp's revenue?",  # Apostrophe
            'Model "Y200" price?',  # Quotes
            "X100 & Y200 comparison",  # Ampersand
            "Price < $50,000?",  # Less than
            "Revenue > $1B?",  # Greater than
            "TechCorp; DROP TABLE users;--",  # SQL injection attempt
            "<script>alert('xss')</script> TechCorp?",  # XSS attempt
        ]

        for query in queries:
            result = self.runner.pipeline.run(query)
            # Should not crash and should not execute injection
            assert result is not None
            answer = result.answer.lower()
            # XSS: script tags should not appear (even quoted is suspicious)
            assert "<script>" not in result.answer
            # SQL injection: if DROP TABLE appears, it should be in context of
            # explaining what wasn't found, not as executed output
            if "drop table" in answer:
                assert any(
                    phrase in answer
                    for phrase in [
                        "do not contain",
                        "does not contain",
                        "no information",
                        "not found",
                        "cannot find",
                        "injection",  # explaining the attack
                    ]
                ), f"DROP TABLE appears without refusal context: {result.answer[:200]}"


class TestInputLengthLimits:
    """Tests for handling various input lengths."""

    @pytest.fixture(autouse=True)
    def setup_pipeline(self, e2e_runner):
        self.runner = e2e_runner

    def test_very_long_query(self):
        """Very long queries should be handled (truncated or rejected gracefully)."""
        # ~1.8KB query - still "very long" but faster to process
        long_query = "What is TechCorp? " * 100

        result = self.runner.pipeline.run(long_query)

        # Should not crash
        assert result is not None

        # Response should be reasonable (not echo the long input)
        assert len(result.answer) < len(long_query)

    def test_query_with_long_word(self):
        """Query with extremely long 'word' should be handled."""
        # Tests robustness, not answer quality (no tiered fallback needed)
        long_word = "a" * 2000
        query = f"What is {long_word}?"

        result = self.runner.pipeline.run(query)
        assert result is not None

    @pytest.mark.slow
    def test_many_short_queries_dont_leak_memory(self):
        """Rapid short queries shouldn't cause memory issues."""
        import gc

        import psutil

        process = psutil.Process()
        gc.collect()
        mem_before = process.memory_info().rss / 1024 / 1024

        # Run 15 queries - enough to detect memory leaks while keeping test fast
        successful = 0
        for i in range(15):
            try:
                self.runner.pipeline.run(f"Query {i}: What is TechCorp?")
                successful += 1
            except Exception:
                pass  # Transient failures are OK for memory testing

        gc.collect()
        mem_after = process.memory_info().rss / 1024 / 1024

        # Need at least 10 successful queries to meaningfully test memory
        assert successful >= 10, f"Only {successful}/15 queries succeeded - too few to test memory"

        mem_growth = mem_after - mem_before
        print(f"\nMemory growth over {successful} queries: {mem_growth:.1f}MB")

        # Should not grow more than 100MB over 15 queries
        assert mem_growth < 100, f"Memory grew {mem_growth:.1f}MB - possible leak"


class TestRepeatedPatterns:
    """Tests for handling repeated/adversarial patterns."""

    @pytest.fixture(autouse=True)
    def setup_pipeline(self, e2e_runner):
        self.runner = e2e_runner

    @with_tiered_fallback
    def test_repeated_question_marks(self):
        """Repeated question marks should be handled."""
        result = self.runner.pipeline.run("What is TechCorp????????")
        assert result is not None

    @with_tiered_fallback
    def test_repeated_spaces(self):
        """Excessive spaces should be normalized."""
        result = self.runner.pipeline.run("What    is      TechCorp      Industries?")
        assert result is not None

    @with_tiered_fallback
    def test_newlines_in_query(self):
        """Newlines in query should be handled."""
        result = self.runner.pipeline.run(
            "What is TechCorp?\nWhere is it located?\nWho is the CEO?"
        )
        assert result is not None

    @with_tiered_fallback
    def test_mixed_case_extremes(self):
        """Alternating case should be handled."""
        result = self.runner.pipeline.run("WhAt Is TeCHcOrP?")
        assert result is not None
        # Should still find relevant info
        assert "techcorp" in result.answer.lower() or "austin" in result.answer.lower()
