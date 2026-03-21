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
        """SQL injection, XSS, excessive punctuation, whitespace should not crash."""
        answer = self.runner.engine.answer(
            Query(
                text="TechCorp; DROP TABLE users;-- <script>alert('xss')</script>"
                "???????? What\n\n  is    TechCorp?"
            )
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
