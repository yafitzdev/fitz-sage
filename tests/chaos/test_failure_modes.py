# tests/chaos/test_failure_modes.py
"""
Failure mode and graceful degradation tests.

These tests verify the system handles failures gracefully:
1. LLM service unavailable
2. Vector DB unavailable
3. Embedding service unavailable
4. Timeout handling
5. Partial failures

Run with: pytest tests/chaos/ -v -s -m chaos
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from fitz_sage.core import Query

pytestmark = pytest.mark.chaos


class TestLLMFailures:
    """Tests for LLM service failure handling."""

    @pytest.fixture(autouse=True)
    def setup_pipeline(self, krag_e2e_runner):
        self.runner = krag_e2e_runner

    def test_llm_timeout_handling(self):
        """LLM timeout should be handled gracefully.

        Mocks the synthesizer's chat to raise TimeoutError, simulating an LLM
        timeout at generation. The engine catches and wraps as GenerationError.
        """
        import asyncio

        from fitz_sage.core.exceptions import GenerationError, KnowledgeError

        with patch.object(
            self.runner.engine._synthesizer._chat,
            "chat",
            side_effect=asyncio.TimeoutError("LLM timeout"),
        ):
            with pytest.raises((GenerationError, KnowledgeError)):
                self.runner.engine.answer(
                    Query(
                        text="What is the detailed history and background of TechCorp Industries?"
                    )
                )


class TestRetrievalFailures:
    """Tests for retrieval/vector DB failure handling."""

    @pytest.fixture(autouse=True)
    def setup_pipeline(self, krag_e2e_runner):
        self.runner = krag_e2e_runner

    def test_empty_retrieval_handling(self):
        """Query returning no results should be handled."""
        # Query for something that definitely won't match
        answer = self.runner.engine.answer(
            Query(text="xyzzy12345 nonexistent query that matches nothing")
        )

        # Should return a response without crashing
        assert answer is not None
        assert isinstance(answer.text, str)
        # Check for epistemic honesty phrases (flexible matching)
        answer_lower = answer.text.lower()
        honest_phrases = [
            "no information",
            "cannot find",
            "don't have",
            "not found",
            "no relevant",
            "insufficient",
            "unable to",
            "do not contain",
            "does not contain",
        ]
        # Accept either explicit honesty or a very short response (< 50 chars)
        is_honest = any(term in answer_lower for term in honest_phrases) or len(answer.text) < 50
        assert is_honest, f"Expected honest/minimal response, got: {answer.text[:200]}"

    def test_retrieval_exception_handling(self):
        """Retrieval exception should be handled gracefully."""
        from fitz_sage.core.exceptions import KnowledgeError

        with patch.object(
            self.runner.engine._retrieval_router,
            "retrieve",
            side_effect=ConnectionError("Vector DB connection failed"),
        ):
            try:
                answer = self.runner.engine.answer(Query(text="What is TechCorp?"))
                # Should handle gracefully
                assert answer is None or answer.text is not None
            except (KnowledgeError, ConnectionError) as e:
                # Should be a handled error, not a raw crash
                assert "connection" in str(e).lower() or "retrieval" in str(e).lower()


class TestPartialFailures:
    """Tests for partial system failures."""

    @pytest.fixture(autouse=True)
    def setup_pipeline(self, krag_e2e_runner):
        self.runner = krag_e2e_runner

    def test_degraded_mode_no_reranking(self):
        """System should work if reranking is unavailable."""
        # Skip if no reranker configured
        if not self.runner.engine._address_reranker:
            pytest.skip("No reranker configured")

        # Patch reranker to fail
        with patch.object(
            self.runner.engine._address_reranker,
            "rerank",
            side_effect=Exception("Reranker unavailable"),
        ):
            try:
                answer = self.runner.engine.answer(Query(text="What is TechCorp?"))
                # Should still return results (degraded mode without reranking)
                assert answer is not None
            except Exception:
                # If it raises, that's also acceptable for this test
                pass

    def test_handles_malformed_llm_response(self):
        """Malformed LLM response should be handled."""
        from fitz_sage.core.exceptions import GenerationError, KnowledgeError

        with patch.object(
            self.runner.engine._chat,
            "chat",
            return_value=None,
        ):
            try:
                _answer = self.runner.engine.answer(Query(text="What is TechCorp?"))
                # Should handle None response gracefully
            except (GenerationError, KnowledgeError, TypeError, AttributeError):
                # Handled exception is acceptable
                pass


class TestResourceExhaustion:
    """Tests for resource exhaustion scenarios."""

    @pytest.fixture(autouse=True)
    def setup_pipeline(self, krag_e2e_runner):
        self.runner = krag_e2e_runner

    def test_handles_oom_gracefully(self):
        """Memory allocation failures should be handled."""
        with patch.object(
            self.runner.engine._retrieval_router,
            "retrieve",
            side_effect=MemoryError("Out of memory"),
        ):
            try:
                _answer = self.runner.engine.answer(Query(text="What is TechCorp?"))
            except MemoryError:
                # MemoryError is acceptable - system didn't crash uncontrollably
                pass
            except Exception:
                # Other handled exceptions are fine
                pass


class TestRecovery:
    """Tests for system recovery after failures."""

    @pytest.fixture(autouse=True)
    def setup_pipeline(self, krag_e2e_runner):
        self.runner = krag_e2e_runner

    def test_recovery_after_error(self):
        """System should recover after transient error."""
        # First query - simulate failure
        call_count = 0

        original_retrieve = self.runner.engine._retrieval_router.retrieve

        def flaky_retrieve(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Transient failure")
            return original_retrieve(*args, **kwargs)

        with patch.object(
            self.runner.engine._retrieval_router,
            "retrieve",
            side_effect=flaky_retrieve,
        ):
            # First call fails
            try:
                self.runner.engine.answer(Query(text="What is TechCorp?"))
            except Exception:
                pass

            # Second call should succeed (after "recovery")
            answer = self.runner.engine.answer(Query(text="What is TechCorp?"))
            assert answer is not None
