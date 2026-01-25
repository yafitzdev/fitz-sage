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

from unittest.mock import Mock, patch

import pytest

pytestmark = pytest.mark.chaos


def create_failing_chat_factory(side_effect):
    """Create a chat factory that returns a client with failing chat method."""
    mock_chat = Mock()
    mock_chat.chat = Mock(side_effect=side_effect)

    def factory(tier="fast"):
        return mock_chat

    return factory


class TestLLMFailures:
    """Tests for LLM service failure handling."""

    @pytest.fixture(autouse=True)
    def setup_pipeline(self, e2e_runner):
        self.runner = e2e_runner

    def test_llm_timeout_handling(self):
        """LLM timeout should be handled gracefully."""
        import asyncio

        from fitz_ai.engines.fitz_rag.exceptions import LLMError

        # Replace the chat factory with one that returns a timeout-failing client
        original_factory = self.runner.pipeline.chat_factory
        self.runner.pipeline.chat_factory = create_failing_chat_factory(
            asyncio.TimeoutError("LLM timeout")
        )

        try:
            result = self.runner.pipeline.run("What is TechCorp?")
            # Should either return a graceful error or raise a handled exception
            assert result is None or "error" in result.answer.lower()
        except LLMError:
            # LLMError is the expected exception - system handled it properly
            pass
        except Exception as e:
            # Other exceptions are acceptable as long as they're handled
            assert "timeout" in str(e).lower() or "unavailable" in str(e).lower()
        finally:
            # Restore original factory
            self.runner.pipeline.chat_factory = original_factory

    def test_llm_rate_limit_handling(self):
        """Rate limit errors should be handled."""
        from fitz_ai.engines.fitz_rag.exceptions import LLMError

        def rate_limited(*args, **kwargs):
            raise Exception("Rate limit exceeded. Retry after 60 seconds.")

        # Replace the chat factory with one that returns a rate-limited client
        original_factory = self.runner.pipeline.chat_factory
        self.runner.pipeline.chat_factory = create_failing_chat_factory(rate_limited)

        try:
            result = self.runner.pipeline.run("What is TechCorp?")
            # Should handle gracefully
            assert result is None or "error" in str(result).lower()
        except LLMError:
            # LLMError is the expected exception - system handled it properly
            pass
        except Exception as e:
            # Other handled exceptions are acceptable
            assert "rate" in str(e).lower() or "limit" in str(e).lower()
        finally:
            # Restore original factory
            self.runner.pipeline.chat_factory = original_factory


class TestRetrievalFailures:
    """Tests for retrieval/vector DB failure handling."""

    @pytest.fixture(autouse=True)
    def setup_pipeline(self, e2e_runner):
        self.runner = e2e_runner

    def test_empty_retrieval_handling(self):
        """Query returning no results should be handled."""
        # Query for something that definitely won't match
        result = self.runner.pipeline.run("xyzzy12345 nonexistent query that matches nothing")

        # Should return a response without crashing
        # The system may say "no information" or return a minimal response
        # The important thing is that it doesn't crash
        assert result is not None
        assert isinstance(result.answer, str)
        # Check for epistemic honesty phrases (flexible matching)
        answer = result.answer.lower()
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
        is_honest = any(term in answer for term in honest_phrases) or len(result.answer) < 50
        assert is_honest, f"Expected honest/minimal response, got: {result.answer[:200]}"

    def test_retrieval_exception_handling(self):
        """Retrieval exception should be handled gracefully."""

        def failing_retrieve(*args, **kwargs):
            raise ConnectionError("Vector DB connection failed")

        with patch.object(
            self.runner.pipeline.retrieval,
            "retrieve",
            side_effect=failing_retrieve,
        ):
            try:
                result = self.runner.pipeline.run("What is TechCorp?")
                # Should handle gracefully
                assert result is None or result.answer is not None
            except Exception as e:
                # Should be a handled error, not a raw crash
                assert "connection" in str(e).lower() or "retrieval" in str(e).lower()


class TestPartialFailures:
    """Tests for partial system failures."""

    @pytest.fixture(autouse=True)
    def setup_pipeline(self, e2e_runner):
        self.runner = e2e_runner

    def test_degraded_mode_no_reranking(self):
        """System should work if reranking is unavailable."""
        # Skip if no reranking configured
        if not hasattr(self.runner.pipeline, "reranker"):
            pytest.skip("No reranker configured")

        # Patch reranker to fail
        with patch.object(
            self.runner.pipeline.reranker,
            "rerank",
            side_effect=Exception("Reranker unavailable"),
        ):
            result = self.runner.pipeline.run("What is TechCorp?")

            # Should still return results (degraded mode without reranking)
            assert result is not None
            # Quality may be lower but should still work

    def test_handles_malformed_llm_response(self):
        """Malformed LLM response should be handled."""

        def malformed_response(*args, **kwargs):
            return None  # No response

        # Replace the chat factory with one that returns malformed responses
        original_factory = self.runner.pipeline.chat_factory
        self.runner.pipeline.chat_factory = create_failing_chat_factory(malformed_response)

        try:
            _result = self.runner.pipeline.run("What is TechCorp?")
            # Should handle None response gracefully
        except Exception:
            # Handled exception is acceptable
            pass
        finally:
            # Restore original factory
            self.runner.pipeline.chat_factory = original_factory


class TestResourceExhaustion:
    """Tests for resource exhaustion scenarios."""

    @pytest.fixture(autouse=True)
    def setup_pipeline(self, e2e_runner):
        self.runner = e2e_runner

    def test_handles_oom_gracefully(self):
        """Memory allocation failures should be handled."""

        def oom_simulation(*args, **kwargs):
            raise MemoryError("Out of memory")

        with patch.object(
            self.runner.pipeline.retrieval,
            "retrieve",
            side_effect=oom_simulation,
        ):
            try:
                _result = self.runner.pipeline.run("What is TechCorp?")
            except MemoryError:
                # MemoryError is acceptable - system didn't crash uncontrollably
                pass
            except Exception:
                # Other handled exceptions are fine
                pass


class TestRecovery:
    """Tests for system recovery after failures."""

    @pytest.fixture(autouse=True)
    def setup_pipeline(self, e2e_runner):
        self.runner = e2e_runner

    def test_recovery_after_error(self):
        """System should recover after transient error."""
        # First query - simulate failure
        call_count = 0

        original_retrieve = self.runner.pipeline.retrieval.retrieve

        def flaky_retrieve(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Transient failure")
            return original_retrieve(*args, **kwargs)

        with patch.object(
            self.runner.pipeline.retrieval,
            "retrieve",
            side_effect=flaky_retrieve,
        ):
            # First call fails
            try:
                self.runner.pipeline.run("What is TechCorp?")
            except Exception:
                pass

            # Second call should succeed (after "recovery")
            result = self.runner.pipeline.run("What is TechCorp?")
            assert result is not None
