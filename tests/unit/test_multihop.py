# tests/test_multihop.py
"""Tests for multi-hop retrieval."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from fitz_ai.core.chunk import Chunk
from fitz_ai.engines.fitz_rag.retrieval.multihop import (
    BridgeExtractor,
    EvidenceEvaluator,
    HopController,
)


def create_mock_chat_factory(mock_chat):
    """Create a mock chat factory that returns the mock chat client."""

    def factory(tier: str = "fast"):
        return mock_chat

    return factory


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_chat():
    """Create a mock chat client."""
    return MagicMock()


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    return [
        Chunk(
            id="c1",
            doc_id="d1",
            chunk_index=0,
            content="Sarah Chen is the CEO of TechCorp.",
            metadata={"source_file": "people.md"},
        ),
        Chunk(
            id="c2",
            doc_id="d2",
            chunk_index=0,
            content="TechCorp manufactures electric vehicles.",
            metadata={"source_file": "companies.md"},
        ),
        Chunk(
            id="c3",
            doc_id="d3",
            chunk_index=0,
            content="AutoMotors is TechCorp's main competitor.",
            metadata={"source_file": "companies.md"},
        ),
    ]


@pytest.fixture
def mock_retrieval(sample_chunks):
    """Create a mock retrieval pipeline."""
    retrieval = MagicMock()
    # First call returns first 2 chunks, second call returns the third
    retrieval.retrieve.side_effect = [
        sample_chunks[:2],  # First hop
        sample_chunks[2:],  # Second hop
    ]
    return retrieval


# =============================================================================
# EvidenceEvaluator Tests
# =============================================================================


class TestEvidenceEvaluator:
    """Tests for EvidenceEvaluator."""

    def test_sufficient_evidence(self, mock_chat, sample_chunks):
        """Test detection of sufficient evidence."""
        mock_chat.chat.return_value = "SUFFICIENT"
        evaluator = EvidenceEvaluator(chat_factory=create_mock_chat_factory(mock_chat))

        result = evaluator.evaluate("What company does Sarah Chen lead?", sample_chunks)

        assert result is True
        mock_chat.chat.assert_called_once()

    def test_insufficient_evidence(self, mock_chat, sample_chunks):
        """Test detection of insufficient evidence."""
        mock_chat.chat.return_value = "INSUFFICIENT"
        evaluator = EvidenceEvaluator(chat_factory=create_mock_chat_factory(mock_chat))

        result = evaluator.evaluate("What is Sarah's competitor's revenue?", sample_chunks)

        assert result is False
        mock_chat.chat.assert_called_once()

    def test_empty_chunks_returns_false(self, mock_chat):
        """Test that empty chunks returns insufficient."""
        evaluator = EvidenceEvaluator(chat_factory=create_mock_chat_factory(mock_chat))

        result = evaluator.evaluate("Any query", [])

        assert result is False
        mock_chat.chat.assert_not_called()

    def test_parses_sufficient_case_insensitive(self, mock_chat, sample_chunks):
        """Test case-insensitive parsing of response."""
        mock_chat.chat.return_value = "sufficient - the evidence is complete"
        evaluator = EvidenceEvaluator(chat_factory=create_mock_chat_factory(mock_chat))

        result = evaluator.evaluate("Query", sample_chunks)

        assert result is True


# =============================================================================
# BridgeExtractor Tests
# =============================================================================


class TestBridgeExtractor:
    """Tests for BridgeExtractor."""

    def test_extracts_bridge_questions(self, mock_chat, sample_chunks):
        """Test extraction of bridge questions."""
        mock_chat.chat.return_value = '["What products does AutoMotors manufacture?"]'
        extractor = BridgeExtractor(chat_factory=create_mock_chat_factory(mock_chat))

        result = extractor.extract("What does Sarah's competitor make?", sample_chunks)

        assert len(result) == 1
        assert "AutoMotors" in result[0]
        mock_chat.chat.assert_called_once()

    def test_handles_markdown_response(self, mock_chat, sample_chunks):
        """Test parsing of markdown-wrapped JSON response."""
        mock_chat.chat.return_value = """```json
["Question about competitor", "Question about products"]
```"""
        extractor = BridgeExtractor(
            chat_factory=create_mock_chat_factory(mock_chat), max_questions=2
        )

        result = extractor.extract("Query", sample_chunks)

        assert len(result) == 2

    def test_returns_empty_list_on_parse_error(self, mock_chat, sample_chunks):
        """Test graceful handling of parse errors."""
        mock_chat.chat.return_value = "This is not valid JSON"
        extractor = BridgeExtractor(chat_factory=create_mock_chat_factory(mock_chat))

        result = extractor.extract("Query", sample_chunks)

        assert result == []

    def test_respects_max_questions(self, mock_chat, sample_chunks):
        """Test that max_questions limit is enforced."""
        mock_chat.chat.return_value = '["q1", "q2", "q3", "q4", "q5"]'
        extractor = BridgeExtractor(
            chat_factory=create_mock_chat_factory(mock_chat), max_questions=2
        )

        result = extractor.extract("Query", sample_chunks)

        assert len(result) == 2

    def test_empty_array_response(self, mock_chat, sample_chunks):
        """Test handling of empty array (no gaps found)."""
        mock_chat.chat.return_value = "[]"
        extractor = BridgeExtractor(chat_factory=create_mock_chat_factory(mock_chat))

        result = extractor.extract("Query", sample_chunks)

        assert result == []


# =============================================================================
# HopController Tests
# =============================================================================


class TestHopController:
    """Tests for HopController."""

    def test_single_hop_sufficient(self, mock_retrieval, sample_chunks):
        """Test that sufficient evidence stops after 1 hop."""
        # Evaluator says sufficient on first hop
        evaluator = MagicMock()
        evaluator.evaluate.return_value = True

        extractor = MagicMock()

        controller = HopController(
            retrieval_pipeline=mock_retrieval,
            evaluator=evaluator,
            extractor=extractor,
            max_hops=3,
        )

        chunks, metadata = controller.retrieve("What company does Sarah lead?")

        assert len(chunks) == 2  # Only first retrieval
        assert metadata.total_hops == 1
        assert not metadata.used_multi_hop
        mock_retrieval.retrieve.assert_called_once()
        extractor.extract.assert_not_called()  # Should not extract if sufficient

    def test_two_hop_with_bridge(self, mock_retrieval, sample_chunks):
        """Test two-hop retrieval with bridge question."""
        # First hop insufficient, second hop sufficient
        evaluator = MagicMock()
        evaluator.evaluate.side_effect = [False, True]

        # Bridge question after first hop
        extractor = MagicMock()
        extractor.extract.return_value = ["What is AutoMotors?"]

        controller = HopController(
            retrieval_pipeline=mock_retrieval,
            evaluator=evaluator,
            extractor=extractor,
            max_hops=3,
        )

        chunks, metadata = controller.retrieve("What does Sarah's competitor make?")

        assert len(chunks) == 3  # All chunks accumulated
        assert metadata.total_hops == 2
        assert metadata.used_multi_hop
        assert mock_retrieval.retrieve.call_count == 2

    def test_max_hops_reached(self, sample_chunks):
        """Test that max_hops limit is enforced."""
        # Always returns same chunks
        mock_retrieval = MagicMock()
        mock_retrieval.retrieve.return_value = sample_chunks[:1]

        # Never sufficient
        evaluator = MagicMock()
        evaluator.evaluate.return_value = False

        # Always has a bridge question
        extractor = MagicMock()
        extractor.extract.return_value = ["Next question"]

        controller = HopController(
            retrieval_pipeline=mock_retrieval,
            evaluator=evaluator,
            extractor=extractor,
            max_hops=3,
        )

        chunks, metadata = controller.retrieve("Complex query")

        assert metadata.total_hops == 3  # Stopped at max
        assert mock_retrieval.retrieve.call_count == 3

    def test_no_bridge_questions_stops_early(self, mock_retrieval, sample_chunks):
        """Test that missing bridge questions stops iteration."""
        # Never sufficient
        evaluator = MagicMock()
        evaluator.evaluate.return_value = False

        # No bridge questions found
        extractor = MagicMock()
        extractor.extract.return_value = []

        controller = HopController(
            retrieval_pipeline=mock_retrieval,
            evaluator=evaluator,
            extractor=extractor,
            max_hops=3,
        )

        chunks, metadata = controller.retrieve("Query")

        assert metadata.total_hops == 1  # Stopped after first hop
        mock_retrieval.retrieve.assert_called_once()

    def test_chunk_deduplication(self, sample_chunks):
        """Test that duplicate chunks are not added."""
        # Returns same chunk on both hops
        mock_retrieval = MagicMock()
        mock_retrieval.retrieve.side_effect = [
            sample_chunks[:2],  # First hop: c1, c2
            [sample_chunks[0], sample_chunks[2]],  # Second hop: c1 (dup), c3
        ]

        # First insufficient, second sufficient
        evaluator = MagicMock()
        evaluator.evaluate.side_effect = [False, True]

        extractor = MagicMock()
        extractor.extract.return_value = ["Follow-up"]

        controller = HopController(
            retrieval_pipeline=mock_retrieval,
            evaluator=evaluator,
            extractor=extractor,
            max_hops=3,
        )

        chunks, metadata = controller.retrieve("Query")

        # Should have 3 unique chunks, not 4
        chunk_ids = [c.id for c in chunks]
        assert len(chunk_ids) == 3
        assert len(set(chunk_ids)) == 3  # All unique

    def test_filter_override_passed_to_retrieval(self, mock_retrieval, sample_chunks):
        """Test that filter_override is passed through to retrieval."""
        evaluator = MagicMock()
        evaluator.evaluate.return_value = True

        extractor = MagicMock()

        controller = HopController(
            retrieval_pipeline=mock_retrieval,
            evaluator=evaluator,
            extractor=extractor,
        )

        filter_override = {"hierarchy_level": 2}
        controller.retrieve("Query", filter_override=filter_override)

        mock_retrieval.retrieve.assert_called_once_with("Query", filter_override=filter_override)

    def test_hop_metadata_trace(self, mock_retrieval, sample_chunks):
        """Test that hop trace captures all iterations."""
        evaluator = MagicMock()
        evaluator.evaluate.side_effect = [False, True]

        extractor = MagicMock()
        extractor.extract.return_value = ["Bridge question"]

        controller = HopController(
            retrieval_pipeline=mock_retrieval,
            evaluator=evaluator,
            extractor=extractor,
        )

        chunks, metadata = controller.retrieve("Query")

        assert len(metadata.trace) == 2

        # First hop
        assert metadata.trace[0].hop_number == 0
        assert metadata.trace[0].is_sufficient is False
        assert metadata.trace[0].bridge_questions == ["Bridge question"]

        # Second hop
        assert metadata.trace[1].hop_number == 1
        assert metadata.trace[1].is_sufficient is True
        assert metadata.trace[1].bridge_questions == []


# =============================================================================
# Config Integration Tests
# =============================================================================
