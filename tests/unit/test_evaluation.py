# tests/unit/test_evaluation.py
"""
Unit tests for the governance observability evaluation module.

Tests:
- GovernanceLog: serialization, from_decision, hash_query, query_text
- ModeDistribution: rate calculations, edge cases
- GovernanceFlip: regression/improvement detection, version tracking
- ConstraintFrequency: trigger rate calculations
- AbstainTrend: serialization
- GovernanceLogger: batching behavior
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from fitz_ai.core.answer_mode import AnswerMode
from fitz_ai.engines.fitz_rag.governance import GovernanceDecision, GovernanceLog
from fitz_ai.evaluation.models import (
    AbstainTrend,
    ConstraintFrequency,
    GovernanceFlip,
    ModeDistribution,
)

# =============================================================================
# GovernanceLog Tests
# =============================================================================


class TestGovernanceLog:
    """Tests for GovernanceLog dataclass."""

    def test_hash_query_normalizes_case(self):
        """Hash should be case-insensitive."""
        hash1 = GovernanceLog.hash_query("What is the revenue?")
        hash2 = GovernanceLog.hash_query("WHAT IS THE REVENUE?")
        hash3 = GovernanceLog.hash_query("what is the revenue?")

        assert hash1 == hash2 == hash3

    def test_hash_query_strips_whitespace(self):
        """Hash should ignore leading/trailing whitespace."""
        hash1 = GovernanceLog.hash_query("query")
        hash2 = GovernanceLog.hash_query("  query  ")
        hash3 = GovernanceLog.hash_query("\n\tquery\n")

        assert hash1 == hash2 == hash3

    def test_hash_query_returns_64_char_hex(self):
        """Hash should be 64 character hex string (SHA256)."""
        hash_val = GovernanceLog.hash_query("test query")

        assert len(hash_val) == 64
        assert all(c in "0123456789abcdef" for c in hash_val)

    def test_hash_query_different_queries_different_hashes(self):
        """Different queries should produce different hashes."""
        hash1 = GovernanceLog.hash_query("query one")
        hash2 = GovernanceLog.hash_query("query two")

        assert hash1 != hash2

    def test_from_decision_creates_log_entry(self):
        """from_decision should create valid GovernanceLog."""
        decision = GovernanceDecision(
            mode=AnswerMode.ABSTAIN,
            triggered_constraints=("insufficient_evidence",),
            reasons=("No evidence found",),
            signals=frozenset({"abstain"}),
        )

        log = GovernanceLog.from_decision(
            decision,
            query_hash="abc123",
            query_text="What is X?",
            chunk_count=5,
            collection="test_collection",
            latency_ms=1.5,
            pipeline_version="0.7.1",
        )

        assert log.mode == "abstain"
        assert log.query_hash == "abc123"
        assert log.query_text == "What is X?"
        assert log.chunk_count == 5
        assert log.collection == "test_collection"
        assert log.latency_ms == 1.5
        assert log.pipeline_version == "0.7.1"
        assert log.triggered_constraints == ("insufficient_evidence",)
        assert log.reasons == ("No evidence found",)
        assert "abstain" in log.signals
        assert isinstance(log.timestamp, datetime)

    def test_from_decision_timestamp_is_utc(self):
        """Timestamp should be UTC."""
        decision = GovernanceDecision.trustworthy()

        log = GovernanceLog.from_decision(
            decision,
            query_hash="abc",
            query_text="test",
            chunk_count=0,
            collection="test",
        )

        assert log.timestamp.tzinfo == timezone.utc

    def test_to_dict_serialization(self):
        """to_dict should serialize all fields correctly."""
        now = datetime.now(timezone.utc)

        log = GovernanceLog(
            timestamp=now,
            query_hash="hash123",
            query_text="What is the answer?",
            mode="trustworthy",
            triggered_constraints=("c1", "c2"),
            signals=("sig1",),
            reasons=("reason1",),
            chunk_count=10,
            collection="my_collection",
            latency_ms=2.5,
            pipeline_version="0.7.1",
        )

        d = log.to_dict()

        assert d["timestamp"] == now.isoformat()
        assert d["query_hash"] == "hash123"
        assert d["query_text"] == "What is the answer?"
        assert d["mode"] == "trustworthy"
        assert d["triggered_constraints"] == ["c1", "c2"]
        assert d["signals"] == ["sig1"]
        assert d["reasons"] == ["reason1"]
        assert d["chunk_count"] == 10
        assert d["collection"] == "my_collection"
        assert d["latency_ms"] == 2.5
        assert d["pipeline_version"] == "0.7.1"

    def test_to_dict_handles_none_optionals(self):
        """to_dict should handle None latency_ms and pipeline_version."""
        log = GovernanceLog(
            timestamp=datetime.now(timezone.utc),
            query_hash="hash",
            query_text=None,
            mode="trustworthy",
            triggered_constraints=(),
            signals=(),
            reasons=(),
            chunk_count=0,
            collection="test",
            latency_ms=None,
            pipeline_version=None,
        )

        d = log.to_dict()

        assert d["query_text"] is None
        assert d["latency_ms"] is None
        assert d["pipeline_version"] is None

    def test_governance_log_is_frozen(self):
        """GovernanceLog should be immutable (frozen dataclass)."""
        log = GovernanceLog(
            timestamp=datetime.now(timezone.utc),
            query_hash="hash",
            query_text="test",
            mode="trustworthy",
            triggered_constraints=(),
            signals=(),
            reasons=(),
            chunk_count=0,
            collection="test",
        )

        with pytest.raises(AttributeError):
            log.mode = "abstain"


# =============================================================================
# ModeDistribution Tests
# =============================================================================


class TestModeDistribution:
    """Tests for ModeDistribution dataclass."""

    def test_rate_calculations_basic(self):
        """Rate calculations should be correct."""
        dist = ModeDistribution(
            period_start=datetime.now(timezone.utc),
            period_end=datetime.now(timezone.utc),
            total_queries=100,
            trustworthy_count=85,
            disputed_count=5,
            abstain_count=10,
        )

        assert dist.trustworthy_rate == 0.85
        assert dist.disputed_rate == 0.05
        assert dist.abstain_rate == 0.10

    def test_rate_calculations_zero_queries(self):
        """Rates should be 0.0 when total_queries is 0."""
        dist = ModeDistribution(
            period_start=datetime.now(timezone.utc),
            period_end=datetime.now(timezone.utc),
            total_queries=0,
            trustworthy_count=0,
            disputed_count=0,
            abstain_count=0,
        )

        assert dist.trustworthy_rate == 0.0
        assert dist.disputed_rate == 0.0
        assert dist.abstain_rate == 0.0

    def test_rate_calculations_single_mode(self):
        """100% in one mode should work correctly."""
        dist = ModeDistribution(
            period_start=datetime.now(timezone.utc),
            period_end=datetime.now(timezone.utc),
            total_queries=50,
            trustworthy_count=50,
            disputed_count=0,
            abstain_count=0,
        )

        assert dist.trustworthy_rate == 1.0
        assert dist.abstain_rate == 0.0

    def test_to_dict_serialization(self):
        """to_dict should serialize all fields including rates."""
        start = datetime.now(timezone.utc)
        end = start + timedelta(days=7)

        dist = ModeDistribution(
            period_start=start,
            period_end=end,
            total_queries=100,
            trustworthy_count=90,
            disputed_count=5,
            abstain_count=5,
        )

        d = dist.to_dict()

        assert d["period_start"] == start.isoformat()
        assert d["period_end"] == end.isoformat()
        assert d["total_queries"] == 100
        assert d["trustworthy_count"] == 90
        assert d["disputed_count"] == 5
        assert d["abstain_count"] == 5
        assert d["trustworthy_rate"] == 0.9
        assert d["disputed_rate"] == 0.05
        assert d["abstain_rate"] == 0.05


# =============================================================================
# GovernanceFlip Tests
# =============================================================================


class TestGovernanceFlip:
    """Tests for GovernanceFlip dataclass."""

    def test_is_regression_trustworthy_to_abstain(self):
        """TRUSTWORTHY -> ABSTAIN is a regression."""
        flip = GovernanceFlip(
            query_hash="abc",
            query_text="test",
            old_mode="trustworthy",
            new_mode="abstain",
            old_timestamp=datetime.now(timezone.utc),
            new_timestamp=datetime.now(timezone.utc),
        )

        assert flip.is_regression is True
        assert flip.is_improvement is False

    def test_is_regression_trustworthy_to_disputed(self):
        """TRUSTWORTHY -> DISPUTED is a regression."""
        flip = GovernanceFlip(
            query_hash="abc",
            query_text="test",
            old_mode="trustworthy",
            new_mode="disputed",
            old_timestamp=datetime.now(timezone.utc),
            new_timestamp=datetime.now(timezone.utc),
        )

        assert flip.is_regression is True
        assert flip.is_improvement is False

    def test_is_improvement_abstain_to_trustworthy(self):
        """ABSTAIN -> TRUSTWORTHY is an improvement."""
        flip = GovernanceFlip(
            query_hash="abc",
            query_text="test",
            old_mode="abstain",
            new_mode="trustworthy",
            old_timestamp=datetime.now(timezone.utc),
            new_timestamp=datetime.now(timezone.utc),
        )

        assert flip.is_improvement is True
        assert flip.is_regression is False

    def test_is_improvement_disputed_to_trustworthy(self):
        """DISPUTED -> TRUSTWORTHY is an improvement."""
        flip = GovernanceFlip(
            query_hash="abc",
            query_text="test",
            old_mode="disputed",
            new_mode="trustworthy",
            old_timestamp=datetime.now(timezone.utc),
            new_timestamp=datetime.now(timezone.utc),
        )

        assert flip.is_improvement is True

    def test_neither_regression_nor_improvement(self):
        """Some flips are neither regression nor improvement."""
        # ABSTAIN -> DISPUTED - unclear if better or worse
        flip = GovernanceFlip(
            query_hash="abc",
            query_text="test",
            old_mode="abstain",
            new_mode="disputed",
            old_timestamp=datetime.now(timezone.utc),
            new_timestamp=datetime.now(timezone.utc),
        )

        assert flip.is_regression is False
        assert flip.is_improvement is False

    def test_version_fields_default_to_none(self):
        """Version fields should default to None."""
        flip = GovernanceFlip(
            query_hash="abc",
            query_text="test",
            old_mode="trustworthy",
            new_mode="abstain",
            old_timestamp=datetime.now(timezone.utc),
            new_timestamp=datetime.now(timezone.utc),
        )

        assert flip.old_version is None
        assert flip.new_version is None

    def test_version_fields_can_be_set(self):
        """Version fields can be explicitly set."""
        flip = GovernanceFlip(
            query_hash="abc",
            query_text="test",
            old_mode="trustworthy",
            new_mode="abstain",
            old_timestamp=datetime.now(timezone.utc),
            new_timestamp=datetime.now(timezone.utc),
            old_version="0.7.0",
            new_version="0.7.1",
        )

        assert flip.old_version == "0.7.0"
        assert flip.new_version == "0.7.1"

    def test_to_dict_serialization(self):
        """to_dict should serialize all fields including versions."""
        old_ts = datetime.now(timezone.utc) - timedelta(days=1)
        new_ts = datetime.now(timezone.utc)

        flip = GovernanceFlip(
            query_hash="abc123",
            query_text="What is X?",
            old_mode="trustworthy",
            new_mode="abstain",
            old_timestamp=old_ts,
            new_timestamp=new_ts,
            old_version="0.7.0",
            new_version="0.7.1",
        )

        d = flip.to_dict()

        assert d["query_hash"] == "abc123"
        assert d["query_text"] == "What is X?"
        assert d["old_mode"] == "trustworthy"
        assert d["new_mode"] == "abstain"
        assert d["old_timestamp"] == old_ts.isoformat()
        assert d["new_timestamp"] == new_ts.isoformat()
        assert d["old_version"] == "0.7.0"
        assert d["new_version"] == "0.7.1"
        assert d["is_regression"] is True
        assert d["is_improvement"] is False

    def test_to_dict_with_none_versions(self):
        """to_dict should handle None versions."""
        flip = GovernanceFlip(
            query_hash="abc",
            query_text=None,
            old_mode="trustworthy",
            new_mode="abstain",
            old_timestamp=datetime.now(timezone.utc),
            new_timestamp=datetime.now(timezone.utc),
        )

        d = flip.to_dict()

        assert d["query_text"] is None
        assert d["old_version"] is None
        assert d["new_version"] is None


# =============================================================================
# ConstraintFrequency Tests
# =============================================================================


class TestConstraintFrequency:
    """Tests for ConstraintFrequency dataclass."""

    def test_trigger_rate_calculation(self):
        """Trigger rate should be trigger_count / total_queries."""
        cf = ConstraintFrequency(
            constraint_name="insufficient_evidence",
            trigger_count=25,
            total_queries=100,
        )

        assert cf.trigger_rate == 0.25

    def test_trigger_rate_zero_queries(self):
        """Trigger rate should be 0.0 when total_queries is 0."""
        cf = ConstraintFrequency(
            constraint_name="test",
            trigger_count=0,
            total_queries=0,
        )

        assert cf.trigger_rate == 0.0

    def test_trigger_rate_all_triggered(self):
        """100% trigger rate should work."""
        cf = ConstraintFrequency(
            constraint_name="test",
            trigger_count=50,
            total_queries=50,
        )

        assert cf.trigger_rate == 1.0

    def test_to_dict_serialization(self):
        """to_dict should serialize correctly."""
        cf = ConstraintFrequency(
            constraint_name="conflict_aware",
            trigger_count=15,
            total_queries=100,
        )

        d = cf.to_dict()

        assert d["constraint_name"] == "conflict_aware"
        assert d["trigger_count"] == 15
        assert d["trigger_rate"] == 0.15


# =============================================================================
# AbstainTrend Tests
# =============================================================================


class TestAbstainTrend:
    """Tests for AbstainTrend dataclass."""

    def test_basic_creation(self):
        """AbstainTrend should store all fields."""
        bucket_start = datetime.now(timezone.utc)

        trend = AbstainTrend(
            bucket_start=bucket_start,
            abstain_rate=0.15,
            total_queries=100,
        )

        assert trend.bucket_start == bucket_start
        assert trend.abstain_rate == 0.15
        assert trend.total_queries == 100

    def test_to_dict_serialization(self):
        """to_dict should serialize correctly."""
        bucket_start = datetime.now(timezone.utc)

        trend = AbstainTrend(
            bucket_start=bucket_start,
            abstain_rate=0.2,
            total_queries=50,
        )

        d = trend.to_dict()

        assert d["bucket_start"] == bucket_start.isoformat()
        assert d["abstain_rate"] == 0.2
        assert d["total_queries"] == 50


# =============================================================================
# GovernanceLogger Tests
# =============================================================================


class TestGovernanceLogger:
    """Tests for GovernanceLogger batching behavior."""

    def test_logger_buffers_entries(self):
        """Logger should buffer entries until batch size reached."""
        from fitz_ai.evaluation.logger import GovernanceLogger

        mock_pool = MagicMock()
        logger = GovernanceLogger(
            pool=mock_pool,
            collection="test",
            batch_size=10,
        )

        # Log fewer than batch_size entries
        decision = GovernanceDecision.trustworthy()
        for i in range(5):
            logger.log(decision, f"query {i}", [])

        # Should have 5 pending entries
        assert logger.pending_count() == 5

        # Pool should not have been called yet (no flush)
        mock_pool.connection.assert_not_called()

    def test_logger_flushes_at_batch_size(self):
        """Logger should auto-flush when batch size is reached."""
        from fitz_ai.evaluation.logger import GovernanceLogger

        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pool.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_pool.connection.return_value.__exit__ = MagicMock(return_value=False)

        logger = GovernanceLogger(
            pool=mock_pool,
            collection="test",
            batch_size=5,
        )
        logger._schema_ensured = True  # Skip schema check

        decision = GovernanceDecision.trustworthy()

        # Log exactly batch_size entries
        for i in range(5):
            logger.log(decision, f"query {i}", [])

        # Should have flushed, buffer should be empty
        assert logger.pending_count() == 0

        # executemany should have been called on cursor
        mock_cursor.executemany.assert_called_once()

    def test_logger_explicit_flush(self):
        """flush() should persist all buffered entries."""
        from fitz_ai.evaluation.logger import GovernanceLogger

        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_pool.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_pool.connection.return_value.__exit__ = MagicMock(return_value=False)

        logger = GovernanceLogger(
            pool=mock_pool,
            collection="test",
            batch_size=100,  # Large batch size
        )
        logger._schema_ensured = True

        decision = GovernanceDecision.trustworthy()

        # Log some entries
        for i in range(3):
            logger.log(decision, f"query {i}", [])

        assert logger.pending_count() == 3

        # Explicit flush
        count = logger.flush()

        assert count == 3
        assert logger.pending_count() == 0
        mock_cursor.executemany.assert_called_once()

    def test_logger_flush_empty_buffer(self):
        """flush() with empty buffer should return 0."""
        from fitz_ai.evaluation.logger import GovernanceLogger

        mock_pool = MagicMock()
        logger = GovernanceLogger(pool=mock_pool, collection="test")

        count = logger.flush()

        assert count == 0

    def test_logger_creates_correct_log_entry(self):
        """Logger should create GovernanceLog with correct fields."""
        from fitz_ai.evaluation.logger import GovernanceLogger

        mock_pool = MagicMock()
        logger = GovernanceLogger(
            pool=mock_pool,
            collection="my_collection",
            pipeline_version="1.0.0",
            batch_size=100,
        )

        decision = GovernanceDecision(
            mode=AnswerMode.ABSTAIN,
            triggered_constraints=("c1",),
            reasons=("reason1",),
            signals=frozenset({"abstain"}),
        )

        # Mock chunks
        mock_chunk = MagicMock()
        chunks = [mock_chunk, mock_chunk, mock_chunk]

        log_entry = logger.log(decision, "test query", chunks, latency_ms=2.5)

        assert log_entry.mode == "abstain"
        assert log_entry.query_text == "test query"
        assert log_entry.chunk_count == 3
        assert log_entry.collection == "my_collection"
        assert log_entry.pipeline_version == "1.0.0"
        assert log_entry.latency_ms == 2.5
        assert log_entry.triggered_constraints == ("c1",)

    def test_logger_uses_fitz_version_as_default(self):
        """Logger should use fitz_ai.__version__ as default pipeline_version."""
        import fitz_ai
        from fitz_ai.evaluation.logger import GovernanceLogger

        mock_pool = MagicMock()
        logger = GovernanceLogger(pool=mock_pool, collection="test")

        assert logger.pipeline_version == fitz_ai.__version__


# =============================================================================
# Integration-style tests (using mocks)
# =============================================================================


class TestGovernanceLogIntegration:
    """Integration-style tests for GovernanceLog with GovernanceDecision."""

    @pytest.mark.parametrize(
        "mode,expected_mode_str",
        [
            (AnswerMode.TRUSTWORTHY, "trustworthy"),
            (AnswerMode.DISPUTED, "disputed"),
            (AnswerMode.ABSTAIN, "abstain"),
        ],
    )
    def test_from_decision_maps_all_modes(self, mode, expected_mode_str):
        """from_decision should correctly map all AnswerMode values."""
        decision = GovernanceDecision(mode=mode)

        log = GovernanceLog.from_decision(
            decision,
            query_hash="abc",
            query_text="test",
            chunk_count=0,
            collection="test",
        )

        assert log.mode == expected_mode_str

    def test_full_round_trip(self):
        """Test creating decision -> log -> dict -> verify."""
        from fitz_ai.engines.fitz_rag.governance import AnswerGovernor
        from fitz_ai.engines.fitz_rag.guardrails.base import ConstraintResult

        # Create constraint results
        results = [
            ConstraintResult(
                allow_decisive_answer=False,
                reason="Sources contradict",
                signal="disputed",
                metadata={"constraint_name": "conflict_aware"},
            ),
        ]

        # Governor decides
        governor = AnswerGovernor()
        decision = governor.decide(results)

        # Create log
        query = "What happened in 2024?"
        log = GovernanceLog.from_decision(
            decision,
            query_hash=GovernanceLog.hash_query(query),
            query_text=query,
            chunk_count=5,
            collection="test",
            latency_ms=1.0,
            pipeline_version="0.7.1",
        )

        # Serialize
        d = log.to_dict()

        # Verify round trip
        assert d["mode"] == "disputed"
        assert d["query_text"] == query
        assert d["triggered_constraints"] == ["conflict_aware"]
        assert d["reasons"] == ["Sources contradict"]
        assert d["chunk_count"] == 5
        assert d["pipeline_version"] == "0.7.1"
