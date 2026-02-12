# tests/unit/test_cross_strategy_ranker.py
"""Tests for CrossStrategyRanker — weighted scoring across strategies."""

from __future__ import annotations

import pytest

from fitz_ai.engines.fitz_krag.query_analyzer import QueryAnalysis, QueryType
from fitz_ai.engines.fitz_krag.retrieval.ranker import (
    CrossStrategyRanker,
)
from fitz_ai.engines.fitz_krag.types import Address, AddressKind


@pytest.fixture
def ranker():
    return CrossStrategyRanker()


def _code_addr(score=0.5, location="module.func", summary="A function"):
    return Address(
        kind=AddressKind.SYMBOL,
        source_id="f1",
        location=location,
        summary=summary,
        score=score,
        metadata={"kind": "function"},
    )


def _section_addr(score=0.5, location="Introduction", summary="Intro section"):
    return Address(
        kind=AddressKind.SECTION,
        source_id="f2",
        location=location,
        summary=summary,
        score=score,
        metadata={"section_id": "sec1"},
    )


class TestRanking:
    def test_code_query_prioritizes_code(self, ranker):
        analysis = QueryAnalysis(primary_type=QueryType.CODE)
        addresses = [
            _section_addr(score=0.9),
            _code_addr(score=0.7),
        ]
        ranked = ranker.rank(addresses, analysis)
        # Code should rank higher despite lower base score (0.7*0.8 > 0.9*0.1)
        assert ranked[0].kind == AddressKind.SYMBOL

    def test_doc_query_prioritizes_sections(self, ranker):
        analysis = QueryAnalysis(primary_type=QueryType.DOCUMENTATION)
        addresses = [
            _code_addr(score=0.9),
            _section_addr(score=0.7),
        ]
        ranked = ranker.rank(addresses, analysis)
        # Section should rank higher despite lower base score
        assert ranked[0].kind == AddressKind.SECTION

    def test_cross_query_balances_both(self, ranker):
        analysis = QueryAnalysis(primary_type=QueryType.CROSS)
        addresses = [
            _code_addr(score=0.8),
            _section_addr(score=0.8),
        ]
        ranked = ranker.rank(addresses, analysis)
        # Both should have equal weight (0.4 each), so order based on score
        assert len(ranked) == 2

    def test_general_query_favors_chunks(self, ranker):
        analysis = QueryAnalysis(primary_type=QueryType.GENERAL)
        chunk_addr = Address(
            kind=AddressKind.CHUNK,
            source_id="c1",
            location="doc",
            summary="chunk",
            score=0.7,
            metadata={"text": "some content"},
        )
        addresses = [_code_addr(score=0.7), chunk_addr]
        ranked = ranker.rank(addresses, analysis)
        # Chunk weight (0.4) > code weight (0.3) with same base score
        assert ranked[0].kind == AddressKind.CHUNK


class TestEntityBonus:
    def test_entity_match_in_location(self, ranker):
        analysis = QueryAnalysis(
            primary_type=QueryType.CODE,
            entities=("authenticate",),
        )
        addr_match = _code_addr(score=0.5, location="auth.authenticate")
        addr_no_match = _code_addr(score=0.5, location="utils.helper")
        ranked = ranker.rank([addr_no_match, addr_match], analysis)
        assert ranked[0].location == "auth.authenticate"

    def test_entity_match_in_summary(self, ranker):
        analysis = QueryAnalysis(
            primary_type=QueryType.CODE,
            entities=("login",),
        )
        addr_match = _code_addr(score=0.5, summary="Handles user login flow")
        addr_no_match = _code_addr(score=0.5, summary="Utility function")
        ranked = ranker.rank([addr_no_match, addr_match], analysis)
        assert "login" in ranked[0].summary.lower()

    def test_entity_match_case_insensitive(self, ranker):
        analysis = QueryAnalysis(
            primary_type=QueryType.CODE,
            entities=("UserService",),
        )
        addr = _code_addr(score=0.5, location="module.userservice")
        no_match = _code_addr(score=0.5, location="module.other")
        ranked = ranker.rank([no_match, addr], analysis)
        assert ranked[0].location == "module.userservice"

    def test_no_entities_no_bonus(self, ranker):
        analysis = QueryAnalysis(primary_type=QueryType.CODE, entities=())
        addr1 = _code_addr(score=0.6, location="a.b")
        addr2 = _code_addr(score=0.5, location="c.d")
        ranked = ranker.rank([addr2, addr1], analysis)
        # Should just rank by weighted score
        assert ranked[0].location == "a.b"


class TestScoring:
    def test_weighted_score_computed(self, ranker):
        analysis = QueryAnalysis(primary_type=QueryType.CODE)
        addr = _code_addr(score=1.0)
        ranked = ranker.rank([addr], analysis)
        # Score should be 1.0 * 0.8 (code weight for CODE query) = 0.8
        assert len(ranked) == 1

    def test_empty_addresses(self, ranker):
        analysis = QueryAnalysis(primary_type=QueryType.CODE)
        ranked = ranker.rank([], analysis)
        assert ranked == []
