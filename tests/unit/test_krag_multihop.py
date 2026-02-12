# tests/unit/test_krag_multihop.py
"""
Unit tests for KragHopController.

Multi-hop retrieval controller for KRAG: iterative retrieve -> read -> evaluate
-> bridge cycle adapted for address-based architecture.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from fitz_ai.engines.fitz_krag.retrieval.multihop import KragHopController
from fitz_ai.engines.fitz_krag.types import Address, AddressKind, ReadResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _addr(
    source_id: str = "src",
    location: str = "mod.func",
    score: float = 0.9,
) -> Address:
    """Build a SYMBOL Address."""
    return Address(
        kind=AddressKind.SYMBOL,
        source_id=source_id,
        location=location,
        summary=f"Symbol {location}",
        score=score,
    )


def _read_result(
    source_id: str = "src",
    location: str = "mod.func",
    content: str = "def func(): return 42",
    file_path: str = "module.py",
) -> ReadResult:
    """Build a ReadResult with a corresponding Address."""
    addr = _addr(source_id=source_id, location=location)
    return ReadResult(
        address=addr,
        content=content,
        file_path=file_path,
        line_range=(1, 3),
    )


def _make_chat_factory(
    is_sufficient: bool = True,
    bridge_questions: list[str] | None = None,
) -> MagicMock:
    """Create a mock ChatFactory that returns a mock chat client.

    The chat client's response is controlled to simulate sufficiency evaluation
    and bridge question extraction.
    """
    factory = MagicMock(name="chat_factory")
    chat = MagicMock(name="chat_client")

    responses = []
    if is_sufficient:
        responses.append("SUFFICIENT")
    else:
        responses.append("INSUFFICIENT")
        if bridge_questions:
            import json

            responses.append(json.dumps(bridge_questions))
        else:
            responses.append("[]")

    # Each call to chat.chat returns the next response
    chat.chat.side_effect = responses
    factory.return_value = chat
    return factory


def _make_multihop_factory(sufficiency_sequence: list[str]) -> MagicMock:
    """Create a factory where the chat returns a sequence of responses.

    Each pair is (sufficiency_response, bridge_response).
    """
    factory = MagicMock(name="chat_factory")
    chat = MagicMock(name="chat_client")
    chat.chat.side_effect = sufficiency_sequence
    factory.return_value = chat
    return factory


def _make_router(address_batches: list[list[Address]]) -> MagicMock:
    """Create a mock router returning different addresses on each call."""
    router = MagicMock(name="router")
    router.retrieve.side_effect = address_batches
    return router


def _make_reader(result_batches: list[list[ReadResult]]) -> MagicMock:
    """Create a mock reader returning different results on each call."""
    reader = MagicMock(name="reader")
    reader.read.side_effect = result_batches
    return reader


# ---------------------------------------------------------------------------
# TestSingleHop
# ---------------------------------------------------------------------------


class TestSingleHop:
    """Tests for single-hop execution (sufficient evidence on first hop)."""

    def test_single_hop_sufficient(self):
        """Stops after one hop when evidence is deemed sufficient."""
        addr1 = _addr(source_id="f1", location="mod.auth")
        result1 = _read_result(source_id="f1", location="mod.auth")

        router = _make_router(
            [
                [addr1],  # hop 1 retrieval
            ]
        )
        reader = _make_reader(
            [
                [result1],  # hop 1 read
            ]
        )
        factory = _make_chat_factory(is_sufficient=True)

        controller = KragHopController(
            router=router,
            reader=reader,
            chat_factory=factory,
            max_hops=3,
            top_read=5,
        )

        results = controller.execute("how does auth work?")

        assert len(results) == 1
        assert results[0].address.location == "mod.auth"
        router.retrieve.assert_called_once()
        reader.read.assert_called_once()

    def test_returns_all_read_results(self):
        """Multiple read results from single hop are all returned."""
        addr1 = _addr(source_id="f1", location="mod.a")
        addr2 = _addr(source_id="f2", location="mod.b")
        result1 = _read_result(source_id="f1", location="mod.a")
        result2 = _read_result(source_id="f2", location="mod.b")

        router = _make_router([[addr1, addr2]])
        reader = _make_reader([[result1, result2]])
        factory = _make_chat_factory(is_sufficient=True)

        controller = KragHopController(
            router=router,
            reader=reader,
            chat_factory=factory,
            max_hops=2,
            top_read=5,
        )

        results = controller.execute("query")

        assert len(results) == 2


# ---------------------------------------------------------------------------
# TestMultipleHops
# ---------------------------------------------------------------------------


class TestMultipleHops:
    """Tests for multi-hop execution with bridge questions."""

    def test_two_hops_with_bridge(self):
        """Insufficient evidence triggers bridge question and second hop."""
        addr1 = _addr(source_id="f1", location="mod.auth")
        addr2 = _addr(source_id="f2", location="mod.session")
        result1 = _read_result(source_id="f1", location="mod.auth")
        result2 = _read_result(source_id="f2", location="mod.session")

        router = _make_router(
            [
                [addr1],  # hop 1
                [addr2],  # hop 2 (bridge query)
            ]
        )
        reader = _make_reader(
            [
                [result1],  # hop 1
                [result2],  # hop 2
            ]
        )

        # Hop 1: INSUFFICIENT + bridge questions
        # Hop 2: SUFFICIENT
        import json

        factory = _make_multihop_factory(
            [
                "INSUFFICIENT",  # hop 1 sufficiency
                json.dumps(["what is the session handler?"]),  # hop 1 bridge
                "SUFFICIENT",  # hop 2 sufficiency
            ]
        )

        controller = KragHopController(
            router=router,
            reader=reader,
            chat_factory=factory,
            max_hops=3,
            top_read=5,
        )

        results = controller.execute("how does auth work with sessions?")

        assert len(results) == 2
        assert results[0].address.location == "mod.auth"
        assert results[1].address.location == "mod.session"
        assert router.retrieve.call_count == 2


# ---------------------------------------------------------------------------
# TestStopConditions
# ---------------------------------------------------------------------------


class TestStopConditions:
    """Tests for hop termination conditions."""

    def test_stops_when_no_new_addresses(self):
        """Stops when router returns same addresses (all filtered as duplicates)."""
        addr1 = _addr(source_id="f1", location="mod.auth")
        result1 = _read_result(source_id="f1", location="mod.auth")

        router = _make_router(
            [
                [addr1],  # hop 1
                [addr1],  # hop 2: same address -> filtered -> empty new_addresses
            ]
        )
        reader = _make_reader(
            [
                [result1],  # hop 1
            ]
        )

        import json

        factory = _make_multihop_factory(
            [
                "INSUFFICIENT",
                json.dumps(["follow-up question"]),
            ]
        )

        controller = KragHopController(
            router=router,
            reader=reader,
            chat_factory=factory,
            max_hops=5,
            top_read=5,
        )

        results = controller.execute("query")

        # Only hop 1 results since hop 2 has no new addresses
        assert len(results) == 1
        reader.read.assert_called_once()

    def test_stops_when_max_hops_reached(self):
        """Stops at max_hops even if evidence is insufficient."""
        addresses_per_hop = [[_addr(source_id=f"f{i}", location=f"mod.func{i}")] for i in range(3)]
        results_per_hop = [
            [_read_result(source_id=f"f{i}", location=f"mod.func{i}")] for i in range(3)
        ]
        router = _make_router(addresses_per_hop)
        reader = _make_reader(results_per_hop)

        import json

        # Always insufficient, always has bridge questions
        responses = []
        for _ in range(3):
            responses.append("INSUFFICIENT")
            responses.append(json.dumps(["next question"]))
        factory = _make_multihop_factory(responses)

        controller = KragHopController(
            router=router,
            reader=reader,
            chat_factory=factory,
            max_hops=2,  # limit to 2 hops
            top_read=5,
        )

        results = controller.execute("complex query")

        # Should have results from exactly 2 hops
        assert len(results) == 2
        assert router.retrieve.call_count == 2

    def test_stops_when_no_addresses_returned(self):
        """Stops immediately if router returns empty list."""
        router = _make_router([[]])  # Empty addresses
        reader = _make_reader([])
        factory = _make_chat_factory(is_sufficient=True)

        controller = KragHopController(
            router=router,
            reader=reader,
            chat_factory=factory,
            max_hops=3,
            top_read=5,
        )

        results = controller.execute("query about nothing")

        assert results == []
        reader.read.assert_not_called()

    def test_stops_when_no_bridge_questions(self):
        """Stops if bridge extraction returns empty list."""
        addr1 = _addr(source_id="f1", location="mod.func")
        result1 = _read_result(source_id="f1", location="mod.func")

        router = _make_router([[addr1]])
        reader = _make_reader([[result1]])

        import json

        factory = _make_multihop_factory(
            [
                "INSUFFICIENT",
                json.dumps([]),  # No bridge questions
            ]
        )

        controller = KragHopController(
            router=router,
            reader=reader,
            chat_factory=factory,
            max_hops=5,
            top_read=5,
        )

        results = controller.execute("query")

        assert len(results) == 1
        router.retrieve.assert_called_once()


# ---------------------------------------------------------------------------
# TestDeduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Tests for address deduplication across hops."""

    def test_deduplicates_across_hops(self):
        """Same address appearing in hop 2 is filtered out."""
        addr_shared = _addr(source_id="f1", location="mod.shared")
        addr_new = _addr(source_id="f2", location="mod.new_func")
        result_shared = _read_result(source_id="f1", location="mod.shared")
        result_new = _read_result(source_id="f2", location="mod.new_func")

        router = _make_router(
            [
                [addr_shared],  # hop 1
                [addr_shared, addr_new],  # hop 2: addr_shared is duplicate
            ]
        )
        reader = _make_reader(
            [
                [result_shared],  # hop 1
                [result_new],  # hop 2: only addr_new read
            ]
        )

        import json

        factory = _make_multihop_factory(
            [
                "INSUFFICIENT",
                json.dumps(["bridge question"]),
                "SUFFICIENT",
            ]
        )

        controller = KragHopController(
            router=router,
            reader=reader,
            chat_factory=factory,
            max_hops=3,
            top_read=5,
        )

        results = controller.execute("query")

        assert len(results) == 2
        locations = [r.address.location for r in results]
        assert "mod.shared" in locations
        assert "mod.new_func" in locations

        # Reader called twice: once per hop
        assert reader.read.call_count == 2

        # Second read should only have the new address
        second_read_addrs = reader.read.call_args_list[1][0][0]
        assert len(second_read_addrs) == 1
        assert second_read_addrs[0].location == "mod.new_func"

    def test_passes_analysis_and_detection(self):
        """execute forwards analysis and detection to the router."""
        addr1 = _addr(source_id="f1", location="mod.func")
        result1 = _read_result(source_id="f1", location="mod.func")

        router = _make_router([[addr1]])
        reader = _make_reader([[result1]])
        factory = _make_chat_factory(is_sufficient=True)

        analysis = MagicMock(name="analysis")
        detection = MagicMock(name="detection")

        controller = KragHopController(
            router=router,
            reader=reader,
            chat_factory=factory,
            max_hops=2,
            top_read=5,
        )

        controller.execute("query", analysis=analysis, detection=detection)

        router.retrieve.assert_called_once_with(
            "query",
            analysis,
            detection=detection,
        )
