# fitz_ai/engines/fitz_rag/retrieval/multihop/controller.py
"""
Multi-hop retrieval controller.

Orchestrates iterative retrieval until sufficient evidence is found.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import RETRIEVER

from .evaluator import EvidenceEvaluator
from .extractor import BridgeExtractor

if TYPE_CHECKING:
    from fitz_ai.core.chunk import Chunk

logger = get_logger(__name__)


@dataclass
class HopResult:
    """Result of a single hop iteration."""

    chunks: list["Chunk"]
    is_sufficient: bool
    bridge_questions: list[str]
    hop_number: int


@dataclass
class HopMetadata:
    """Metadata about the multi-hop retrieval process."""

    total_hops: int
    trace: list[HopResult] = field(default_factory=list)

    @property
    def used_multi_hop(self) -> bool:
        """True if more than one hop was needed."""
        return self.total_hops > 1


class HopController:
    """
    Orchestrates multi-hop retrieval.

    Iteratively retrieves evidence until:
    1. Evidence is sufficient to answer the query
    2. No bridge questions can be generated
    3. Maximum hops reached

    Usage:
        controller = HopController(
            retrieval_pipeline=retrieval,
            evaluator=EvidenceEvaluator(chat=fast_chat),
            extractor=BridgeExtractor(chat=fast_chat),
        )
        chunks, metadata = controller.retrieve("What does Sarah's company make?")
    """

    def __init__(
        self,
        retrieval_pipeline: Any,  # RetrievalPipelineFromYaml
        evaluator: EvidenceEvaluator,
        extractor: BridgeExtractor,
        max_hops: int = 2,
    ):
        """
        Initialize the controller.

        Args:
            retrieval_pipeline: The retrieval pipeline to use for each hop
            evaluator: Evaluates if evidence is sufficient
            extractor: Extracts bridge questions for follow-up
            max_hops: Maximum retrieval iterations (default 2)
        """
        self.retrieval = retrieval_pipeline
        self.evaluator = evaluator
        self.extractor = extractor
        self.max_hops = max_hops

    def retrieve(
        self,
        query: str,
        filter_override: dict[str, Any] | None = None,
    ) -> tuple[list["Chunk"], HopMetadata]:
        """
        Execute multi-hop retrieval.

        Args:
            query: User query
            filter_override: Optional filter for vector search (e.g., for L2 routing)

        Returns:
            chunks: All accumulated chunks across hops
            metadata: Hop trace for debugging/observability
        """
        all_chunks: list["Chunk"] = []
        seen_ids: set[str] = set()
        hop_trace: list[HopResult] = []

        current_query = query

        for hop in range(self.max_hops):
            logger.debug(f"{RETRIEVER} Multi-hop: starting hop {hop + 1}/{self.max_hops}")

            # Step 1: Retrieve with current query
            new_chunks = self.retrieval.retrieve(
                current_query,
                filter_override=filter_override,
            )

            # Dedupe and accumulate
            added = 0
            for chunk in new_chunks:
                if chunk.id not in seen_ids:
                    seen_ids.add(chunk.id)
                    all_chunks.append(chunk)
                    added += 1

            logger.debug(
                f"{RETRIEVER} Multi-hop: hop {hop + 1} retrieved {len(new_chunks)} chunks "
                f"({added} new, {len(all_chunks)} total)"
            )

            # Step 2: Evaluate if we have enough evidence
            is_sufficient = self.evaluator.evaluate(query, all_chunks)

            if is_sufficient:
                hop_trace.append(
                    HopResult(
                        chunks=new_chunks,
                        is_sufficient=True,
                        bridge_questions=[],
                        hop_number=hop,
                    )
                )
                logger.info(
                    f"{RETRIEVER} Multi-hop: SUFFICIENT after {hop + 1} hop(s), "
                    f"{len(all_chunks)} total chunks"
                )
                break

            # Step 3: Extract bridge questions for follow-up
            bridge_questions = self.extractor.extract(query, all_chunks)

            hop_trace.append(
                HopResult(
                    chunks=new_chunks,
                    is_sufficient=False,
                    bridge_questions=bridge_questions,
                    hop_number=hop,
                )
            )

            if not bridge_questions:
                # No more questions to ask - stop even if insufficient
                logger.info(
                    f"{RETRIEVER} Multi-hop: no bridge questions after {hop + 1} hop(s), "
                    f"stopping with {len(all_chunks)} chunks"
                )
                break

            # Use first bridge question for next hop
            current_query = bridge_questions[0]
            logger.debug(f"{RETRIEVER} Multi-hop: bridging to: {current_query}")

        else:
            # Max hops reached
            logger.info(
                f"{RETRIEVER} Multi-hop: max hops ({self.max_hops}) reached, "
                f"{len(all_chunks)} total chunks"
            )

        metadata = HopMetadata(
            total_hops=len(hop_trace),
            trace=hop_trace,
        )

        return all_chunks, metadata
