# tests/test_rag_pipeline_llm_failure.py
"""
Test that RAGPipeline properly handles LLM failures.
"""

from dataclasses import dataclass

import pytest

from fitz_ai.core.chunk import Chunk
from fitz_ai.engines.fitz_rag.exceptions import LLMError
from fitz_ai.engines.fitz_rag.generation.retrieval_guided.synthesis import (
    RGS,
    RGSConfig,
)
from fitz_ai.engines.fitz_rag.pipeline.components import PipelineComponents
from fitz_ai.engines.fitz_rag.pipeline.engine import RAGPipeline


def create_mock_chat_factory(mock_chat):
    """Create a mock chat factory that returns the mock chat client."""

    def factory(tier: str = "fast"):
        return mock_chat

    return factory


@dataclass
class MockRetrievalPipeline:
    """Mock retrieval pipeline that returns fixed chunks."""

    plugin_name: str = "mock"

    def retrieve(
        self, query: str, filter_override: dict | None = None, conversation_context=None
    ) -> list[Chunk]:
        return [
            Chunk(
                id="chunk_1",
                doc_id="doc_1",
                content="Some content.",
                chunk_index=0,
                metadata={},
            ),
        ]


class FailingLLM:
    """Mock LLM that always raises an exception."""

    def chat(self, messages: list[dict]) -> str:
        raise RuntimeError("LLM service unavailable")


def test_rag_pipeline_llm_failure():
    components = PipelineComponents(
        retrieval=MockRetrievalPipeline(),
        chat_factory=create_mock_chat_factory(FailingLLM()),
        rgs=RGS(RGSConfig()),
    )
    pipe = RAGPipeline(components)

    with pytest.raises(LLMError):
        pipe.run("Any query")
