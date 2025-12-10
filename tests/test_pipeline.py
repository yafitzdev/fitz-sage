import pytest

from fitz_rag.pipeline.engine import RAGPipeline
from fitz_rag.generation.rgs import RGS, RGSConfig
from fitz_rag.models.chunk import Chunk
from fitz_rag.models.document import Document


# ---------------------------------------------------------------------
# Dummy LLM + Retriever for testing
# ---------------------------------------------------------------------

class DummyLLM:
    """
    Fake LLM that records the last messages and returns a predictable answer.
    """

    def __init__(self):
        self.last_messages = None

    def chat(self, messages):
        self.last_messages = messages
        return "[DUMMY-ANSWER] success"


class DummyRetriever:
    """
    Minimal retriever returning a single predictable chunk.
    """

    def retrieve(self, query, top_k=None):
        doc = Document(
            id="doc1",
            path="dummy.txt",
            metadata={"source": "unit-test"},
            content="This is a test document.",
        )
        chunk = Chunk(
            id="chunk1",
            doc_id="doc1",
            content=f"retrieved text for: {query}",
            metadata={"file": "dummy.txt"},
            chunk_index=0,
        )
        return [chunk]


# ---------------------------------------------------------------------
# Pipeline test
# ---------------------------------------------------------------------

def test_pipeline_end_to_end():
    """
    Full pipeline test using:
    - DummyRetriever
    - DummyLLM
    - RGS prompt builder
    - RAGPipeline coordinator
    """

    retriever = DummyRetriever()
    llm = DummyLLM()
    rgs = RGS(RGSConfig())

    pipeline = RAGPipeline(
        retriever=retriever,
        llm=llm,
        rgs=rgs,
    )

    query = "why did it fail?"

    result = pipeline.run(query=query)

    # ------------------------------------------------------------------
    # Validate pipeline output
    # ------------------------------------------------------------------

    # Check final answer
    assert result.answer.startswith("[DUMMY-ANSWER]")

    # Check RGS-added source metadata
    assert len(result.sources) == 1
    assert result.sources[0].source_id == "chunk1"

    # Check prompt was passed to LLM
    assert llm.last_messages is not None
    system_msg = llm.last_messages[0]["content"]
    user_msg = llm.last_messages[1]["content"]

    assert "retrieval-grounded assistant" in system_msg
    assert "User question:" in user_msg
    assert query in user_msg
    assert "retrieved text for" in user_msg  # chunk content present

    # RGS label check (S1)
    assert "[S1]" in user_msg
