# tests/test_rerank_engine_basic.py

from fitz_rag.llm.rerank.engine import RerankEngine

class DummyRerankPlugin:
    # Return new order: reverse
    def rerank(self, query, chunks):
        return list(reversed(chunks))

def test_rerank_engine_basic_flow():
    engine = RerankEngine(DummyRerankPlugin())

    chunks = ["A", "B", "C"]
    ranked = engine.rerank("q", chunks)

    assert ranked == ["C", "B", "A"]
