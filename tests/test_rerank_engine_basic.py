from fitz_stack.llm.rerank import RerankEngine

class DummyRerankPlugin:
    def rerank(self, query, chunks):
        # return indices
        return [1, 0, 2]

def test_rerank_engine_basic_flow():
    engine = RerankEngine(DummyRerankPlugin())
    chunks = ["A", "B", "C"]
    ranked = engine.rerank("q", chunks)
    assert ranked == ["B", "A", "C"]
