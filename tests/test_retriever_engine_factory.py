from fitz_rag.retriever.engine import RetrieverEngine
from fitz_rag.retriever.plugins.dense import DenseRetrievalPlugin

def test_retriever_engine_from_name():
    engine = RetrieverEngine.from_name(
        "dense",
        client="dummy",
        embed_cfg=type("Cfg", (), {"api_key": "k", "model": "m", "output_dimension": None}),
        retriever_cfg=type("Cfg", (), {"collection": "col", "top_k": 2}),
        rerank_cfg=None,
    )

    assert isinstance(engine.plugin, DenseRetrievalPlugin)
    assert engine.plugin.client == "dummy"
