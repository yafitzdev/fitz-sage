import pytest
from unittest.mock import MagicMock, patch

# retriever exceptions
from fitz_rag.exceptions.retriever import (
    EmbeddingError,
    VectorSearchError,
    RerankError,
)

# prompt builder
from fitz_rag.sourcer.prompt_builder import (
    build_user_prompt,
    PromptBuilderError,
    build_rag_block,
)

# rag base
from fitz_rag.sourcer.rag_base import (
    RAGContextBuilder,
    RetrievalContext,
    SourceConfig,
    ArtefactRetrievalStrategy,
    PluginLoadError,
)


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

class DummyStrategy(ArtefactRetrievalStrategy):
    """Strategy that throws an exception on retrieve()."""
    def retrieve(self, trf, query):
        raise RuntimeError("strategy blew up")


class FakeStrategy(ArtefactRetrievalStrategy):
    """Strategy that returns dict chunks."""
    def retrieve(self, trf, query):
        return [
            {
                "id": "1",
                "text": "hello",
                "file": "doc",
                "metadata": {"chunk_index": 0},
                "score": None,
            }
        ]


# ---------------------------------------------------------------------
# TEST: RAGRetriever exception propagation
# ---------------------------------------------------------------------

def test_embedding_error_message():
    """Retriever should wrap embedding failures in EmbeddingError."""
    from fitz_rag.retriever.plugins.dense import RAGRetriever
    from fitz_rag.config.schema import EmbeddingConfig, RetrieverConfig

    embed_cfg = EmbeddingConfig(provider="cohere", api_key="x", model="y")
    retr_cfg = RetrieverConfig(collection="test", top_k=5)

    with patch("fitz_rag.llm.embedding.plugins.cohere.CohereEmbeddingClient.embed") as m:
        m.side_effect = RuntimeError("embedding crash")

        retr = RAGRetriever(
            client=MagicMock(),
            embed_cfg=embed_cfg,
            retriever_cfg=retr_cfg,
            rerank_cfg=None,
        )

        with pytest.raises(EmbeddingError, match="Failed to embed query:"):
            retr.retrieve("hello")


def test_vector_search_error_message():
    """Retriever should wrap vector-search failures in VectorSearchError."""
    from fitz_rag.retriever.plugins.dense import RAGRetriever
    from fitz_rag.config.schema import EmbeddingConfig, RetrieverConfig

    embed_cfg = EmbeddingConfig(provider="cohere", api_key="x", model="y")
    retr_cfg = RetrieverConfig(collection="test", top_k=5)

    mock_client = MagicMock()
    mock_client.search.side_effect = RuntimeError("qdrant boom")

    with patch("fitz_rag.llm.embedding.plugins.cohere.CohereEmbeddingClient.embed") as embed_mock:
        embed_mock.return_value = [0.1, 0.2, 0.3]

        retr = RAGRetriever(
            client=mock_client,
            embed_cfg=embed_cfg,
            retriever_cfg=retr_cfg,
            rerank_cfg=None,
        )

        with pytest.raises(VectorSearchError, match="Vector search failed for collection 'test'"):
            retr.retrieve("hello")


def test_rerank_error_message():
    """Retriever should wrap rerank failures in RerankError."""
    from fitz_rag.retriever.plugins.dense import RAGRetriever
    from fitz_rag.config.schema import EmbeddingConfig, RetrieverConfig, RerankConfig

    embed_cfg = EmbeddingConfig(provider="cohere", api_key="x", model="y")
    retr_cfg = RetrieverConfig(collection="test", top_k=5)
    rerank_cfg = RerankConfig(provider="cohere", api_key="x", model="rerank", enabled=True)

    with patch("fitz_rag.llm.embedding.plugins.cohere.CohereEmbeddingClient.embed") as embed_mock:
        embed_mock.return_value = [0.1, 0.2, 0.3]

        mock_hit = MagicMock()
        mock_hit.id = "1"
        mock_hit.payload = {"text": "a", "metadata": {}}

        mock_client = MagicMock()
        mock_client.search.return_value = [mock_hit]

        with patch("fitz_rag.llm.rerank.plugins.cohere.CohereRerankClient.rerank") as rerank_mock:
            rerank_mock.side_effect = RuntimeError("rerank boom")

            retr = RAGRetriever(
                client=mock_client,
                embed_cfg=embed_cfg,
                retriever_cfg=retr_cfg,
                rerank_cfg=rerank_cfg,
            )

            with pytest.raises(RerankError, match="Reranking failed"):
                retr.retrieve("hello")


# ---------------------------------------------------------------------
# TEST: PromptBuilderError
# ---------------------------------------------------------------------

def test_prompt_builder_invalid_json():
    """build_user_prompt should raise PromptBuilderError for unserializable TRF."""
    bad_obj = {"x": set([1, 2])}  # sets are not JSON serializable

    ctx = RetrievalContext(query="q", artefacts={})
    sources = []

    with pytest.raises(PromptBuilderError, match="Invalid TRF JSON structure"):
        build_user_prompt(bad_obj, ctx, "task", sources)


def test_prompt_builder_chunk_format_error():
    """Formatting chunks should raise PromptBuilderError if chunk fields fail."""
    bad_chunk = {"text": "hello", "metadata": None, "score": 0.5}

    ctx = RetrievalContext(query="q", artefacts={"src": [bad_chunk]})
    sources = [SourceConfig(name="src", order=0, strategy=MagicMock())]

    with pytest.raises(PromptBuilderError, match="Failed to format chunks for label 'SRC'"):
        build_rag_block(ctx, sources)


# ---------------------------------------------------------------------
# TEST: RAG Base — PluginLoadError & StrategyError
# ---------------------------------------------------------------------

def test_plugin_load_error():
    """load_source_configs should raise PluginLoadError for import failure."""
    with patch("importlib.import_module") as imp:
        imp.side_effect = RuntimeError("import crash")

        with pytest.raises(PluginLoadError, match="Failed to import plugin module"):
            from fitz_rag.sourcer.rag_base import load_source_configs
            load_source_configs()


def test_retrieval_strategy_error_attached():
    """
    RAGContextBuilder should not crash on strategy failure.
    Instead, it should store an error chunk (dict) with a message.
    """
    ctx_builder = RAGContextBuilder(
        sources=[SourceConfig(name="fail_src", order=0, strategy=DummyStrategy())]
    )

    ctx = ctx_builder.retrieve_for(trf={}, query="hello")

    # fail_src should be empty list
    assert ctx.artefacts["fail_src"] == []

    # fail_src_error should contain a dict with a text field
    err_chunks = ctx.artefacts["fail_src_error"]
    assert len(err_chunks) == 1
    assert "Retrieval error in fail_src" in err_chunks[0]["text"]
