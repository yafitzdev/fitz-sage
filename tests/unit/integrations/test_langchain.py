# tests/unit/integrations/test_langchain.py
"""Tests for LangChain integration."""

from unittest.mock import MagicMock, patch

import pytest

# Skip if langchain-core not installed
pytest.importorskip("langchain_core")

from langchain_core.documents import Document

from fitz_ai.integrations.langchain.runnable import FitzRAGChain


class TestFitzRAGChain:
    """Tests for FitzRAGChain."""

    def test_extract_chunk_ids_with_id(self):
        """Extracts IDs from documents with 'id' metadata."""
        docs = [
            Document(page_content="text1", metadata={"id": "chunk_1"}),
            Document(page_content="text2", metadata={"id": "chunk_2"}),
        ]

        # Create minimal chain for testing
        chain = FitzRAGChain.__new__(FitzRAGChain)
        ids = chain._extract_chunk_ids(docs)

        assert ids == ["chunk_1", "chunk_2"]

    def test_extract_chunk_ids_with_chunk_id(self):
        """Extracts IDs from documents with 'chunk_id' metadata."""
        docs = [
            Document(page_content="text1", metadata={"chunk_id": "c1"}),
            Document(page_content="text2", metadata={"chunk_id": "c2"}),
        ]

        chain = FitzRAGChain.__new__(FitzRAGChain)
        ids = chain._extract_chunk_ids(docs)

        assert ids == ["c1", "c2"]

    def test_extract_chunk_ids_fallback(self):
        """Generates fallback IDs for documents without ID metadata."""
        docs = [
            Document(page_content="text1", metadata={}),
            Document(page_content="text2", metadata={"title": "Doc 2"}),
        ]

        chain = FitzRAGChain.__new__(FitzRAGChain)
        ids = chain._extract_chunk_ids(docs)

        assert ids == ["chunk_0", "chunk_1"]

    def test_doc_to_source(self):
        """Converts Document to source dict."""
        doc = Document(
            page_content="This is a long text that should be truncated..." * 50,
            metadata={"id": "doc_1", "source": "file.pdf", "page": 5},
        )

        chain = FitzRAGChain.__new__(FitzRAGChain)
        source = chain._doc_to_source(doc)

        assert source["source_id"] == "doc_1"
        assert len(source["excerpt"]) <= 500
        assert source["metadata"]["page"] == 5

    def test_doc_to_source_fallback_id(self):
        """Uses 'source' as fallback for source_id."""
        doc = Document(
            page_content="text",
            metadata={"source": "file.pdf"},  # No 'id', but has 'source'
        )

        chain = FitzRAGChain.__new__(FitzRAGChain)
        source = chain._doc_to_source(doc)

        assert source["source_id"] == "file.pdf"

    @patch("fitz_ai.integrations.langchain.runnable.FitzOptimizer")
    def test_cache_hit_skips_llm(self, mock_optimizer_class):
        """Cache hit returns answer WITHOUT calling LLM."""
        # Setup: cache returns a hit
        mock_optimizer = MagicMock()
        mock_optimizer_class.return_value = mock_optimizer

        mock_lookup_result = MagicMock()
        mock_lookup_result.hit = True
        mock_lookup_result.answer = "Cached answer"
        mock_lookup_result.sources = [{"source_id": "doc1", "excerpt": "text"}]
        mock_optimizer.lookup.return_value = mock_lookup_result

        # Setup retriever and LLM
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            Document(page_content="doc content", metadata={"id": "doc1"})
        ]

        mock_llm = MagicMock()  # Should NOT be called

        mock_embed_fn = MagicMock(return_value=[1.0] * 1536)

        chain = FitzRAGChain(
            retriever=mock_retriever,
            llm=mock_llm,
            api_key="fitz_test",
            org_key="a" * 64,
            embedding_fn=mock_embed_fn,
            llm_model="gpt-4o",
        )

        result = chain.invoke({"question": "What is X?"})

        # Verify: LLM was NOT called (this is the key assertion!)
        mock_llm.invoke.assert_not_called()

        # Verify: cached answer returned
        assert result["answer"] == "Cached answer"
        assert result["_fitz_cache_hit"] is True

        # Verify: retriever WAS called (we need docs for fingerprint)
        mock_retriever.invoke.assert_called_once()

    @patch("fitz_ai.integrations.langchain.runnable.FitzOptimizer")
    def test_cache_miss_runs_llm_and_stores(self, mock_optimizer_class):
        """Cache miss runs LLM and stores result."""
        # Setup mocks
        mock_optimizer = MagicMock()
        mock_optimizer_class.return_value = mock_optimizer

        # Cache miss
        mock_lookup_result = MagicMock()
        mock_lookup_result.hit = False
        mock_lookup_result.routing_advice = None
        mock_optimizer.lookup.return_value = mock_lookup_result

        # Setup retriever
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            Document(page_content="source text", metadata={"id": "doc1"})
        ]

        # Setup LLM
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "LLM generated answer"
        mock_llm.invoke.return_value = mock_response

        mock_embed_fn = MagicMock(return_value=[1.0] * 1536)

        chain = FitzRAGChain(
            retriever=mock_retriever,
            llm=mock_llm,
            api_key="fitz_test",
            org_key="a" * 64,
            embedding_fn=mock_embed_fn,
            llm_model="gpt-4o",
        )

        result = chain.invoke({"question": "What is X?"})

        # LLM was called
        mock_llm.invoke.assert_called_once()

        # Cache was checked
        mock_optimizer.lookup.assert_called_once()

        # Result was stored
        mock_optimizer.store.assert_called_once()
        store_call = mock_optimizer.store.call_args
        assert store_call.kwargs["answer_text"] == "LLM generated answer"

        # Result has correct structure
        assert result["answer"] == "LLM generated answer"
        assert result["_fitz_cache_hit"] is False

    @patch("fitz_ai.integrations.langchain.runnable.FitzOptimizer")
    def test_invoke_includes_routing_advice(self, mock_optimizer_class):
        """Invoke includes routing advice in result on cache miss."""
        # Setup mocks
        mock_optimizer = MagicMock()
        mock_optimizer_class.return_value = mock_optimizer

        # Cache miss with routing
        mock_lookup_result = MagicMock()
        mock_lookup_result.hit = False
        mock_lookup_result.routing_advice = {
            "complexity": "simple",
            "recommended_model": "fast",
        }
        mock_optimizer.lookup.return_value = mock_lookup_result

        # Setup retriever
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []

        # Setup LLM
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Test answer"
        mock_llm.invoke.return_value = mock_response

        mock_embed_fn = MagicMock(return_value=[1.0] * 1536)

        chain = FitzRAGChain(
            retriever=mock_retriever,
            llm=mock_llm,
            api_key="fitz_test",
            org_key="a" * 64,
            embedding_fn=mock_embed_fn,
            llm_model="gpt-4o",
        )

        result = chain.invoke({"question": "What is X?"})

        # Routing advice is included
        assert result.get("_fitz_routing") is not None
        assert result["_fitz_routing"]["complexity"] == "simple"

    @patch("fitz_ai.integrations.langchain.runnable.FitzOptimizer")
    def test_invoke_empty_question_fallback(self, mock_optimizer_class):
        """Empty question falls back to uncached execution."""
        mock_optimizer = MagicMock()
        mock_optimizer_class.return_value = mock_optimizer

        # Setup retriever
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []

        # Setup LLM
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "result"
        mock_llm.invoke.return_value = mock_response

        chain = FitzRAGChain(
            retriever=mock_retriever,
            llm=mock_llm,
            api_key="fitz_test",
            org_key="a" * 64,
            embedding_fn=lambda x: [1.0] * 1536,
            llm_model="gpt-4o",
        )

        _ = chain.invoke({})  # No question - test LLM invocation path

        # LLM was called (uncached path)
        mock_llm.invoke.assert_called_once()

        # Cache was NOT checked
        mock_optimizer.lookup.assert_not_called()

    @patch("fitz_ai.integrations.langchain.runnable.FitzOptimizer")
    def test_was_cache_hit_property(self, mock_optimizer_class):
        """was_cache_hit property reflects last invocation."""
        mock_optimizer = MagicMock()
        mock_optimizer_class.return_value = mock_optimizer

        mock_lookup_result = MagicMock()
        mock_lookup_result.hit = True
        mock_lookup_result.answer = "Cached"
        mock_lookup_result.sources = []
        mock_optimizer.lookup.return_value = mock_lookup_result

        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []

        mock_llm = MagicMock()

        chain = FitzRAGChain(
            retriever=mock_retriever,
            llm=mock_llm,
            api_key="fitz_test",
            org_key="a" * 64,
            embedding_fn=lambda x: [1.0] * 1536,
            llm_model="gpt-4o",
        )

        chain.invoke({"question": "test"})

        assert chain.was_cache_hit is True

    @patch("fitz_ai.integrations.langchain.runnable.FitzOptimizer")
    def test_context_manager(self, mock_optimizer_class):
        """Context manager closes optimizer."""
        mock_optimizer = MagicMock()
        mock_optimizer_class.return_value = mock_optimizer

        mock_retriever = MagicMock()
        mock_llm = MagicMock()

        with FitzRAGChain(
            retriever=mock_retriever,
            llm=mock_llm,
            api_key="fitz_test",
            org_key="a" * 64,
            embedding_fn=lambda x: [1.0] * 1536,
            llm_model="gpt-4o",
        ) as chain:  # noqa: F841
            pass

        mock_optimizer.close.assert_called_once()


# Backwards compatibility alias test
class TestFitzCacheRunnableAlias:
    """Test that FitzCacheRunnable alias works."""

    def test_alias_exists(self):
        """FitzCacheRunnable is an alias for FitzRAGChain."""
        from fitz_ai.integrations.langchain import FitzCacheRunnable, FitzRAGChain

        assert FitzCacheRunnable is FitzRAGChain


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
