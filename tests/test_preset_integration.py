# tests/test_preset_integration.py
from fitz.core.config.presets import get_preset
from fitz.pipeline.pipeline.engine import RAGPipeline


def test_preset_works_with_pipeline():
    """Test that presets work with RAGPipeline."""
    config = get_preset("local")

    # Should be able to create pipeline from preset
    pipeline = RAGPipeline.from_dict(config)

    # Verify pipeline was created with expected components
    assert pipeline is not None
    assert pipeline.retriever is not None
    assert pipeline.llm is not None
    assert pipeline.rgs is not None
    assert pipeline.context is not None

    # Note: We don't actually run queries here because that would require
    # a real vector database, embeddings, and LLM. Those are tested separately
    # in integration tests with mocked components.