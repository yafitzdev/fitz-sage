# tests/unit/test_entity_extraction.py
"""Tests for entity extraction via ChunkEnricher."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from fitz_ai.core.chunk import Chunk
from fitz_ai.ingestion.enrichment import EnrichmentConfig, EnrichmentPipeline


class TestEnrichmentPipelineChunkEnrichment:
    """Tests for chunk enrichment integration in EnrichmentPipeline.

    Entity extraction is baked into the ChunkEnricher bus along with
    summary and keyword extraction. All enrichments run automatically when
    a chat_client is provided.
    """

    def test_chunk_enrichment_enabled_with_chat_client(self, tmp_path):
        """Chunk enrichment is enabled when chat_client is provided."""
        config = EnrichmentConfig()
        mock_chat = MagicMock()
        mock_chat.chat.return_value = "[]"

        pipeline = EnrichmentPipeline(
            config=config,
            project_root=tmp_path,
            chat_client=mock_chat,
        )

        assert pipeline.chunk_enrichment_enabled

    def test_chunk_enrichment_disabled_without_chat_client(self, tmp_path):
        """Chunk enrichment is disabled when no chat_client is provided."""
        config = EnrichmentConfig()
        pipeline = EnrichmentPipeline(
            config=config,
            project_root=tmp_path,
            chat_client=None,
        )

        assert not pipeline.chunk_enrichment_enabled

    def test_enrich_extracts_entities(self, tmp_path):
        """ChunkEnricher extracts entities as part of batch enrichment."""
        config = EnrichmentConfig()

        mock_chat = MagicMock()
        # ChunkEnricher expects batch response with summary, keywords, entities
        mock_chat.chat.return_value = json.dumps(
            [
                {
                    "summary": "A test class for unit testing.",
                    "keywords": ["TestClass"],
                    "entities": [{"name": "TestClass", "type": "class"}],
                },
            ]
        )

        pipeline = EnrichmentPipeline(
            config=config,
            project_root=tmp_path,
            chat_client=mock_chat,
        )

        # Create test chunks
        chunks = [
            Chunk(
                id="chunk1",
                doc_id="doc1",
                chunk_index=0,
                content="class TestClass:\n    pass",
                metadata={"source_file": "/path/to/file.py"},
            ),
        ]

        result = pipeline.enrich(chunks)

        # Find the original chunk (hierarchy enrichment may add L2 summary chunks)
        original_chunk = next(c for c in result.chunks if c.id == "chunk1")

        # Verify entities extracted
        assert "entities" in original_chunk.metadata
        entities = original_chunk.metadata["entities"]
        assert len(entities) == 1
        assert entities[0]["name"] == "TestClass"
        assert entities[0]["type"] == "class"

        # Verify summary also extracted (baked in)
        assert "summary" in original_chunk.metadata
        assert "test class" in original_chunk.metadata["summary"].lower()

    def test_enrich_extracts_all_enrichments(self, tmp_path):
        """ChunkEnricher extracts summary, keywords, and entities together."""
        config = EnrichmentConfig()

        mock_chat = MagicMock()
        mock_chat.chat.return_value = json.dumps(
            [
                {
                    "summary": "User authentication module.",
                    "keywords": ["UserAuth", "authenticate", "AUTH_TOKEN"],
                    "entities": [
                        {"name": "UserAuth", "type": "class"},
                        {"name": "authenticate", "type": "function"},
                    ],
                },
            ]
        )

        pipeline = EnrichmentPipeline(
            config=config,
            project_root=tmp_path,
            chat_client=mock_chat,
        )

        chunks = [
            Chunk(
                id="chunk1",
                doc_id="doc1",
                chunk_index=0,
                content="class UserAuth:\n    def authenticate(self): pass",
                metadata={"source_file": "/path/to/auth.py"},
            ),
        ]

        result = pipeline.enrich(chunks)

        # Find the original chunk (hierarchy enrichment may add L2 summary chunks)
        original_chunk = next(c for c in result.chunks if c.id == "chunk1")

        # All enrichments should be present
        assert "summary" in original_chunk.metadata
        assert "entities" in original_chunk.metadata
        assert len(original_chunk.metadata["entities"]) == 2
