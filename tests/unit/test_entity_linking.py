# tests/unit/test_entity_linking.py
"""Tests for entity extraction via ChunkEnricher pipeline integration."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from fitz_ai.core.chunk import Chunk
from fitz_ai.ingestion.enrichment import EnrichmentConfig, EnrichmentPipeline


class TestEntityLinkingPipelineIntegration:
    """Tests for entity extraction integration with EnrichmentPipeline.

    Entity extraction is baked into the ChunkEnricher bus.
    """

    def test_pipeline_extracts_entities(self, tmp_path):
        """Pipeline should extract entities via ChunkEnricher."""
        mock_chat = MagicMock()
        # ChunkEnricher expects batch response format
        mock_chat.chat.return_value = json.dumps(
            [
                {
                    "summary": "Authentication service using OAuth2.",
                    "keywords": ["UserService", "OAuth2"],
                    "entities": [
                        {"name": "UserService", "type": "class"},
                        {"name": "OAuth2", "type": "technology"},
                    ],
                }
            ]
        )

        config = EnrichmentConfig()

        pipeline = EnrichmentPipeline(
            config=config,
            project_root=tmp_path,
            chat_client=mock_chat,
        )

        chunks = [
            Chunk(
                id="c1",
                doc_id="d1",
                chunk_index=0,
                content="UserService uses OAuth2 for authentication",
                metadata={"source_file": "auth.py"},
            )
        ]

        result = pipeline.enrich(chunks)

        # Find original chunk (hierarchy enrichment may add summary chunks)
        original_chunk = next(c for c in result.chunks if c.id == "c1")

        # Should have entities extracted
        assert "entities" in original_chunk.metadata
        assert len(original_chunk.metadata["entities"]) == 2
        entity_names = {e["name"] for e in original_chunk.metadata["entities"]}
        assert "UserService" in entity_names
        assert "OAuth2" in entity_names
