# tests/test_entity_linking.py
"""Tests for entity linking module."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from fitz_ai.core.chunk import Chunk
from fitz_ai.ingestion.enrichment import EnrichmentConfig, EnrichmentPipeline
from fitz_ai.ingestion.enrichment.entities import EntityLink, EntityLinker


class TestEntityLink:
    """Tests for EntityLink dataclass."""

    def test_to_dict(self):
        """Test EntityLink serialization."""
        link = EntityLink(
            source="UserService",
            target="OAuth2",
            source_type="class",
            target_type="api",
            chunk_id="chunk_123",
        )

        d = link.to_dict()

        assert d == {
            "source": "UserService",
            "target": "OAuth2",
            "source_type": "class",
            "target_type": "api",
            "chunk_id": "chunk_123",
        }


class TestEntityLinker:
    """Tests for EntityLinker class."""

    def test_link_three_entities_creates_three_links(self):
        """Chunk with 3 entities should create 3 pairwise links."""
        linker = EntityLinker()

        chunks = [
            Chunk(
                id="c1",
                doc_id="d1",
                chunk_index=0,
                content="UserService uses OAuth2 and calls login()",
                metadata={
                    "entities": [
                        {"name": "UserService", "type": "class"},
                        {"name": "OAuth2", "type": "api"},
                        {"name": "login", "type": "function"},
                    ]
                },
            )
        ]

        result = linker.link(chunks)

        assert len(result) == 1
        links = result[0].metadata.get("entity_links", [])
        assert len(links) == 3

        # Verify all pairs are present
        pairs = {(link["source"], link["target"]) for link in links}
        assert ("UserService", "OAuth2") in pairs
        assert ("UserService", "login") in pairs
        assert ("OAuth2", "login") in pairs

    def test_link_two_entities_creates_one_link(self):
        """Chunk with 2 entities should create 1 link."""
        linker = EntityLinker()

        chunks = [
            Chunk(
                id="c1",
                doc_id="d1",
                chunk_index=0,
                content="UserService uses OAuth2",
                metadata={
                    "entities": [
                        {"name": "UserService", "type": "class"},
                        {"name": "OAuth2", "type": "api"},
                    ]
                },
            )
        ]

        result = linker.link(chunks)

        links = result[0].metadata.get("entity_links", [])
        assert len(links) == 1
        assert links[0]["source"] == "UserService"
        assert links[0]["target"] == "OAuth2"

    def test_link_single_entity_no_links(self):
        """Chunk with 1 entity should have no links."""
        linker = EntityLinker()

        chunks = [
            Chunk(
                id="c1",
                doc_id="d1",
                chunk_index=0,
                content="UserService handles users",
                metadata={
                    "entities": [
                        {"name": "UserService", "type": "class"},
                    ]
                },
            )
        ]

        result = linker.link(chunks)

        assert "entity_links" not in result[0].metadata

    def test_link_no_entities_no_links(self):
        """Chunk with no entities should have no links."""
        linker = EntityLinker()

        chunks = [
            Chunk(
                id="c1",
                doc_id="d1",
                chunk_index=0,
                content="Some content without entities",
                metadata={},
            )
        ]

        result = linker.link(chunks)

        assert "entity_links" not in result[0].metadata

    def test_link_multiple_chunks_independent(self):
        """Multiple chunks should have independent links."""
        linker = EntityLinker()

        chunks = [
            Chunk(
                id="c1",
                doc_id="d1",
                chunk_index=0,
                content="UserService uses OAuth2",
                metadata={
                    "entities": [
                        {"name": "UserService", "type": "class"},
                        {"name": "OAuth2", "type": "api"},
                    ]
                },
            ),
            Chunk(
                id="c2",
                doc_id="d1",
                chunk_index=1,
                content="PaymentService uses Stripe",
                metadata={
                    "entities": [
                        {"name": "PaymentService", "type": "class"},
                        {"name": "Stripe", "type": "api"},
                    ]
                },
            ),
        ]

        result = linker.link(chunks)

        # Each chunk should have its own links
        links1 = result[0].metadata.get("entity_links", [])
        links2 = result[1].metadata.get("entity_links", [])

        assert len(links1) == 1
        assert len(links2) == 1

        assert links1[0]["source"] == "UserService"
        assert links1[0]["chunk_id"] == "c1"

        assert links2[0]["source"] == "PaymentService"
        assert links2[0]["chunk_id"] == "c2"

    def test_link_preserves_entity_types(self):
        """Links should preserve entity types from source entities."""
        linker = EntityLinker()

        chunks = [
            Chunk(
                id="c1",
                doc_id="d1",
                chunk_index=0,
                content="UserService uses OAuth2",
                metadata={
                    "entities": [
                        {"name": "UserService", "type": "class"},
                        {"name": "OAuth2", "type": "api"},
                    ]
                },
            )
        ]

        result = linker.link(chunks)

        links = result[0].metadata.get("entity_links", [])
        assert links[0]["source_type"] == "class"
        assert links[0]["target_type"] == "api"

    def test_link_all_fields_present(self):
        """All required fields should be present in link."""
        linker = EntityLinker()

        chunks = [
            Chunk(
                id="c1",
                doc_id="d1",
                chunk_index=0,
                content="UserService uses OAuth2",
                metadata={
                    "entities": [
                        {"name": "UserService", "type": "class"},
                        {"name": "OAuth2", "type": "api"},
                    ]
                },
            )
        ]

        result = linker.link(chunks)

        links = result[0].metadata.get("entity_links", [])
        link = links[0]

        assert "source" in link
        assert "target" in link
        assert "source_type" in link
        assert "target_type" in link
        assert "chunk_id" in link


class TestEntityLinkingPipelineIntegration:
    """Tests for entity linking integration with EnrichmentPipeline.

    Note: Entity extraction is now baked into the ChunkEnricher bus.
    Entity linking is a separate post-processing step that runs on extracted entities.
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
