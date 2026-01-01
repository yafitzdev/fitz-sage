# tests/test_entity_linking.py
"""Tests for entity linking module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

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
    """Tests for entity linking integration with EnrichmentPipeline."""

    def test_pipeline_links_entities(self, tmp_path):
        """Pipeline should link entities after extraction."""
        mock_chat = MagicMock()
        # Return entities in expected JSON format
        mock_chat.chat.return_value = """[
            {"name": "UserService", "type": "class", "description": "Handles users"},
            {"name": "OAuth2", "type": "api", "description": "Auth protocol"}
        ]"""

        config = EnrichmentConfig.from_dict(
            {
                "enabled": True,
                "entities": {
                    "enabled": True,
                    "types": ["class", "api"],
                },
            }
        )

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
                metadata={"file_path": "auth.py"},
            )
        ]

        result = pipeline.enrich(chunks)

        # Should have entities
        assert "entities" in result.chunks[0].metadata
        assert len(result.chunks[0].metadata["entities"]) == 2

        # Should have entity links
        assert "entity_links" in result.chunks[0].metadata
        links = result.chunks[0].metadata["entity_links"]
        assert len(links) == 1
        assert links[0]["source"] == "UserService"
        assert links[0]["target"] == "OAuth2"
