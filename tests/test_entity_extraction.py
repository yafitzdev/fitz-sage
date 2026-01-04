# tests/test_entity_extraction.py
"""Tests for entity extraction module."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from fitz_ai.core.chunk import Chunk
from fitz_ai.ingestion.enrichment import EnrichmentConfig, EnrichmentPipeline
from fitz_ai.ingestion.enrichment.entities import (
    ALL_ENTITY_TYPES,
    DOMAIN_ENTITY_TYPES,
    NAMED_ENTITY_TYPES,
    Entity,
    EntityCache,
    EntityExtractor,
)


class TestEntityModel:
    """Tests for Entity dataclass."""

    def test_entity_creation(self):
        entity = Entity(name="UserAuth", type="class", description="Handles authentication")
        assert entity.name == "UserAuth"
        assert entity.type == "class"
        assert entity.description == "Handles authentication"

    def test_entity_without_description(self):
        entity = Entity(name="OAuth2", type="api")
        assert entity.name == "OAuth2"
        assert entity.type == "api"
        assert entity.description is None

    def test_entity_to_dict(self):
        entity = Entity(name="process_data", type="function", description="Processes data")
        d = entity.to_dict()
        assert d["name"] == "process_data"
        assert d["type"] == "function"
        assert d["description"] == "Processes data"

    def test_entity_types_defined(self):
        assert "class" in DOMAIN_ENTITY_TYPES
        assert "function" in DOMAIN_ENTITY_TYPES
        assert "api" in DOMAIN_ENTITY_TYPES
        assert "person" in NAMED_ENTITY_TYPES
        assert "organization" in NAMED_ENTITY_TYPES
        assert ALL_ENTITY_TYPES == DOMAIN_ENTITY_TYPES | NAMED_ENTITY_TYPES


class TestEntityCache:
    """Tests for EntityCache."""

    def test_get_set_basic(self, tmp_path):
        cache_path = tmp_path / "entity_cache.json"
        cache = EntityCache(cache_path)

        # Initially empty
        assert cache.get("hash1", "extractor1") is None

        # Set and get
        entities = [Entity(name="Foo", type="class")]
        cache.set("hash1", "extractor1", entities)
        result = cache.get("hash1", "extractor1")
        assert result is not None
        assert len(result) == 1
        assert result[0].name == "Foo"

    def test_different_extractor_id(self, tmp_path):
        cache_path = tmp_path / "entity_cache.json"
        cache = EntityCache(cache_path)

        entities = [Entity(name="Bar", type="function")]
        cache.set("hash1", "extractor1", entities)

        # Different extractor_id should not match
        assert cache.get("hash1", "extractor2") is None

    def test_persistence(self, tmp_path):
        cache_path = tmp_path / "entity_cache.json"

        # Write to cache
        cache1 = EntityCache(cache_path)
        entities = [Entity(name="MyClass", type="class", description="A class")]
        cache1.set("hash1", "extractor1", entities)
        cache1.save()

        # Load from cache
        cache2 = EntityCache(cache_path)
        result = cache2.get("hash1", "extractor1")
        assert result is not None
        assert len(result) == 1
        assert result[0].name == "MyClass"
        assert result[0].description == "A class"

    def test_context_manager(self, tmp_path):
        cache_path = tmp_path / "entity_cache.json"

        with EntityCache(cache_path) as cache:
            entities = [Entity(name="Func", type="function")]
            cache.set("hash1", "extractor1", entities)

        # Should be saved on exit
        cache2 = EntityCache(cache_path)
        assert cache2.get("hash1", "extractor1") is not None

    def test_clear(self, tmp_path):
        cache_path = tmp_path / "entity_cache.json"
        cache = EntityCache(cache_path)

        cache.set("hash1", "ext1", [Entity(name="A", type="class")])
        cache.set("hash2", "ext1", [Entity(name="B", type="function")])
        assert len(cache) == 2

        cache.clear()
        assert len(cache) == 0


class TestEntityExtractor:
    """Tests for EntityExtractor."""

    def test_basic_extraction(self, tmp_path):
        cache = EntityCache(tmp_path / "cache.json")

        # Mock chat client that returns JSON entities
        mock_chat = MagicMock()
        mock_chat.chat.return_value = json.dumps(
            [
                {"name": "UserAuth", "type": "class", "description": "Auth handler"},
                {"name": "login", "type": "function", "description": "Logs in user"},
            ]
        )

        extractor = EntityExtractor(
            chat_client=mock_chat,
            cache=cache,
            extractor_id="test:v1",
        )

        entities = extractor.extract(
            content="class UserAuth:\n    def login(self): pass",
            file_path="/path/to/file.py",
            content_hash="abc123",
        )

        assert len(entities) == 2
        assert entities[0].name == "UserAuth"
        assert entities[0].type == "class"
        assert entities[1].name == "login"
        mock_chat.chat.assert_called_once()

    def test_cache_hit(self, tmp_path):
        cache = EntityCache(tmp_path / "cache.json")

        # Pre-populate cache
        cached_entities = [Entity(name="CachedClass", type="class")]
        cache.set("abc123", "test:v1", cached_entities)

        mock_chat = MagicMock()

        extractor = EntityExtractor(
            chat_client=mock_chat,
            cache=cache,
            extractor_id="test:v1",
        )

        entities = extractor.extract(
            content="class Something: pass",
            file_path="/path/to/file.py",
            content_hash="abc123",
        )

        assert len(entities) == 1
        assert entities[0].name == "CachedClass"
        mock_chat.chat.assert_not_called()

    def test_type_filtering(self, tmp_path):
        cache = EntityCache(tmp_path / "cache.json")

        mock_chat = MagicMock()
        mock_chat.chat.return_value = json.dumps(
            [
                {"name": "UserAuth", "type": "class", "description": "Auth"},
                {"name": "John", "type": "person", "description": "Developer"},
                {"name": "login", "type": "function", "description": "Login func"},
            ]
        )

        # Only extract class and function types
        extractor = EntityExtractor(
            chat_client=mock_chat,
            cache=cache,
            extractor_id="test:v1",
            entity_types=["class", "function"],
        )

        entities = extractor.extract(
            content="class UserAuth: ...",
            file_path="/path/to/file.py",
            content_hash="abc123",
        )

        assert len(entities) == 2
        entity_types = {e.type for e in entities}
        assert "person" not in entity_types
        assert "class" in entity_types
        assert "function" in entity_types

    def test_handles_markdown_response(self, tmp_path):
        cache = EntityCache(tmp_path / "cache.json")

        mock_chat = MagicMock()
        # LLM returns JSON wrapped in markdown code block
        mock_chat.chat.return_value = """```json
[{"name": "API", "type": "api", "description": "REST API"}]
```"""

        extractor = EntityExtractor(
            chat_client=mock_chat,
            cache=cache,
            extractor_id="test:v1",
        )

        entities = extractor.extract(
            content="# API endpoint",
            file_path="/path/to/readme.md",
            content_hash="def456",
        )

        assert len(entities) == 1
        assert entities[0].name == "API"

    def test_handles_invalid_json(self, tmp_path):
        cache = EntityCache(tmp_path / "cache.json")

        mock_chat = MagicMock()
        mock_chat.chat.return_value = "This is not valid JSON"

        extractor = EntityExtractor(
            chat_client=mock_chat,
            cache=cache,
            extractor_id="test:v1",
        )

        entities = extractor.extract(
            content="some content",
            file_path="/path/to/file.txt",
            content_hash="ghi789",
        )

        # Should return empty list on parse failure
        assert entities == []

    def test_code_vs_doc_prompt(self, tmp_path):
        cache = EntityCache(tmp_path / "cache.json")

        mock_chat = MagicMock()
        mock_chat.chat.return_value = "[]"

        extractor = EntityExtractor(
            chat_client=mock_chat,
            cache=cache,
            extractor_id="test:v1",
        )

        # Extract from Python file
        extractor.extract("class Foo: pass", "/path/file.py", "hash1")
        py_prompt = mock_chat.chat.call_args_list[0][0][0][0]["content"]

        # Extract from markdown file
        extractor.extract("# Title", "/path/file.md", "hash2")
        md_prompt = mock_chat.chat.call_args_list[1][0][0][0]["content"]

        # Prompts should be different (code vs doc)
        assert py_prompt != md_prompt


class TestEnrichmentPipelineWithEntities:
    """Tests for entity extraction integration in EnrichmentPipeline."""

    def test_entities_disabled_by_default(self, tmp_path):
        config = EnrichmentConfig(enabled=True)
        pipeline = EnrichmentPipeline(
            config=config,
            project_root=tmp_path,
            chat_client=MagicMock(),
        )

        assert not pipeline.entities_enabled

    def test_entities_enabled_with_config(self, tmp_path):
        config = EnrichmentConfig.from_dict(
            {
                "enabled": True,
                "entities": {"enabled": True},
            }
        )

        mock_chat = MagicMock()
        mock_chat.chat.return_value = "[]"

        pipeline = EnrichmentPipeline(
            config=config,
            project_root=tmp_path,
            chat_client=mock_chat,
        )

        assert pipeline.entities_enabled

    def test_entities_require_chat_client(self, tmp_path):
        config = EnrichmentConfig.from_dict(
            {
                "enabled": True,
                "entities": {"enabled": True},
            }
        )

        # No chat client provided
        pipeline = EnrichmentPipeline(
            config=config,
            project_root=tmp_path,
            chat_client=None,
        )

        assert not pipeline.entities_enabled

    def test_enrich_extracts_entities(self, tmp_path):
        config = EnrichmentConfig.from_dict(
            {
                "enabled": True,
                "entities": {"enabled": True},
            }
        )

        mock_chat = MagicMock()
        mock_chat.chat.return_value = json.dumps(
            [
                {"name": "TestClass", "type": "class", "description": "A test class"},
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
                metadata={"file_path": "/path/to/file.py"},
            ),
        ]

        result = pipeline.enrich(chunks)

        assert len(result.chunks) == 1
        chunk = result.chunks[0]
        assert "entities" in chunk.metadata
        entities = chunk.metadata["entities"]
        assert len(entities) == 1
        assert entities[0]["name"] == "TestClass"
        assert entities[0]["type"] == "class"

    def test_enrich_with_type_filter(self, tmp_path):
        config = EnrichmentConfig.from_dict(
            {
                "enabled": True,
                "entities": {
                    "enabled": True,
                    "types": ["class", "function"],  # Only extract these
                },
            }
        )

        mock_chat = MagicMock()
        mock_chat.chat.return_value = json.dumps(
            [
                {"name": "MyClass", "type": "class", "description": "A class"},
                {"name": "John", "type": "person", "description": "A person"},
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
                content="class MyClass: ...",
                metadata={"file_path": "/path/to/file.py"},
            ),
        ]

        result = pipeline.enrich(chunks)

        chunk = result.chunks[0]
        entities = chunk.metadata["entities"]
        # Should only have class, not person
        assert len(entities) == 1
        assert entities[0]["type"] == "class"
