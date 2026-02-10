# tests/unit/vector_db/test_loader.py
"""Tests for vector_db loader, VectorDBSpec, and GenericVectorDBPlugin."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
import yaml

pytestmark = pytest.mark.tier1

from fitz_ai.vector_db.base import SearchResult
from fitz_ai.vector_db.loader import (
    GenericVectorDBPlugin,
    VectorDBSpec,
    _string_to_uuid,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MINIMAL_SPEC = {
    "name": "test_db",
    "type": "vector_db",
    "description": "Test vector DB",
    "connection": {
        "type": "http",
        "base_url": "http://{{host}}:{{port}}",
        "default_host": "localhost",
        "default_port": 6333,
    },
    "operations": {
        "search": {
            "method": "POST",
            "endpoint": "/collections/{{collection}}/points/search",
            "body": {"vector": "{{query_vector}}", "limit": "{{limit}}"},
            "response": {
                "results_path": "result",
                "mapping": {"id": "id", "score": "score", "payload": "payload"},
            },
        },
        "upsert": {
            "method": "PUT",
            "endpoint": "/collections/{{collection}}/points",
            "body": {"points": "{{points}}"},
        },
        "count": {
            "method": "POST",
            "endpoint": "/collections/{{collection}}/points/count",
            "body": {},
            "response": {"count_path": "result.count"},
        },
        "create_collection": {
            "method": "PUT",
            "endpoint": "/collections/{{collection}}",
            "body": {"vectors": {"size": "{{vector_size}}", "distance": "Cosine"}},
        },
        "delete_collection": {
            "method": "DELETE",
            "endpoint": "/collections/{{collection}}",
            "response": {"success_codes": [200]},
        },
        "list_collections": {
            "method": "GET",
            "endpoint": "/collections",
            "response": {"collections_path": "result.collections", "name_field": "name"},
        },
    },
    "features": {
        "requires_uuid_ids": False,
        "auto_detect": None,
        "supports_namespaces": False,
    },
}

_UUID_SPEC = {**_MINIMAL_SPEC, "features": {**_MINIMAL_SPEC["features"], "requires_uuid_ids": True}}


@pytest.fixture
def spec_path(tmp_path):
    """Write minimal spec YAML and return path."""
    p = tmp_path / "test_db.yaml"
    p.write_text(yaml.dump(_MINIMAL_SPEC))
    return p


@pytest.fixture
def uuid_spec_path(tmp_path):
    """Write UUID-requiring spec YAML and return path."""
    p = tmp_path / "uuid_db.yaml"
    p.write_text(yaml.dump(_UUID_SPEC))
    return p


@pytest.fixture
def spec(spec_path):
    return VectorDBSpec(spec_path)


@pytest.fixture
def uuid_spec(uuid_spec_path):
    return VectorDBSpec(uuid_spec_path)


# ---------------------------------------------------------------------------
# _string_to_uuid
# ---------------------------------------------------------------------------

class TestStringToUuid:
    def test_deterministic(self):
        assert _string_to_uuid("hello") == _string_to_uuid("hello")

    def test_different_inputs_different_uuids(self):
        assert _string_to_uuid("a") != _string_to_uuid("b")

    def test_returns_valid_uuid(self):
        result = _string_to_uuid("test")
        uuid.UUID(result)  # should not raise


# ---------------------------------------------------------------------------
# VectorDBSpec
# ---------------------------------------------------------------------------

class TestVectorDBSpec:
    def test_name_and_type(self, spec):
        assert spec.name == "test_db"
        assert spec.type == "vector_db"

    def test_is_local_false_for_http(self, spec):
        assert not spec.is_local()

    def test_is_local_true(self, tmp_path):
        local_spec = {**_MINIMAL_SPEC, "connection": {"type": "local"}, "operations": {"python_class": "a.b.C"}}
        p = tmp_path / "local.yaml"
        p.write_text(yaml.dump(local_spec))
        s = VectorDBSpec(p)
        assert s.is_local()

    def test_get_local_class_path(self, tmp_path):
        local_spec = {**_MINIMAL_SPEC, "connection": {"type": "local"}, "operations": {"python_class": "my.module.MyClass"}}
        p = tmp_path / "local.yaml"
        p.write_text(yaml.dump(local_spec))
        s = VectorDBSpec(p)
        assert s.get_local_class_path() == "my.module.MyClass"

    def test_requires_uuid_ids(self, spec, uuid_spec):
        assert not spec.requires_uuid_ids()
        assert uuid_spec.requires_uuid_ids()

    def test_supports_namespaces(self, spec):
        assert not spec.supports_namespaces()

    def test_get_auto_detect_service(self, spec):
        assert spec.get_auto_detect_service() is None

    def test_build_base_url_defaults(self, spec):
        url = spec.build_base_url()
        assert url == "http://localhost:6333"

    def test_build_base_url_overrides(self, spec):
        url = spec.build_base_url(host="myhost", port=9999)
        assert url == "http://myhost:9999"

    def test_get_auth_headers_no_auth(self, spec):
        assert spec.get_auth_headers() == {}

    def test_get_auth_headers_bearer(self, tmp_path, monkeypatch):
        auth_spec = {
            **_MINIMAL_SPEC,
            "connection": {
                **_MINIMAL_SPEC["connection"],
                "auth": {"type": "bearer", "env_var": "TEST_API_KEY", "scheme": "Bearer"},
            },
        }
        p = tmp_path / "auth.yaml"
        p.write_text(yaml.dump(auth_spec))
        monkeypatch.setenv("TEST_API_KEY", "secret123")
        s = VectorDBSpec(p)
        headers = s.get_auth_headers()
        assert headers == {"Authorization": "Bearer secret123"}

    def test_get_auth_headers_missing_env_raises(self, tmp_path, monkeypatch):
        auth_spec = {
            **_MINIMAL_SPEC,
            "connection": {
                **_MINIMAL_SPEC["connection"],
                "auth": {"type": "bearer", "env_var": "MISSING_KEY"},
            },
        }
        p = tmp_path / "auth.yaml"
        p.write_text(yaml.dump(auth_spec))
        monkeypatch.delenv("MISSING_KEY", raising=False)
        s = VectorDBSpec(p)
        with pytest.raises(ValueError, match="MISSING_KEY not set"):
            s.get_auth_headers()

    def test_get_auth_headers_optional_missing(self, tmp_path, monkeypatch):
        auth_spec = {
            **_MINIMAL_SPEC,
            "connection": {
                **_MINIMAL_SPEC["connection"],
                "auth": {"type": "bearer", "env_var": "OPT_KEY", "optional": True},
            },
        }
        p = tmp_path / "auth.yaml"
        p.write_text(yaml.dump(auth_spec))
        monkeypatch.delenv("OPT_KEY", raising=False)
        s = VectorDBSpec(p)
        assert s.get_auth_headers() == {}

    def test_render_template_plain_string(self, spec):
        assert spec.render_template("hello", {}) == "hello"

    def test_render_template_jinja(self, spec):
        result = spec.render_template("{{name}}", {"name": "world"})
        assert result == "world"

    def test_render_template_direct_variable(self, spec):
        """Direct {{var}} substitution preserves type (not stringified)."""
        result = spec.render_template("{{ items }}", {"items": [1, 2, 3]})
        assert result == [1, 2, 3]

    def test_render_template_dict(self, spec):
        template = {"key": "{{val}}"}
        result = spec.render_template(template, {"val": "hello"})
        assert result == {"key": "hello"}

    def test_render_template_list(self, spec):
        template = ["{{a}}", "{{b}}"]
        result = spec.render_template(template, {"a": "x", "b": "y"})
        assert result == ["x", "y"]

    def test_render_template_passthrough(self, spec):
        assert spec.render_template(42, {}) == 42

    def test_transform_points_identity(self, spec):
        points = [{"id": "1", "vector": [0.1], "payload": {}}]
        result = spec.transform_points(points, "nonexistent_op")
        assert result == points

    def test_transform_points_mapping(self, tmp_path):
        mapped_spec = {
            **_MINIMAL_SPEC,
            "operations": {
                **_MINIMAL_SPEC["operations"],
                "upsert": {
                    "method": "PUT",
                    "endpoint": "/points",
                    "point_transform": {"_id": "id", "embedding": "vector"},
                },
            },
        }
        p = tmp_path / "mapped.yaml"
        p.write_text(yaml.dump(mapped_spec))
        s = VectorDBSpec(p)
        points = [{"id": "abc", "vector": [0.1], "payload": {"k": "v"}}]
        result = s.transform_points(points, "upsert")
        assert result[0]["_id"] == "abc"
        assert result[0]["embedding"] == [0.1]
        assert "id" not in result[0]


# ---------------------------------------------------------------------------
# GenericVectorDBPlugin
# ---------------------------------------------------------------------------

class TestGenericVectorDBPlugin:
    @pytest.fixture
    def plugin(self, spec):
        """Create plugin with mocked HTTP client."""
        plugin = GenericVectorDBPlugin(spec)
        plugin.client = MagicMock(spec=httpx.Client)
        return plugin

    def test_search_returns_results(self, plugin):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": [
                {"id": "p1", "score": 0.95, "payload": {"content": "hello"}},
                {"id": "p2", "score": 0.80, "payload": {"content": "world"}},
            ]
        }
        plugin.client.request.return_value = mock_response

        results = plugin.search("my_col", [0.1, 0.2], limit=10)

        assert len(results) == 2
        assert isinstance(results[0], SearchResult)
        assert results[0].id == "p1"
        assert results[0].score == 0.95
        assert results[0].payload == {"content": "hello"}

    def test_search_empty_results(self, plugin):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": []}
        plugin.client.request.return_value = mock_response

        results = plugin.search("col", [0.1], limit=5)
        assert results == []

    def test_search_not_supported_raises(self, tmp_path):
        no_search_spec = {**_MINIMAL_SPEC, "operations": {}}
        p = tmp_path / "no_search.yaml"
        p.write_text(yaml.dump(no_search_spec))
        s = VectorDBSpec(p)
        plugin = GenericVectorDBPlugin(s)
        plugin.client = MagicMock()
        with pytest.raises(NotImplementedError):
            plugin.search("col", [0.1], 5)

    def test_upsert_sends_request(self, plugin):
        mock_response = MagicMock()
        mock_response.status_code = 200
        plugin.client.request.return_value = mock_response

        points = [{"id": "p1", "vector": [0.1, 0.2], "payload": {"content": "hi"}}]
        plugin.upsert("my_col", points)

        plugin.client.request.assert_called_once()
        call = plugin.client.request.call_args
        assert call.kwargs["method"] == "PUT"

    def test_upsert_auto_create_on_404(self, tmp_path):
        auto_spec = {
            **_MINIMAL_SPEC,
            "operations": {
                **_MINIMAL_SPEC["operations"],
                "upsert": {
                    **_MINIMAL_SPEC["operations"]["upsert"],
                    "auto_create_collection": True,
                    "create_collection_endpoint": "/collections/{{collection}}",
                    "create_collection_method": "PUT",
                    "create_collection_body": {},
                },
            },
        }
        p = tmp_path / "auto.yaml"
        p.write_text(yaml.dump(auto_spec))
        s = VectorDBSpec(p)
        plugin = GenericVectorDBPlugin(s)
        plugin.client = MagicMock(spec=httpx.Client)

        # First call returns 404, second succeeds (after auto-create)
        resp_404 = MagicMock(status_code=404)
        resp_200 = MagicMock(status_code=200)
        resp_create = MagicMock(status_code=201)
        plugin.client.request.side_effect = [resp_404, resp_create, resp_200]

        points = [{"id": "p1", "vector": [0.1]}]
        plugin.upsert("new_col", points)

        assert plugin.client.request.call_count == 3

    def test_count_returns_int(self, plugin):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": {"count": 42}}
        plugin.client.request.return_value = mock_response

        count = plugin.count("my_col")
        assert count == 42

    def test_create_collection(self, plugin):
        mock_response = MagicMock()
        mock_response.status_code = 200
        plugin.client.request.return_value = mock_response

        plugin.create_collection("new_col", vector_size=768)
        plugin.client.request.assert_called_once()

    def test_delete_collection(self, plugin):
        mock_response = MagicMock()
        mock_response.status_code = 200
        plugin.client.request.return_value = mock_response

        plugin.delete_collection("old_col")
        plugin.client.request.assert_called_once()

    def test_list_collections(self, plugin):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": {"collections": [{"name": "col1"}, {"name": "col2"}]}
        }
        plugin.client.get.return_value = mock_response

        result = plugin.list_collections()
        assert result == ["col1", "col2"]

    def test_convert_point_ids_no_uuid(self, plugin):
        points = [{"id": "abc", "vector": [0.1]}]
        result = plugin._convert_point_ids(points)
        assert result[0]["id"] == "abc"

    def test_convert_point_ids_with_uuid_spec(self, uuid_spec):
        plugin = GenericVectorDBPlugin(uuid_spec)
        plugin.client = MagicMock()

        points = [{"id": "not-a-uuid", "vector": [0.1]}]
        result = plugin._convert_point_ids(points)

        assert result[0]["id"] != "not-a-uuid"
        uuid.UUID(result[0]["id"])  # should be valid UUID
        assert result[0]["payload"]["_original_id"] == "not-a-uuid"

    def test_convert_point_ids_keeps_existing_uuid(self, uuid_spec):
        plugin = GenericVectorDBPlugin(uuid_spec)
        plugin.client = MagicMock()

        valid_uuid = str(uuid.uuid4())
        points = [{"id": valid_uuid, "vector": [0.1]}]
        result = plugin._convert_point_ids(points)
        assert result[0]["id"] == valid_uuid

    def test_retrieve_returns_points(self, plugin):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": [
                {"id": "p1", "payload": {"content": "hello"}},
            ]
        }
        # Add retrieve operation to spec
        plugin.spec.operations["retrieve"] = {
            "method": "POST",
            "endpoint": "/collections/{{collection}}/points",
            "body": {"ids": "{{ids}}"},
            "response": {
                "results_path": "result",
                "mapping": {"id": "id", "payload": "payload"},
            },
        }
        plugin.client.request.return_value = mock_response

        results = plugin.retrieve("col", ["p1"])
        assert len(results) == 1
        assert results[0]["id"] == "p1"

    def test_retrieve_empty_ids(self, plugin):
        plugin.spec.operations["retrieve"] = {
            "method": "POST",
            "endpoint": "/points",
            "response": {"results_path": "result", "mapping": {"id": "id", "payload": "payload"}},
        }
        assert plugin.retrieve("col", []) == []
