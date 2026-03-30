# tests/unit/vector_db/test_custom_plugin.py
"""Tests for CustomVectorDB plugin."""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest

pytestmark = pytest.mark.tier1

from fitz_sage.vector_db.base import SearchResult
from fitz_sage.vector_db.custom import CustomVectorDB, _extract_path, _substitute_vars

# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


class TestSubstituteVars:
    def test_simple_substitution(self):
        result = _substitute_vars("/items/{collection}", {"collection": "my_col"})
        assert result == "/items/my_col"

    def test_list_serialization(self):
        result = _substitute_vars("{items}", {"items": [1, 2, 3]})
        assert result == "[1, 2, 3]"

    def test_dict_serialization(self):
        result = _substitute_vars("{data}", {"data": {"key": "val"}})
        assert '"key"' in result and '"val"' in result

    def test_missing_var_kept(self):
        result = _substitute_vars("{missing}", {})
        assert result == "{missing}"

    def test_multiple_vars(self):
        result = _substitute_vars("{a}/{b}", {"a": "x", "b": "y"})
        assert result == "x/y"


class TestExtractPath:
    def test_simple_key(self):
        assert _extract_path({"a": 1}, "a") == 1

    def test_nested_key(self):
        assert _extract_path({"a": {"b": 2}}, "a.b") == 2

    def test_missing_key_returns_default(self):
        assert _extract_path({"a": 1}, "b", "fallback") == "fallback"

    def test_list_index(self):
        assert _extract_path({"items": [10, 20, 30]}, "items.1") == 20

    def test_empty_path_returns_data(self):
        data = {"x": 1}
        assert _extract_path(data, "") == data

    def test_deeply_nested(self):
        data = {"a": {"b": {"c": {"d": 42}}}}
        assert _extract_path(data, "a.b.c.d") == 42

    def test_list_index_out_of_range(self):
        assert _extract_path([1, 2], "5", "default") == "default"


# ---------------------------------------------------------------------------
# CustomVectorDB
# ---------------------------------------------------------------------------


def _make_custom_db(**overrides) -> CustomVectorDB:
    """Create CustomVectorDB with mocked HTTP client."""
    kwargs = {
        "base_url": "http://localhost:8000",
        "upsert": {
            "method": "POST",
            "endpoint": "/collections/{collection}/points",
            "body": '{"points": {points}}',
        },
        "search": {
            "method": "POST",
            "endpoint": "/collections/{collection}/search",
            "body": '{"vector": {query_vector}, "limit": {limit}}',
            "results_path": "results",
            "mapping": {"id": "id", "score": "score", "payload": "metadata"},
        },
        **overrides,
    }
    db = CustomVectorDB(**kwargs)
    db.client = MagicMock(spec=httpx.Client)
    return db


class TestCustomVectorDBInit:
    def test_requires_base_url(self):
        with pytest.raises(ValueError, match="base_url"):
            CustomVectorDB(upsert={}, search={})

    def test_requires_upsert(self):
        with pytest.raises(ValueError, match="upsert"):
            CustomVectorDB(base_url="http://localhost", search={})

    def test_requires_search(self):
        with pytest.raises(ValueError, match="search"):
            CustomVectorDB(
                base_url="http://localhost", upsert={"method": "POST", "endpoint": "/up"}
            )

    def test_auth_from_env(self, monkeypatch):
        monkeypatch.setenv("MY_API_KEY", "secret")
        db = CustomVectorDB(
            base_url="http://localhost",
            upsert={"method": "POST", "endpoint": "/upsert"},
            search={"method": "POST", "endpoint": "/search"},
            auth={"header": "X-Api-Key", "value_env": "MY_API_KEY"},
        )
        # Client was created with headers
        assert db.client is not None

    def test_auth_missing_env_raises(self, monkeypatch):
        monkeypatch.delenv("MISSING_KEY", raising=False)
        with pytest.raises(ValueError, match="MISSING_KEY"):
            CustomVectorDB(
                base_url="http://localhost",
                upsert={"method": "POST", "endpoint": "/upsert"},
                search={"method": "POST", "endpoint": "/search"},
                auth={"value_env": "MISSING_KEY"},
            )


class TestCustomVectorDBSearch:
    def test_returns_search_results(self):
        db = _make_custom_db()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"id": "p1", "score": 0.9, "metadata": {"content": "hello"}},
                {"id": "p2", "score": 0.7, "metadata": {"content": "world"}},
            ]
        }
        db.client.request.return_value = mock_response

        results = db.search("col", [0.1, 0.2], limit=5)

        assert len(results) == 2
        assert isinstance(results[0], SearchResult)
        assert results[0].id == "p1"
        assert results[0].score == 0.9
        assert results[0].payload == {"content": "hello"}

    def test_empty_results(self):
        db = _make_custom_db()
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        db.client.request.return_value = mock_response

        assert db.search("col", [0.1], limit=5) == []


class TestCustomVectorDBUpsert:
    def test_sends_points(self):
        db = _make_custom_db()
        mock_response = MagicMock(status_code=200)
        db.client.request.return_value = mock_response

        points = [{"id": "p1", "vector": [0.1], "payload": {}}]
        db.upsert("col", points)

        db.client.request.assert_called_once()

    def test_point_transform(self):
        db = _make_custom_db(
            upsert={
                "method": "POST",
                "endpoint": "/points",
                "body": '{"data": {points}}',
                "point_transform": {"_id": "id", "embedding": "vector"},
            }
        )
        mock_response = MagicMock(status_code=200)
        db.client.request.return_value = mock_response

        points = [{"id": "p1", "vector": [0.1], "payload": {}}]
        db.upsert("col", points)
        db.client.request.assert_called_once()


class TestCustomVectorDBOptionalOps:
    def test_list_collections_not_configured(self):
        db = _make_custom_db()
        with pytest.raises(NotImplementedError, match="list_collections not configured"):
            db.list_collections()

    def test_get_stats_not_configured(self):
        db = _make_custom_db()
        with pytest.raises(NotImplementedError, match="get_collection_stats not configured"):
            db.get_collection_stats("col")

    def test_delete_not_configured(self):
        db = _make_custom_db()
        with pytest.raises(NotImplementedError, match="delete_collection not configured"):
            db.delete_collection("col")

    def test_list_collections_configured(self):
        db = _make_custom_db(
            list_collections={
                "method": "GET",
                "endpoint": "/collections",
                "collections_path": "data",
                "name_field": "name",
            }
        )
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"name": "a"}, {"name": "b"}]}
        db.client.request.return_value = mock_response

        result = db.list_collections()
        assert result == ["a", "b"]

    def test_delete_collection_accepts_404(self):
        db = _make_custom_db(
            delete_collection={"method": "DELETE", "endpoint": "/collections/{collection}"}
        )
        mock_response = MagicMock(status_code=404)
        db.client.request.return_value = mock_response

        db.delete_collection("gone_col")  # should not raise
