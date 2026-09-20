from __future__ import annotations

from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
from fitz_sage.engines.fitz_krag.ingestion.pipeline import KragIngestPipeline
from fitz_sage.engines.fitz_krag.ingestion.raw_file_store import RawFileStore
from fitz_sage.engines.fitz_krag.ingestion.schema import ensure_schema
from fitz_sage.engines.fitz_krag.semantic_index import SemanticIndex
from fitz_sage.storage.config import StorageConfig
from fitz_sage.storage.sqlite import SqliteConnectionManager


def _index(tmp_path):
    manager = SqliteConnectionManager(StorageConfig(storage_path=tmp_path / "sqlite"))
    collection = "semantic_test"
    ensure_schema(manager, collection)
    return manager, RawFileStore(manager, collection), SemanticIndex(manager, collection)


def _store(raw_store, semantic_index, file_id: str, content: str) -> None:
    raw_store.upsert(
        file_id=file_id,
        path=f"{file_id}.md",
        content=content,
        content_hash=file_id,
        file_type=".md",
        size_bytes=len(content.encode()),
    )
    semantic_index.index_file(file_id, content)


def test_collects_bidirectional_abbreviation_forms(tmp_path):
    _manager, raw_store, semantic_index = _index(tmp_path)
    _store(
        raw_store,
        semantic_index,
        "sla",
        "A Service Level Agreement (SLA) defines the expected uptime. "
        "The SLA also specifies response times.",
    )

    short_query = semantic_index.expand("What uptime does the SLA promise?")
    long_query = semantic_index.expand("Show the Service Level Agreement")

    assert any(item.term == "Service Level Agreement" for item in short_query)
    assert any(item.term == "SLA" for item in long_query)
    assert semantic_index.stats()["forms"] >= 3


def test_ingestion_pipeline_populates_the_query_expansion_graph(tmp_path):
    manager = SqliteConnectionManager(StorageConfig(storage_path=tmp_path / "sqlite"))
    collection = "semantic_pipeline"
    semantic_index = SemanticIndex(manager, collection)
    pipeline = KragIngestPipeline(
        FitzKragConfig(collection=collection),
        chat=None,
        connection_manager=manager,
        collection=collection,
        semantic_index=semantic_index,
    )
    source = tmp_path / "operations.md"
    source.write_text(
        "# Operations\n\nA Service Level Agreement (SLA) defines response time.",
        encoding="utf-8",
    )

    counts = pipeline.parse_file("operations.md", source, "operations")

    assert counts["sections"] == 1
    assert any(
        item.term == "Service Level Agreement"
        for item in semantic_index.expand("What does the SLA require?")
    )


def test_builds_contextual_error_cluster(tmp_path):
    _manager, raw_store, semantic_index = _index(tmp_path)
    _store(
        raw_store,
        semantic_index,
        "installer-1",
        "Windows Installer packages use the MSI format. "
        "MSI installation can fail with error 1722 when a custom action fails.",
    )
    _store(
        raw_store,
        semantic_index,
        "installer-2",
        "Error 1722 is reported by Windows Installer when the custom action stops.",
    )

    expansions = semantic_index.expand("Why does MSI report error 1722?", max_terms=10)
    normalized = {item.term.casefold() for item in expansions}

    assert "windows installer" in normalized
    assert "custom action" in normalized
    assert "installation fail" not in normalized
    assert semantic_index.stats()["clusters"] >= 1


def test_expands_identifier_spelling_variants_in_both_directions(tmp_path):
    _manager, raw_store, semantic_index = _index(tmp_path)
    _store(
        raw_store,
        semantic_index,
        "auth",
        "AuthService processes session tokens.",
    )

    spaced = {item.term for item in semantic_index.expand("AuthService")}
    compact = {item.term for item in semantic_index.expand("Auth Service")}

    assert "Auth Service" in spaced
    assert "AuthService" in compact


def test_reindex_removes_stale_semantic_evidence(tmp_path):
    _manager, raw_store, semantic_index = _index(tmp_path)
    _store(
        raw_store,
        semantic_index,
        "policy",
        "A Recovery Time Objective (RTO) defines acceptable downtime.",
    )
    assert any(item.term == "Recovery Time Objective" for item in semantic_index.expand("RTO"))

    _store(
        raw_store,
        semantic_index,
        "policy",
        "This replacement document only discusses backup retention.",
    )

    assert semantic_index.expand("RTO") == []


def test_remove_file_retracts_relationships(tmp_path):
    _manager, raw_store, semantic_index = _index(tmp_path)
    _store(
        raw_store,
        semantic_index,
        "policy",
        "A Recovery Point Objective (RPO) defines tolerable data loss.",
    )
    assert semantic_index.expand("RPO")

    semantic_index.remove_file("policy")
    raw_store.delete("policy")

    assert semantic_index.expand("RPO") == []
