# tests/e2e/debug_retrieval.py
"""Debug script to investigate retrieval failures."""

import uuid
from pathlib import Path

from fitz_ai.cli.context import CLIContext
from fitz_ai.engines.fitz_rag.config.schema import (
    ChunkingRouterConfig,
    ExtensionChunkerConfig,
)
from fitz_ai.ingestion.chunking.router import ChunkingRouter
from fitz_ai.ingestion.diff import run_diff_ingest
from fitz_ai.ingestion.parser import ParserRouter
from fitz_ai.ingestion.state import IngestStateManager
from fitz_ai.llm.registry import get_llm_plugin
from fitz_ai.vector_db.registry import get_vector_db_plugin

FIXTURES_DIR = Path(__file__).parent / "fixtures_rag"


def setup_collection():
    """Set up a debug collection with fixtures."""
    ctx = CLIContext.load()
    config = ctx.raw_config

    embedding_plugin = config.get("embedding", {}).get("plugin_name", "openai")
    embedding_kwargs = config.get("embedding", {}).get("kwargs", {})
    vector_db_plugin_name = config.get("vector_db", {}).get("plugin_name", "pgvector")
    vector_db_kwargs = config.get("vector_db", {}).get("kwargs", {})

    collection = f"e2e_debug_{uuid.uuid4().hex[:8]}"
    print(f"Creating debug collection: {collection}")

    vector_client = get_vector_db_plugin(vector_db_plugin_name, **vector_db_kwargs)
    embedder = get_llm_plugin(
        plugin_type="embedding",
        plugin_name=embedding_plugin,
        **embedding_kwargs,
    )

    # Parser and chunking
    parser_router = ParserRouter(docling_parser="docling")
    router_config = ChunkingRouterConfig(
        default=ExtensionChunkerConfig(
            plugin_name="simple",
            # Larger chunks to keep product sections/facts together
            kwargs={"chunk_size": 2000, "chunk_overlap": 200},
        ),
        by_extension={},
    )
    chunking_router = ChunkingRouter.from_config(router_config)

    # State manager
    state_manager = IngestStateManager()
    state_manager.load()

    # Writer adapter
    class VectorDBWriterAdapter:
        def __init__(self, client):
            self._client = client

        def upsert(self, collection, points, defer_persist=False):
            self._client.upsert(collection, points, defer_persist=defer_persist)

        def flush(self):
            if hasattr(self._client, "flush"):
                self._client.flush()

    writer = VectorDBWriterAdapter(vector_client)

    # Run ingestion (no enrichment for speed)
    print(f"Ingesting fixtures from: {FIXTURES_DIR}")
    summary = run_diff_ingest(
        source=str(FIXTURES_DIR),
        state_manager=state_manager,
        vector_db_writer=writer,
        embedder=embedder,
        parser_router=parser_router,
        chunking_router=chunking_router,
        collection=collection,
        embedding_id=embedding_plugin,
        vector_db_id=vector_db_plugin_name,
        enrichment_pipeline=None,  # Skip enrichment for debug
        force=True,
    )

    print(f"Ingested {summary.ingested} files")
    return collection, vector_client, embedder


def debug_retrieval():
    """Debug what's being retrieved for failing queries."""
    collection, vector_client, embedder = setup_collection()

    # Check collection size
    count = vector_client.count(collection)
    print(f"Collection has {count} vectors")

    # Dump all chunks to see what's stored
    print("\n" + "=" * 70)
    print("ALL CHUNKS IN COLLECTION")
    print("=" * 70)

    # Get all vectors by doing a random search with high limit
    dummy_embedding = embedder.embed("test query")
    all_results = vector_client.search(
        collection_name=collection,
        query_vector=dummy_embedding,
        limit=count,
    )

    # Count chunks with key terms
    price_chunks = []
    y200_chunks = []
    austin_chunks = []

    for i, result in enumerate(all_results, 1):
        payload = result.payload or {}
        content = payload.get("content", "")
        source = payload.get("source_file", "unknown")

        # Check for key terms
        has_austin = "austin" in content.lower()
        has_price = "$55" in content or "55,000" in content
        has_y200 = "y200" in content.lower()

        if has_price:
            price_chunks.append(i)
        if has_y200:
            y200_chunks.append(i)
        if has_austin:
            austin_chunks.append(i)

        print(f"\n[Chunk {i}] source={source[:60] if source else 'unknown'}")
        print(f"  austin={has_austin}, price={has_price}, y200={has_y200}, len={len(content)}")
        print(f"  Content: {content[:150]}...")

    print("\n\nSUMMARY:")
    print(f"  Chunks with price: {price_chunks}")
    print(f"  Chunks with y200: {y200_chunks}")
    print(f"  Chunks with austin: {austin_chunks}")

    # Failing queries to debug
    failing_queries = [
        ("E20", "Where is TechCorp Industries headquartered?"),
        ("E21", "What is the price of the Model Y200?"),
        ("E05", "Compare the Model X100 vs Model Y200"),
        ("E06", "What is the price difference between Model X100 and Model Z50?"),
        ("E07", "What are the battery capacities and ranges for all TechCorp vehicle models?"),
    ]

    print("\n" + "=" * 70)
    print("DEBUGGING RETRIEVAL FOR FAILING QUERIES")
    print("=" * 70)

    for query_id, query in failing_queries:
        print(f"\n--- {query_id}: {query} ---")

        # Get embedding
        query_embedding = embedder.embed(query)

        # Search
        results = vector_client.search(
            collection_name=collection,
            query_vector=query_embedding,
            limit=5,
        )

        print(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            score = result.score or 0
            payload = result.payload or {}
            content = payload.get("content", "")[:150]
            source = payload.get("source_file", "unknown")
            print(f"  [{i}] score={score:.3f} source={source}")
            print(f"      {content}...")

        print()

    # Also check what content is actually in the collection
    print("\n" + "=" * 70)
    print("CHECKING FOR EXPECTED CONTENT")
    print("=" * 70)

    # Search for specific terms that should be in the collection
    search_terms = [
        ("Austin", "Should find TechCorp HQ"),
        ("$55,000", "Should find Model Y200 price"),
        ("55,000", "Should find Model Y200 price (no $)"),
        ("Price: $55", "Should find Model Y200 price line"),
        ("Model Y200", "Should find Y200 section"),
        ("Model X100", "Should find product specs"),
        ("75 kWh", "Should find battery specs"),
    ]

    for term, description in search_terms:
        print(f"\nSearching for '{term}' ({description}):")
        query_embedding = embedder.embed(term)
        results = vector_client.search(
            collection_name=collection,
            query_vector=query_embedding,
            limit=3,
        )

        for i, result in enumerate(results, 1):
            score = result.score or 0
            payload = result.payload or {}
            content = payload.get("content", "")
            has_term = term.lower() in content.lower()
            print(f"  [{i}] score={score:.3f} contains_term={has_term}")
            if has_term:
                # Show context around the term
                idx = content.lower().find(term.lower())
                snippet = content[max(0, idx - 50) : idx + len(term) + 50]
                print(f"      ...{snippet}...")

    # Cleanup
    print("\n" + "=" * 70)
    print("CLEANUP")
    print("=" * 70)
    try:
        deleted = vector_client.delete_collection(collection)
        print(f"Deleted collection {collection} ({deleted} vectors)")
    except Exception as e:
        print(f"Failed to cleanup: {e}")


if __name__ == "__main__":
    debug_retrieval()
