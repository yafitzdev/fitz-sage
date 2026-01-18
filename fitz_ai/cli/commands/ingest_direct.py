# fitz_ai/cli/commands/ingest_direct.py
"""
Direct text ingestion for ingest command.

Handles ingesting raw text strings directly into the vector database.
"""

from __future__ import annotations

import hashlib

from fitz_ai.cli.context import CLIContext
from fitz_ai.cli.ui import ui


def ingest_direct_text(
    text: str,
    collection: str,
    ctx: CLIContext,
    config: dict,
) -> None:
    """
    Ingest direct text into the vector database.

    Args:
        text: The text to ingest
        collection: Collection name
        ctx: CLI context
        config: Raw config dict
    """
    from fitz_ai.core.chunk import Chunk
    from fitz_ai.core.document import DocumentElement, ElementType, ParsedDocument
    from fitz_ai.ingestion.chunking.plugins.default.simple import SimpleChunker
    from fitz_ai.llm.registry import get_llm_plugin
    from fitz_ai.vector_db.registry import get_vector_db_plugin

    ui.header("Fitz Ingest", "Direct text ingestion")
    ui.info(f"Collection: {collection}")

    # Generate a unique ID for this text
    text_hash = hashlib.sha256(text.encode()).hexdigest()[:12]
    doc_id = f"text_{text_hash}"

    # Show preview
    preview = text[:100] + "..." if len(text) > 100 else text
    ui.info(f"Text: {preview}")
    print()

    # Get components
    embedding_kwargs = config.get("embedding", {}).get("kwargs", {})
    embedder = get_llm_plugin(
        plugin_type="embedding",
        plugin_name=ctx.embedding_plugin,
        **embedding_kwargs,
    )

    vector_client = get_vector_db_plugin(ctx.vector_db_plugin)

    # Create a ParsedDocument from the text
    document = ParsedDocument(
        source="direct_input",
        elements=[DocumentElement(type=ElementType.TEXT, content=text)],
        metadata={"doc_id": doc_id, "source_type": "direct_text"},
    )

    # Chunk the text (use simple chunker)
    chunker = SimpleChunker(chunk_size=1000, chunk_overlap=100)
    chunks = chunker.chunk(document)

    # If text is small enough, it might be a single chunk or empty
    if not chunks:
        # Text was too small or empty - create a single chunk manually
        chunks = [
            Chunk(
                id=f"{doc_id}:0",
                doc_id=doc_id,
                chunk_index=0,
                content=text,
                metadata={"source_file": "direct_input", "doc_id": doc_id},
            )
        ]

    ui.step(1, 3, f"Created {len(chunks)} chunk(s)")

    # Embed chunks
    ui.step(2, 3, "Embedding...")
    chunk_texts = [chunk.content for chunk in chunks]
    embeddings = embedder.embed(chunk_texts)

    # Build points for vector DB
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point_id = f"{doc_id}_chunk_{i}"
        points.append({
            "id": point_id,
            "vector": embedding,
            "payload": {
                "text": chunk.content,
                "source": "direct_input",
                "doc_id": doc_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "original_text": text if len(text) <= 500 else None,
            },
        })

    # Upsert to vector DB
    ui.step(3, 3, "Storing...")
    vector_client.upsert(collection, points)

    print()
    ui.success(f"Ingested {len(chunks)} chunk(s) into '{collection}'")
    ui.info(f"Document ID: {doc_id}")
