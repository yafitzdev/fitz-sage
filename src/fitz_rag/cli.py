from __future__ import annotations

import typer
from typing import Optional
import yaml

from fitz_rag.config import get_config
from fitz_rag.retriever.qdrant_client import create_qdrant_client
from fitz_rag.retriever.rag_retriever import RAGRetriever
from fitz_rag.llm.embedding_client import DummyEmbeddingClient  # replace later with real embedder

app = typer.Typer(help="🔥 Fitz-RAG — Retrieval-Augmented Generation Toolkit")


# ---------------------------------------------------------
# Version
# ---------------------------------------------------------
@app.command()
def version():
    """Show current Fitz-RAG version."""
    typer.echo("fitz-rag version 0.1.0")


# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
@app.command("config-show")
def config_show():
    """Display the loaded configuration."""
    cfg = get_config()
    typer.echo(yaml.dump(cfg, sort_keys=False))


@app.command("config-path")
def config_path():
    """Show the active config file path (if any)."""
    import os

    path = os.getenv("FITZ_RAG_CONFIG")
    if path:
        typer.echo(f"Using override config at: {path}")
    else:
        typer.echo("Using default embedded config.")


# ---------------------------------------------------------
# Collections
# ---------------------------------------------------------
collections_app = typer.Typer(help="Manage Qdrant collections")
app.add_typer(collections_app, name="collections")


@collections_app.command("list")
def collections_list():
    """List all collections in Qdrant."""
    client = create_qdrant_client()
    cols = client.get_collections().collections
    for c in cols:
        typer.echo(f"- {c.name}")


@collections_app.command("drop")
def collections_drop(name: str):
    """Drop a Qdrant collection."""
    client = create_qdrant_client()

    if typer.confirm(f"Are you sure you want to delete collection '{name}'?"):
        client.delete_collection(name)
        typer.echo(f"Deleted collection: {name}")


# ---------------------------------------------------------
# Query (core RAG retrieval)
# ---------------------------------------------------------
@app.command()
def query(text: str):
    """
    Run a basic RAG retrieval using the configured embedding model.
    """
    cfg = get_config()

    # 1. Qdrant client
    client = create_qdrant_client()

    # 2. Embedding client
    embedder = DummyEmbeddingClient()  # <-- replace later with real model integration

    # 3. Retriever
    retr = RAGRetriever(
        client=client,
        embedder=embedder,
        collection=cfg["qdrant"]["collection"],
        top_k=cfg["retriever"]["top_k"],
    )

    # 4. Perform retrieval
    chunks = retr.retrieve(text)

    typer.echo(f"\n🔍 Retrieved {len(chunks)} chunks:")
    typer.echo("--------------------------------------------------")
    for c in chunks:
        path = c.metadata.get("file") or c.metadata.get("source") or "<no-file>"
        typer.echo(f"- score={c.score:.4f} | {path}")
    typer.echo("--------------------------------------------------")


if __name__ == "__main__":
    app()
