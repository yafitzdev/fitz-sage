from __future__ import annotations

import os
import typer
import yaml

from fitz_rag.config import get_config

from fitz_rag.vector_db.qdrant_client import create_qdrant_client
from fitz_rag.retriever.dense_retriever import RAGRetriever

from fitz_rag.config.schema import (
    EmbeddingConfig,
    RetrieverConfig,
    RerankConfig,
)

from fitz_rag.exceptions.retriever import (
    EmbeddingError,
    VectorSearchError,
    RerankError,
)

from fitz_stack.logging import get_logger
from fitz_stack.logging_tags import CLI


app = typer.Typer(help="🔥 Fitz-RAG — Retrieval-Augmented Generation Toolkit")

logger = get_logger(__name__)


class CLIError(Exception):
    """Raised for CLI-level operational failures."""


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
    logger.debug(f"{CLI} Loading configuration for display")

    try:
        cfg = get_config()
        typer.echo(yaml.dump(cfg, sort_keys=False))
    except Exception as e:
        logger.error(f"{CLI} Failed to load configuration: {e}")
        raise CLIError(f"Failed to load configuration: {e}") from e


@app.command("config-path")
def config_path():
    """Show the active config file path (if any)."""
    logger.debug(f"{CLI} Checking config path")

    try:
        path = os.getenv("FITZ_RAG_CONFIG")
        if path:
            typer.echo(f"Using override config at: {path}")
        else:
            typer.echo("Using default embedded config.")
    except Exception as e:
        logger.error(f"{CLI} Failed resolving config path: {e}")
        raise CLIError(f"Failed to resolve config path: {e}") from e


# ---------------------------------------------------------
# Collections
# ---------------------------------------------------------
collections_app = typer.Typer(help="Manage Qdrant collections")
app.add_typer(collections_app, name="collections")


@collections_app.command("list")
def collections_list():
    """List all collections in Qdrant."""
    logger.debug(f"{CLI} Listing Qdrant collections")

    try:
        client = create_qdrant_client()
        cols = client.get_collections().collections
    except Exception as e:
        logger.error(f"{CLI} Could not connect to Qdrant: {e}")
        raise CLIError(f"Could not connect to Qdrant: {e}") from e

    for c in cols:
        typer.echo(f"- {c.name}")


@collections_app.command("drop")
def collections_drop(name: str):
    """Drop a Qdrant collection."""
    logger.debug(f"{CLI} Request to drop collection '{name}'")

    try:
        client = create_qdrant_client()
    except Exception as e:
        logger.error(f"{CLI} Qdrant connection failed: {e}")
        raise CLIError(f"Qdrant connection failed: {e}") from e

    if typer.confirm(f"Are you sure you want to delete collection '{name}'?"):
        try:
            client.delete_collection(name)
            typer.echo(f"Deleted collection: {name}")
        except Exception as e:
            logger.error(f"{CLI} Could not delete collection '{name}': {e}")
            raise CLIError(f"Failed to delete collection '{name}': {e}") from e


# ---------------------------------------------------------
# Query (core RAG retrieval + rerank)
# ---------------------------------------------------------
@app.command()
def query(text: str):
    """
    Run a full retrieval using the structured RAGRetriever.

    Steps:
      - load config
      - build embedding/retriever/rerank config objects
      - connect to Qdrant
      - run retrieval
    """

    logger.info(f"{CLI} Starting RAG query: '{text[:50]}...'")

    cfg = get_config()

    # -----------------------------------------------------
    # 1. Build config objects
    # -----------------------------------------------------
    logger.debug(f"{CLI} Building config objects")

    try:
        embed_cfg = EmbeddingConfig.from_dict(cfg.get("embedding", {}))
        retr_cfg = RetrieverConfig.from_dict(cfg.get("retriever", {}))
        rerank_cfg = None

        if "rerank" in cfg and cfg["rerank"].get("enabled", False):
            rerank_cfg = RerankConfig.from_dict(cfg["rerank"])

    except Exception as e:
        logger.error(f"{CLI} Invalid configuration: {e}")
        raise CLIError(f"Invalid configuration structure: {e}") from e

    # -----------------------------------------------------
    # 2. Qdrant client
    # -----------------------------------------------------
    logger.debug(f"{CLI} Connecting to Qdrant")

    try:
        client = create_qdrant_client()
    except Exception as e:
        logger.error(f"{CLI} Failed to connect to Qdrant: {e}")
        raise CLIError(f"Failed to connect to Qdrant: {e}") from e

    # -----------------------------------------------------
    # 3. Construct RAGRetriever
    # -----------------------------------------------------
    logger.debug(f"{CLI} Initializing retriever")

    try:
        retr = RAGRetriever(
            client=client,
            embed_cfg=embed_cfg,
            retriever_cfg=retr_cfg,
            rerank_cfg=rerank_cfg,
        )
    except Exception as e:
        logger.error(f"{CLI} Retriever initialization failed: {e}")
        raise CLIError(f"Failed to initialize retriever: {e}") from e

    # -----------------------------------------------------
    # 4. Perform retrieval
    # -----------------------------------------------------
    logger.debug(f"{CLI} Running retrieval")

    try:
        chunks = retr.retrieve(text)
    except EmbeddingError as e:
        logger.error(f"{CLI} Embedding failed: {e}")
        raise CLIError(f"Embedding failed: {e}") from e
    except VectorSearchError as e:
        logger.error(f"{CLI} Vector search failed: {e}")
        raise CLIError(f"Vector search failed: {e}") from e
    except RerankError as e:
        logger.error(f"{CLI} Reranking failed: {e}")
        raise CLIError(f"Reranking failed: {e}") from e
    except Exception as e:
        logger.error(f"{CLI} Unexpected error during retrieval: {e}")
        raise CLIError(f"Unexpected retrieval error: {e}") from e

    logger.info(f"{CLI} Retrieved {len(chunks)} chunks")

    # -----------------------------------------------------
    # 5. Display results
    # -----------------------------------------------------
    typer.echo(f"\n🔍 Retrieved {len(chunks)} chunks (after rerank if enabled):")
    typer.echo("--------------------------------------------------")
    for c in chunks:
        path = c.metadata.get("file") or c.metadata.get("source") or "<no-file>"
        typer.echo(f"- score=? | {path}")
    typer.echo("--------------------------------------------------")


if __name__ == "__main__":
    app()
