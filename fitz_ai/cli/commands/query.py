# fitz_ai/cli/commands/query.py
"""
Query command - Engine-agnostic query interface.

Uses the default engine (set via 'fitz engine'). Override with --engine.

Usage:
    fitz query                          # Interactive mode
    fitz query "What is RAG?"           # Direct query
    fitz query -c my_collection         # Specify collection
    fitz query --engine custom          # Use a custom engine
"""

from __future__ import annotations

from typing import Optional

import typer

from fitz_ai.cli.ui import display_answer, ui
from fitz_ai.logging.logger import get_logger
from fitz_ai.runtime import create_engine, get_default_engine, get_engine_registry
from fitz_ai.services import FitzService
from fitz_ai.services.fitz_service import CollectionNotFoundError, QueryError

logger = get_logger(__name__)


# =============================================================================
# Helpers
# =============================================================================


def _get_root_cause(exc: Exception) -> str:
    """
    Extract the root cause message from a chain of exceptions.

    Walks the __cause__ chain to find the most specific error message,
    filtering out generic wrapper messages like "Retrieval failed".
    """
    messages = []
    current = exc

    while current is not None:
        msg = str(current)
        # Skip generic wrapper messages
        if msg and msg not in ("Retrieval failed", "Retrieval failed: Retrieval failed"):
            messages.append(msg)
        current = current.__cause__

    # Return the most specific (deepest) meaningful message
    if messages:
        # Prefer the deepest cause that has useful info
        for msg in reversed(messages):
            if "timed out" in msg.lower() or "error" in msg.lower() or len(msg) > 20:
                return msg
        return messages[-1]

    return str(exc)


# =============================================================================
# Main Command
# =============================================================================


def command(
    question: Optional[str] = typer.Argument(
        None,
        help="Question to ask (will prompt if not provided).",
    ),
    collection: Optional[str] = typer.Option(
        None,
        "--collection",
        "-c",
        help="Collection to query (engines with collection support).",
    ),
    engine: Optional[str] = typer.Option(
        None,
        "--engine",
        "-e",
        help="Engine to use. Uses default from 'fitz engine' if not specified.",
    ),
) -> None:
    """
    Query your knowledge base.

    Uses the default engine (set via 'fitz engine'). Override with --engine.

    Examples:
        fitz query                         # Interactive mode
        fitz query "What is RAG?"          # Direct query
        fitz query "question" -e custom    # Use a custom engine
        fitz query "question" -c my_coll   # Specify collection (fitz_rag)
    """
    # =========================================================================
    # Header
    # =========================================================================

    ui.header("Fitz Query", "Query your knowledge base")

    # =========================================================================
    # Engine selection (use default if not specified)
    # =========================================================================

    registry = get_engine_registry()

    if engine is None:
        engine = get_default_engine()
    elif engine not in registry.list():
        ui.error(f"Unknown engine: '{engine}'. Available: {', '.join(registry.list())}")
        raise typer.Exit(1)

    ui.info(f"Engine: {engine}")

    # =========================================================================
    # Capabilities-based routing
    # =========================================================================

    caps = registry.get_capabilities(engine)

    # Engines that need documents loaded first can't do standalone queries
    if caps.requires_documents_at_query:
        _show_documents_required_message(engine, caps)
        raise typer.Exit(0)

    # Engines with persistent ingest support
    if caps.supports_persistent_ingest:
        _run_persistent_ingest_query(question, collection, engine)
    # Engines with collection support use FitzService
    elif caps.supports_collections:
        _run_collection_query(question, collection, engine)
    else:
        # Engines without collections use generic runtime path
        _run_generic_query(question, engine)


def _show_documents_required_message(engine_name: str, caps) -> None:
    """Show message for engines that require documents at query time."""
    if caps.cli_query_message:
        ui.warning(caps.cli_query_message)
    else:
        ui.warning(f"Engine '{engine_name}' requires documents to be loaded first.")
        ui.info(
            f"Use 'fitz quickstart <folder> \"question\" --engine {engine_name}' for one-off queries."
        )
    print()
    ui.info("Or use the Python API:")
    print()
    print("  from fitz_ai.runtime import create_engine")
    print("  from fitz_ai.core import Query")
    print()
    print(f"  engine = create_engine('{engine_name}')")
    print("  engine.add_documents(['doc1...', 'doc2...'])")
    print("  answer = engine.answer(Query(text='your question'))")
    print()


def _select_collection(service: FitzService, requested: Optional[str]) -> str:
    """Select collection interactively or use requested."""
    collections = service.list_collections()

    if not collections:
        print()
        ui.warning("No collections found in vector database.")
        ui.info("Run 'fitz ingest ./docs' first to ingest documents.")
        raise typer.Exit(0)

    collection_names = [c.name for c in collections]

    if requested is not None:
        if requested not in collection_names:
            print()
            ui.warning(f"Collection '{requested}' not found.")
            ui.info(f"Available collections: {', '.join(collection_names)}")
            ui.info("Use -c <collection> to specify another, or run 'fitz ingest' to create it.")
            raise typer.Exit(0)
        return requested

    # Auto-select if only one
    if len(collection_names) == 1:
        return collection_names[0]

    # Prompt user
    print()
    return ui.prompt_numbered_choice("Collection", collection_names, collection_names[0])


def _run_persistent_ingest_query(
    question: Optional[str], collection: Optional[str], engine_name: str
) -> None:
    """Run query using an engine with persistent ingest support."""

    # Get list of available collections via registry (no hardcoded imports)
    registry = get_engine_registry()
    collections = registry.get_list_collections(engine_name)

    if not collections:
        ui.warning(f"No {engine_name} collections found.")
        ui.info(f"Run 'fitz ingest ./docs -e {engine_name}' first to ingest documents.")
        raise typer.Exit(0)

    # Collection selection
    if collection is None:
        if len(collections) == 1:
            collection = collections[0]
            ui.info(f"Using collection: {collection}")
        else:
            print()
            collection = ui.prompt_numbered_choice("Collection", collections, collections[0])
    elif collection not in collections:
        ui.error(f"Collection '{collection}' not found. Available: {', '.join(collections)}")
        raise typer.Exit(1)

    # Prompt for question if not provided
    if question is None:
        question_text = ui.prompt_text("Question")
    else:
        question_text = question

    print()
    ui.info(f"Engine: {engine_name} | Collection: {collection}")
    print()

    # Create engine, load collection, and query
    try:
        engine_instance = create_engine(engine_name)

        ui.info("Loading collection...")
        engine_instance.load(collection)

        ui.info("Querying...")
        from fitz_ai.core import Query

        query = Query(text=question_text)
        answer = engine_instance.answer(query)
        display_answer(answer)
    except Exception as e:
        # Show clean error message, full traceback only at debug level
        ui.error(f"Query failed: {_get_root_cause(e)}")
        logger.debug("Query error", exc_info=True)
        raise typer.Exit(1)


def _run_collection_query(
    question: Optional[str], collection: Optional[str], engine_name: str
) -> None:
    """Run query using FitzService for fitz_rag engine."""
    service = FitzService()

    # Collection selection
    selected_collection = _select_collection(service, collection)

    # Prompt for question if not provided
    if question is None:
        question_text = ui.prompt_text("Question")
    else:
        question_text = question

    # Display info
    print()
    ui.info(f"Engine: {engine_name} | Collection: {selected_collection}")
    print()

    # Execute query via FitzService
    try:
        answer = service.query(
            question=question_text,
            collection=selected_collection,
            engine=engine_name,
        )
        display_answer(answer)
    except CollectionNotFoundError as e:
        ui.error(f"Collection '{e.collection}' not found.")
        ui.info("Run 'fitz collections' to see available collections.")
        raise typer.Exit(1)
    except QueryError as e:
        # Show clean error message, full traceback only at debug level
        ui.error(f"Query failed: {_get_root_cause(e)}")
        logger.debug("Query error", exc_info=True)
        raise typer.Exit(1)


def _run_generic_query(question: Optional[str], engine_name: str) -> None:
    """Run query using generic runtime path."""
    # Prompt for question if not provided
    if question is None:
        question_text = ui.prompt_text("Question")
    else:
        question_text = question

    print()
    ui.info(f"Using engine: {engine_name}")
    print()

    try:
        engine_instance = create_engine(engine_name)
        from fitz_ai.core import Query

        query = Query(text=question_text)
        answer = engine_instance.answer(query)
        display_answer(answer)
    except Exception as e:
        # Show clean error message, full traceback only at debug level
        ui.error(f"Query failed: {_get_root_cause(e)}")
        logger.debug("Query error", exc_info=True)
        raise typer.Exit(1)
