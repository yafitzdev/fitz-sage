# fitz_ai/cli/commands/query.py
"""
Interactive query command.

Usage:
    fitz query                     # Interactive mode
    fitz query "What is RAG?"      # Direct query
    fitz query -c my_collection    # Specify collection
"""

from __future__ import annotations

from typing import List, Optional

import typer

from fitz_ai.cli.ui import RICH, Markdown, Panel, Table, console, ui
from fitz_ai.core.config import ConfigNotFoundError, load_config_dict
from fitz_ai.core.paths import FitzPaths
from fitz_ai.engines.classic_rag.config import ClassicRagConfig, load_config
from fitz_ai.engines.classic_rag.pipeline.pipeline.engine import RAGPipeline
from fitz_ai.logging.logger import get_logger
from fitz_ai.vector_db.registry import get_vector_db_plugin

logger = get_logger(__name__)


# =============================================================================
# Config Loading
# =============================================================================


def _load_config_safe() -> tuple[dict, ClassicRagConfig]:
    """Load config or exit with helpful message."""
    try:
        config_path = FitzPaths.config()
        raw_config = load_config_dict(config_path)
        typed_config = load_config(config_path)
        return raw_config, typed_config
    except ConfigNotFoundError:
        ui.error("No config found. Run 'fitz init' first.")
        raise typer.Exit(1)
    except Exception as e:
        ui.error(f"Failed to load config: {e}")
        raise typer.Exit(1)


def _get_collections(raw_config: dict) -> List[str]:
    """Get list of collections from vector DB."""
    try:
        vdb_plugin = raw_config.get("vector_db", {}).get("plugin_name", "qdrant")
        vdb = get_vector_db_plugin(vdb_plugin)
        return sorted(vdb.list_collections())
    except Exception:
        return []


def _select_collection(collections: List[str], default: str) -> str:
    """Let user select a collection via numbered menu."""
    if not collections:
        ui.warning("No collections found. Using config default.")
        return default

    if len(collections) == 1:
        ui.info(f"Collection: {collections[0]} (only one available)")
        return collections[0]

    # Sort collections: default first, then alphabetically
    sorted_collections = sorted(collections)
    if default in sorted_collections:
        sorted_collections.remove(default)
        sorted_collections.insert(0, default)

    # Show available collections with numbers
    ui.info("Available collections:")
    for idx, collection in enumerate(sorted_collections, 1):
        marker = " (default)" if collection == default else ""
        ui.info(f"  [{idx}] {collection}{marker}")

    # Prompt for number selection (default is always 1)
    while True:
        if RICH:
            from rich.prompt import Prompt
            response = Prompt.ask("Select collection number", default="1")
        else:
            response = input(f"Select collection number [1]: ").strip()
            if not response:
                response = "1"

        try:
            selection = int(response)
            if 1 <= selection <= len(sorted_collections):
                return sorted_collections[selection - 1]
            else:
                ui.error(f"Please enter a number between 1 and {len(sorted_collections)}")
        except ValueError:
            ui.error("Please enter a valid number")


# =============================================================================
# Answer Display
# =============================================================================


def _display_answer(answer, show_sources: bool = True) -> None:
    """Display the answer with optional sources."""
    print()

    if RICH:
        # Answer panel
        console.print(
            Panel(
                Markdown(answer.answer),
                title="[bold green]Answer[/bold green]",
                border_style="green",
            )
        )

        # Sources table
        if show_sources and hasattr(answer, "sources") and answer.sources:
            print()
            table = Table(title="Sources")
            table.add_column("#", style="dim", width=3)
            table.add_column("Document", style="cyan")
            table.add_column("Excerpt", style="dim", max_width=60)

            for i, source in enumerate(answer.sources[:5], 1):
                doc_id = getattr(source, "doc_id", getattr(source, "source_file", "?"))
                content = getattr(source, "content", getattr(source, "text", ""))
                excerpt = content[:100] + "..." if len(content) > 100 else content
                excerpt = excerpt.replace("\n", " ")
                table.add_row(str(i), doc_id, excerpt)

            console.print(table)
    else:
        # Plain text output
        print("Answer:")
        print("-" * 40)
        print(answer.answer)
        print()

        if show_sources and hasattr(answer, "sources") and answer.sources:
            print("Sources:")
            for i, source in enumerate(answer.sources[:5], 1):
                doc_id = getattr(source, "doc_id", getattr(source, "source_file", "?"))
                print(f"  [{i}] {doc_id}")


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
            help="Collection to query (uses config default if not specified).",
        ),
        top_k: Optional[int] = typer.Option(
            None,
            "--top-k",
            "-k",
            help="Number of chunks to retrieve.",
        ),
        no_rerank: bool = typer.Option(
            False,
            "--no-rerank",
            help="Disable reranking.",
        ),
        no_sources: bool = typer.Option(
            False,
            "--no-sources",
            help="Don't show source documents.",
        ),
        interactive: bool = typer.Option(
            False,
            "--interactive",
            "-i",
            help="Interactive mode (continuous Q&A).",
        ),
) -> None:
    """
    Query your knowledge base.

    Run without arguments for interactive mode:
        fitz query

    Or ask directly:
        fitz query "What is RAG?"

    Options:
        fitz query "question" -c my_collection
        fitz query "question" -k 10
        fitz query "question" --no-rerank
    """
    # =========================================================================
    # Load config
    # =========================================================================

    raw_config, typed_config = _load_config_safe()
    default_collection = typed_config.retriever.collection

    # =========================================================================
    # Collection selection (if not specified via -c flag)
    # =========================================================================

    if collection:
        # User specified collection via flag
        typed_config.retriever.collection = collection
    else:
        # Show collection menu (whether question provided or not)
        collections = _get_collections(raw_config)
        if collections and len(collections) > 1:
            selected = _select_collection(collections, default_collection)
            typed_config.retriever.collection = selected
            print()
        elif collections:
            # Only one collection available
            ui.info(f"Collection: {collections[0]} (only one available)")
            typed_config.retriever.collection = collections[0]

    # Override top_k if specified
    if top_k:
        typed_config.retriever.top_k = top_k

    # Disable rerank if requested
    if no_rerank:
        typed_config.rerank.enabled = False

    # Get display info
    display_collection = typed_config.retriever.collection
    display_chat = raw_config.get("chat", {}).get("plugin_name", "?")

    # =========================================================================
    # Query loop
    # =========================================================================

    if interactive:
        ui.info(f"Interactive mode - Collection: {display_collection}")
        ui.info("Type 'quit' or 'exit' to end session")
        print()

        pipeline = RAGPipeline.from_config(typed_config)

        while True:
            question_text = ui.prompt_text("Question")
            if question_text.lower() in ("quit", "exit", "q"):
                ui.info("Goodbye!")
                break

            try:
                answer = pipeline.run(question_text)
                _display_answer(answer, show_sources=not no_sources)
            except Exception as e:
                ui.error(f"Query failed: {e}")
                logger.exception("Query error")

    else:
        # Single query mode
        if question is None:
            question_text = ui.prompt_text("Question")
        else:
            question_text = question

        ui.info(f"Collection: {display_collection}")
        ui.info(f"Chat: {display_chat}")
        print()

        try:
            pipeline = RAGPipeline.from_config(typed_config)
            answer = pipeline.run(question_text)
            _display_answer(answer, show_sources=not no_sources)
        except Exception as e:
            ui.error(f"Query failed: {e}")
            logger.exception("Query error")
            raise typer.Exit(1)