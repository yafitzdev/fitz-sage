# fitz_ai/cli/v2/commands/query.py
"""
Interactive query command.

Usage:
    fitz query                     # Interactive mode
    fitz query "What is RAG?"      # Direct query
    fitz query -c my_collection    # Specify collection
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from fitz_ai.core.config import load_config_dict, ConfigNotFoundError
from fitz_ai.core.paths import FitzPaths
from fitz_ai.engines.classic_rag.config import load_config, ClassicRagConfig
from fitz_ai.engines.classic_rag.pipeline.pipeline.engine import RAGPipeline
from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)

# Rich for UI (optional)
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    from rich.markdown import Markdown
    from rich.table import Table

    console = Console()
    RICH = True
except ImportError:
    console = None
    RICH = False


# =============================================================================
# UI Helpers
# =============================================================================


def _print(msg: str, style: str = "") -> None:
    if RICH and style:
        console.print(f"[{style}]{msg}[/{style}]")
    else:
        print(msg)


def _header(title: str) -> None:
    if RICH:
        console.print(Panel.fit(f"[bold]{title}[/bold]", border_style="blue"))
    else:
        print(f"\n{'=' * 50}")
        print(title)
        print('=' * 50)


def _error(msg: str) -> None:
    if RICH:
        console.print(f"[red]✗[/red] {msg}")
    else:
        print(f"✗ {msg}")


def _prompt_text(prompt: str, default: str = "") -> str:
    if RICH:
        return Prompt.ask(prompt, default=default) if default else Prompt.ask(prompt)
    else:
        if default:
            response = input(f"{prompt} [{default}]: ").strip()
            return response if response else default
        else:
            return input(f"{prompt}: ").strip()


def _prompt_confirm(prompt: str, default: bool = True) -> bool:
    if RICH:
        return Confirm.ask(prompt, default=default)
    else:
        yn = "Y/n" if default else "y/N"
        response = input(f"{prompt} [{yn}]: ").strip().lower()
        if not response:
            return default
        return response in ("y", "yes")


# =============================================================================
# Config Loading
# =============================================================================


def _load_config_safe() -> tuple[dict, ClassicRagConfig]:
    """Load config or exit with helpful message."""
    try:
        config_path = FitzPaths.config()
        raw_config = load_config_dict(config_path)
        typed_config = load_config(str(config_path))
        return raw_config, typed_config
    except (ConfigNotFoundError, FileNotFoundError):
        _error("No config found. Run 'fitz init' first.")
        raise typer.Exit(1)
    except Exception as e:
        _error(f"Failed to load config: {e}")
        raise typer.Exit(1)


# =============================================================================
# Answer Display
# =============================================================================


def _display_answer(answer, show_sources: bool = True) -> None:
    """Display the answer with optional sources."""
    # Extract answer text
    answer_text = getattr(answer, "answer", None) or getattr(answer, "text", None) or str(answer)

    if RICH:
        # Display answer
        console.print()
        console.print(Panel(
            Markdown(answer_text),
            title="[bold green]Answer[/bold green]",
            border_style="green",
            padding=(1, 2),
        ))

        # Display sources if available
        if show_sources:
            sources = getattr(answer, "sources", None) or getattr(answer, "citations", None) or []
            if sources:
                console.print()
                table = Table(title="Sources", show_header=True, header_style="bold cyan")
                table.add_column("#", style="dim", width=3)
                table.add_column("Source", style="cyan")
                table.add_column("Excerpt", style="dim", max_width=60)

                for i, source in enumerate(sources[:5], 1):
                    source_id = getattr(source, "source_id", None) or getattr(source, "doc_id", None) or "?"
                    text = getattr(source, "text", None) or getattr(source, "content", None) or ""
                    # Truncate excerpt
                    excerpt = text[:100] + "..." if len(text) > 100 else text
                    excerpt = excerpt.replace("\n", " ")
                    table.add_row(str(i), str(source_id), excerpt)

                console.print(table)
    else:
        # Plain text output
        print()
        print("=" * 60)
        print("ANSWER")
        print("=" * 60)
        print()
        print(answer_text)
        print()

        if show_sources:
            sources = getattr(answer, "sources", None) or getattr(answer, "citations", None) or []
            if sources:
                print("-" * 60)
                print("SOURCES")
                print("-" * 60)
                for i, source in enumerate(sources[:5], 1):
                    source_id = getattr(source, "source_id", None) or getattr(source, "doc_id", None) or "?"
                    print(f"  [{i}] {source_id}")
                print()


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
        help="Collection to query (default: from config).",
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
        help="Disable reranking even if configured.",
    ),
    show_sources: bool = typer.Option(
        True,
        "--sources/--no-sources",
        help="Show/hide source citations.",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Enter interactive chat mode.",
    ),
) -> None:
    """
    Query your knowledge base.

    Run without arguments for interactive mode:
        fitz query

    Or ask directly:
        fitz query "What is RAG?"

    Options:
        fitz query "question" -c my_collection   # Specific collection
        fitz query "question" -k 10              # More results
        fitz query "question" --no-rerank        # Skip reranking
        fitz query -i                            # Interactive chat mode
    """
    # =========================================================================
    # Load config
    # =========================================================================

    raw_config, typed_config = _load_config_safe()

    # Extract settings from config
    chat_plugin = raw_config.get("chat", {}).get("plugin_name", "?")
    embedding_plugin = raw_config.get("embedding", {}).get("plugin_name", "?")
    vector_db_plugin = raw_config.get("vector_db", {}).get("plugin_name", "?")
    rerank_enabled = raw_config.get("rerank", {}).get("enabled", False)
    rerank_plugin = raw_config.get("rerank", {}).get("plugin_name", None)
    default_collection = raw_config.get("retriever", {}).get("collection", "default")
    default_top_k = raw_config.get("retriever", {}).get("top_k", 5)

    # Apply CLI overrides
    if collection:
        typed_config.retriever.collection = collection
    else:
        collection = default_collection

    if top_k:
        typed_config.retriever.top_k = top_k
    else:
        top_k = default_top_k

    if no_rerank:
        typed_config.rerank.enabled = False
        rerank_enabled = False

    # =========================================================================
    # Header
    # =========================================================================

    _header("Fitz Query")

    if RICH:
        info_parts = [
            f"[dim]Collection:[/dim] {collection}",
            f"[dim]Chat:[/dim] {chat_plugin}",
            f"[dim]Embedding:[/dim] {embedding_plugin}",
        ]
        if rerank_enabled and rerank_plugin:
            info_parts.append(f"[dim]Rerank:[/dim] {rerank_plugin}")
        console.print("  ".join(info_parts))
        console.print()
    else:
        print(f"Collection: {collection} | Chat: {chat_plugin} | Rerank: {'on' if rerank_enabled else 'off'}")
        print()

    # =========================================================================
    # Build pipeline
    # =========================================================================

    try:
        if RICH:
            with console.status("[bold blue]Loading pipeline...", spinner="dots"):
                pipeline = RAGPipeline.from_config(typed_config)
        else:
            print("Loading pipeline...")
            pipeline = RAGPipeline.from_config(typed_config)
    except Exception as e:
        _error(f"Failed to initialize pipeline: {e}")
        raise typer.Exit(1)

    # =========================================================================
    # Interactive mode
    # =========================================================================

    if interactive or question is None:
        _print("Enter your questions (type 'exit' or 'quit' to stop):", "dim")
        print()

        while True:
            try:
                if RICH:
                    q = Prompt.ask("[bold cyan]You[/bold cyan]")
                else:
                    q = input("You: ").strip()

                if not q:
                    continue

                if q.lower() in ("exit", "quit", "q"):
                    _print("Goodbye!", "dim")
                    break

                # Run query
                if RICH:
                    with console.status("[bold blue]Thinking...", spinner="dots"):
                        answer = pipeline.run(q)
                else:
                    print("Thinking...")
                    answer = pipeline.run(q)

                _display_answer(answer, show_sources=show_sources)
                print()

            except KeyboardInterrupt:
                print()
                _print("Interrupted. Goodbye!", "dim")
                break
            except Exception as e:
                _error(f"Query failed: {e}")
                logger.exception("Query error")

        return

    # =========================================================================
    # Single query mode
    # =========================================================================

    if not question or not question.strip():
        _error("No question provided.")
        raise typer.Exit(1)

    _print(f"[bold]Question:[/bold] {question}" if RICH else f"Question: {question}")
    print()

    try:
        if RICH:
            with console.status("[bold blue]Thinking...", spinner="dots"):
                answer = pipeline.run(question)
        else:
            print("Thinking...")
            answer = pipeline.run(question)

        _display_answer(answer, show_sources=show_sources)

    except Exception as e:
        _error(f"Query failed: {e}")
        logger.exception("Query error")
        raise typer.Exit(1)