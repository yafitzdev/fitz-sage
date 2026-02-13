# fitz_ai/cli/cli.py
"""
Fitz CLI - Main application.

Commands:
    fitz query         Query knowledge base (--source to register, --chat for interactive)
    fitz init          Setup wizard
    fitz collections   Manage collections (list, info, delete)
    fitz config        View config, run diagnostics (--doctor, --test)
    fitz serve         Start the REST API server
    fitz reset         Reset pgserver database (when stuck/corrupted)
    fitz eval          Evaluation tools

NOTE: Commands use lazy loading - heavy imports only happen when a command is invoked.
"""

from __future__ import annotations

import logging

# Suppress noisy third-party and internal INFO logs in CLI
logging.basicConfig(level=logging.WARNING)

# Platform configuration - must run before any HuggingFace imports
from fitz_ai.core.platform import configure_huggingface_windows

configure_huggingface_windows()

from pathlib import Path  # noqa: E402
from typing import Optional  # noqa: E402

import typer  # noqa: E402

app = typer.Typer(
    name="fitz",
    help='Fitz - local-first RAG framework. Start with: fitz query "your question" --source ./docs',
    no_args_is_help=True,
    add_completion=False,
)


# =============================================================================
# LAZY COMMANDS
# =============================================================================
# Each command is a thin wrapper that imports the real implementation only when invoked.
# This keeps CLI startup fast by avoiding heavy imports (torch, pydantic models, etc.).


@app.command("query")
def query(
    question: Optional[str] = typer.Argument(None, help="Question to ask."),
    source: Optional[Path] = typer.Option(None, "--source", "-s", help="Path to documents (registers before querying)."),
    collection: Optional[str] = typer.Option(None, "--collection", "-c", help="Collection name."),
    engine: Optional[str] = typer.Option(None, "--engine", "-e", help="Engine to use."),
    chat: bool = typer.Option(False, "--chat", help="Interactive chat mode."),
) -> None:
    """Query the knowledge base. Use --source to register docs, --chat for interactive mode."""
    from fitz_ai.cli.commands import query as mod

    mod.command(question=question, source=source, collection=collection, engine=engine, chat=chat)


@app.command("init")
def init(
    non_interactive: bool = typer.Option(False, "--non-interactive", "-y", help="Use defaults."),
    show_config: bool = typer.Option(False, "--show", "-s", help="Preview config without saving."),
) -> None:
    """Initialize Fitz with an interactive setup wizard."""
    from fitz_ai.cli.commands import init as mod

    mod.command(non_interactive=non_interactive, show_config=show_config)


@app.command("collections")
def collections() -> None:
    """Manage collections (list, info, delete)."""
    from fitz_ai.cli.commands import collections as mod

    mod.command()


@app.command("config")
def config(
    show_path: bool = typer.Option(False, "--path", "-p", help="Show config file path."),
    as_json: bool = typer.Option(False, "--json", help="Output as JSON."),
    raw: bool = typer.Option(False, "--raw", help="Show raw YAML."),
    edit: bool = typer.Option(False, "--edit", "-e", help="Open config in editor."),
    doctor: bool = typer.Option(False, "--doctor", "-d", help="Run system diagnostics."),
    test: bool = typer.Option(False, "--test", "-t", help="Test actual connections."),
) -> None:
    """View configuration and run diagnostics."""
    from fitz_ai.cli.commands import config as mod

    mod.command(show_path=show_path, as_json=as_json, raw=raw, edit=edit, doctor=doctor, test=test)


@app.command("serve")
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind to."),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to."),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload."),
) -> None:
    """Start the REST API server."""
    from fitz_ai.cli.commands import serve as mod

    mod.command(host=host, port=port, reload=reload)


@app.command("reset")
def reset(
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompt."),
) -> None:
    """Reset pgserver database (use when pgserver hangs or gets corrupted)."""
    from fitz_ai.cli.commands import reset as mod

    mod.reset(force=force)


# =============================================================================
# SUBCOMMAND GROUPS
# =============================================================================


def _register_subcommands() -> None:
    """Register subcommand groups with lazy imports."""
    from fitz_ai.cli.commands.eval import app as eval_app

    app.add_typer(eval_app, name="eval")


_register_subcommands()


# =============================================================================
# ENTERPRISE PLUGIN DISCOVERY
# =============================================================================
# If fitz-ai-enterprise is installed, add its commands to the main CLI.

try:
    from fitz_ai_enterprise.cli import benchmark_app  # noqa: E402

    app.add_typer(benchmark_app, name="benchmark")
except ImportError:
    pass  # Enterprise not installed


if __name__ == "__main__":
    app()
