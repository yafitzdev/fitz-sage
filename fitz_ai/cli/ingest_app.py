# fitz_ai/cli/ingest_app.py
"""
Ingest sub-application for Fitz CLI.

Provides the `fitz ingest` command group:
- fitz ingest ./docs collection    # Ingest documents
- fitz ingest validate ./docs      # Validate before ingesting
- fitz ingest stats collection     # Show collection stats
- fitz ingest plugins              # List ingest plugins
"""

from __future__ import annotations

import typer

# Create ingest sub-app
ingest_app = typer.Typer(
    help="Document ingestion commands",
    no_args_is_help=True,
)


def _register_ingest_commands():
    """Register ingest commands."""
    from fitz_ai.cli.commands import ingest, validate, stats, list_plugins

    # Main ingest command (fitz ingest ./docs collection)
    # Use callback to make source/collection positional args work at app level
    ingest_app.command("run", help="Ingest documents into a collection")(ingest.command)

    # Also register as default command when called with args
    # This allows: fitz ingest ./docs collection (without 'run')
    @ingest_app.callback(invoke_without_command=True)
    def ingest_callback(
            ctx: typer.Context,
            source: str = typer.Argument(None, help="Source file or directory"),
            collection: str = typer.Argument(None, help="Target collection name"),
    ):
        """
        Ingest documents into a vector database.

        Examples:
            fitz ingest ./docs default
            fitz ingest run ./docs my_knowledge
        """
        # If subcommand was invoked, let it run
        if ctx.invoked_subcommand is not None:
            return

        # If no args provided, show help
        if source is None:
            # Show help
            typer.echo(ctx.get_help())
            raise typer.Exit(0)

        # Otherwise, call ingest.command with the args
        # This handles: fitz ingest ./docs collection
        from pathlib import Path
        ingest.command(
            source=Path(source),
            collection=collection or "default",
        )

    # Validate command (fitz ingest validate ./docs)
    ingest_app.command("validate", help="Validate documents before ingesting")(validate.command)

    # Stats command (fitz ingest stats collection)
    ingest_app.command("stats", help="Show collection statistics")(stats.command)

    # List plugins command (fitz ingest plugins)
    ingest_app.command("plugins", help="List available ingestion plugins")(list_plugins.command)


# Register commands
_register_ingest_commands()