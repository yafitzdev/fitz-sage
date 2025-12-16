"""
CLI module for ingestion commands.

This module provides a modular command-line interface where each command
is defined in its own file for better organization and maintainability.

Available commands:
- run: Ingest documents into vector database
- list-plugins: Show available plugins
- validate: Validate documents before ingestion (dry-run)
- stats: Show statistics about a collection
"""

import typer

# Create main app first
app = typer.Typer(
    help="Ingestion CLI commands",
    no_args_is_help=True,
)


# Import and register commands AFTER app creation to avoid circular imports
def _register_commands():
    """Register commands after module initialization to avoid circular imports."""
    from fitz.ingest.cli import list_plugins, run, stats, validate

    app.command("run")(run.command)
    app.command("list-plugins")(list_plugins.command)
    app.command("validate")(validate.command)
    app.command("stats")(stats.command)


# Register commands immediately
_register_commands()


if __name__ == "__main__":
    app()
