# fitz_ai/cli/cli.py
"""
Main Fitz CLI.

Goals:
- Discoverability first
- Zero magic
- User-friendly errors
"""

from __future__ import annotations

import typer

# Try to import error handler, but don't fail if not present
try:
    from fitz_ai.cli.utils.errors import friendly_errors, install_global_handler

    install_global_handler()
    HAS_ERROR_HANDLER = True
except ImportError:
    HAS_ERROR_HANDLER = False

    # Dummy decorator if errors.py not installed
    def friendly_errors(func):
        return func


app = typer.Typer(
    help="Fitz — local-first RAG framework",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# Register sub-apps with lazy imports to avoid circular dependency issues
# ---------------------------------------------------------------------------


def _register_sub_apps():
    """Register ingest sub-app after module initialization."""
    # Create ingest sub-app with its own commands
    ingest_app = typer.Typer(
        help="Document ingestion commands",
        no_args_is_help=True,
    )

    # Register ingest sub-commands
    try:
        from fitz_ai.cli.commands import ingest as ingest_module
        from fitz_ai.cli.commands import validate as validate_module
        from fitz_ai.cli.commands import stats as stats_module
        from fitz_ai.cli.commands import list_plugins as list_plugins_module

        # Main ingest command (fitz ingest ./docs collection)
        ingest_app.command("run")(friendly_errors(ingest_module.command))
        # Also register as default when calling "fitz ingest ./docs collection"
        ingest_app.callback(invoke_without_command=True)(
            lambda: None  # Allow subcommands
        )

        # Validate command (fitz ingest validate ./docs)
        ingest_app.command("validate")(friendly_errors(validate_module.command))

        # Stats command (fitz ingest stats collection)
        ingest_app.command("stats")(friendly_errors(stats_module.command))

        # List plugins command (fitz ingest plugins)
        ingest_app.command("plugins")(friendly_errors(list_plugins_module.command))

        # Add the ingest sub-app to main app
        app.add_typer(ingest_app, name="ingest")

    except (ImportError, ModuleNotFoundError) as e:
        # Log the error for debugging but don't fail
        import sys
        print(f"Warning: Could not register ingest commands: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Commands with friendly error handling
# ---------------------------------------------------------------------------


def _register_commands():
    """Register commands after module initialization to avoid circular imports."""
    from fitz_ai.cli.commands import chunk, db, doctor
    from fitz_ai.cli.commands import help as help_module
    from fitz_ai.cli.commands import init, plugins, quickstart
    from fitz_ai.cli.commands import config as config_module
    from fitz_ai.cli.commands import query as query_module

    # Wrap commands with friendly error handling
    app.command("help")(friendly_errors(help_module.command))
    app.command("init")(friendly_errors(init.command))
    app.command("plugins")(friendly_errors(plugins.command))
    app.command("doctor")(friendly_errors(doctor.command))
    app.command("quickstart")(friendly_errors(quickstart.command))

    # TOP-LEVEL query command (the main one users will use!)
    app.command("query")(friendly_errors(query_module.command))

    # TOP-LEVEL config command
    app.command("config")(friendly_errors(config_module.command))

    # Database inspection command
    app.command("db")(friendly_errors(db.command))

    # Chunking preview command
    app.command("chunk")(friendly_errors(chunk.command))


# Call immediately during module initialization
_register_sub_apps()
_register_commands()


if __name__ == "__main__":
    app()