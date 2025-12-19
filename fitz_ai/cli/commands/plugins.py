# fitz_ai/cli/commands/plugins.py
"""
Plugins command for Fitz CLI.

Lists all discovered plugins across all registries.

Usage:
    fitz plugins
"""

import typer


def command() -> None:
    """
    List all discovered plugins.

    Shows available plugins for:
    - Chat (LLM providers)
    - Embedding (text to vectors)
    - Rerank (result scoring)
    - Vector DB (storage)
    """
    # Import from correct locations:
    # - LLM plugins (chat, embedding, rerank) are in fitz_ai.llm.registry
    # - Vector DB plugins are in fitz_ai.vector_db.registry
    from fitz_ai.llm.registry import available_llm_plugins
    from fitz_ai.vector_db.registry import available_vector_db_plugins

    typer.echo()
    typer.echo("=" * 50)
    typer.echo("DISCOVERED PLUGINS")
    typer.echo("=" * 50)

    def show_llm(title: str, plugin_type: str):
        """Show LLM plugins (chat, embedding, rerank)."""
        typer.echo()
        typer.echo(f"{title}:")
        names = available_llm_plugins(plugin_type)
        if not names:
            typer.echo("  (none)")
        else:
            for name in sorted(names):
                typer.echo(f"  - {name}")

    def show_vector_db(title: str):
        """Show Vector DB plugins (separate registry)."""
        typer.echo()
        typer.echo(f"{title}:")
        names = available_vector_db_plugins()
        if not names:
            typer.echo("  (none)")
        else:
            for name in sorted(names):
                typer.echo(f"  - {name}")

    # LLM plugins
    show_llm("Chat (LLM)", "chat")
    show_llm("Embedding", "embedding")
    show_llm("Rerank", "rerank")

    # Vector DB plugins (separate registry!)
    show_vector_db("Vector DB")

    typer.echo()