# fitz_ai/cli/v2/commands/init.py
"""
Interactive setup wizard for Fitz.

Usage:
    fitz init              # Interactive wizard
    fitz init -y           # Auto-detect and use defaults
    fitz init --show       # Preview config without saving
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from fitz_ai.core.detect import detect_all
from fitz_ai.llm.registry import available_llm_plugins
from fitz_ai.vector_db.registry import available_vector_db_plugins

# Rich for pretty output (optional)
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Confirm, Prompt
    from rich.syntax import Syntax
    from rich.table import Table

    console = Console()
    RICH = True
except ImportError:
    console = None
    RICH = False


# =============================================================================
# Helpers
# =============================================================================


def _is_local_plugin(name: str) -> bool:
    """Check if plugin is local/Ollama-based."""
    return any(x in name.lower() for x in ("ollama", "local", "offline"))


def _is_faiss_plugin(name: str) -> bool:
    """Check if plugin is FAISS-based."""
    return "faiss" in name.lower()


def _print(msg: str, style: str = "") -> None:
    """Print with optional Rich styling."""
    if RICH and style:
        console.print(f"[{style}]{msg}[/{style}]")
    else:
        print(msg)


def _header(title: str) -> None:
    """Print section header."""
    if RICH:
        console.print(f"\n[bold blue]{title}[/bold blue]")
    else:
        print(f"\n{title}")
        print("-" * len(title))


def _status(name: str, available: bool, detail: str = "") -> None:
    """Print status line."""
    icon = "✓" if available else "✗"
    color = "green" if available else "dim"
    if RICH:
        console.print(f"  [{color}]{icon}[/{color}] {name}" + (f" [dim]({detail})[/dim]" if detail else ""))
    else:
        print(f"  {icon} {name}" + (f" ({detail})" if detail else ""))


def _prompt_choice(prompt: str, choices: list[str], default: str) -> str:
    """Prompt user to select from choices."""
    if RICH:
        return Prompt.ask(prompt, choices=choices, default=default)
    else:
        choice_str = "/".join(choices)
        while True:
            response = input(f"{prompt} [{choice_str}] ({default}): ").strip()
            if not response:
                return default
            if response in choices:
                return response
            print(f"Invalid. Choose from: {', '.join(choices)}")


def _prompt_confirm(prompt: str, default: bool = True) -> bool:
    """Prompt for yes/no."""
    if RICH:
        return Confirm.ask(prompt, default=default)
    else:
        yn = "Y/n" if default else "y/N"
        response = input(f"{prompt} [{yn}]: ").strip().lower()
        if not response:
            return default
        return response in ("y", "yes")


def _prompt_text(prompt: str, default: str) -> str:
    """Prompt for text input."""
    if RICH:
        return Prompt.ask(prompt, default=default)
    else:
        response = input(f"{prompt} ({default}): ").strip()
        return response if response else default


def _auto_or_prompt(category: str, available: list[str], default: str, prompt: str) -> str:
    """Auto-select if one option, otherwise prompt."""
    if len(available) == 1:
        choice = available[0]
        if RICH:
            console.print(f"  [dim]{category}:[/dim] [green]{choice}[/green] [dim](auto)[/dim]")
        else:
            print(f"  {category}: {choice} (auto)")
        return choice
    return _prompt_choice(prompt, available, default)


# =============================================================================
# Plugin Filtering
# =============================================================================


def _filter_available_plugins(
    plugins: list[str],
    plugin_type: str,
    system: Any,
) -> list[str]:
    """Filter plugins to only those that are available."""
    available = []

    for plugin in plugins:
        is_available = False

        # Check local/Ollama plugins
        if _is_local_plugin(plugin):
            is_available = system.ollama.available

        # Check FAISS
        elif _is_faiss_plugin(plugin):
            is_available = system.faiss.available

        # Check Qdrant
        elif plugin == "qdrant":
            is_available = system.qdrant.available

        # Check API key providers
        elif plugin in system.api_keys:
            is_available = system.api_keys[plugin].available

        if is_available:
            available.append(plugin)

    return available


# =============================================================================
# Config Generation
# =============================================================================


def _generate_config(
    chat: str,
    embedding: str,
    vector_db: str,
    collection: str,
    rerank: str | None,
    qdrant_host: str,
    qdrant_port: int,
) -> str:
    """Generate YAML configuration."""

    # Vector DB kwargs
    vdb_kwargs = ""
    if vector_db == "qdrant":
        vdb_kwargs = f"""
    host: "{qdrant_host}"
    port: {qdrant_port}"""

    # Rerank section
    if rerank:
        rerank_section = f"""
# Reranker (improves retrieval quality)
rerank:
  enabled: true
  plugin_name: {rerank}
  kwargs: {{}}
"""
    else:
        rerank_section = """
# Reranker (disabled)
rerank:
  enabled: false
"""

    return f"""# Fitz RAG Configuration
# Generated by: fitz init

# Chat (LLM for answering questions)
chat:
  plugin_name: {chat}
  kwargs:
    temperature: 0.2

# Embedding (text to vectors)
embedding:
  plugin_name: {embedding}
  kwargs: {{}}

# Vector Database
vector_db:
  plugin_name: {vector_db}
  kwargs:{vdb_kwargs if vdb_kwargs else " {}"}

# Retriever
retriever:
  plugin_name: dense
  collection: {collection}
  top_k: 5
{rerank_section}
# RGS (Retrieval-Guided Synthesis)
rgs:
  enable_citations: true
  strict_grounding: true
  max_chunks: 8

# Logging
logging:
  level: INFO
"""


# =============================================================================
# Main Command
# =============================================================================


def command(
    non_interactive: bool = typer.Option(
        False,
        "--non-interactive",
        "-y",
        help="Use detected defaults without prompting.",
    ),
    show_config: bool = typer.Option(
        False,
        "--show",
        "-s",
        help="Preview config without saving.",
    ),
) -> None:
    """
    Initialize Fitz with an interactive setup wizard.

    Detects available providers (API keys, Ollama, Qdrant) and
    creates a working configuration file.

    Examples:
        fitz init           # Interactive wizard
        fitz init -y        # Auto-detect and use defaults
        fitz init --show    # Preview config without saving
    """
    from fitz_ai.core.paths import FitzPaths

    # =========================================================================
    # Header
    # =========================================================================

    if RICH:
        console.print(Panel.fit(
            "[bold]Fitz Setup Wizard[/bold]\n[dim]Let's configure your RAG pipeline[/dim]",
            border_style="blue",
        ))
    else:
        print("\n" + "=" * 50)
        print("Fitz Setup Wizard")
        print("=" * 50)

    # =========================================================================
    # Detect System
    # =========================================================================

    _header("Detecting System")

    system = detect_all()

    # Show detection results
    _status("Ollama", system.ollama.available, system.ollama.details if not system.ollama.available else "")
    _status("Qdrant", system.qdrant.available, f"{system.qdrant.host}:{system.qdrant.port}" if system.qdrant.available else "")
    _status("FAISS", system.faiss.available)

    for name, key_status in system.api_keys.items():
        _status(name.capitalize(), key_status.available)

    # =========================================================================
    # Discover Plugins
    # =========================================================================

    all_chat = available_llm_plugins("chat")
    all_embedding = available_llm_plugins("embedding")
    all_rerank = available_llm_plugins("rerank")
    all_vector_db = available_vector_db_plugins()

    # Filter to available only
    avail_chat = _filter_available_plugins(all_chat, "chat", system)
    avail_embedding = _filter_available_plugins(all_embedding, "embedding", system)
    avail_rerank = _filter_available_plugins(all_rerank, "rerank", system)
    avail_vector_db = _filter_available_plugins(all_vector_db, "vector_db", system)

    # =========================================================================
    # Validate Minimum Requirements
    # =========================================================================

    if not avail_chat:
        _print("\n✗ No chat plugins available!", "red")
        _print("  Set an API key (COHERE_API_KEY, OPENAI_API_KEY) or start Ollama.", "dim")
        raise typer.Exit(1)

    if not avail_embedding:
        _print("\n✗ No embedding plugins available!", "red")
        raise typer.Exit(1)

    if not avail_vector_db:
        _print("\n✗ No vector database available!", "red")
        _print("  Install FAISS (pip install faiss-cpu) or start Qdrant.", "dim")
        raise typer.Exit(1)

    # =========================================================================
    # User Selection
    # =========================================================================

    _header("Configuration")

    if non_interactive:
        # Use first available for each
        chat_choice = avail_chat[0]
        embedding_choice = avail_embedding[0]
        vector_db_choice = avail_vector_db[0]
        rerank_choice = avail_rerank[0] if avail_rerank else None
        collection_name = "default"

        _print(f"  Chat: {chat_choice}", "dim")
        _print(f"  Embedding: {embedding_choice}", "dim")
        _print(f"  Vector DB: {vector_db_choice}", "dim")
        _print(f"  Rerank: {rerank_choice or 'disabled'}", "dim")
        _print(f"  Collection: {collection_name}", "dim")
    else:
        # Interactive selection
        chat_choice = _auto_or_prompt("Chat", avail_chat, avail_chat[0], "Select chat plugin")
        embedding_choice = _auto_or_prompt("Embedding", avail_embedding, avail_embedding[0], "Select embedding plugin")
        vector_db_choice = _auto_or_prompt("Vector DB", avail_vector_db, avail_vector_db[0], "Select vector database")

        # Rerank is optional
        if avail_rerank:
            if _prompt_confirm("Enable reranking?", default=True):
                rerank_choice = _auto_or_prompt("Rerank", avail_rerank, avail_rerank[0], "Select rerank plugin")
            else:
                rerank_choice = None
        else:
            rerank_choice = None
            _print("  Rerank: not available", "dim")

        collection_name = _prompt_text("Collection name", "default")

    # =========================================================================
    # Generate Config
    # =========================================================================

    qdrant_host = system.qdrant.host if system.qdrant.available else "localhost"
    qdrant_port = system.qdrant.port if system.qdrant.available else 6333

    config_yaml = _generate_config(
        chat=chat_choice,
        embedding=embedding_choice,
        vector_db=vector_db_choice,
        collection=collection_name,
        rerank=rerank_choice,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
    )

    # =========================================================================
    # Show Config
    # =========================================================================

    _header("Generated Configuration")

    if RICH:
        console.print(Syntax(config_yaml, "yaml", theme="monokai"))
    else:
        print(config_yaml)

    if show_config:
        return

    # =========================================================================
    # Save Config
    # =========================================================================

    _header("Saving")

    fitz_dir = FitzPaths.workspace()
    fitz_config = FitzPaths.config()

    # Confirm overwrite if exists
    if fitz_config.exists() and not non_interactive:
        if not _prompt_confirm("Config exists. Overwrite?", default=False):
            _print("Aborted.", "yellow")
            raise typer.Exit(0)

    fitz_dir.mkdir(parents=True, exist_ok=True)
    fitz_config.write_text(config_yaml)
    _print(f"✓ Saved to {fitz_config}", "green")

    # =========================================================================
    # Next Steps
    # =========================================================================

    _header("Done!")

    if RICH:
        console.print(f"""
[green]Your configuration is ready![/green]

Next steps:
  [cyan]fitz ingest ./docs {collection_name}[/cyan]  # Ingest documents
  [cyan]fitz query "your question"[/cyan]           # Query knowledge base
  [cyan]fitz doctor[/cyan]                          # Verify setup
""")
    else:
        print(f"""
Your configuration is ready!

Next steps:
  fitz ingest ./docs {collection_name}  # Ingest documents
  fitz query "your question"            # Query knowledge base
  fitz doctor                           # Verify setup
""")