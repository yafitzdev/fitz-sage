# fitz_ai/cli/commands/init.py
# PLUGIN-AGNOSTIC VERSION
# Uses plugin registries for auto-discovery - NO hardcoded plugin names!

"""
Interactive setup wizard for Fitz.

Detects available providers and creates a working configuration.

Usage:
    fitz init              # Interactive wizard
    fitz init -y           # Auto-detect and use defaults
    fitz init --show-config  # Preview config without saving
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import typer

# Import centralized detection
from fitz_ai.core.detect import detect_all

# Import plugin registries
from fitz_ai.llm.registry import available_llm_plugins
from fitz_ai.vector_db.registry import available_vector_db_plugins

# Rich for pretty output (optional, falls back gracefully)
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Confirm, Prompt

    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


# =============================================================================
# Engine Discovery
# =============================================================================


def _is_local_plugin(plugin_name: str) -> bool:
    """Check if a plugin is local/Ollama-based (doesn't require API keys)."""
    plugin_lower = plugin_name.lower()
    return any(x in plugin_lower for x in ("ollama", "local", "offline"))


def _is_faiss_plugin(plugin_name: str) -> bool:
    """Check if a plugin is FAISS-based (local vector DB)."""
    plugin_lower = plugin_name.lower()
    return "faiss" in plugin_lower or plugin_lower == "local"


def detect_engines() -> dict[str, str]:
    """Auto-discover all registered engines."""
    # Import engines to trigger registration
    try:
        import fitz_ai.engines.classic_rag  # noqa
    except ImportError:
        pass

    try:
        import fitz_ai.engines.clara  # noqa
    except ImportError:
        pass

    # Get engines from registry
    try:
        from fitz_ai.runtime import list_engines_with_info
        return list_engines_with_info()
    except ImportError:
        # Fallback if runtime not available
        return {"classic_rag": "Retrieval-augmented generation using vector search"}


def prompt_engine_choice(engines: dict[str, str]) -> str:
    """Prompt user to select an engine."""
    if not engines:
        raise ValueError("No engines available")

    # If only one engine, auto-select
    if len(engines) == 1:
        engine = list(engines.keys())[0]
        if RICH_AVAILABLE:
            console.print(f"  [dim]Engine:[/dim] [green]{engine}[/green] [dim](auto-selected)[/dim]")
        else:
            print(f"  Engine: {engine} (auto-selected)")
        return engine

    # Show available engines
    if RICH_AVAILABLE:
        console.print("\n[bold blue]Available Engines[/bold blue]")
        for i, (name, desc) in enumerate(engines.items(), 1):
            status = "[bold green]Production[/bold green]" if name == "classic_rag" else "[yellow]Experimental[/yellow]"
            console.print(f"  {i}. [bold]{name}[/bold] ({status})")
            console.print(f"     [dim]{desc}[/dim]")
    else:
        print("\nAvailable Engines:")
        for i, (name, desc) in enumerate(engines.items(), 1):
            status = "Production" if name == "classic_rag" else "Experimental"
            print(f"  {i}. {name} ({status})")
            print(f"     {desc}")

    # Get user choice
    choices = list(engines.keys())
    default_idx = 0
    if "classic_rag" in choices:
        default_idx = choices.index("classic_rag")

    if RICH_AVAILABLE:
        choice = Prompt.ask(
            "\nSelect engine",
            choices=[str(i) for i in range(1, len(choices) + 1)],
            default=str(default_idx + 1),
        )
    else:
        choice = input(f"\nSelect engine [1-{len(choices)}] ({default_idx + 1}): ").strip()
        if not choice:
            choice = str(default_idx + 1)

    return choices[int(choice) - 1]


# =============================================================================
# Plugin Discovery (AGNOSTIC!)
# =============================================================================


def get_plugin_defaults(plugin_name: str, plugin_type: str) -> Dict[str, Any]:
    """Get default configuration for a plugin."""
    defaults = {
        "temperature": 0.2,
    }
    return defaults


def generate_plugin_config(plugin_name: str, plugin_type: str, **kwargs) -> str:
    """Generate YAML config for any plugin dynamically."""
    # Build kwargs section
    if kwargs:
        kwargs_lines = []
        for key, value in kwargs.items():
            if isinstance(value, str):
                kwargs_lines.append(f'    {key}: "{value}"')
            else:
                kwargs_lines.append(f"    {key}: {value}")
        kwargs_str = "\n".join(kwargs_lines)
    else:
        kwargs_str = "    {}"

    return f"""  plugin_name: {plugin_name}
  kwargs:
{kwargs_str}"""


# =============================================================================
# Config Generation (AGNOSTIC!)
# =============================================================================


def generate_config(
    chat_provider: str,
    embedding_provider: str,
    vector_db: str,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    collection: str = "default",
    enable_rerank: bool = False,
    rerank_provider: str | None = None,
) -> str:
    """Generate YAML config dynamically based on selected plugins."""

    # Generate each section dynamically
    chat_config = generate_plugin_config(chat_provider, "chat")
    embedding_config = generate_plugin_config(embedding_provider, "embedding")

    # Vector DB might need host/port
    vector_db_kwargs = {}
    if vector_db == "qdrant":
        vector_db_kwargs = {"host": qdrant_host, "port": qdrant_port}
    vector_db_config = generate_plugin_config(vector_db, "vector_db", **vector_db_kwargs)

    # Rerank config
    if enable_rerank and rerank_provider:
        rerank_config = f"""rerank:
  enabled: true
{generate_plugin_config(rerank_provider, "rerank")}"""
    else:
        rerank_config = """rerank:
  enabled: false"""

    # Build full config
    config = f"""# Fitz RAG Configuration
# Generated by: fitz init
# 
# Edit this file to customize your setup.
# Documentation: https://github.com/yafitzdev/fitz-ai

# =============================================================================
# Chat Configuration
# =============================================================================
chat:
{chat_config}

# =============================================================================
# Embedding Configuration  
# =============================================================================
embedding:
{embedding_config}

# =============================================================================
# Vector Database Configuration
# =============================================================================
vector_db:
{vector_db_config}

# =============================================================================
# Retriever Configuration
# =============================================================================
retriever:
  plugin_name: dense
  collection: {collection}
  top_k: 5

# =============================================================================
# Reranker (improves retrieval quality)
# =============================================================================
{rerank_config}

# =============================================================================
# RGS (Retrieval-Guided Synthesis)
# =============================================================================
rgs:
  enable_citations: true
  strict_grounding: true
  max_chunks: 8
  include_query_in_context: true
  source_label_prefix: S

# =============================================================================
# Logging
# =============================================================================
logging:
  level: INFO
"""
    return config


# =============================================================================
# Display Helpers
# =============================================================================


def print_header(text: str) -> None:
    """Print a header."""
    if RICH_AVAILABLE:
        console.print(f"\n[bold blue]{text}[/bold blue]")
    else:
        print(f"\n{text}")


def print_status(name: str, available: bool, details: str = "") -> None:
    """Print status of a provider."""
    if RICH_AVAILABLE:
        icon = "✓" if available else "✗"
        color = "green" if available else "red"
        console.print(f"  [{color}]{icon}[/{color}] {name}: {details}")
    else:
        icon = "✓" if available else "✗"
        print(f"  {icon} {name}: {details}")


def prompt_choice(prompt: str, choices: list[str], default: str = None) -> str:
    """Prompt user for a choice."""
    choices_str = "/".join(choices)
    default_str = f" ({default})" if default else ""

    while True:
        if RICH_AVAILABLE:
            response = Prompt.ask(f"{prompt} ", default=default or "", choices=choices)
        else:
            response = input(f"{prompt} [{choices_str}]{default_str}: ").strip()
            if not response and default:
                response = default

        if response in choices:
            return response

        print(f"Invalid choice. Please enter one of: {', '.join(choices)}")


def prompt_confirm(prompt: str, default: bool = True) -> bool:
    """Prompt user for yes/no."""
    if RICH_AVAILABLE:
        return Confirm.ask(prompt, default=default)
    else:
        default_str = "Y/n" if default else "y/N"
        response = input(f"{prompt} [{default_str}]: ").strip().lower()
        if not response:
            return default
        return response in ("y", "yes")


def auto_select_or_prompt(
    category: str,
    available: list[str],
    default: str,
    prompt_text: str,
) -> str:
    """Auto-select if only one option, otherwise prompt user."""
    if len(available) == 1:
        choice = available[0]
        if RICH_AVAILABLE:
            console.print(f"  [dim]{category}:[/dim] [green]{choice}[/green] [dim](auto-selected)[/dim]")
        else:
            print(f"  {category}: {choice} (auto-selected)")
        return choice
    else:
        return prompt_choice(prompt_text, available, default=default)


# =============================================================================
# Main Command
# =============================================================================


def command(
    non_interactive: bool = typer.Option(
        False,
        "--non-interactive",
        "-y",
        help="Use detected defaults without prompting",
    ),
    show_config: bool = typer.Option(
        False,
        "--show-config",
        help="Only show what config would be generated",
    ),
) -> None:
    """
    Initialize Fitz with an interactive setup wizard.

    Detects available providers (API keys, Ollama, Qdrant) and
    creates a working configuration file.

    Examples:
        fitz init              # Interactive wizard
        fitz init -y           # Auto-detect and use defaults
        fitz init --show-config  # Preview config without saving
    """
    from fitz_ai.core.paths import FitzPaths

    # Header
    if RICH_AVAILABLE:
        console.print(
            Panel.fit(
                "[bold]Fitz Setup Wizard[/bold]\nLet's configure your RAG pipeline!",
                border_style="blue",
            )
        )
    else:
        print("\n" + "=" * 60)
        print("Fitz Setup Wizard")
        print("Let's configure your RAG pipeline!")
        print("=" * 60)

    # ==========================================================================
    # Engine Selection
    # ==========================================================================

    engines = detect_engines()

    if non_interactive or len(engines) == 1:
        engine = "classic_rag" if "classic_rag" in engines else list(engines.keys())[0]
    else:
        engine = prompt_engine_choice(engines)

    # For now, only classic_rag is fully supported
    if engine != "classic_rag":
        if RICH_AVAILABLE:
            console.print(f"\n[yellow]Setup wizard for '{engine}' coming soon![/yellow]")
            console.print("Please create config manually or use defaults.")
        else:
            print(f"\nSetup wizard for '{engine}' coming soon!")
            print("Please create config manually or use defaults.")
        raise typer.Exit(code=0)

    # ==========================================================================
    # Plugin Discovery (AGNOSTIC!)
    # ==========================================================================

    print_header("Detecting Available Plugins...")

    # Discover plugins from registries
    available_chat = available_llm_plugins("chat")
    available_embeddings = available_llm_plugins("embedding")
    available_rerank = available_llm_plugins("rerank")
    available_vector_dbs = available_vector_db_plugins()

    # System detection for API keys and services
    system = detect_all()

    # ==========================================================================
    # Display Available Plugins (AGNOSTIC!)
    # ==========================================================================

    print_header("Available Chat Plugins")
    for plugin in sorted(available_chat):
        available = False
        if _is_local_plugin(plugin):
            available = system.ollama.available
        elif plugin in system.api_keys:
            available = system.api_keys[plugin].available

        status = "✓ Available" if available else "✗ Not configured"
        print_status(plugin, available, status)

    print_header("Available Embedding Plugins")
    for plugin in sorted(available_embeddings):
        available = False
        if _is_local_plugin(plugin):
            available = system.ollama.available
        elif plugin in system.api_keys:
            available = system.api_keys[plugin].available

        status = "✓ Available" if available else "✗ Not configured"
        print_status(plugin, available, status)

    print_header("Available Rerank Plugins")
    for plugin in sorted(available_rerank):
        available = False
        if plugin in system.api_keys:
            available = system.api_keys[plugin].available

        status = "✓ Available" if available else "✗ Not configured"
        print_status(plugin, available, status)

    print_header("Available Vector Database Plugins")
    for plugin in sorted(available_vector_dbs):
        available = False
        if _is_faiss_plugin(plugin):
            available = system.faiss.available
        elif plugin == "qdrant":
            available = system.qdrant.available

        status = "✓ Available" if available else "✗ Not configured"
        print_status(plugin, available, status)

    # ==========================================================================
    # Filter to Available Only
    # ==========================================================================

    available_chat_filtered = []
    for plugin in available_chat:
        if plugin in ["ollama", "local"] and system.ollama.available:
            available_chat_filtered.append(plugin)
        elif plugin in system.api_keys and system.api_keys[plugin].available:
            available_chat_filtered.append(plugin)

    available_embeddings_filtered = []
    for plugin in available_embeddings:
        if plugin in ["ollama", "local"] and system.ollama.available:
            available_embeddings_filtered.append(plugin)
        elif plugin in system.api_keys and system.api_keys[plugin].available:
            available_embeddings_filtered.append(plugin)

    available_rerank_filtered = []
    for plugin in available_rerank:
        if plugin in system.api_keys and system.api_keys[plugin].available:
            available_rerank_filtered.append(plugin)

    available_vector_dbs_filtered = []
    for plugin in available_vector_dbs:
        if plugin in ["local-faiss", "faiss"] and system.faiss.available:
            available_vector_dbs_filtered.append(plugin)
        elif plugin == "qdrant" and system.qdrant.available:
            available_vector_dbs_filtered.append(plugin)

    # ==========================================================================
    # Check we have minimum required plugins
    # ==========================================================================

    if not available_chat_filtered:
        if RICH_AVAILABLE:
            console.print("\n[red]No chat plugins available![/red]")
            console.print("Set an API key or install Ollama.")
        else:
            print("\nNo chat plugins available!")
            print("Set an API key or install Ollama.")
        raise typer.Exit(code=1)

    if not available_embeddings_filtered:
        if RICH_AVAILABLE:
            console.print("\n[red]No embedding plugins available![/red]")
        else:
            print("\nNo embedding plugins available!")
        raise typer.Exit(code=1)

    if not available_vector_dbs_filtered:
        if RICH_AVAILABLE:
            console.print("\n[red]No vector database plugins available![/red]")
            console.print("Install FAISS (pip install faiss-cpu) or start Qdrant.")
        else:
            print("\nNo vector database plugins available!")
            print("Install FAISS (pip install faiss-cpu) or start Qdrant.")
        raise typer.Exit(code=1)

    # ==========================================================================
    # User Selection
    # ==========================================================================

    chat_default = available_chat_filtered[0]
    embedding_default = available_embeddings_filtered[0]
    vector_db_default = available_vector_dbs_filtered[0]
    rerank_default = available_rerank_filtered[0] if available_rerank_filtered else None

    if non_interactive:
        chat_choice = chat_default
        embedding_choice = embedding_default
        vector_db_choice = vector_db_default
        collection_name = "default"
        enable_rerank = rerank_default is not None
        rerank_choice = rerank_default
    else:
        print_header("Configuration Options")

        chat_choice = auto_select_or_prompt(
            "Chat",
            available_chat_filtered,
            chat_default,
            "Select chat plugin",
        )

        embedding_choice = auto_select_or_prompt(
            "Embedding",
            available_embeddings_filtered,
            embedding_default,
            "Select embedding plugin",
        )

        vector_db_choice = auto_select_or_prompt(
            "Vector DB",
            available_vector_dbs_filtered,
            vector_db_default,
            "Select vector database plugin",
        )

        # Rerank
        if available_rerank_filtered:
            enable_rerank = prompt_confirm("Enable reranking?", default=True)
            if enable_rerank:
                rerank_choice = auto_select_or_prompt(
                    "Rerank",
                    available_rerank_filtered,
                    rerank_default,
                    "Select rerank plugin",
                )
            else:
                rerank_choice = None
        else:
            enable_rerank = False
            rerank_choice = None
            if RICH_AVAILABLE:
                console.print("  [dim]Rerank:[/dim] [yellow]No rerank plugins available[/yellow]")
            else:
                print("  Rerank: No rerank plugins available")

        # Collection name
        if RICH_AVAILABLE:
            collection_name = Prompt.ask("Collection name", default="default")
        else:
            collection_name = input("Collection name [default]: ").strip() or "default"

    # ==========================================================================
    # Generate Config (AGNOSTIC!)
    # ==========================================================================

    qdrant_host = system.qdrant.host if system.qdrant.available else "localhost"
    qdrant_port = system.qdrant.port if system.qdrant.available else 6333

    config = generate_config(
        chat_provider=chat_choice,
        embedding_provider=embedding_choice,
        vector_db=vector_db_choice,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        collection=collection_name,
        enable_rerank=enable_rerank,
        rerank_provider=rerank_choice,
    )

    # ==========================================================================
    # Output
    # ==========================================================================

    print_header("Generated Configuration")

    if RICH_AVAILABLE:
        from rich.syntax import Syntax
        console.print(Syntax(config, "yaml", theme="monokai", line_numbers=False))
    else:
        print(config)

    if show_config:
        return

    # Save config
    print_header("Saving Configuration")

    # Determine save location
    fitz_dir = FitzPaths.workspace()
    fitz_config = FitzPaths.config()

    if not non_interactive:
        if fitz_config.exists():
            if not prompt_confirm("Overwrite existing config file?", default=False):
                print("Aborted.")
                raise typer.Exit(code=0)

    fitz_dir.mkdir(parents=True, exist_ok=True)
    fitz_config.write_text(config)
    print(f"✓ Saved to {fitz_config}")

    # ==========================================================================
    # Next Steps
    # ==========================================================================

    print_header("Setup Complete!")

    next_steps = f"""
Your configuration is ready! Next steps:

1. Ingest some documents:
   fitz ingest ./your_docs {collection_name}

2. Query your documents:
   fitz query "What is in my documents?"

3. Check your setup:
   fitz doctor
"""
    print(next_steps)