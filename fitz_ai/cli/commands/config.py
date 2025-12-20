# fitz_ai/cli/commands/config.py
"""
Configuration command.

Usage:
    fitz config                # Show current config
    fitz config --json         # Output as JSON
    fitz config --path         # Show config file path
    fitz config --edit         # Open config in editor
"""

from __future__ import annotations

from pathlib import Path
import os
import subprocess

import typer

from fitz_ai.core.config import load_config_dict, ConfigNotFoundError
from fitz_ai.core.paths import FitzPaths
from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)

# Rich for UI (optional)
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
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


def _success(msg: str) -> None:
    if RICH:
        console.print(f"[green]✓[/green] {msg}")
    else:
        print(f"✓ {msg}")


# =============================================================================
# Display Functions
# =============================================================================


def _show_config_summary(config: dict) -> None:
    """Show a summary table of config settings."""
    print()

    if RICH:
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Component", style="cyan")
        table.add_column("Plugin", style="green")
        table.add_column("Details", style="dim")

        # Chat
        chat = config.get("chat", {})
        chat_plugin = chat.get("plugin_name", "?")
        chat_model = chat.get("kwargs", {}).get("model", "")
        table.add_row("Chat", chat_plugin, chat_model)

        # Embedding
        emb = config.get("embedding", {})
        emb_plugin = emb.get("plugin_name", "?")
        emb_model = emb.get("kwargs", {}).get("model", "")
        table.add_row("Embedding", emb_plugin, emb_model)

        # Vector DB
        vdb = config.get("vector_db", {})
        vdb_plugin = vdb.get("plugin_name", "?")
        vdb_host = vdb.get("kwargs", {}).get("host", "")
        vdb_port = vdb.get("kwargs", {}).get("port", "")
        vdb_details = f"{vdb_host}:{vdb_port}" if vdb_host else ""
        table.add_row("Vector DB", vdb_plugin, vdb_details)

        # Retriever
        ret = config.get("retriever", {})
        ret_plugin = ret.get("plugin_name", "dense")
        ret_collection = ret.get("collection", "default")
        ret_top_k = ret.get("top_k", 5)
        table.add_row("Retriever", ret_plugin, f"collection={ret_collection}, top_k={ret_top_k}")

        # Rerank
        rerank = config.get("rerank", {})
        rerank_enabled = rerank.get("enabled", False)
        if rerank_enabled:
            rerank_plugin = rerank.get("plugin_name", "?")
            rerank_model = rerank.get("kwargs", {}).get("model", "")
            table.add_row("Rerank", rerank_plugin, rerank_model)
        else:
            table.add_row("Rerank", "[dim]disabled[/dim]", "")

        # RGS
        rgs = config.get("rgs", {})
        citations = "on" if rgs.get("enable_citations", True) else "off"
        grounding = "strict" if rgs.get("strict_grounding", True) else "relaxed"
        table.add_row("RGS", f"citations={citations}", f"grounding={grounding}")

        console.print(table)
    else:
        print("Configuration Summary:")
        print("-" * 40)

        chat = config.get("chat", {})
        print(f"  Chat:      {chat.get('plugin_name', '?')}")

        emb = config.get("embedding", {})
        print(f"  Embedding: {emb.get('plugin_name', '?')}")

        vdb = config.get("vector_db", {})
        print(f"  Vector DB: {vdb.get('plugin_name', '?')}")

        ret = config.get("retriever", {})
        print(f"  Retriever: {ret.get('plugin_name', 'dense')} (collection={ret.get('collection', 'default')})")

        rerank = config.get("rerank", {})
        if rerank.get("enabled"):
            print(f"  Rerank:    {rerank.get('plugin_name', '?')}")
        else:
            print("  Rerank:    disabled")

    print()


def _show_config_yaml(config_path: Path) -> None:
    """Show raw YAML config."""
    try:
        content = config_path.read_text()
    except Exception as e:
        _error(f"Failed to read config: {e}")
        return

    print()
    if RICH:
        console.print(Syntax(content, "yaml", theme="monokai", line_numbers=True))
    else:
        print(content)
    print()


def _show_config_json(config: dict) -> None:
    """Show config as JSON."""
    import json

    print()
    if RICH:
        json_str = json.dumps(config, indent=2)
        console.print(Syntax(json_str, "json", theme="monokai"))
    else:
        print(json.dumps(config, indent=2))
    print()


def _open_in_editor(config_path: Path) -> None:
    """Open config file in default editor."""
    # Try common editors
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL")

    if not editor:
        # Try to find a common editor
        for candidate in ["code", "nano", "vim", "vi", "notepad"]:
            try:
                # Check if editor exists
                result = subprocess.run(
                    ["which", candidate] if os.name != "nt" else ["where", candidate],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    editor = candidate
                    break
            except Exception:
                continue

    if not editor:
        _error("No editor found. Set EDITOR environment variable.")
        _print(f"Config path: {config_path}", "dim")
        return

    try:
        _print(f"Opening in {editor}...", "dim")
        subprocess.run([editor, str(config_path)])
    except Exception as e:
        _error(f"Failed to open editor: {e}")
        _print(f"Config path: {config_path}", "dim")


# =============================================================================
# Main Command
# =============================================================================


def command(
    show_path: bool = typer.Option(
        False,
        "--path",
        "-p",
        help="Show config file path only.",
    ),
    as_json: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON.",
    ),
    raw: bool = typer.Option(
        False,
        "--raw",
        "-r",
        help="Show raw YAML file.",
    ),
    edit: bool = typer.Option(
        False,
        "--edit",
        "-e",
        help="Open config in editor.",
    ),
) -> None:
    """
    View and manage Fitz configuration.

    Shows a summary of your current configuration settings.

    Examples:
        fitz config              # Show config summary
        fitz config --raw        # Show raw YAML
        fitz config --json       # Output as JSON
        fitz config --path       # Show file path
        fitz config --edit       # Open in editor
    """
    # =========================================================================
    # Get config path
    # =========================================================================

    config_path = FitzPaths.config()

    # =========================================================================
    # Path only mode
    # =========================================================================

    if show_path:
        if config_path.exists():
            print(str(config_path))
        else:
            _error(f"Config not found: {config_path}")
            _print("Run 'fitz init' to create one.", "dim")
            raise typer.Exit(1)
        return

    # =========================================================================
    # Edit mode
    # =========================================================================

    if edit:
        if not config_path.exists():
            _error("No config file to edit.")
            _print("Run 'fitz init' first.", "dim")
            raise typer.Exit(1)

        _open_in_editor(config_path)
        return

    # =========================================================================
    # Load config
    # =========================================================================

    if not config_path.exists():
        _error("No configuration found.")
        print()
        _print(f"Expected: {config_path}", "dim")
        _print("Run 'fitz init' to create a configuration.", "dim")
        raise typer.Exit(1)

    try:
        config = load_config_dict(config_path)
    except Exception as e:
        _error(f"Failed to load config: {e}")
        raise typer.Exit(1)

    # =========================================================================
    # Header
    # =========================================================================

    _header("Fitz Configuration")
    _print(f"File: {config_path}", "dim")

    # =========================================================================
    # JSON output
    # =========================================================================

    if as_json:
        _show_config_json(config)
        return

    # =========================================================================
    # Raw YAML output
    # =========================================================================

    if raw:
        _show_config_yaml(config_path)
        return

    # =========================================================================
    # Summary view (default)
    # =========================================================================

    _show_config_summary(config)

    # Show helpful hints
    if RICH:
        console.print("[dim]Tip: Use --raw for full YAML, --edit to modify[/dim]")
    else:
        print("Tip: Use --raw for full YAML, --edit to modify")