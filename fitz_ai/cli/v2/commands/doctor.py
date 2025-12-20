# fitz_ai/cli/v2/commands/doctor.py
"""
System diagnostics command.

Usage:
    fitz doctor              # Run all checks
    fitz doctor --verbose    # Show more details
    fitz doctor --test       # Test actual connections
"""

from __future__ import annotations

import sys
from typing import Optional

import typer

from fitz_ai.core.config import load_config_dict, ConfigNotFoundError
from fitz_ai.core.detect import detect_all
from fitz_ai.core.paths import FitzPaths
from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)

# Rich for UI (optional)
try:
    from rich.console import Console
    from rich.panel import Panel
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


def _section(title: str) -> None:
    if RICH:
        console.print(f"\n[bold cyan]{title}[/bold cyan]")
    else:
        print(f"\n{title}")
        print("-" * len(title))


def _status(name: str, ok: bool, detail: str = "") -> None:
    """Print a status line."""
    if RICH:
        icon = "✓" if ok else "✗"
        color = "green" if ok else "red"
        detail_str = f" [dim]({detail})[/dim]" if detail else ""
        console.print(f"  [{color}]{icon}[/{color}] {name}{detail_str}")
    else:
        icon = "✓" if ok else "✗"
        detail_str = f" ({detail})" if detail else ""
        print(f"  {icon} {name}{detail_str}")


def _warning(name: str, detail: str = "") -> None:
    """Print a warning line."""
    if RICH:
        detail_str = f" [dim]({detail})[/dim]" if detail else ""
        console.print(f"  [yellow]⚠[/yellow] {name}{detail_str}")
    else:
        detail_str = f" ({detail})" if detail else ""
        print(f"  ⚠ {name}{detail_str}")


def _info(msg: str) -> None:
    """Print info line."""
    if RICH:
        console.print(f"  [dim]{msg}[/dim]")
    else:
        print(f"    {msg}")


# =============================================================================
# Check Functions
# =============================================================================


def _check_python() -> tuple[bool, str]:
    """Check Python version."""
    version = sys.version.split()[0]
    ok = sys.version_info >= (3, 10)
    return ok, f"Python {version}"


def _check_workspace() -> tuple[bool, str]:
    """Check if workspace exists."""
    workspace = FitzPaths.workspace()
    if workspace.exists():
        return True, str(workspace)
    return False, "Not found (run 'fitz init')"


def _check_config() -> tuple[bool, str, Optional[dict]]:
    """Check if config exists and is valid."""
    config_path = FitzPaths.config()

    if not config_path.exists():
        return False, "Not found (run 'fitz init')", None

    try:
        config = load_config_dict(config_path)
        return True, "Valid", config
    except Exception as e:
        return False, f"Invalid: {e}", None


def _check_dependencies() -> list[tuple[str, bool, str]]:
    """Check required Python packages."""
    packages = [
        ("typer", "CLI framework"),
        ("httpx", "HTTP client"),
        ("pydantic", "Config validation"),
        ("yaml", "YAML parsing", "pyyaml"),
        ("rich", "Pretty output", "rich"),
    ]

    results = []
    for item in packages:
        name = item[0]
        desc = item[1]
        import_name = item[2] if len(item) > 2 else item[0]

        try:
            mod = __import__(import_name.replace("-", "_"))
            version = getattr(mod, "__version__", "")
            results.append((name, True, version))
        except ImportError:
            results.append((name, False, f"pip install {import_name}"))

    return results


def _check_optional_dependencies() -> list[tuple[str, bool, str]]:
    """Check optional Python packages."""
    packages = [
        ("faiss", "Local vector DB", "faiss-cpu"),
        ("qdrant_client", "Qdrant client", "qdrant-client"),
        ("cohere", "Cohere API", "cohere"),
        ("openai", "OpenAI API", "openai"),
    ]

    results = []
    for item in packages:
        name = item[0]
        desc = item[1]
        import_name = item[2] if len(item) > 2 else item[0]

        try:
            mod = __import__(name.replace("-", "_"))
            version = getattr(mod, "__version__", "installed")
            results.append((desc, True, version))
        except ImportError:
            results.append((desc, False, "not installed"))

    return results


def _test_embedding(config: dict) -> tuple[bool, str]:
    """Test embedding plugin."""
    try:
        from fitz_ai.llm.registry import get_llm_plugin

        plugin_name = config.get("embedding", {}).get("plugin_name")
        if not plugin_name:
            return False, "Not configured"

        plugin = get_llm_plugin(plugin_type="embedding", plugin_name=plugin_name)

        # Try a test embedding
        vector = plugin.embed("test")
        if vector and len(vector) > 0:
            return True, f"{plugin_name} (dim={len(vector)})"
        return False, "Empty response"

    except Exception as e:
        return False, str(e)[:50]


def _test_chat(config: dict) -> tuple[bool, str]:
    """Test chat plugin (without making actual call)."""
    try:
        from fitz_ai.llm.registry import get_llm_plugin

        plugin_name = config.get("chat", {}).get("plugin_name")
        if not plugin_name:
            return False, "Not configured"

        # Just try to instantiate
        plugin = get_llm_plugin(plugin_type="chat", plugin_name=plugin_name)
        return True, f"{plugin_name} ready"

    except Exception as e:
        return False, str(e)[:50]


def _test_vector_db(config: dict) -> tuple[bool, str]:
    """Test vector DB connection."""
    try:
        from fitz_ai.vector_db.registry import get_vector_db_plugin

        plugin_name = config.get("vector_db", {}).get("plugin_name")
        if not plugin_name:
            return False, "Not configured"

        plugin = get_vector_db_plugin(plugin_name)

        # Try to list collections
        collections = plugin.list_collections()
        return True, f"{plugin_name} ({len(collections)} collections)"

    except Exception as e:
        return False, str(e)[:50]


def _test_rerank(config: dict) -> tuple[bool, str]:
    """Test rerank plugin."""
    rerank_config = config.get("rerank", {})

    if not rerank_config.get("enabled"):
        return True, "Disabled (optional)"

    try:
        from fitz_ai.llm.registry import get_llm_plugin

        plugin_name = rerank_config.get("plugin_name")
        if not plugin_name:
            return False, "Enabled but no plugin"

        # Just try to instantiate
        plugin = get_llm_plugin(plugin_type="rerank", plugin_name=plugin_name)
        return True, f"{plugin_name} ready"

    except Exception as e:
        return False, str(e)[:50]


# =============================================================================
# Main Command
# =============================================================================


def command(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed output.",
    ),
    test: bool = typer.Option(
        False,
        "--test",
        "-t",
        help="Test actual connections.",
    ),
) -> None:
    """
    Run diagnostics on your Fitz setup.

    Checks system requirements, configuration, and connections.

    Examples:
        fitz doctor           # Quick check
        fitz doctor -v        # Verbose output
        fitz doctor --test    # Test actual connections
    """
    issues = []
    warnings = []

    # =========================================================================
    # Header
    # =========================================================================

    _header("Fitz Doctor")

    # =========================================================================
    # Core Checks
    # =========================================================================

    _section("System")

    # Python
    ok, detail = _check_python()
    _status("Python", ok, detail)
    if not ok:
        issues.append("Python 3.10+ required")

    # Workspace
    ok, detail = _check_workspace()
    _status("Workspace", ok, detail)
    if not ok:
        warnings.append("Run 'fitz init' to create workspace")

    # Config
    ok, detail, config = _check_config()
    _status("Config", ok, detail)
    if not ok:
        warnings.append("Run 'fitz init' to create config")
        config = {}

    # =========================================================================
    # Dependencies
    # =========================================================================

    _section("Dependencies")

    for name, ok, detail in _check_dependencies():
        _status(name, ok, detail)
        if not ok:
            issues.append(f"Missing: {name}")

    # =========================================================================
    # Optional Dependencies (verbose only)
    # =========================================================================

    if verbose:
        _section("Optional Packages")
        for name, ok, detail in _check_optional_dependencies():
            if ok:
                _status(name, True, detail)
            else:
                _warning(name, detail)

    # =========================================================================
    # Services
    # =========================================================================

    _section("Services")

    system = detect_all()

    # Ollama
    if system.ollama.available:
        _status("Ollama", True, system.ollama.details)
    else:
        _warning("Ollama", system.ollama.details)

    # Qdrant
    if system.qdrant.available:
        _status("Qdrant", True, f"{system.qdrant.host}:{system.qdrant.port}")
    else:
        _warning("Qdrant", system.qdrant.details)

    # FAISS
    if system.faiss.available:
        _status("FAISS", True, "installed")
    else:
        _warning("FAISS", system.faiss.details)

    # =========================================================================
    # API Keys
    # =========================================================================

    _section("API Keys")

    for name, key_status in system.api_keys.items():
        if key_status.available:
            _status(name.capitalize(), True, "configured")
        else:
            _warning(name.capitalize(), f"${key_status.env_var} not set")

    # =========================================================================
    # Connection Tests (if --test)
    # =========================================================================

    if test and config:
        _section("Connection Tests")

        # Embedding
        ok, detail = _test_embedding(config)
        _status("Embedding", ok, detail)
        if not ok:
            issues.append(f"Embedding failed: {detail}")

        # Chat
        ok, detail = _test_chat(config)
        _status("Chat", ok, detail)
        if not ok:
            issues.append(f"Chat failed: {detail}")

        # Vector DB
        ok, detail = _test_vector_db(config)
        _status("Vector DB", ok, detail)
        if not ok:
            issues.append(f"Vector DB failed: {detail}")

        # Rerank
        ok, detail = _test_rerank(config)
        _status("Rerank", ok, detail)

    # =========================================================================
    # Config Summary (verbose)
    # =========================================================================

    if verbose and config:
        _section("Config Summary")

        chat = config.get("chat", {}).get("plugin_name", "?")
        embedding = config.get("embedding", {}).get("plugin_name", "?")
        vector_db = config.get("vector_db", {}).get("plugin_name", "?")
        rerank = config.get("rerank", {})
        rerank_str = rerank.get("plugin_name", "disabled") if rerank.get("enabled") else "disabled"

        _info(f"Chat: {chat}")
        _info(f"Embedding: {embedding}")
        _info(f"Vector DB: {vector_db}")
        _info(f"Rerank: {rerank_str}")

    # =========================================================================
    # Summary
    # =========================================================================

    print()

    if issues:
        if RICH:
            console.print(Panel(
                "\n".join(f"[red]✗[/red] {issue}" for issue in issues),
                title="[red]Issues Found[/red]",
                border_style="red",
            ))
        else:
            print("Issues Found:")
            for issue in issues:
                print(f"  ✗ {issue}")

        raise typer.Exit(1)

    elif warnings:
        if RICH:
            console.print("[yellow]Some warnings, but Fitz should work.[/yellow]")
        else:
            print("Some warnings, but Fitz should work.")

        if not test:
            _info("Run 'fitz doctor --test' to verify connections")

    else:
        if RICH:
            console.print("[green bold]✓ All checks passed![/green bold]")
        else:
            print("✓ All checks passed!")

        if not test:
            _info("Run 'fitz doctor --test' to verify connections")