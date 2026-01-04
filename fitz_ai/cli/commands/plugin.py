# fitz_ai/cli/commands/plugin.py
"""
Plugin command - LLM-powered plugin generator.

Usage:
    fitz plugin              # Interactive wizard
    fitz plugin llm-chat anthropic  # Direct generation
"""

from __future__ import annotations

from typing import Optional

import typer

from fitz_ai.cli.ui import ui
from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Plugin Type Selection
# =============================================================================


def _get_plugin_type_choices() -> list[tuple[str, str]]:
    """Get available plugin types with descriptions."""
    from fitz_ai.plugin_gen.types import PluginType

    return [(pt.value, pt.description) for pt in PluginType]


def _select_plugin_type() -> str:
    """Interactive plugin type selection."""
    choices = _get_plugin_type_choices()

    ui.print("Plugin types:", "bold")
    for i, (value, desc) in enumerate(choices, 1):
        ui.print(f"  {i}. {value}: {desc}")

    print()
    default_idx = 1
    result = ui.prompt_int("Select plugin type", default=default_idx)

    idx = max(1, min(result, len(choices))) - 1
    return choices[idx][0]


# =============================================================================
# Progress Display
# =============================================================================


def _create_progress_callback():
    """Create a progress callback for generation."""

    def callback(message: str) -> None:
        ui.info(message)

    return callback


# =============================================================================
# Main Command
# =============================================================================


def command(
    plugin_type: Optional[str] = typer.Argument(
        None,
        help="Plugin type (llm-chat, llm-embedding, chunker, etc.).",
    ),
    description: Optional[str] = typer.Argument(
        None,
        help="Description of the plugin to generate (e.g., 'anthropic', 'sentence-based chunker').",
    ),
    chat_plugin: Optional[str] = typer.Option(
        None,
        "--chat-plugin",
        "-p",
        help="Chat plugin to use for generation (default: from config).",
    ),
    tier: str = typer.Option(
        "smart",
        "--tier",
        "-t",
        help="Model tier for generation (smart, fast, balanced).",
    ),
) -> None:
    """
    Generate a plugin using LLM.

    The LLM generates complete, working plugins from natural language
    descriptions. Generated plugins are validated and saved to the
    user plugins directory (~/.fitz/plugins/).

    Examples:
        fitz plugin                            # Interactive wizard
        fitz plugin llm-chat anthropic         # Generate Anthropic chat plugin
        fitz plugin chunker "sentence-based"   # Generate custom chunker
        fitz plugin constraint "source check"  # Generate constraint plugin
    """
    from fitz_ai.cli.context import CLIContext
    from fitz_ai.plugin_gen import PluginGenerator, PluginType

    # =========================================================================
    # Load Config
    # =========================================================================

    ctx = CLIContext.load()

    # Use configured chat plugin if not specified
    if chat_plugin is None:
        chat_plugin = ctx.chat_plugin

    # =========================================================================
    # Header
    # =========================================================================

    ui.header("Fitz Plugin Generator", "Create plugins with AI")

    # =========================================================================
    # Plugin Type Selection
    # =========================================================================

    if plugin_type is None:
        ui.section("Plugin Type")
        plugin_type = _select_plugin_type()
    else:
        # Validate plugin type
        try:
            PluginType(plugin_type)
        except ValueError:
            valid_types = [pt.value for pt in PluginType]
            ui.error(f"Invalid plugin type: {plugin_type}")
            ui.info(f"Valid types: {', '.join(valid_types)}")
            raise typer.Exit(1)

    # =========================================================================
    # Description Input
    # =========================================================================

    if description is None:
        ui.section("Description")
        ui.info(
            "Describe what you want the plugin to do.\n"
            "For API providers, just use the provider name (e.g., 'anthropic').\n"
            "For custom logic, describe the behavior."
        )
        print()
        description = ui.prompt_text("Description")

    if not description.strip():
        ui.error("Description cannot be empty")
        raise typer.Exit(1)

    # =========================================================================
    # Generation
    # =========================================================================

    ui.section("Generating Plugin")

    pt = PluginType(plugin_type)
    ui.info(f"Type: {pt.display_name}")
    ui.info(f"Description: {description}")
    print()

    generator = PluginGenerator(chat_plugin=chat_plugin, tier=tier)

    progress_callback = _create_progress_callback()

    result = generator.generate(
        plugin_type=pt,
        description=description,
        progress_callback=progress_callback,
    )

    # =========================================================================
    # Results
    # =========================================================================

    print()
    ui.section("Validation Results")

    for v in result.validations:
        if v.success:
            ui.status(v.level.value, True)
        else:
            ui.status(v.level.value, False)
            if v.error:
                ui.print(f"    {v.error}", "dim")

    print()

    if result.success:
        ui.success("Plugin created successfully!")
        ui.info(f"Name: {result.plugin_name}")
        ui.info(f"Path: {result.path}")
        print()
        ui.info("The plugin is now available for use. Set any required API keys.")
    else:
        ui.error(f"Plugin generation failed after {result.attempts} attempts")
        if result.first_error:
            ui.info(f"Error: {result.first_error}")
        raise typer.Exit(1)
