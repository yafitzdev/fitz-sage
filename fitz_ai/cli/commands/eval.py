# fitz_ai/cli/commands/eval.py
"""
Evaluation and observability commands.

Commands:
    fitz eval governance-stats    - Show governance decision statistics
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Optional

import typer

from fitz_ai.cli.ui import RICH, Panel, Table, console, ui
from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)

app = typer.Typer(
    name="eval",
    help="Evaluation and observability commands.",
    no_args_is_help=True,
)


def _get_collection(collection: Optional[str]) -> str:
    """Get collection name, using default from context if not specified."""
    if collection is None:
        from fitz_ai.cli.context import CLIContext

        ctx = CLIContext.load()
        return ctx.retrieval_collection
    return collection


def _get_stats_pool(collection: str):
    """Get a connection pool for the given collection."""
    from fitz_ai.storage.postgres import get_connection_manager

    manager = get_connection_manager()
    return manager.get_pool(collection)


@app.command("governance-stats")
def governance_stats(
    collection: Optional[str] = typer.Option(
        None,
        "--collection",
        "-c",
        help="Filter by collection. Uses default if not specified.",
    ),
    days: int = typer.Option(
        7,
        "--days",
        "-d",
        help="Stats for last N days.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed breakdown including constraints and flips.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output as JSON.",
    ),
) -> None:
    """
    Show governance decision statistics.

    Displays mode distribution, triggered constraints, and behavioral changes
    over the specified time period.

    Examples:

        # Basic stats for last 7 days
        fitz eval governance-stats

        # Stats for a specific collection
        fitz eval governance-stats -c my_collection

        # Last 30 days with detailed breakdown
        fitz eval governance-stats --days 30 --verbose

        # JSON output for scripting
        fitz eval governance-stats --json
    """
    from fitz_ai.evaluation import GovernanceStats

    collection = _get_collection(collection)

    try:
        pool = _get_stats_pool(collection)
    except Exception as e:
        ui.error(f"Failed to connect to database: {e}")
        raise typer.Exit(1)

    stats = GovernanceStats(pool)

    # Get mode distribution
    try:
        distribution = stats.get_mode_distribution(collection=collection, days=days)
    except Exception as e:
        ui.error(f"Failed to get statistics: {e}")
        raise typer.Exit(1)

    # Check if there's any data
    if distribution.total_queries == 0:
        if json_output:
            print(json.dumps({"error": "No governance logs found", "collection": collection}))
        else:
            ui.warning(f"No governance logs found for collection '{collection}'.")
            ui.info("Run some queries with 'fitz query' to generate governance data.")
        return

    # Get additional data for verbose mode
    constraints = None
    flips = None
    if verbose:
        try:
            constraints = stats.get_constraint_frequency(collection=collection, days=days)
            flips = stats.detect_flips(days=days)
        except Exception as e:
            logger.warning(f"Failed to get detailed stats: {e}")

    # JSON output
    if json_output:
        output = {
            "collection": collection,
            "days": days,
            "distribution": distribution.to_dict(),
        }
        if constraints:
            output["constraints"] = [c.to_dict() for c in constraints]
        if flips:
            output["flips"] = [f.to_dict() for f in flips]
        print(json.dumps(output, indent=2))
        return

    # Console output
    _display_stats(collection, days, distribution, constraints, flips, verbose)


def _display_stats(
    collection: str,
    days: int,
    distribution,
    constraints,
    flips,
    verbose: bool,
) -> None:
    """Display stats in console with Rich formatting."""
    # Header
    ui.header(f"Governance Statistics (last {days} days)")
    print()
    ui.info(f"Collection: {collection}")
    ui.info(f"Total queries: {distribution.total_queries}")
    print()

    if RICH:
        _display_rich(distribution, constraints, flips, verbose)
    else:
        _display_plain(distribution, constraints, flips, verbose)


def _display_rich(distribution, constraints, flips, verbose: bool) -> None:
    """Rich console output with tables."""
    # Mode distribution table
    table = Table(title="Answer Modes", show_header=True, header_style="bold")
    table.add_column("Mode", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Rate", justify="right")

    # Add rows with color coding based on mode
    table.add_row(
        "CONFIDENT",
        str(distribution.confident_count),
        f"{distribution.confident_rate:.1%}",
    )
    table.add_row(
        "QUALIFIED",
        str(distribution.qualified_count),
        f"{distribution.qualified_rate:.1%}",
    )
    table.add_row(
        "DISPUTED",
        str(distribution.disputed_count),
        f"{distribution.disputed_rate:.1%}",
    )

    # Highlight abstain if high
    abstain_style = "red" if distribution.abstain_rate > 0.2 else None
    table.add_row(
        "ABSTAIN",
        str(distribution.abstain_count),
        f"{distribution.abstain_rate:.1%}",
        style=abstain_style,
    )

    console.print(table)
    print()

    # Verbose: constraint frequency
    if verbose and constraints:
        _display_constraints_rich(constraints)

    # Verbose: flips
    if verbose and flips:
        _display_flips_rich(flips)


def _display_constraints_rich(constraints) -> None:
    """Display constraint frequency table with Rich."""
    print()
    table = Table(title="Triggered Constraints", show_header=True, header_style="bold")
    table.add_column("Constraint", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Rate", justify="right")

    for cf in constraints:
        table.add_row(
            cf.constraint_name,
            str(cf.trigger_count),
            f"{cf.trigger_rate:.1%}",
        )

    console.print(table)


def _display_flips_rich(flips) -> None:
    """Display governance flips table with Rich."""
    print()

    # Separate regressions and improvements
    regressions = [f for f in flips if f.is_regression]
    improvements = [f for f in flips if f.is_improvement]
    other = [f for f in flips if not f.is_regression and not f.is_improvement]

    if regressions:
        table = Table(title="⚠️  Regressions", show_header=True, header_style="bold red")
        table.add_column("Query", style="dim", max_width=40)
        table.add_column("Change", style="red")
        table.add_column("When")

        for flip in regressions[:10]:  # Limit to 10
            query_preview = (
                flip.query_text[:37] + "..." if flip.query_text else flip.query_hash[:10]
            )
            table.add_row(
                query_preview,
                f"{flip.old_mode} → {flip.new_mode}",
                flip.new_timestamp.strftime("%Y-%m-%d %H:%M"),
            )

        console.print(table)

    if improvements:
        print()
        table = Table(title="✓ Improvements", show_header=True, header_style="bold green")
        table.add_column("Query", style="dim", max_width=40)
        table.add_column("Change", style="green")
        table.add_column("When")

        for flip in improvements[:10]:
            query_preview = (
                flip.query_text[:37] + "..." if flip.query_text else flip.query_hash[:10]
            )
            table.add_row(
                query_preview,
                f"{flip.old_mode} → {flip.new_mode}",
                flip.new_timestamp.strftime("%Y-%m-%d %H:%M"),
            )

        console.print(table)

    # Summary
    print()
    if regressions:
        ui.warning(f"{len(regressions)} regression(s) detected")
    if improvements:
        ui.success(f"{len(improvements)} improvement(s) detected")
    if other:
        ui.info(f"{len(other)} other mode change(s)")


def _display_plain(distribution, constraints, flips, verbose: bool) -> None:
    """Plain text output without Rich."""
    print("Answer Modes:")
    print("-" * 40)
    print(f"  CONFIDENT: {distribution.confident_count} ({distribution.confident_rate:.1%})")
    print(f"  QUALIFIED: {distribution.qualified_count} ({distribution.qualified_rate:.1%})")
    print(f"  DISPUTED:  {distribution.disputed_count} ({distribution.disputed_rate:.1%})")
    print(f"  ABSTAIN:   {distribution.abstain_count} ({distribution.abstain_rate:.1%})")

    if verbose and constraints:
        print()
        print("Triggered Constraints:")
        print("-" * 40)
        for cf in constraints:
            print(f"  {cf.constraint_name}: {cf.trigger_count} ({cf.trigger_rate:.1%})")

    if verbose and flips:
        print()
        regressions = [f for f in flips if f.is_regression]
        improvements = [f for f in flips if f.is_improvement]

        if regressions:
            print("Regressions:")
            print("-" * 40)
            for flip in regressions[:10]:
                query_preview = (
                    flip.query_text[:30] + "..." if flip.query_text else flip.query_hash[:10]
                )
                print(f"  {query_preview}: {flip.old_mode} -> {flip.new_mode}")

        if improvements:
            print()
            print("Improvements:")
            print("-" * 40)
            for flip in improvements[:10]:
                query_preview = (
                    flip.query_text[:30] + "..." if flip.query_text else flip.query_hash[:10]
                )
                print(f"  {query_preview}: {flip.old_mode} -> {flip.new_mode}")


# Export for main CLI
def command() -> None:
    """Run the eval command group."""
    app()


__all__ = ["app", "command"]
