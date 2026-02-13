# fitz_ai/cli/commands/eval.py
"""
Evaluation and observability commands.

Commands:
    fitz eval governance-stats    - Show governance decision statistics
    fitz eval beir                - Run BEIR retrieval benchmark (cross-RAG comparison)
    fitz eval rgb                 - Run RGB robustness tests
    fitz eval fitz-gov            - Run fitz-gov governance benchmark (Fitz's moat)
    fitz eval dashboard           - Display benchmark results dashboard
    fitz eval all                 - Run all benchmarks
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import typer

from fitz_ai.cli.ui import RICH, Table, console, ui
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
        "TRUSTWORTHY",
        str(distribution.trustworthy_count),
        f"{distribution.trustworthy_rate:.1%}",
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
        table.add_column("Version", style="dim")
        table.add_column("When")

        for flip in regressions[:10]:  # Limit to 10
            query_preview = (
                flip.query_text[:37] + "..." if flip.query_text else flip.query_hash[:10]
            )
            version_change = ""
            if flip.old_version or flip.new_version:
                old_v = flip.old_version or "?"
                new_v = flip.new_version or "?"
                version_change = f"{old_v} → {new_v}"
            table.add_row(
                query_preview,
                f"{flip.old_mode} → {flip.new_mode}",
                version_change,
                flip.new_timestamp.strftime("%Y-%m-%d %H:%M"),
            )

        console.print(table)

    if improvements:
        print()
        table = Table(title="✓ Improvements", show_header=True, header_style="bold green")
        table.add_column("Query", style="dim", max_width=40)
        table.add_column("Change", style="green")
        table.add_column("Version", style="dim")
        table.add_column("When")

        for flip in improvements[:10]:
            query_preview = (
                flip.query_text[:37] + "..." if flip.query_text else flip.query_hash[:10]
            )
            version_change = ""
            if flip.old_version or flip.new_version:
                old_v = flip.old_version or "?"
                new_v = flip.new_version or "?"
                version_change = f"{old_v} → {new_v}"
            table.add_row(
                query_preview,
                f"{flip.old_mode} → {flip.new_mode}",
                version_change,
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
    print(f"  TRUSTWORTHY: {distribution.trustworthy_count} ({distribution.trustworthy_rate:.1%})")
    print(f"  DISPUTED:    {distribution.disputed_count} ({distribution.disputed_rate:.1%})")
    print(f"  ABSTAIN:     {distribution.abstain_count} ({distribution.abstain_rate:.1%})")

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
                version_info = ""
                if flip.old_version or flip.new_version:
                    version_info = f" [v{flip.old_version or '?'} -> v{flip.new_version or '?'}]"
                print(f"  {query_preview}: {flip.old_mode} -> {flip.new_mode}{version_info}")

        if improvements:
            print()
            print("Improvements:")
            print("-" * 40)
            for flip in improvements[:10]:
                query_preview = (
                    flip.query_text[:30] + "..." if flip.query_text else flip.query_hash[:10]
                )
                version_info = ""
                if flip.old_version or flip.new_version:
                    version_info = f" [v{flip.old_version or '?'} -> v{flip.new_version or '?'}]"
                print(f"  {query_preview}: {flip.old_mode} -> {flip.new_mode}{version_info}")


# =============================================================================
# Benchmark Commands
# =============================================================================


def _get_engine(collection: str, engine_name: str | None = None):
    """Get an engine instance for the given collection.

    Uses the default engine unless overridden.
    """
    from fitz_ai.config import load_engine_config
    from fitz_ai.runtime import create_engine, get_default_engine

    if engine_name is None:
        engine_name = get_default_engine()

    config = load_engine_config(engine_name)
    config.collection = collection
    return create_engine(engine_name, config=config)


@app.command("beir")
def beir_benchmark(
    dataset: Annotated[
        list[str],
        typer.Option(
            "--dataset",
            "-d",
            help="BEIR dataset(s) to evaluate. Can specify multiple.",
        ),
    ] = ["scifact"],
    data_dir: Annotated[
        Optional[str],
        typer.Option(
            "--data-dir",
            help="Directory to store downloaded datasets. Defaults to ~/.fitz/beir_data/",
        ),
    ] = None,
    collection: Annotated[
        Optional[str],
        typer.Option(
            "--collection",
            "-c",
            help="Collection to use for evaluation.",
        ),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output file path for results (JSON).",
        ),
    ] = None,
) -> None:
    """
    Run BEIR retrieval benchmark.

    Evaluates retrieval quality using industry-standard BEIR datasets.
    Measures nDCG@10, Recall@100, and other metrics.

    Requires: pip install fitz-ai[benchmarks]

    Examples:

        # Run on default dataset (scifact)
        fitz eval beir

        # Run on multiple datasets
        fitz eval beir -d scifact -d nfcorpus

        # Save results
        fitz eval beir -d scifact -o results/beir_scifact.json
    """
    try:
        from fitz_ai.evaluation.benchmarks.beir import BEIRBenchmark
    except ImportError:
        ui.error("BEIR benchmark requires: pip install fitz-ai[benchmarks]")
        raise typer.Exit(1)

    collection = _get_collection(collection)
    engine = _get_engine(collection)

    benchmark = BEIRBenchmark(data_dir=data_dir)

    ui.header("BEIR Retrieval Benchmark")
    ui.info(f"Datasets: {', '.join(dataset)}")
    print()

    if len(dataset) == 1:
        result = benchmark.evaluate(engine, dataset[0])
        ui.success(f"nDCG@10: {result.ndcg_at_10:.4f}")
        ui.info(f"Recall@100: {result.recall_at_100:.4f}")
        ui.info(f"Queries: {result.num_queries}")

        if output:
            benchmark.save_results(result, output)
            ui.info(f"Results saved to {output}")
    else:
        result = benchmark.evaluate_suite(engine, dataset)
        ui.success(f"Average nDCG@10: {result.average_ndcg_at_10:.4f}")
        ui.info(f"Average Recall@100: {result.average_recall_at_100:.4f}")

        for r in result.results:
            print(f"  {r.dataset}: nDCG@10={r.ndcg_at_10:.4f}")

        if output:
            benchmark.save_results(result, output)
            ui.info(f"Results saved to {output}")


@app.command("rgb")
def rgb_benchmark(
    collection: Annotated[
        Optional[str],
        typer.Option(
            "--collection",
            "-c",
            help="Collection to use for evaluation.",
        ),
    ] = None,
    test_set: Annotated[
        Optional[Path],
        typer.Option(
            "--test-set",
            "-t",
            help="Path to custom test cases (JSON).",
        ),
    ] = None,
    test_type: Annotated[
        Optional[str],
        typer.Option(
            "--type",
            help="Specific test type to run (noise_robustness, negative_rejection, etc.)",
        ),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output file path for results (JSON).",
        ),
    ] = None,
) -> None:
    """
    Run RGB robustness benchmark.

    Tests RAG robustness across four dimensions:
    - noise_robustness: Handling irrelevant context
    - negative_rejection: Abstaining when appropriate (maps to ABSTAIN mode)
    - information_integration: Synthesizing multiple sources
    - counterfactual_robustness: Detecting conflicts (maps to DISPUTED mode)

    Examples:

        # Run all RGB tests
        fitz eval rgb

        # Run specific test type
        fitz eval rgb --type negative_rejection

        # Use custom test set
        fitz eval rgb -t my_rgb_tests.json
    """
    from fitz_ai.evaluation.benchmarks.rgb import RGBEvaluator, RGBTestType

    collection = _get_collection(collection)
    engine = _get_engine(collection)

    evaluator = RGBEvaluator()

    ui.header("RGB Robustness Benchmark")
    print()

    # Parse test type if specified
    rgb_type = None
    if test_type:
        try:
            rgb_type = RGBTestType(test_type)
        except ValueError:
            ui.error(f"Invalid test type: {test_type}")
            ui.info(f"Valid types: {', '.join(t.value for t in RGBTestType)}")
            raise typer.Exit(1)

    if test_set:
        result = evaluator.evaluate_from_file(engine, test_set, rgb_type)
    else:
        # Use built-in test cases
        ui.warning("No test set provided. Using empty test set.")
        ui.info("Create test cases with rgb.create_negative_rejection_case() etc.")
        result = evaluator.evaluate(engine, [], rgb_type)

    ui.success(f"Overall Score: {result.overall_score:.2%}")

    if result.noise_robustness:
        ui.info(f"Noise Robustness: {result.noise_robustness.score:.2%}")
    if result.negative_rejection:
        ui.info(f"Negative Rejection: {result.negative_rejection.score:.2%}")
    if result.information_integration:
        ui.info(f"Information Integration: {result.information_integration.score:.2%}")
    if result.counterfactual_robustness:
        ui.info(f"Counterfactual Robustness: {result.counterfactual_robustness.score:.2%}")

    if output:
        evaluator.save_results(result, output)
        ui.info(f"Results saved to {output}")


@app.command("fitz-gov")
def fitz_gov_benchmark(
    collection: Annotated[
        Optional[str],
        typer.Option(
            "--collection",
            "-c",
            help="Collection to use for evaluation.",
        ),
    ] = None,
    category: Annotated[
        Optional[list[str]],
        typer.Option(
            "--category",
            help="Categories to test: abstention, dispute, trustworthy_hedged, trustworthy_direct, grounding, relevance.",
        ),
    ] = None,
    data_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--data-dir",
            help="Directory containing test case JSON files.",
        ),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output file path for results (JSON).",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output as JSON to stdout.",
        ),
    ] = False,
    full: Annotated[
        bool,
        typer.Option(
            "--full",
            help="Run full LLM generation for answer quality tests (slower).",
        ),
    ] = False,
    enrich: Annotated[
        bool,
        typer.Option(
            "--enrich/--no-enrich",
            help="Enrich chunks with metadata (summary, keywords, entities) before constraints. Default: enabled for realistic production simulation.",
        ),
    ] = True,
    deterministic: Annotated[
        bool,
        typer.Option(
            "--deterministic",
            help="Use deterministic constraints (embeddings + regex). No LLM variance.",
        ),
    ] = False,
    fusion: Annotated[
        bool,
        typer.Option(
            "--fusion",
            help="Use 3-prompt fusion for contradiction detection. Reduces variance via majority voting.",
        ),
    ] = False,
    adaptive: Annotated[
        bool,
        typer.Option(
            "--adaptive/--no-adaptive",
            help="Auto-select detection method: fusion for uncertainty queries, pairwise for factual.",
        ),
    ] = True,
    model: Annotated[
        Optional[str],
        typer.Option(
            "--model",
            "-m",
            help="Override chat model for constraints. Format: provider or provider/model (e.g., ollama, ollama/qwen2.5:3b, cohere).",
        ),
    ] = None,
) -> None:
    """
    Run fitz-gov governance calibration benchmark.

    Fitz's core differentiator - tests governance mode classification:
    - abstention: Should refuse when evidence insufficient
    - dispute: Should flag conflicting information
    - trustworthy_hedged: Should hedge uncertain claims
    - trustworthy_direct: Should answer directly when evidence is clear

    Outputs accuracy by category and a confusion matrix.

    Examples:

        # Run full benchmark
        fitz eval fitz-gov

        # Specify model for reproducible results
        fitz eval fitz-gov --model ollama/qwen2.5:3b

        # Run specific categories
        fitz eval fitz-gov --category abstention --category dispute

        # JSON output
        fitz eval fitz-gov --json

        # Save results
        fitz eval fitz-gov -o results/fitz_gov.json
    """
    from fitz_ai.evaluation.benchmarks.fitz_gov import FitzGovBenchmark, FitzGovCategory

    collection = _get_collection(collection)
    engine = _get_engine(collection)

    benchmark = FitzGovBenchmark(
        data_dir=data_dir,
        full_mode=full,
        enrich_chunks=enrich,
        use_fusion=fusion if not deterministic else False,
        adaptive=adaptive if not deterministic else False,
        model_override=model,
    )

    # Parse categories
    categories = None
    if category:
        try:
            categories = [FitzGovCategory(c) for c in category]
        except ValueError as e:
            ui.error(f"Invalid category: {e}")
            ui.info(f"Valid categories: {', '.join(FitzGovBenchmark.get_available_categories())}")
            raise typer.Exit(1)

    if not json_output:
        ui.header("fitz-gov Governance Benchmark")
        if model:
            ui.info(f"Model: {model}")
        if categories:
            ui.info(f"Categories: {', '.join(c.value for c in categories)}")
        print()

    result = benchmark.evaluate(engine, categories=categories)

    if json_output:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        ui.success(f"Overall Accuracy: {result.overall_accuracy:.2%}")
        print()

        if RICH:
            _display_fitz_gov_rich(result)
        else:
            _display_fitz_gov_plain(result)

    if output:
        benchmark.save_results(result, output)
        if not json_output:
            ui.info(f"Results saved to {output}")


def _display_fitz_gov_rich(result) -> None:
    """Display fitz-gov results with Rich."""
    # Governance Mode Categories
    table = Table(title="Governance Mode Accuracy", show_header=True, header_style="bold")
    table.add_column("Category", style="cyan")
    table.add_column("Accuracy", justify="right")
    table.add_column("Correct", justify="right")
    table.add_column("Total", justify="right")

    # Import FitzGovCategory for dict access
    from fitz_ai.evaluation.benchmarks.fitz_gov import FitzGovCategory

    gov_cats = [
        FitzGovCategory.ABSTENTION,
        FitzGovCategory.DISPUTE,
        FitzGovCategory.TRUSTWORTHY_HEDGED,
        FitzGovCategory.TRUSTWORTHY_DIRECT,
    ]
    for cat in gov_cats:
        cat_result = result.category_results.get(cat)
        if cat_result:
            style = (
                "green"
                if cat_result.accuracy >= 0.8
                else "yellow" if cat_result.accuracy >= 0.5 else "red"
            )
            table.add_row(
                cat_result.category.value.title(),
                f"[{style}]{cat_result.accuracy:.2%}[/{style}]",
                str(cat_result.num_correct),
                str(cat_result.num_total),
            )

    console.print(table)
    print()

    # Answer Quality Categories (grounding + relevance)
    quality_cats = [FitzGovCategory.GROUNDING, FitzGovCategory.RELEVANCE]
    has_quality = any(cat in result.category_results for cat in quality_cats)
    if has_quality:
        quality_table = Table(title="Answer Quality", show_header=True, header_style="bold")
        quality_table.add_column("Category", style="cyan")
        quality_table.add_column("Accuracy", justify="right")
        quality_table.add_column("Correct", justify="right")
        quality_table.add_column("Total", justify="right")

        for cat in quality_cats:
            cat_result = result.category_results.get(cat)
            if cat_result:
                style = (
                    "green"
                    if cat_result.accuracy >= 0.8
                    else "yellow" if cat_result.accuracy >= 0.5 else "red"
                )
                quality_table.add_row(
                    cat_result.category.value.title(),
                    f"[{style}]{cat_result.accuracy:.2%}[/{style}]",
                    str(cat_result.num_correct),
                    str(cat_result.num_total),
                )

        console.print(quality_table)
        print()

    # Confusion matrix
    console.print(str(result.confusion_matrix))


def _display_fitz_gov_plain(result) -> None:
    """Display fitz-gov results in plain text."""
    from fitz_ai.evaluation.benchmarks.fitz_gov import FitzGovCategory

    print("Governance Mode Categories:")
    print("-" * 40)
    gov_cats = [
        FitzGovCategory.ABSTENTION,
        FitzGovCategory.DISPUTE,
        FitzGovCategory.TRUSTWORTHY_HEDGED,
        FitzGovCategory.TRUSTWORTHY_DIRECT,
    ]
    for cat in gov_cats:
        cat_result = result.category_results.get(cat)
        if cat_result:
            print(
                f"  {cat_result.category.value.title()}: "
                f"{cat_result.accuracy:.2%} ({cat_result.num_correct}/{cat_result.num_total})"
            )

    quality_cats = [FitzGovCategory.GROUNDING, FitzGovCategory.RELEVANCE]
    has_quality = any(cat in result.category_results for cat in quality_cats)
    if has_quality:
        print()
        print("Answer Quality Categories:")
        print("-" * 40)
        for cat in quality_cats:
            cat_result = result.category_results.get(cat)
            if cat_result:
                print(
                    f"  {cat_result.category.value.title()}: "
                    f"{cat_result.accuracy:.2%} ({cat_result.num_correct}/{cat_result.num_total})"
                )

    print()
    print(str(result.confusion_matrix))


@app.command("dashboard")
def benchmark_dashboard(
    results_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory containing benchmark result JSON files.",
        ),
    ],
) -> None:
    """
    Display unified benchmark results dashboard.

    Loads results from all benchmarks (BEIR, RAGAS, RGB, fitz-gov)
    and displays a consolidated view.

    Expected directory structure:
        results_dir/
            beir/*.json
            ragas/*.json
            rgb/*.json
            fitz_gov/*.json

    Examples:

        # View dashboard
        fitz eval dashboard ./benchmark_results
    """
    from fitz_ai.evaluation.dashboard import BenchmarkDashboard

    if not results_dir.exists():
        ui.error(f"Results directory not found: {results_dir}")
        raise typer.Exit(1)

    dashboard = BenchmarkDashboard.load_from_directory(results_dir)
    dashboard.print_summary()


@app.command("all")
def run_all_benchmarks(
    collection: Annotated[
        Optional[str],
        typer.Option(
            "--collection",
            "-c",
            help="Collection to use for evaluation.",
        ),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory to save all results.",
        ),
    ] = Path("./benchmark_results"),
    beir_datasets: Annotated[
        list[str],
        typer.Option(
            "--beir-dataset",
            help="BEIR datasets to run (can specify multiple).",
        ),
    ] = ["scifact"],
    skip_beir: Annotated[
        bool,
        typer.Option(
            "--skip-beir",
            help="Skip BEIR benchmark (requires external package).",
        ),
    ] = False,
) -> None:
    """
    Run all benchmarks and save results.

    Runs BEIR (cross-RAG comparison) and fitz-gov (Fitz's governance benchmark).
    Results are saved to subdirectories for dashboard viewing.

    Examples:

        # Run all benchmarks
        fitz eval all -c my_collection -o ./results

        # Skip BEIR (no external deps needed)
        fitz eval all --skip-beir
    """
    collection = _get_collection(collection)

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "beir").mkdir(exist_ok=True)
    (output_dir / "rgb").mkdir(exist_ok=True)
    (output_dir / "fitz_gov").mkdir(exist_ok=True)

    ui.header("Running All Benchmarks")
    print()

    # BEIR
    if not skip_beir:
        try:
            from fitz_ai.evaluation.benchmarks.beir import BEIRBenchmark

            ui.info("Running BEIR benchmark...")
            engine = _get_engine(collection)
            benchmark = BEIRBenchmark()
            result = benchmark.evaluate_suite(engine, beir_datasets)
            benchmark.save_results(result, output_dir / "beir" / "results.json")
            ui.success(f"BEIR: nDCG@10 = {result.average_ndcg_at_10:.4f}")
        except ImportError:
            ui.warning("Skipping BEIR (requires: pip install fitz-ai[benchmarks])")
    else:
        ui.info("Skipping BEIR benchmark")

    # fitz-gov (no external deps)
    ui.info("Running fitz-gov benchmark...")
    from fitz_ai.evaluation.benchmarks.fitz_gov import FitzGovBenchmark

    engine = _get_engine(collection)
    benchmark = FitzGovBenchmark()
    result = benchmark.evaluate(engine)
    benchmark.save_results(result, output_dir / "fitz_gov" / "results.json")
    ui.success(f"fitz-gov: Accuracy = {result.overall_accuracy:.2%}")

    print()
    ui.success(f"All results saved to {output_dir}")
    ui.info(f"View dashboard with: fitz eval dashboard {output_dir}")


# Export for main CLI
def command() -> None:
    """Run the eval command group."""
    app()


__all__ = ["app", "command"]
