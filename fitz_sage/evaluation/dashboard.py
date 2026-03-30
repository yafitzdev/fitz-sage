# fitz_sage/evaluation/dashboard.py
"""
Unified benchmark dashboard for displaying evaluation results.

Aggregates results from BEIR, RGB, and fitz-gov benchmarks
into a single, formatted display.

Usage:
    from fitz_sage.evaluation.dashboard import BenchmarkDashboard

    dashboard = BenchmarkDashboard.load_from_directory("./results")
    dashboard.print_summary()
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fitz_sage.cli.ui import RICH, Panel, Table, console, ui
from fitz_sage.logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkDashboard:
    """Aggregated view of all benchmark results."""

    beir_results: list[Any] = field(default_factory=list)
    """BEIR retrieval benchmark results (cross-RAG comparison)."""

    rgb_results: list[Any] = field(default_factory=list)
    """RGB robustness benchmark results."""

    fitz_gov_results: list[Any] = field(default_factory=list)
    """fitz-gov governance benchmark results (Fitz's moat)."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """When dashboard was created."""

    @classmethod
    def load_from_directory(cls, results_dir: Path | str) -> BenchmarkDashboard:
        """
        Load all benchmark results from a directory.

        Expected structure:
            results_dir/
                beir/
                    *.json
                rgb/
                    *.json
                fitz_gov/
                    *.json

        Args:
            results_dir: Path to directory containing benchmark results

        Returns:
            BenchmarkDashboard with loaded results
        """
        results_dir = Path(results_dir)

        beir_results = cls._load_results(results_dir / "beir", "beir")
        rgb_results = cls._load_results(results_dir / "rgb", "rgb")
        fitz_gov_results = cls._load_results(results_dir / "fitz_gov", "fitz_gov")

        return cls(
            beir_results=beir_results,
            rgb_results=rgb_results,
            fitz_gov_results=fitz_gov_results,
        )

    @staticmethod
    def _load_results(subdir: Path, benchmark_type: str) -> list[Any]:
        """Load results from a benchmark subdirectory."""
        results = []

        if not subdir.exists():
            return results

        for json_file in subdir.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    data["_source_file"] = str(json_file)
                    results.append(data)
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")

        return results

    def print_summary(self) -> None:
        """Print formatted summary of all benchmark results."""
        if RICH:
            self._print_rich_summary()
        else:
            self._print_plain_summary()

    def _print_rich_summary(self) -> None:
        """Print Rich-formatted summary."""
        ui.header("Fitz Benchmark Dashboard")
        print()

        # Overview panel
        total_benchmarks = sum(
            [
                len(self.beir_results),
                len(self.rgb_results),
                len(self.fitz_gov_results),
            ]
        )
        console.print(
            Panel(
                f"Total benchmark runs: {total_benchmarks}\n"
                f"BEIR: {len(self.beir_results)} | "
                f"RGB: {len(self.rgb_results)} | "
                f"fitz-gov: {len(self.fitz_gov_results)}",
                title="Overview",
            )
        )
        print()

        # BEIR results
        if self.beir_results:
            self._print_beir_rich()

        # RGB results
        if self.rgb_results:
            self._print_rgb_rich()

        # fitz-gov results
        if self.fitz_gov_results:
            self._print_fitz_gov_rich()

    def _print_beir_rich(self) -> None:
        """Print BEIR results with Rich."""
        table = Table(title="BEIR Retrieval Quality", show_header=True, header_style="bold cyan")
        table.add_column("Dataset", style="dim")
        table.add_column("nDCG@10", justify="right")
        table.add_column("Recall@100", justify="right")
        table.add_column("MRR@10", justify="right")
        table.add_column("Queries", justify="right")

        for result in self.beir_results:
            # Handle both single result and suite result formats
            if "results" in result:
                # Suite result
                for r in result["results"]:
                    table.add_row(
                        r["dataset"],
                        f"{r['ndcg_at_10']:.4f}",
                        f"{r['recall_at_100']:.4f}",
                        f"{r.get('mrr_at_10', 0):.4f}",
                        str(r["num_queries"]),
                    )
            else:
                # Single result
                table.add_row(
                    result["dataset"],
                    f"{result['ndcg_at_10']:.4f}",
                    f"{result['recall_at_100']:.4f}",
                    f"{result.get('mrr_at_10', 0):.4f}",
                    str(result["num_queries"]),
                )

        console.print(table)
        print()

    def _print_rgb_rich(self) -> None:
        """Print RGB results with Rich."""
        table = Table(title="RGB Robustness Tests", show_header=True, header_style="bold cyan")
        table.add_column("Test Type", style="dim")
        table.add_column("Score", justify="right")
        table.add_column("Passed", justify="right")
        table.add_column("Total", justify="right")

        for result in self.rgb_results:
            # Add each test type
            for test_type in [
                "noise_robustness",
                "negative_rejection",
                "information_integration",
                "counterfactual_robustness",
            ]:
                type_result = result.get(test_type)
                if type_result:
                    table.add_row(
                        test_type.replace("_", " ").title(),
                        f"{type_result['score']:.2%}",
                        str(type_result["num_passed"]),
                        str(type_result["num_total"]),
                    )

        console.print(table)
        print()

    def _print_fitz_gov_rich(self) -> None:
        """Print fitz-gov results with Rich."""
        for result in self.fitz_gov_results:
            # Summary table
            table = Table(
                title="fitz-gov Governance Calibration", show_header=True, header_style="bold cyan"
            )
            table.add_column("Category", style="dim")
            table.add_column("Accuracy", justify="right")
            table.add_column("Correct", justify="right")
            table.add_column("Total", justify="right")

            overall_acc = result.get("overall_accuracy", 0)
            table.add_row(
                "OVERALL",
                f"{overall_acc:.2%}",
                "-",
                str(result.get("num_cases", 0)),
                style="bold",
            )

            # Governance mode categories
            for cat in ["abstention", "dispute", "trustworthy_hedged", "trustworthy_direct"]:
                cat_result = result.get(cat)
                if cat_result:
                    table.add_row(
                        cat.title(),
                        f"{cat_result['accuracy']:.2%}",
                        str(cat_result["num_correct"]),
                        str(cat_result["num_total"]),
                    )

            # Answer quality categories
            for cat in ["grounding", "relevance"]:
                cat_result = result.get(cat)
                if cat_result:
                    table.add_row(
                        cat.title(),
                        f"{cat_result['accuracy']:.2%}",
                        str(cat_result["num_correct"]),
                        str(cat_result["num_total"]),
                    )

            console.print(table)
            print()

            # Confusion matrix
            if "confusion_matrix" in result:
                self._print_confusion_matrix_rich(result["confusion_matrix"])

    def _print_confusion_matrix_rich(self, matrix: dict[str, dict[str, int]]) -> None:
        """Print confusion matrix with Rich."""
        modes = ["trustworthy", "disputed", "abstain"]

        table = Table(title="Mode Confusion Matrix", show_header=True, header_style="bold")
        table.add_column("Expected \\ Actual", style="dim")
        for mode in modes:
            table.add_column(mode[:8].title(), justify="right")

        for exp in modes:
            row = [exp.title()]
            for act in modes:
                count = matrix.get(exp, {}).get(act, 0)
                if exp == act and count > 0:
                    row.append(f"[green]{count}[/green]")
                elif count > 0:
                    row.append(f"[red]{count}[/red]")
                else:
                    row.append("0")
            table.add_row(*row)

        console.print(table)
        print()

    def _print_plain_summary(self) -> None:
        """Print plain text summary."""
        print("=" * 60)
        print("Fitz Benchmark Dashboard")
        print("=" * 60)
        print()

        total = sum(
            [
                len(self.beir_results),
                len(self.rgb_results),
                len(self.fitz_gov_results),
            ]
        )
        print(f"Total benchmark runs: {total}")
        print(
            f"BEIR: {len(self.beir_results)} | "
            f"RGB: {len(self.rgb_results)} | "
            f"fitz-gov: {len(self.fitz_gov_results)}"
        )
        print()

        if self.beir_results:
            print("BEIR Results:")
            print("-" * 40)
            for result in self.beir_results:
                if "results" in result:
                    for r in result["results"]:
                        print(
                            f"  {r['dataset']}: nDCG@10={r['ndcg_at_10']:.4f}, "
                            f"Recall@100={r['recall_at_100']:.4f}"
                        )
                else:
                    print(
                        f"  {result['dataset']}: nDCG@10={result['ndcg_at_10']:.4f}, "
                        f"Recall@100={result['recall_at_100']:.4f}"
                    )
            print()

        if self.rgb_results:
            print("RGB Results:")
            print("-" * 40)
            for result in self.rgb_results:
                print(f"  Overall Score: {result.get('overall_score', 0):.2%}")
            print()

        if self.fitz_gov_results:
            print("fitz-gov Results:")
            print("-" * 40)
            for result in self.fitz_gov_results:
                print(f"  Overall Accuracy: {result.get('overall_accuracy', 0):.2%}")
            print()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "beir_results": self.beir_results,
            "rgb_results": self.rgb_results,
            "fitz_gov_results": self.fitz_gov_results,
            "timestamp": self.timestamp.isoformat(),
        }

    def save(self, path: Path | str) -> None:
        """Save dashboard state to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> BenchmarkDashboard:
        """Load dashboard state from JSON."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        return cls(
            beir_results=data.get("beir_results", []),
            rgb_results=data.get("rgb_results", []),
            fitz_gov_results=data.get("fitz_gov_results", []),
            timestamp=(
                datetime.fromisoformat(data["timestamp"])
                if "timestamp" in data
                else datetime.now(timezone.utc)
            ),
        )


def print_benchmark_summary(results_dir: Path | str) -> None:
    """Convenience function to print benchmark summary."""
    dashboard = BenchmarkDashboard.load_from_directory(results_dir)
    dashboard.print_summary()
