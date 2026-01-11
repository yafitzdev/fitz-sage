# tests/e2e/reporter.py
"""
Report generation for E2E tests.

Generates summary reports in various formats (console, markdown, dict).
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .runner import E2ERunResult, ScenarioResult
    from .scenarios import Feature


class E2EReporter:
    """Generates E2E test reports."""

    def __init__(self, result: "E2ERunResult"):
        """
        Initialize the reporter.

        Args:
            result: E2ERunResult from a test run
        """
        self.result = result

    def summary_dict(self) -> dict:
        """
        Generate summary as a dictionary.

        Returns:
            Dictionary with test statistics
        """
        by_feature = self._group_by_feature()

        return {
            "timestamp": datetime.now().isoformat(),
            "collection": self.result.collection,
            "total": self.result.total,
            "passed": self.result.passed,
            "failed": self.result.failed,
            "pass_rate": f"{self.result.pass_rate:.1f}%",
            "ingestion_duration_s": round(self.result.ingestion_duration_s, 1),
            "total_duration_s": round(self.result.total_duration_s, 1),
            "by_feature": {
                feature: {
                    "passed": stats["passed"],
                    "failed": stats["failed"],
                    "total": stats["total"],
                }
                for feature, stats in by_feature.items()
            },
        }

    def _group_by_feature(self) -> dict[str, dict]:
        """Group results by feature."""
        by_feature: dict[str, dict] = defaultdict(
            lambda: {"passed": 0, "failed": 0, "total": 0}
        )

        for r in self.result.scenario_results:
            feature_name = r.scenario.feature.value
            by_feature[feature_name]["total"] += 1
            if r.validation.passed:
                by_feature[feature_name]["passed"] += 1
            else:
                by_feature[feature_name]["failed"] += 1

        return dict(by_feature)

    def markdown_report(self) -> str:
        """
        Generate a markdown report.

        Returns:
            Markdown formatted report string
        """
        lines = [
            "# E2E Retrieval Intelligence Test Report",
            "",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Collection:** `{self.result.collection}`",
            "",
            "## Summary",
            "",
            f"- **Total Scenarios:** {self.result.total}",
            f"- **Passed:** {self.result.passed}",
            f"- **Failed:** {self.result.failed}",
            f"- **Pass Rate:** {self.result.pass_rate:.1f}%",
            f"- **Ingestion Time:** {self.result.ingestion_duration_s:.1f}s",
            f"- **Total Time:** {self.result.total_duration_s:.1f}s",
            "",
            "## Results by Feature",
            "",
            "| Feature | Passed | Failed | Total | Rate |",
            "|---------|--------|--------|-------|------|",
        ]

        by_feature = self._group_by_feature()
        for feature, stats in sorted(by_feature.items()):
            rate = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
            lines.append(
                f"| {feature} | {stats['passed']} | {stats['failed']} | "
                f"{stats['total']} | {rate:.0f}% |"
            )

        lines.extend(
            [
                "",
                "## Scenario Details",
                "",
                "| ID | Name | Status | Duration | Reason |",
                "|----|------|--------|----------|--------|",
            ]
        )

        for r in self.result.scenario_results:
            status = "PASS" if r.validation.passed else "FAIL"
            reason = r.validation.reason[:50] if not r.validation.passed else "-"
            lines.append(
                f"| {r.scenario.id} | {r.scenario.name} | {status} | "
                f"{r.duration_ms:.0f}ms | {reason} |"
            )

        # Failed scenarios detail
        failed = [r for r in self.result.scenario_results if not r.validation.passed]
        if failed:
            lines.extend(
                [
                    "",
                    "## Failed Scenarios",
                    "",
                ]
            )

            for r in failed:
                lines.extend(
                    [
                        f"### {r.scenario.id}: {r.scenario.name}",
                        "",
                        f"**Query:** {r.scenario.query}",
                        "",
                        f"**Reason:** {r.validation.reason}",
                        "",
                        f"**Answer Preview:**",
                        "```",
                        r.answer_text[:300] if r.answer_text else "(no answer)",
                        "```",
                        "",
                    ]
                )

        return "\n".join(lines)

    def console_report(self) -> None:
        """Print a colored console report."""
        print()
        print("=" * 70)
        print("E2E RETRIEVAL INTELLIGENCE TEST REPORT")
        print("=" * 70)
        print()
        print(f"Collection: {self.result.collection}")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print("-" * 70)
        print("SUMMARY")
        print("-" * 70)
        print(f"  Total:      {self.result.total}")
        print(f"  Passed:     {self.result.passed}")
        print(f"  Failed:     {self.result.failed}")
        print(f"  Pass Rate:  {self.result.pass_rate:.1f}%")
        print(f"  Ingestion:  {self.result.ingestion_duration_s:.1f}s")
        print(f"  Total Time: {self.result.total_duration_s:.1f}s")
        print()

        # By feature
        print("-" * 70)
        print("BY FEATURE")
        print("-" * 70)
        by_feature = self._group_by_feature()
        for feature, stats in sorted(by_feature.items()):
            rate = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
            status = "OK" if stats["failed"] == 0 else "!!"
            print(
                f"  [{status}] {feature:<25} "
                f"{stats['passed']}/{stats['total']} ({rate:.0f}%)"
            )
        print()

        # Scenario details
        print("-" * 70)
        print("SCENARIO RESULTS")
        print("-" * 70)
        for r in self.result.scenario_results:
            status = "PASS" if r.validation.passed else "FAIL"
            print(f"  [{status}] {r.scenario.id}: {r.scenario.name} ({r.duration_ms:.0f}ms)")
            if not r.validation.passed:
                print(f"         -> {r.validation.reason}")
        print()

        # Failed details
        failed = [r for r in self.result.scenario_results if not r.validation.passed]
        if failed:
            print("-" * 70)
            print("FAILED SCENARIO DETAILS")
            print("-" * 70)
            for r in failed:
                print(f"\n  {r.scenario.id}: {r.scenario.name}")
                print(f"  Query: {r.scenario.query}")
                print(f"  Reason: {r.validation.reason}")
                print(f"  Answer: {r.answer_text[:200]}..." if r.answer_text else "  (no answer)")
            print()

        print("=" * 70)
