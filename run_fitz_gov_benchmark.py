#!/usr/bin/env python
"""Run fitz-gov 2.0 benchmark with progress tracking."""

import sys
import time
from datetime import datetime

import fitz_gov
from fitz_ai.config.loader import load_engine_config
from fitz_ai.engines.fitz_rag import FitzRagEngine
from fitz_ai.evaluation.benchmarks import FitzGovBenchmark


def main():
    print("=" * 70)
    print("fitz-gov 2.0 Benchmark".center(70))
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: ollama/qwen2.5:3b")
    print(f"Configuration: Adaptive mode with 3-prompt fusion")
    print("-" * 70)

    # Load all test cases
    all_cases = fitz_gov.load_cases()
    print(f"\nLoaded {len(all_cases)} test cases:")

    # Show breakdown by category
    from collections import Counter
    categories = Counter(c.category.value for c in all_cases)
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

    print("\n" + "-" * 70)
    print("Initializing engine...")

    # Initialize engine
    config = load_engine_config('fitz_rag')
    config.chat = 'ollama/qwen2.5:3b'
    engine = FitzRagEngine(config)

    # Initialize benchmark
    benchmark = FitzGovBenchmark(
        model_override='ollama/qwen2.5:3b',
        adaptive=True,
        use_fusion=True,
        llm_validation=False  # Disable LLM validation for speed
    )

    print("Engine initialized.")
    print("\n" + "-" * 70)
    print("Running benchmark...")
    print("Processing categories:")

    # Process by category for progress tracking
    all_results = []
    category_results = {}

    for category in fitz_gov.FitzGovCategory:
        cat_cases = [c for c in all_cases if c.category == category]
        if not cat_cases:
            continue

        print(f"\n  {category.value}: Processing {len(cat_cases)} cases...", end="", flush=True)
        start_time = time.time()

        # Run evaluation for this category
        result = benchmark.evaluate(engine, test_cases=cat_cases)

        elapsed = time.time() - start_time
        print(f" Done in {elapsed:.1f}s")

        # Parse result to get accuracy
        result_str = str(result)
        if "Overall Accuracy:" in result_str:
            accuracy_line = [l for l in result_str.split('\n') if "Overall Accuracy:" in l][0]
            accuracy = accuracy_line.split(":")[1].strip()
            category_results[category.value] = accuracy

    print("\n" + "=" * 70)
    print("FINAL RESULTS - fitz-gov 2.0")
    print("=" * 70)

    # Run full evaluation to get complete results
    print("\nGenerating complete evaluation report...")
    final_results = benchmark.evaluate(engine, test_cases=all_cases)
    print(final_results)

    print("\n" + "-" * 70)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Save results to file
    with open("fitz-gov-2.0-results.txt", "w") as f:
        f.write(f"fitz-gov 2.0 Benchmark Results\n")
        f.write(f"{'=' * 70}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: ollama/qwen2.5:3b\n")
        f.write(f"Test cases: {len(all_cases)}\n")
        f.write(f"\n{final_results}\n")

    print("\nResults saved to: fitz-gov-2.0-results.txt")


if __name__ == "__main__":
    main()