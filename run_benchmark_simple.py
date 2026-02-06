#!/usr/bin/env python
"""Run fitz-gov 2.0 benchmark - simple version."""

import sys
from datetime import datetime

import fitz_gov
from fitz_ai.config.loader import load_engine_config
from fitz_ai.engines.fitz_rag import FitzRagEngine
from fitz_ai.evaluation.benchmarks import FitzGovBenchmark


def main():
    print("=" * 70)
    print("fitz-gov 2.0 Benchmark")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: ollama/qwen2.5:3b")
    print("-" * 70)

    # Initialize
    config = load_engine_config('fitz_rag')
    config.chat = 'ollama/qwen2.5:3b'
    engine = FitzRagEngine(config)

    benchmark = FitzGovBenchmark(
        model_override='ollama/qwen2.5:3b',
        adaptive=True,
        use_fusion=True,
        llm_validation=False
    )

    print("Running complete benchmark on 331 test cases...")
    print("This will take approximately 20-30 minutes...")
    sys.stdout.flush()

    # Run full evaluation
    results = benchmark.evaluate(engine)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(results)

    # Save to file
    with open("fitz-gov-2.0-results.txt", "w") as f:
        f.write(str(results))

    print("\nResults saved to: fitz-gov-2.0-results.txt")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()