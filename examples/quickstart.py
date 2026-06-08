"""Quickstart: evaluate a sentence-transformers model on a single MTEB-PT task.

Usage::

    pip install mteb-pt sentence-transformers
    python examples/quickstart.py

Default model and task can be changed via CLI flags. This script runs in under
5 minutes on a CPU for the default ``HateBR`` task with ``mE5-base``.

Output: a JSON file in ``./results/HateBR.json`` with the headline metric and
the full per-experiment score distribution.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mteb_pt.register  # noqa: F401  -- side-effect import: registers tasks
import mteb
from sentence_transformers import SentenceTransformer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a single MTEB-PT task on a sentence-transformers model.",
    )
    parser.add_argument(
        "--model",
        default="intfloat/multilingual-e5-base",
        help="HuggingFace model identifier (default: intfloat/multilingual-e5-base)",
    )
    parser.add_argument(
        "--task",
        default="HateBR",
        help="MTEB-PT task name (default: HateBR)",
    )
    parser.add_argument(
        "--output",
        default="./results",
        help="Output directory for the result JSON (default: ./results)",
    )
    args = parser.parse_args()

    print(f"Model:  {args.model}")
    print(f"Task:   {args.task}")
    print(f"Output: {args.output}")
    print()

    model = SentenceTransformer(args.model)
    task = mteb.get_task(args.task)
    evaluator = mteb.MTEB(tasks=[task])
    evaluator.run(model, output_folder=args.output, verbosity=1)

    # Locate the result JSON
    result_file = next(Path(args.output).rglob(f"{args.task}.json"), None)
    if result_file:
        print(f"\nResult written to: {result_file}")
    else:
        print("\nNo result JSON found — check the output directory.")


if __name__ == "__main__":
    main()
