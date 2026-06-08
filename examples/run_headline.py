"""Run the full MTEB-PT headline evaluation (16 tasks) on a single model.

Usage::

    pip install mteb-pt sentence-transformers
    python examples/run_headline.py --model intfloat/multilingual-e5-large-instruct

Output: 16 per-task JSON files under ``./results/{model_slug}/`` plus a
``mean_16.json`` aggregate.

This is the script used to reproduce a row of the public leaderboard at
https://huggingface.co/spaces/mteb-pt/leaderboard. Total runtime on a single
A10G GPU is approximately 30-60 minutes depending on model size.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mteb_pt
import mteb_pt.register  # noqa: F401
import mteb
from sentence_transformers import SentenceTransformer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full 16-task MTEB-PT headline evaluation on a model.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model identifier (e.g. intfloat/multilingual-e5-large-instruct)",
    )
    parser.add_argument(
        "--output",
        default="./results",
        help="Output directory for result JSONs (default: ./results)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Encoding batch size (default: 32)",
    )
    args = parser.parse_args()

    slug = args.model.replace("/", "__")
    out_dir = Path(args.output) / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model:      {args.model}")
    print(f"Output dir: {out_dir}")
    print(f"Tasks:      {len(mteb_pt.HEADLINE_TASKS)} headline (mean_16)")
    print()

    model = SentenceTransformer(args.model)

    summary: dict[str, dict] = {}
    for task_name in mteb_pt.HEADLINE_TASKS:
        print(f"\n=== {task_name} ===")
        try:
            task = mteb.get_task(task_name)
            evaluator = mteb.MTEB(tasks=[task])
            evaluator.run(
                model,
                output_folder=str(out_dir),
                verbosity=1,
                encode_kwargs={"batch_size": args.batch_size},
                overwrite_results=True,
            )
            result_file = next(out_dir.rglob(f"{task_name}.json"), None)
            if result_file:
                with open(result_file) as f:
                    data = json.load(f)
                main_score = _extract_main_score(data)
                summary[task_name] = {"main_score": main_score, "status": "ok"}
                print(f"  main_score = {main_score:.4f}")
            else:
                summary[task_name] = {"status": "no_output_json"}
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {str(e)[:200]}")
            summary[task_name] = {"status": "failed", "error": str(e)[:200]}

    # Compute mean_16
    ok_scores = [
        v["main_score"] for v in summary.values()
        if v.get("status") == "ok" and v.get("main_score") is not None
    ]
    mean_16 = sum(ok_scores) / len(ok_scores) if ok_scores else None

    summary_path = out_dir / "mean_16.json"
    with open(summary_path, "w") as f:
        json.dump({
            "model_id": args.model,
            "mean_16": mean_16,
            "n_tasks_evaluated": len(ok_scores),
            "n_tasks_total": len(mteb_pt.HEADLINE_TASKS),
            "per_task": summary,
        }, f, indent=2)

    print(f"\n========== SUMMARY ==========")
    print(f"mean_16 = {mean_16:.4f}" if mean_16 is not None else "mean_16 = N/A")
    print(f"Tasks ok = {len(ok_scores)}/{len(mteb_pt.HEADLINE_TASKS)}")
    print(f"Written to: {summary_path}")


def _extract_main_score(result_json: dict) -> float | None:
    """Pull the headline score from the standard mteb result JSON shape."""
    if "scores" in result_json:
        for split_data in result_json["scores"].values():
            for entry in split_data:
                if "main_score" in entry:
                    return float(entry["main_score"])
    return None


if __name__ == "__main__":
    main()
