"""Compute paired-bootstrap p-values between two model rankings on MTEB-PT.

This reproduces Finding F1 from the paper: a paired bootstrap over the 16
headline tasks resamples task scores with replacement and computes the
fraction of resamples in which the sign of the score difference flips.

Usage::

    python examples/compute_bootstrap_ci.py \\
        --results-a ./results/intfloat__multilingual-e5-large-instruct/mean_16.json \\
        --results-b ./results/Qwen__Qwen3-Embedding-8B/mean_16.json

Output: two-sided p-value, observed mean difference, and a 95% bootstrap
confidence interval on each model's mean_16.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import mteb_pt


def load_per_task_scores(results_path: Path) -> dict[str, float]:
    """Load per-task main_score dict from a `mean_16.json` summary."""
    with open(results_path) as f:
        data = json.load(f)
    out: dict[str, float] = {}
    for task, info in data.get("per_task", {}).items():
        if info.get("status") == "ok" and info.get("main_score") is not None:
            out[task] = float(info["main_score"])
    return out


def paired_bootstrap_pvalue(
    scores_a: list[float],
    scores_b: list[float],
    n_resamples: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Two-sided paired bootstrap p-value + observed mean difference.

    Resamples the matched task indices with replacement and recomputes
    mean(A) - mean(B) on each resample. The two-sided p-value is twice
    the smaller tail mass relative to the observed sign.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    assert a.shape == b.shape, "score arrays must be the same length"
    n = a.shape[0]
    observed = float(a.mean() - b.mean())
    n_pos = n_neg = 0
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        diff = a[idx].mean() - b[idx].mean()
        if diff > 0:
            n_pos += 1
        elif diff < 0:
            n_neg += 1
    if observed >= 0:
        p_one_sided = n_neg / n_resamples
    else:
        p_one_sided = n_pos / n_resamples
    p_two_sided = min(1.0, 2 * p_one_sided)
    return p_two_sided, observed


def bootstrap_ci(
    scores: list[float],
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """95% percentile-bootstrap CI on the mean of a score vector."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(scores, dtype=float)
    n = arr.shape[0]
    means = np.array([
        arr[rng.integers(0, n, size=n)].mean() for _ in range(n_resamples)
    ])
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired-bootstrap significance test between two model evaluations.",
    )
    parser.add_argument("--results-a", required=True, type=Path,
                        help="Path to mean_16.json for model A")
    parser.add_argument("--results-b", required=True, type=Path,
                        help="Path to mean_16.json for model B")
    parser.add_argument("--n-resamples", type=int, default=10_000,
                        help="Number of bootstrap resamples (default: 10000)")
    args = parser.parse_args()

    scores_a = load_per_task_scores(args.results_a)
    scores_b = load_per_task_scores(args.results_b)
    common = [t for t in mteb_pt.HEADLINE_TASKS if t in scores_a and t in scores_b]
    if len(common) < 10:
        raise SystemExit(
            f"Only {len(common)} headline tasks present in both result files "
            f"(out of {len(mteb_pt.HEADLINE_TASKS)}). Aborting."
        )

    arr_a = [scores_a[t] for t in common]
    arr_b = [scores_b[t] for t in common]

    mean_a, mean_b = float(np.mean(arr_a)), float(np.mean(arr_b))
    ci_a = bootstrap_ci(arr_a, n_resamples=args.n_resamples)
    ci_b = bootstrap_ci(arr_b, n_resamples=args.n_resamples)
    p, diff = paired_bootstrap_pvalue(arr_a, arr_b, n_resamples=args.n_resamples)

    print(f"Model A:  {args.results_a}")
    print(f"  mean_16 = {mean_a:.4f}  95% CI = [{ci_a[0]:.4f}, {ci_a[1]:.4f}]")
    print(f"Model B:  {args.results_b}")
    print(f"  mean_16 = {mean_b:.4f}  95% CI = [{ci_b[0]:.4f}, {ci_b[1]:.4f}]")
    print()
    print(f"Paired bootstrap (n_resamples = {args.n_resamples}, n_tasks = {len(common)}):")
    print(f"  observed mean difference (A - B) = {diff:+.4f}")
    print(f"  two-sided p-value                = {p:.4f}")
    print(f"  {'tied at alpha=0.05' if p > 0.05 else 'significantly different at alpha=0.05'}")


if __name__ == "__main__":
    main()
