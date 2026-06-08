"""Statistical utilities: bootstrap confidence intervals and paired tests.

Transplanted verbatim from huggingface-embedding/src/stats.py. Used for the
paper's per-task confidence intervals and model-vs-model significance tests.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass
class BootstrapResult:
    mean: float
    ci_low: float
    ci_high: float
    n_resamples: int
    confidence: float

    def as_str(self, decimals: int = 4) -> str:
        return (
            f"{self.mean:.{decimals}f} "
            f"[{self.ci_low:.{decimals}f}, {self.ci_high:.{decimals}f}] "
            f"(CI {int(self.confidence * 100)}%)"
        )


def bootstrap_metric(
    per_query_scores: Sequence[float],
    *,
    metric_fn: Callable[[FloatArray], float] = np.mean,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Bootstrap a metric over per-query scores.

    Standard recipe for IR papers: each query contributes one score (MRR, nDCG, etc.),
    resample queries with replacement, recompute the metric, derive a CI.
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(per_query_scores, dtype=np.float64)
    n = len(arr)
    if n == 0:
        raise ValueError("per_query_scores is empty")

    resampled = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resampled[i] = metric_fn(arr[idx])

    alpha = (1.0 - confidence) / 2.0
    ci_low = float(np.quantile(resampled, alpha))
    ci_high = float(np.quantile(resampled, 1.0 - alpha))
    return BootstrapResult(
        mean=float(metric_fn(arr)),
        ci_low=ci_low,
        ci_high=ci_high,
        n_resamples=n_resamples,
        confidence=confidence,
    )


def paired_bootstrap_pvalue(
    system_a: Sequence[float],
    system_b: Sequence[float],
    *,
    n_resamples: int = 1000,
    seed: int = 42,
) -> float:
    """Paired bootstrap test: probability that system A is NOT better than system B.

    Lower p means stronger evidence that A > B. Each pair corresponds to the
    same query under both systems.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(system_a, dtype=np.float64)
    b = np.asarray(system_b, dtype=np.float64)
    if len(a) != len(b):
        raise ValueError("system_a and system_b must have the same length")
    diffs = a - b
    observed = diffs.mean()

    if observed <= 0:
        return 1.0

    n = len(diffs)
    centered = diffs - observed
    count_extreme = 0
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        if centered[idx].mean() >= observed:
            count_extreme += 1
    return (count_extreme + 1) / (n_resamples + 1)
