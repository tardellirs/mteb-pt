"""Numerical sanity checks for mteb_pt.stats.

These guard the foundation of paper Finding F1 (paired bootstrap) and the
per-task confidence intervals reported across the leaderboard. They are
fast (no models, no I/O) and catch regressions where a refactor silently
breaks the statistical primitives.
"""
from __future__ import annotations

import pytest

from mteb_pt.stats import BootstrapResult, bootstrap_metric, paired_bootstrap_pvalue


def test_bootstrap_metric_returns_well_formed_result() -> None:
    """CI must bracket the point estimate and respect the requested confidence."""
    scores = [0.5, 0.55, 0.6, 0.45, 0.5, 0.52, 0.48, 0.51, 0.53, 0.49]
    res = bootstrap_metric(scores, n_resamples=500, confidence=0.95, seed=42)
    assert isinstance(res, BootstrapResult)
    assert res.ci_low <= res.mean <= res.ci_high
    assert res.n_resamples == 500
    assert res.confidence == 0.95


def test_bootstrap_metric_mean_close_to_arithmetic_mean() -> None:
    """The reported `mean` field is the deterministic point estimate."""
    scores = [0.1, 0.2, 0.3, 0.4, 0.5]
    res = bootstrap_metric(scores, n_resamples=200, seed=0)
    assert res.mean == pytest.approx(0.3, abs=1e-9)


def test_bootstrap_metric_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        bootstrap_metric([], n_resamples=10)


def test_paired_bootstrap_clear_winner_yields_small_p() -> None:
    """When A strictly dominates B on every paired observation, p must be tiny."""
    a = [0.9] * 30
    b = [0.1] * 30
    p = paired_bootstrap_pvalue(a, b, n_resamples=500, seed=42)
    assert 0.0 < p < 0.02, f"expected p << 0.05 for dominated B, got {p}"


def test_paired_bootstrap_inverted_returns_one() -> None:
    """If B dominates A (observed diff <= 0), the test returns 1.0 by definition."""
    a = [0.1] * 20
    b = [0.9] * 20
    p = paired_bootstrap_pvalue(a, b, n_resamples=200, seed=0)
    assert p == 1.0


def test_paired_bootstrap_rejects_unequal_lengths() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap_pvalue([0.5, 0.6], [0.5], n_resamples=10)


def test_paired_bootstrap_tied_systems_give_large_p() -> None:
    """If A and B are identical, observed diff = 0 so p must be 1.0."""
    a = [0.5, 0.6, 0.55, 0.52]
    b = [0.5, 0.6, 0.55, 0.52]
    p = paired_bootstrap_pvalue(a, b, n_resamples=200, seed=0)
    assert p == 1.0
