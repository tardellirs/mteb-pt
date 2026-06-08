"""Smoke tests: verify all 16 headline tasks register and resolve through mteb."""
from __future__ import annotations

import pytest

import mteb_pt
import mteb_pt.register  # noqa: F401  -- side-effect import: registers tasks


def test_headline_tasks_count() -> None:
    """HEADLINE_TASKS should list exactly 16 task names."""
    assert len(mteb_pt.HEADLINE_TASKS) == 16


def test_tasks_by_category_matches_headline_set() -> None:
    """Per-category groupings, taken together, must equal HEADLINE_TASKS exactly.

    Catches the two-source-of-truth drift bug where someone updates
    HEADLINE_TASKS without updating TASKS_BY_CATEGORY (or vice versa).
    """
    flat = {t for v in mteb_pt.TASKS_BY_CATEGORY.values() for t in v}
    assert flat == set(mteb_pt.HEADLINE_TASKS)


@pytest.mark.parametrize("task_name", mteb_pt.HEADLINE_TASKS)
def test_task_resolves_via_mteb_get_task(task_name: str) -> None:
    """Every headline task must be retrievable via ``mteb.get_task(name)``."""
    import mteb

    task = mteb.get_task(task_name)
    assert task.metadata.name == task_name


@pytest.mark.parametrize("task_name", mteb_pt.HEADLINE_TASKS)
def test_task_has_portuguese_eval_lang(task_name: str) -> None:
    """Every headline task must declare Portuguese as an evaluation language."""
    import mteb

    task = mteb.get_task(task_name)
    assert "por-Latn" in task.metadata.eval_langs, (
        f"{task_name} does not declare por-Latn in eval_langs"
    )


@pytest.mark.parametrize("task_name", mteb_pt.HEADLINE_TASKS)
def test_task_dataset_is_pinned(task_name: str) -> None:
    """Every headline task must pin its source dataset to a specific revision SHA."""
    import mteb

    task = mteb.get_task(task_name)
    rev = task.metadata.dataset.get("revision")
    assert rev is not None and len(rev) >= 7, (
        f"{task_name} has no revision SHA (got {rev!r})"
    )
