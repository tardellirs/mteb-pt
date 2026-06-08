"""Import-time smoke tests for the example scripts.

These do not run the scripts (which would require GPU + dataset downloads).
They only verify that each example parses, imports, and exposes a `main()`
callable — catching SyntaxError, broken imports, or accidental top-level
side effects in the public copy-paste examples.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
EXAMPLE_SCRIPTS = ["quickstart", "run_headline", "compute_bootstrap_ci"]


@pytest.mark.parametrize("name", EXAMPLE_SCRIPTS)
def test_example_script_imports_and_has_main(name: str) -> None:
    """Each example script must import cleanly and expose a `main()` function."""
    script_path = EXAMPLES_DIR / f"{name}.py"
    assert script_path.exists(), f"example script {script_path} missing"

    spec = importlib.util.spec_from_file_location(f"examples.{name}", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(getattr(module, "main", None)), (
        f"examples/{name}.py must define a callable `main()`"
    )
