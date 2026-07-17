"""Executable quality gates for reader-facing recommendation code."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ALGORITHM_ROOTS = (Path("chapter_code"), Path("recsys_lab"))


def test_algorithm_python_sources_compile() -> None:
    """Compile in memory so syntax errors fail without leaving ``__pycache__`` files."""
    paths = [path for root in ALGORITHM_ROOTS for path in root.rglob("*.py")]
    assert paths, "algorithm source files were not found"
    for path in paths:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_algorithm_python_sources_pass_ruff() -> None:
    """Keep the same correctness-oriented lint gate in CI and code-server."""
    subprocess.run(
        [sys.executable, "-m", "ruff", "check", *(str(root) for root in ALGORITHM_ROOTS)],
        check=True,
    )
