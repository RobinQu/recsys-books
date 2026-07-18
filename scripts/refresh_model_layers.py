#!/usr/bin/env python3
"""Refresh the 关键模块 bullet list inside the Model Structure cell of deep notebooks.

Reads ``config/model_layers.json`` and rewrites the block under the embedded
paper figure so the text matches the current shared config. Preserves figure,
caption, and everything below ``### 结构``. Idempotent.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import nbformat  # noqa: E402
from tutorial_deep_specs import figure_markdown  # noqa: E402

NOTEBOOKS = ROOT / "notebooks"
FIGURE_RE = re.compile(r"/static/paper-figures/([a-z0-9_]+)\.webp")
# Match everything from the image line through the end of the bullet list,
# stopping at the first ### heading that follows (### 结构).
BLOCK_RE = re.compile(
    r"!\[Figure[^\n]+\]\(/static/paper-figures/(?P<pid>[a-z0-9_]+)\.webp\)\n\n"
    r"> \*\*论文原图节选\*\*[^\n]*\n\n"
    r"### 关键模块\n\n"
    r"(?:- \*\*[^*]+\*\*：[^\n]+\n)+",
)


def refresh(path: Path) -> str:
    nb = nbformat.read(path, as_version=4)
    for cell in nb.cells:
        if cell.cell_type != "markdown" or not cell.source.startswith("## Model Structure & Formula Walkthrough"):
            continue
        m = BLOCK_RE.search(cell.source)
        if not m:
            return f"skip (no figure block): {path.name}"
        pid = m.group("pid")
        fresh = figure_markdown(pid).rstrip() + "\n"
        cell.source = cell.source[: m.start()] + fresh + "\n" + cell.source[m.end():].lstrip("\n")
        # collapse extra blank lines
        cell.source = re.sub(r"\n{3,}", "\n\n", cell.source)
        nbformat.write(nb, path)
        return f"refreshed ({pid}): {path.name}"
    return f"skip (no Model Structure cell): {path.name}"


def main() -> None:
    for path in sorted(NOTEBOOKS.glob("*.ipynb")):
        print(refresh(path))


if __name__ == "__main__":
    main()
