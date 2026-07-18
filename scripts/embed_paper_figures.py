#!/usr/bin/env python3
"""Embed the paper structure figure into already-executed deep-learning notebooks.

This patches the committed ``.ipynb`` files in place so they match the updated
generator (``tutorial_deep_specs.figure_markdown``) without re-executing any code
cell -- all executed outputs (loss curves, metrics, tables) are preserved.

For every notebook whose ``Paper & Context`` cell still carries the in-notebook
PDF-reader jump link ``**本地原文：** [在站内 PDF 阅读器打开...](/papers/<id>)``:

1. extract the ``paper_id`` from that link, then remove the link line;
2. inject the cropped paper figure + key-module list at the top of the
   ``## Model Structure & Formula Walkthrough`` cell (idempotent).

Re-running on an already-patched notebook is a no-op.
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
# **本地原文：** [在站内 PDF 阅读器打开并保留证据页码](/papers/<id>)  (： is full-width)
LINK_RE = re.compile(
    r"\*\*本地原文：\*\* \[在站内 PDF 阅读器打开并保留证据页码\]\(/papers/([a-z0-9]+)\)"
)
FIGURE_RE = re.compile(r"/static/paper-figures/([a-z0-9]+)\.webp")


def _collapse_blank_runs(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text)


def patch_notebook(path: Path) -> str:
    nb = nbformat.read(path, as_version=4)
    paper_id: str | None = None
    changed = False

    for cell in nb.cells:
        if cell.cell_type != "markdown":
            continue
        match = LINK_RE.search(cell.source)
        if match:
            paper_id = match.group(1)
            cell.source = _collapse_blank_runs(LINK_RE.sub("", cell.source))
            changed = True

    # Already-patched notebooks lost the link; recover paper_id from the figure.
    if paper_id is None:
        for cell in nb.cells:
            if cell.cell_type == "markdown" and cell.source.startswith("## Model Structure"):
                figure_match = FIGURE_RE.search(cell.source)
                if figure_match:
                    paper_id = figure_match.group(1)
                break

    if paper_id is None:
        return f"skip (no paper evidence): {path.name}"

    for cell in nb.cells:
        if cell.cell_type != "markdown":
            continue
        if not cell.source.startswith("## Model Structure & Formula Walkthrough"):
            continue
        if f"/static/paper-figures/{paper_id}.webp" in cell.source:
            break  # figure already embedded
        idx = cell.source.find("### 结构")
        if idx == -1:
            return f"skip (no 结构 anchor): {path.name}"
        head = "## Model Structure & Formula Walkthrough"
        rest = cell.source[idx:]
        cell.source = f"{head}\n\n{figure_markdown(paper_id)}\n\n{rest}"
        changed = True
        break

    if changed:
        nbformat.write(nb, path)
        return f"patched ({paper_id}): {path.name}"
    return f"unchanged ({paper_id}): {path.name}"


def main() -> None:
    for path in sorted(NOTEBOOKS.glob("*.ipynb")):
        print(patch_notebook(path))


if __name__ == "__main__":
    main()
