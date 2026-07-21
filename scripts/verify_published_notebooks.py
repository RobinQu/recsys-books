"""Fail when checked notebook previews are stale, unexecuted, or host-specific."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import nbformat

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from app.content import NOTEBOOKS

FORBIDDEN_MARKERS = (
    str(ROOT),
    "/private/tmp/recsys-",
    "/var/folders/",
)


def verify(notebook_dir: Path, preview_dir: Path) -> None:
    expected = {item["slug"] for item in NOTEBOOKS}
    notebooks = {path.stem for path in notebook_dir.glob("*.ipynb")}
    previews = {path.stem for path in preview_dir.glob("*.html")}
    if notebooks != expected:
        raise AssertionError(f"notebook slugs differ: missing={expected - notebooks}, extra={notebooks - expected}")
    if previews != expected:
        raise AssertionError(f"preview slugs differ: missing={expected - previews}, extra={previews - expected}")

    for slug in sorted(expected):
        notebook_path = notebook_dir / f"{slug}.ipynb"
        preview_path = preview_dir / f"{slug}.html"
        notebook = nbformat.read(notebook_path, as_version=4)
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        if not code_cells or not any(cell.get("execution_count") is not None for cell in code_cells):
            raise AssertionError(f"{slug} has no executed code cell")
        if not any(cell.get("outputs") for cell in code_cells):
            raise AssertionError(f"{slug} has no recorded output")
        if any(
            output.get("output_type") == "error"
            for cell in code_cells
            for output in cell.get("outputs", [])
        ):
            raise AssertionError(f"{slug} contains an error output")
        if preview_path.stat().st_mtime_ns < notebook_path.stat().st_mtime_ns:
            raise AssertionError(f"{slug} preview is older than its notebook")

        notebook_text = notebook_path.read_text(encoding="utf-8")
        preview_text = preview_path.read_text(encoding="utf-8")
        for marker in FORBIDDEN_MARKERS:
            if marker in notebook_text or marker in preview_text:
                raise AssertionError(f"{slug} leaks machine-specific path {marker!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook-dir", type=Path, default=ROOT / "notebooks")
    parser.add_argument("--preview-dir", type=Path, default=ROOT / "notebook_previews")
    args = parser.parse_args()
    verify(args.notebook_dir.resolve(), args.preview_dir.resolve())
    print(f"verified {len(NOTEBOOKS)} executed notebooks and previews")


if __name__ == "__main__":
    main()
