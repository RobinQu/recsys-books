#!/usr/bin/env python3
"""Render model-figure excerpts from locally initialized papers via pymupdf."""
from __future__ import annotations

import json
from pathlib import Path

import fitz  # pymupdf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "paper_figures.json"
RESOURCES = json.loads((ROOT / "config" / "resources.json").read_text(encoding="utf-8"))
OUT = ROOT / "app" / "static" / "paper-figures"
DPI = 200  # retina-sharp for notebook embedding


def _paper_paths() -> dict[str, Path]:
    return {row["id"]: ROOT / "resources" / row["path"] for row in RESOURCES["papers"]}


def main() -> None:
    figures = json.loads(CONFIG.read_text(encoding="utf-8"))
    papers = _paper_paths()
    OUT.mkdir(parents=True, exist_ok=True)
    for paper_id, spec in figures.items():
        paper = papers[paper_id]
        if not paper.exists():
            raise SystemExit(f"missing initialized paper: {paper}")
        doc = fitz.open(paper)
        page = doc[spec["page"] - 1]
        pix = page.get_pixmap(dpi=DPI)
        mode = "RGB" if pix.n < 4 else "RGBA"
        img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
        if mode == "RGBA":
            img = img.convert("RGB")
        left, top, right, bottom = spec["crop"]
        box = (
            round(left * img.width),
            round(top * img.height),
            round(right * img.width),
            round(bottom * img.height),
        )
        excerpt = img.crop(box)
        out_path = OUT / f"{paper_id}.webp"
        excerpt.save(out_path, "WEBP", quality=92, method=6)
        print(f"{paper_id}: p.{spec['page']} crop={box} size={excerpt.size} -> {out_path.name}")


if __name__ == "__main__":
    main()
