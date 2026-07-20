#!/usr/bin/env python3
"""Bake highlight + sticky-comment annotations into local paper PDFs at each
evidence anchor's quote location, so the EmbedPDF reader shows the marked
passage and the tutorial's interpretation as a comment.

Idempotent: annotations authored by ANNOT_AUTHOR are removed before re-adding,
so re-running never stacks duplicates. Uses PyMuPDF (fitz).

Examples:
    python scripts/annotate_papers.py --paper grouplens --paper he2014 --paper word2vec
    python scripts/annotate_papers.py --all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "config" / "paper_evidence.json"
PAPERS = ROOT / "resources" / "papers"
ANNOT_AUTHOR = "RecSys Atlas"

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    print("PyMuPDF (fitz) is required: uv add pymupdf", file=sys.stderr)
    sys.exit(1)


def _chapter_items(raw):
    if isinstance(raw, dict):
        return list(raw.get("items", []))
    return list(raw or [])


def collect_anchors() -> dict[str, list[dict]]:
    """paper_id -> list of anchor dicts (quote, page, note, label, keyword)."""
    data = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    by_paper: dict[str, list[dict]] = {}
    for raw in data.values():
        for item in _chapter_items(raw):
            by_paper.setdefault(item["paper_id"], []).append(item)
    return by_paper


def _find_quote_rects(doc, quote: str, page_no: int):
    """Search the quote on the declared page first, then neighbours. Returns (rects, page_idx) or ([], None)."""
    for offset in (0, -1, 1, -2, 2):
        idx = page_no + offset
        if 0 <= idx < doc.page_count:
            found = doc[idx].search_for(quote)
            if found:
                return found, idx
    return [], None


def annotate_paper(paper_id: str, anchors: list[dict]) -> dict:
    pdf_path = PAPERS / f"{paper_id}.pdf"
    if not pdf_path.exists():
        return {"paper": paper_id, "status": "missing"}
    doc = fitz.open(pdf_path)
    baked, skipped = 0, 0
    for anchor in anchors:
        quote = anchor.get("quote", "")
        declared = int(anchor.get("page", 1)) - 1
        note = anchor.get("note", "")
        label = anchor.get("label", anchor.get("keyword", ""))
        if not quote:
            skipped += 1
            continue
        rects, page_idx = _find_quote_rects(doc, quote, declared)
        if not rects:
            # fall back to the short searchable phrase
            short = anchor.get("search", "")
            if short:
                rects, page_idx = _find_quote_rects(doc, short, declared)
        if not rects:
            skipped += 1
            print(f"  [skip] {anchor['id']}: quote not found in {pdf_path.name}", file=sys.stderr)
            continue
        page = doc[page_idx]
        marker = f"[{anchor['id']}]"
        # idempotency: remove our prior highlight + sticky for this anchor
        for annot in list(page.annots()):
            info = annot.info or {}
            if info.get("title") == ANNOT_AUTHOR and marker in (info.get("content") or ""):
                page.delete_annot(annot)
        # highlight the quote passage (multi-rect supported)
        highlight = page.add_highlight_annot(rects)
        highlight.set_info(title=ANNOT_AUTHOR, content=marker)
        highlight.update()
        # sticky comment carrying the tutorial's interpretation
        comment = f"{marker} {label}\n{note}".strip()
        sticky = page.add_text_annot(rects[0].tl, comment, icon="Comment")
        sticky.set_info(title=ANNOT_AUTHOR, content=comment)
        sticky.update()
        baked += 1
    tmp = pdf_path.with_suffix(".annotated.pdf")
    doc.save(tmp, garbage=4, deflate=True)
    doc.close()
    tmp.replace(pdf_path)
    return {"paper": paper_id, "status": "annotated", "baked": baked, "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", action="append", help="annotate a single paper id (repeatable)")
    parser.add_argument("--all", action="store_true", help="annotate every paper with anchors + local PDF")
    args = parser.parse_args()
    by_paper = collect_anchors()
    targets = args.paper or (list(by_paper) if args.all else [])
    if not targets:
        parser.error("pass --paper <id> (repeatable) or --all")
    for paper_id in targets:
        if paper_id not in by_paper:
            print({"paper": paper_id, "status": "no-anchors"})
            continue
        print(annotate_paper(paper_id, by_paper[paper_id]))


if __name__ == "__main__":
    main()
