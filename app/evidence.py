"""Paper-to-tutorial evidence anchors used by the chapter reader."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from recsys_lab.resources import ROOT, inspect_resource, paper_catalog

EVIDENCE_PATH = ROOT / "config" / "paper_evidence.json"
FIGURE_PATH = ROOT / "config" / "paper_figures.json"
MODEL_LAYERS_PATH = ROOT / "config" / "model_layers.json"
ANNOTATIONS_PATH = ROOT / "config" / "paper_annotations.json"


def load_evidence() -> dict[str, Any]:
    # A chapter value is either the legacy list of anchors, or an object
    # {"items": [...], "guide": {...}, "quick_index": [...]} for the enriched
    # paper-guide pages. Helpers below normalise both to the anchor list.
    return json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))


def load_paper_annotations() -> dict[str, list[dict]]:
    """Runtime EmbedPDF annotation geometry per paper, drawn by the reader.

    Produced by scripts/build_pdf_annotations.py from visually verified regions.
    Missing file means no runtime annotations (reader still works)."""
    if not ANNOTATIONS_PATH.exists():
        return {}
    raw = json.loads(ANNOTATIONS_PATH.read_text(encoding="utf-8"))
    return {paper_id: payload.get("annotations", []) for paper_id, payload in raw.items()}


def load_figures() -> dict[str, dict]:
    return json.loads(FIGURE_PATH.read_text(encoding="utf-8"))


def load_model_layers() -> dict[str, list[list[str]]]:
    """Key-module explanations shared by the web guide and the notebook generator."""
    return json.loads(MODEL_LAYERS_PATH.read_text(encoding="utf-8"))


# Frozen at import: the web paper-guide cards and the notebook figure block both
# read this same source of truth in config/model_layers.json.
MODEL_LAYERS = load_model_layers()

# Inline link markup inside guide prose: [[term|page|evidence_id]]
_LINK_RE = re.compile(r"\[\[([^|\]]+)\|(\d+)\|([^|\]]+)\]\]")


def _chapter_items(raw: Any) -> list[dict]:
    """Anchor list whether the chapter value is the legacy list or the new object."""
    if isinstance(raw, dict):
        return list(raw.get("items", []))
    return list(raw or [])


def _chapter_raw(slug: str, evidence: dict[str, Any]) -> dict:
    raw = evidence.get(slug)
    if isinstance(raw, dict):
        return raw
    return {"items": list(raw or [])}


def _parse_prose_links(text: str, paper_id: str | None) -> list[dict]:
    """Split prose into text/link segments so the template can render inline jumps."""
    segments: list[dict] = []
    pos = 0
    for match in _LINK_RE.finditer(text or ""):
        if match.start() > pos:
            segments.append({"type": "text", "value": text[pos:match.start()]})
        segments.append({
            "type": "link",
            "term": match.group(1).strip(),
            "page": int(match.group(2)),
            "evidence_id": match.group(3).strip(),
            "paper_id": paper_id,
        })
        pos = match.end()
    if pos < len(text or ""):
        segments.append({"type": "text", "value": text[pos:]})
    return segments


def _family_prefix(slug: str) -> str:
    if slug.startswith("3_0_"):
        return "3_0_"
    if slug.startswith("4_"):
        return "4_"
    return "_".join(slug.split("_")[:2]) + "_"


def _fallback_evidence(slug: str, evidence: dict[str, Any]) -> list[dict]:
    """Give every detail page a useful paper route, including openings and summaries."""
    prefix = _family_prefix(slug)
    if prefix == "3_0_":
        keys = ["3_1_2_matrix_factorization", "3_1_3_factorization_machine", "3_2_1_dssm"]
    else:
        keys = [key for key in evidence if key.startswith(prefix)]
    return [item for key in keys for item in _chapter_items(evidence.get(key))][:6]


def chapter_evidence(slug: str) -> list[dict]:
    catalog = paper_catalog()
    evidence = load_evidence()
    selected = _chapter_items(evidence.get(slug))
    if len(selected) < 2:
        # A practice page may only own one experiment anchor. Add another anchor
        # from the same paper before falling back to the broader chapter route.
        paper_ids = {item["paper_id"] for item in selected}
        related = [
            item
            for raw in evidence.values()
            for item in _chapter_items(raw)
            if item["paper_id"] in paper_ids and item not in selected
        ]
        selected.extend(related[: 2 - len(selected)])
    if not selected:
        selected = _fallback_evidence(slug, evidence)
    rows = []
    for item in selected:
        paper = catalog[item["paper_id"]]
        label_term = item["label"].split("：", 1)[-1]
        rows.append({
            **item,
            "paper_title": paper["title"],
            "paper_url": f"/papers/{item['paper_id']}",
            # Each evidence location deliberately exposes several semantic
            # entry points. They all resolve to the same verified PDF anchor.
            "annotation_terms": [item["keyword"], label_term, item["search"]],
        })
    return rows


def paper_guide(slug: str) -> dict:
    rows = chapter_evidence(slug)
    evidence = load_evidence()
    raw = _chapter_raw(slug, evidence)
    paper_id = rows[0]["paper_id"] if rows else None
    guide_raw = raw.get("guide", {}) if isinstance(raw, dict) else {}
    # Each guide field is a prose string that may embed [[term|page|evidence_id]]
    # inline links; parse them into renderable segments for the template.
    guide = {
        field: _parse_prose_links(value, paper_id)
        for field, value in guide_raw.items()
        if isinstance(value, str)
    }
    quick_index_raw = raw.get("quick_index", []) if isinstance(raw, dict) else []
    quick_index = [
        {**entry, "paper_id": entry.get("paper_id") or paper_id}
        for entry in quick_index_raw
    ]
    papers = list(dict.fromkeys(item["paper_title"] for item in rows))
    exact = slug in evidence
    diagram_id = paper_id
    diagram_path = ROOT / "app" / "static" / "diagrams" / f"{diagram_id}.svg"
    figure = load_figures().get(diagram_id or "")
    architecture_item = next(
        (item for item in rows if any(token in item["id"] for token in ("fig", "architecture", "model", "equation", "unification"))),
        rows[0] if rows else None,
    )
    layers = [
        {"name": name, "explanation": explanation,
         "evidence_id": architecture_item["id"], "paper_id": architecture_item["paper_id"],
         "page": architecture_item["page"], "label": architecture_item["label"]}
        for name, explanation in MODEL_LAYERS.get(diagram_id or "", [])
    ] if architecture_item else []
    return {
        "items": rows,
        "papers": papers,
        "exact": exact,
        "diagram_id": diagram_id if diagram_path.exists() else None,
        "structure_figure": ({**figure, "paper_id": diagram_id,
                              "url": f"/static/paper-figures/{diagram_id}.webp"}
                             if figure else None),
        "architecture_item": architecture_item,
        "layers": layers,
        "guide": guide,
        "quick_index": quick_index,
        "intro": (
            "本页按“研究问题-模型机制-实验结论-适用边界”阅读原论文。"
            if exact
            else "本页是导读或总结章节，证据路线聚合本章核心论文，先建立共同概念，再进入各算法的专属页面。"
        ),
    }


def paper_payload(paper_id: str) -> dict | None:
    paper = paper_catalog().get(paper_id)
    if paper is None:
        return None
    state = inspect_resource("papers", paper)
    anchors = [
        {**anchor, "chapter_slug": slug}
        for slug, raw in load_evidence().items()
        for anchor in _chapter_items(raw)
        if anchor["paper_id"] == paper_id
    ]
    return {
        **paper,
        "local_url": f"/resources/{paper['path']}",
        "ready": state.status == "ready",
        "state": state,
        "anchors": anchors,
        "annotations": load_paper_annotations().get(paper_id, []),
    }


SOURCE_PAPER_IDS = {
    "Matrix Factorization Techniques": "mf",
    "Factorization Machines": "fm",
    "word2vec": "word2vec",
    "DSSM": "dssm", "MIND": "mind", "DeepFM": "deepfm", "DIN": "din", "DIEN": "dien",
    "MMoE": "mmoe", "PLE": "ple", "SASRec": "sasrec", "TIGER": "tiger", "OneRec": "onerec", "HSTU": "hstu",
    "BERT4Rec": "bert4rec", "KuaiRand": "kuairand-paper",
}


def source_paper_links() -> dict[str, str]:
    return {name: f"/papers/{paper_id}" for name, paper_id in SOURCE_PAPER_IDS.items()}
