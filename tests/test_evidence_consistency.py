"""证据链一致性：quick_index / 散文链接 / 标注之间的 id 与页码必须对齐。

这些断言把"关键字链接 ↔ PDF Annotations"的脏数据挡在提交前：
- quick_index 的 evidence_id 必须能解析到本章 items 或对应论文的标注；
- guide 散文中的 [[term|page|evidence_id]] 链接同理（paper_id 取本章首条 item 的论文，
  与 app/evidence.py:paper_guide 的回退逻辑一致）；
- evidence item 的 paper_id 必须在 paper_annotations.json 中有键；
- 同一 id 同时是 item 与标注时，页码必须一致。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LINK_RE = re.compile(r"\[\[([^|\]]+)\|(\d+)\|([^|\]]+)\]\]")

# 历史问题均已修复：
# - 4.3 的论文 id 是 mf，paper_annotations.json 现已通过 OUTPUT_ALIASES 写入 mf 键；
# - 4.2 的 scalability 证据与标注已对齐到第 9 页同一段话；
# - 4_7_classic_summary 的 mf 合成 id 随 mf 键修复而通过。
# 修复新章节时优先修数据，不要往这个集合里加新条目。
KNOWN_ISSUES: set[tuple[str, str]] = set()


def _load():
    evidence = json.loads((ROOT / "config/paper_evidence.json").read_text(encoding="utf-8"))
    annotations = {
        paper_id: {a["id"]: a for a in payload["annotations"]}
        for paper_id, payload in json.loads(
            (ROOT / "config/paper_annotations.json").read_text(encoding="utf-8")
        ).items()
    }
    return evidence, annotations


def test_quick_index_and_prose_links_resolve_to_items_or_annotations():
    evidence, annotations = _load()
    problems = []
    for slug, raw in evidence.items():
        if not isinstance(raw, dict):
            continue
        items = raw.get("items", [])
        item_ids = {item["id"] for item in items}
        first_paper = items[0]["paper_id"] if items else None

        def resolves(evidence_id: str, paper_id: str | None) -> bool:
            return evidence_id in item_ids or evidence_id in annotations.get(paper_id or "", {})

        for entry in raw.get("quick_index", []):
            pid = entry.get("paper_id") or first_paper
            if not resolves(entry["evidence_id"], pid):
                problems.append(f"{slug}: quick_index {entry['evidence_id']} -> {pid} 无法解析")
        for field, text in (raw.get("guide") or {}).items():
            if not isinstance(text, str):
                continue
            for match in LINK_RE.finditer(text):
                if not resolves(match.group(3), first_paper):
                    problems.append(f"{slug}: guide.{field} 链接 {match.group(3)} 无法解析")
    assert problems == []


def test_evidence_items_have_annotations_and_matching_pages():
    evidence, annotations = _load()
    problems = []
    for slug, raw in evidence.items():
        if not isinstance(raw, dict):
            continue
        for item in raw.get("items", []):
            if (slug, item["id"]) in KNOWN_ISSUES:
                continue
            paper = annotations.get(item["paper_id"])
            if paper is None:
                problems.append(f"{slug}: {item['id']} 的论文 {item['paper_id']} 没有标注键")
            elif item["id"] in paper and paper[item["id"]]["page"] != item["page"]:
                problems.append(
                    f"{slug}: {item['id']} 页码 {item['page']} != 标注页码 {paper[item['id']]['page']}"
                )
    assert problems == []
