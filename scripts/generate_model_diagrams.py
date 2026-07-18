#!/usr/bin/env python3
"""Generate original SVG summaries from paper descriptions (not copied figures)."""
from __future__ import annotations

from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "app" / "static" / "diagrams"

FLOWS = {
    "dssm": (["用户历史/查询", "User Tower", "用户向量 u", "内积/余弦", "物品向量 v", "Item Tower", "物品特征"], "双塔在打分前保持独立，因此物品向量可预计算并建立 ANN 索引。"),
    "mind": (["行为序列", "动态路由", "K 个兴趣向量", "多路 ANN", "合并去重"], "训练时由目标物品选择兴趣，推理时每个兴趣独立召回。"),
    "sasrec": (["有序历史", "位置编码", "因果自注意力", "当前状态 hₜ", "Next-item 检索"], "因果遮罩阻止模型读取未来行为。"),
    "deepfm": (["稀疏字段", "共享 Embedding", "FM 二阶分支", "DNN 高阶分支", "联合 CTR"], "FM 与 DNN 共享表示并端到端优化。"),
    "din": (["候选 item", "候选-历史匹配", "加权兴趣", "特征拼接", "CTR"], "同一用户面对不同候选会产生不同兴趣表示。"),
    "dien": (["行为 Embedding", "GRU + 辅助损失", "兴趣状态", "AUGRU", "目标兴趣", "CTR"], "辅助目标监督兴趣抽取，候选注意力控制兴趣演化。"),
    "mmoe": (["共享输入", "多个 Expert", "Task A Gate", "Task A Tower", "Task B Gate", "Task B Tower"], "任务 gate 学习不同专家混合，缓解硬共享限制。"),
    "ple": (["共享/专属输入", "CGC Layer 1", "共享与任务流", "CGC Layer 2", "任务 Towers"], "每层限制信息路由，逐步分离公共与任务专属知识。"),
    "tiger": (["Item 内容", "RQ-VAE", "Semantic ID", "自回归生成", "合法 ID Trie", "Top-K"], "语义量化与受约束解码共同组成生成式召回。"),
    "onerec": (["行为 Session", "生成式推荐器", "候选列表", "奖励模型", "IPA / DPO", "对齐列表"], "先学习列表生成，再用偏好信号优化整页质量。"),
    "hstu": (["长行为流", "HSTU Blocks", "序列状态", "Next-action 目标", "检索/排序迁移"], "面向高基数非平稳行为流的统一序列转换。"),
}


def svg(model: str, nodes: list[str], caption: str) -> str:
    width, box_height, gap = 720, 62, 26
    height = 88 + len(nodes) * (box_height + gap) + 72
    box_width = 620
    boxes = []
    arrows = []
    for index, label in enumerate(nodes):
        x = 50
        y = 54 + index * (box_height + gap)
        boxes.append(f'<rect x="{x}" y="{y}" width="{box_width}" height="{box_height}" rx="13"/><circle cx="{x + 32}" cy="{y + box_height / 2}" r="17"/><text class="number" x="{x + 32}" y="{y + box_height / 2 + 6}">{index + 1}</text><text x="{x + box_width / 2 + 10}" y="{y + box_height / 2 + 7}">{escape(label)}</text>')
        if index:
            arrow_y = y - gap
            arrows.append(f'<path d="M360 {arrow_y - 2} V{y - 8}"/><path d="M352 {y - 16} L360 {y - 8} L368 {y - 16}"/>')
    return f'''<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="title desc" viewBox="0 0 {width} {height}">
<title id="title">{escape(model.upper())} 教学结构图</title><desc id="desc">{escape(caption)}</desc>
<style>rect{{fill:#18201b;stroke:#91c940;stroke-width:2}}circle{{fill:#91c940}}text{{fill:#edf4ee;font:650 20px system-ui;text-anchor:middle}}.number{{fill:#102015;font-size:17px;font-weight:850}}path{{fill:none;stroke:#91c940;stroke-width:3}}.caption{{fill:#9aa79d;font-size:15px;font-weight:400;text-anchor:start}}</style>
<rect width="{width}" height="{height}" rx="16" fill="#0f1511" stroke="#2d3930"/>{''.join(arrows)}{''.join(boxes)}
<text class="caption" x="50" y="{height - 34}">原创示意图 · {escape(caption)}</text></svg>'''


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for model, (nodes, caption) in FLOWS.items():
        (OUT / f"{model}.svg").write_text(svg(model, nodes, caption), encoding="utf-8")


if __name__ == "__main__":
    main()
