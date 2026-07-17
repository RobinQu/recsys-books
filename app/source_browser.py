"""Read-only source index for the tutorial code walkthrough."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "recsys_lab" / "data.py"
EXPERIMENT_FILE = ROOT / "recsys_lab" / "industrial_experiments.py"
CLASSIC_FILE = ROOT / "recsys_lab" / "experiments.py"
RUN_FUNCTIONS = {
    "3_2_1_dssm": "run_dssm", "3_2_2_mind": "run_mind",
    "3_3_1_deepfm": "run_deepfm", "3_3_2_din": "run_din", "3_3_3_dien": "run_dien",
    "3_4_1_mmoe": "run_mmoe", "3_4_2_ple": "run_ple", "3_2_3_sasrec": "run_sasrec",
    "4_2_openonerec_practice": "run_openonerec", "4_3_dlrm_hstu_practice": "run_hstu",
}
EXPERIMENT_HELPERS = {
    "3_2_1_dssm": ["_real_amazon"],
    "3_2_2_mind": ["_real_amazon", "_mind_rows"],
    "3_3_1_deepfm": ["_real_kuairand", "_ranking_fields"],
    "3_3_2_din": ["_real_kuairand", "_run_sequence_ranker"],
    "3_3_3_dien": ["_real_kuairand", "_run_sequence_ranker"],
    "3_4_1_mmoe": ["_real_kuairand", "_multitask_view", "_run_multitask"],
    "3_4_2_ple": ["_real_kuairand", "_multitask_view", "_run_multitask"],
    "3_2_3_sasrec": ["_real_amazon", "_sequence_windows_from_sequences"],
    "4_2_openonerec_practice": ["_real_kuairand", "_semantic_catalog"],
    "4_3_dlrm_hstu_practice": ["_real_kuairand", "_sequence_windows_from_sequences"],
}


def ide_source_target(slug: str) -> str:
    """Map a tutorial page to one of the IDE opener's whitelisted files."""
    if slug == "3_0_data_pipeline":
        return "data"
    if slug.startswith("3_1_"):
        return "classic"
    if slug in RUN_FUNCTIONS:
        return "industrial"
    return "notebook_generator"


def _extract(path: Path, names: list[str]) -> str:
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    blocks = []
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            start, end = node.lineno, node.end_lineno or node.lineno
            numbered = "\n".join(f"{number:4d}  {lines[number - 1]}" for number in range(start, end + 1))
            blocks.append(f"# {path.relative_to(ROOT)}:{start} · {node.name}\n{numbered}")
    return "\n\n".join(blocks)


def source_sections(slug: str) -> list[dict[str, str]]:
    sections = [{
        "title": "数据加载、来源记录与时间切分", "path": "recsys_lab/data.py",
        "explanation": "检查数据来自哪个固定文件、怎样选取确定性切片、ID 如何映射，以及测试事件为何只来自时间末端。",
        "code": _extract(DATA_FILE, ["_load_cached", "load_movielens", "movielens_provenance", "_load_amazon_cached", "load_amazon_2023", "amazon_provenance", "_load_kuairand_cached", "load_kuairand", "kuairand_provenance", "leave_last_out"]),
    }]
    run_name = RUN_FUNCTIONS.get(slug)
    names = ["seed_everything", "_safe_auc", "_train_binary", "_recall_single_target", *EXPERIMENT_HELPERS.get(slug, [])] + ([run_name] if run_name else [])
    sections.append({
        "title": f"完整实验入口：{run_name}" if run_name else "实验框架公共骨架",
        "path": "recsys_lab/industrial_experiments.py",
        "explanation": "按模型实例化、张量构造、loss、optimizer、推理模式和指标的顺序阅读。辅助函数与入口函数均完整显示。",
        "code": _extract(EXPERIMENT_FILE, names),
    })
    return sections
