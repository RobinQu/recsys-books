from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.content import NOTEBOOKS


def _sanitize_output_value(value: Any, replacements: tuple[tuple[str, str], ...]) -> Any:
    """Remove machine-specific paths while preserving the reader-facing output."""
    if isinstance(value, str):
        for source, label in replacements:
            value = value.replace(source, label)
        value = re.sub(
            r"/(?:private/)?var/folders/[^\s:'\"]+|/(?:private/)?tmp/ipykernel_[^\s:'\"]+",
            "<KERNEL_TEMP>",
            value,
        )
        return value
    if isinstance(value, list):
        return [_sanitize_output_value(item, replacements) for item in value]
    if isinstance(value, dict):
        return {
            key: _sanitize_output_value(item, replacements)
            for key, item in value.items()
        }
    return value


def sanitize_notebook_outputs(
    notebook: nbformat.NotebookNode,
    *,
    project_root: Path,
    artifact_root: Path,
    execution_cwd: Path,
) -> None:
    """Replace host and staging paths in outputs before a notebook is published."""
    replacements = tuple(
        sorted(
            {
                str(project_root.resolve()): "<PROJECT_ROOT>",
                str(artifact_root.resolve()): "<ARTIFACT_ROOT>",
                str(execution_cwd.resolve()): "<EXECUTION_CWD>",
            }.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        )
    )
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        cell.outputs = [
            nbformat.from_dict(_sanitize_output_value(dict(output), replacements))
            for output in cell.get("outputs", [])
        ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="smoke", choices=["smoke", "full"])
    parser.add_argument("--only", help="只执行一个 Notebook slug")
    parser.add_argument(
        "--notebook-dir",
        type=Path,
        default=ROOT / "notebooks",
        help="Notebook 输入与写回目录；默认使用仓库 notebooks/",
    )
    parser.add_argument(
        "--execution-cwd",
        type=Path,
        default=ROOT / "notebooks",
        help="Notebook 内核工作目录；默认使用仓库 notebooks/，以便从工作区导入项目代码",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=ROOT,
        help="结果等运行产物的根目录；默认使用仓库根目录",
    )
    args = parser.parse_args()
    os.environ["RECSYS_PROFILE"] = args.profile
    notebook_dir = args.notebook_dir.resolve()
    execution_cwd = args.execution_cwd.resolve()
    artifact_root = args.artifact_root.resolve()
    if not execution_cwd.is_dir():
        parser.error(f"内核工作目录不存在: {execution_cwd}")
    artifact_root.mkdir(parents=True, exist_ok=True)
    os.environ["RECSYS_ARTIFACT_ROOT"] = str(artifact_root)
    slugs = [item["slug"] for item in NOTEBOOKS]
    summary_slugs = {"3_1_summary", "3_2_summary", "3_3_summary", "3_4_summary", "4_3_generative_summary"}
    execution_order = [slug for slug in slugs if slug not in summary_slugs] + [slug for slug in slugs if slug in summary_slugs]
    paths = [notebook_dir / f"{slug}.ipynb" for slug in execution_order]
    if args.only:
        paths = [notebook_dir / f"{args.only}.ipynb"]
    for path in paths:
        nb = nbformat.read(path, as_version=4)
        client = NotebookClient(
            nb,
            timeout=600 if args.profile == "smoke" else 7200,
            kernel_name="python3",
            resources={"metadata": {"path": str(execution_cwd)}},
        )
        client.execute()
        sanitize_notebook_outputs(
            nb,
            project_root=ROOT,
            artifact_root=artifact_root,
            execution_cwd=execution_cwd,
        )
        nbformat.write(nb, path)
        print(f"executed {path.name}")


if __name__ == "__main__":
    main()
