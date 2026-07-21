from __future__ import annotations

import argparse
from pathlib import Path
import sys

import nbformat
from nbconvert import HTMLExporter
from traitlets.config import Config
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from app.notebook_preview import polish_preview

OUT = ROOT / "notebook_previews"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="只构建一个 Notebook slug 的预览")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "notebooks",
        help="Notebook 输入目录；默认使用仓库 notebooks/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT,
        help="HTML 预览输出目录；默认使用仓库 notebook_previews/",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.only:
        paths = [input_dir / f"{args.only}.ipynb"]
        if not paths[0].is_file():
            parser.error(f"Notebook 不存在: {args.only}")
    else:
        for stale in output_dir.glob("*.html"):
            stale.unlink()
        paths = sorted(input_dir.glob("*.ipynb"))

    config = Config(); config.HTMLExporter.exclude_input_prompt = True; config.HTMLExporter.exclude_output_prompt = True
    exporter = HTMLExporter(template_name="lab", config=config)
    for path in paths:
        body, _ = exporter.from_notebook_node(nbformat.read(path, as_version=4))
        (output_dir / f"{path.stem}.html").write_text(polish_preview(body), encoding="utf-8")
        print(f"preview {path.stem}.html")


if __name__ == "__main__":
    main()
