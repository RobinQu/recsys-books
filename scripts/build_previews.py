from __future__ import annotations

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
    OUT.mkdir(exist_ok=True)
    for stale in OUT.glob("*.html"):
        stale.unlink()
    config = Config(); config.HTMLExporter.exclude_input_prompt = True; config.HTMLExporter.exclude_output_prompt = True
    exporter = HTMLExporter(template_name="lab", config=config)
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        body, _ = exporter.from_notebook_node(nbformat.read(path, as_version=4))
        (OUT / f"{path.stem}.html").write_text(polish_preview(body), encoding="utf-8")
        print(f"preview {path.stem}.html")


if __name__ == "__main__":
    main()
