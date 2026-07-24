import json
from pathlib import Path

from PIL import Image


def test_original_paper_figure_excerpts_are_readable_assets():
    figures = json.loads(Path("config/paper_figures.json").read_text(encoding="utf-8"))
    assert {"dssm", "mind", "sasrec", "deepfm", "din", "dien", "mmoe", "ple", "tiger", "onerec", "hstu"} <= set(figures)
    for paper_id, spec in figures.items():
        assert spec["page"] >= 1 and len(spec["crop"]) == 4
        path = Path("app/static/paper-figures") / f"{paper_id}.webp"
        assert path.exists(), paper_id
        with Image.open(path) as image:
            assert image.width >= 500 and image.height >= 180, (paper_id, image.size)


def test_preview_html_has_formula_overflow_guards():
    html = Path("notebook_previews/3_1_math_foundations.html").read_text(encoding="utf-8")
    assert 'id="recsys-preview-style"' in html
    assert 'mjx-container[display="true"]' in html
