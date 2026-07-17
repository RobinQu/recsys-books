"""Selected-file source browser for every tutorial chapter."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer, TextLexer

ROOT = Path(__file__).resolve().parents[1]
CHAPTER_ROOT = ROOT / "chapter_code"

FRAMEWORK_MODULES = {
    "3_2_1_dssm": ["torch_rechub.models.matching.dssm", "torch_rechub.trainers.match_trainer"],
    "3_2_2_mind": ["torch_rechub.models.matching.mind", "torch_rechub.trainers.match_trainer"],
    "3_2_3_sasrec": ["torch_rechub.models.matching.sasrec", "torch_rechub.trainers.seq_trainer"],
    "3_3_1_deepfm": ["torch_rechub.models.ranking.deepfm", "torch_rechub.trainers.ctr_trainer"],
    "3_3_2_din": ["torch_rechub.models.ranking.din", "torch_rechub.trainers.ctr_trainer"],
    "3_3_3_dien": ["torch_rechub.models.ranking.dien", "torch_rechub.trainers.ctr_trainer"],
    "3_4_1_mmoe": ["torch_rechub.models.multi_task.mmoe", "torch_rechub.trainers.mtl_trainer"],
    "3_4_2_ple": ["torch_rechub.models.multi_task.ple", "torch_rechub.trainers.mtl_trainer"],
    "4_3_dlrm_hstu_practice": ["torch_rechub.models.generative.hstu", "torch_rechub.trainers.seq_trainer"],
}


def _framework_path(module: str) -> Path | None:
    spec = importlib.util.find_spec(module)
    return Path(spec.origin) if spec and spec.origin else None


def _file(path: Path, label: str, origin: str) -> dict[str, str]:
    code = path.read_text(encoding="utf-8")
    lexer = PythonLexer() if path.suffix == ".py" else TextLexer()
    key_markers = (
        "class ", "def forward", "def run_", "def train_", "model =", "optimizer =",
        "loss =", "losses =", "loss.backward", "with torch.no_grad", "scores =",
        "probability =", "topk", "return {",
    )
    important_lines = [
        number for number, line in enumerate(code.splitlines(), start=1)
        if any(marker in line for marker in key_markers)
    ]
    return {
        "id": f"source-{abs(hash(str(path)))}",
        "label": label,
        "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "origin": origin,
        "code": code,
        "highlighted": highlight(
            code,
            lexer,
            HtmlFormatter(linenos="inline", hl_lines=important_lines, cssclass="source-highlight"),
        ),
    }


def source_files(slug: str) -> list[dict]:
    groups: list[dict] = []
    shared_paths = [ROOT / "recsys_lab" / "data.py", ROOT / "recsys_lab" / "runtime.py"]
    if slug.startswith("3_1_"):
        shared_paths.append(ROOT / "recsys_lab" / "experiments.py")
    shared_paths.append(ROOT / "tests" / "test_experiments.py")
    groups.append({"name": "Notebook 公用代码", "files": [_file(path, path.name, "shared") for path in shared_paths]})

    chapter_dir = CHAPTER_ROOT / slug
    order = ["model.py", "train.py", "inference.py", "test_model.py", "__init__.py"]
    chapter_files = [_file(chapter_dir / name, name, "chapter") for name in order if (chapter_dir / name).exists()]
    groups.append({"name": "本章节独立目录", "files": chapter_files})

    framework_files = []
    for module in FRAMEWORK_MODULES.get(slug, []):
        path = _framework_path(module)
        if path and path.exists():
            framework_files.append(_file(path, path.name, f"third-party · {module}"))
    if framework_files:
        groups.append({"name": "Torch-RecHub 框架源码", "files": framework_files})
    return groups
