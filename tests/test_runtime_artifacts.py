from __future__ import annotations

import json

from recsys_lab import runtime


def test_save_records_uses_default_project_root(monkeypatch, tmp_path):
    monkeypatch.delenv("RECSYS_ARTIFACT_ROOT", raising=False)
    monkeypatch.setattr(runtime, "ROOT", tmp_path)

    path = runtime.save_records("chapter_test", "default", [{"value": 1}])

    assert path == tmp_path / "results" / "chapter_test" / "default.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {"records": [{"value": 1}]}


def test_save_records_honors_artifact_root(monkeypatch, tmp_path):
    staged_root = tmp_path / "staged-artifacts"
    monkeypatch.setenv("RECSYS_ARTIFACT_ROOT", str(staged_root))

    path = runtime.save_records("chapter_test", "staged", [{"value": 2}])

    assert path == staged_root / "results" / "chapter_test" / "staged.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {"records": [{"value": 2}]}
