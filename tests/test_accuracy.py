"""Full-profile effect regression gates.

These tests run real training on the complete paper-aligned datasets, so they
are excluded from the default CPU suite: run them explicitly with
`RECSYS_PROFILE=full pytest -m accuracy` (dispatch-level) or
`RECSYS_PROFILE=full pytest -m cuda_accuracy` on a CUDA host. Thresholds are
conservative sanity floors, not paper-value reproductions; every gap between
this repository's protocol and the paper's must stay visible in the notebook's
paper-comparison guard.
"""
from __future__ import annotations

import importlib
import os

import pytest
import torch

requires_full = pytest.mark.skipif(os.getenv("RECSYS_PROFILE") != "full", reason="run with RECSYS_PROFILE=full")

pytestmark = [requires_full]


def _chapter(slug: str):
    return importlib.import_module(f"chapter_code.{slug}.train")


@pytest.mark.accuracy
def test_classic_chapters_on_complete_movielens_latest():
    cf = _chapter("4_2_collaborative_filtering").train_and_evaluate(epochs=4)
    assert cf["recall@5"] > 0.01
    mf = _chapter("4_3_matrix_factorization").train_and_evaluate(epochs=2)
    assert mf["rmse"] < 1.2
    fm = _chapter("4_4_factorization_machine").train_and_evaluate(epochs=2)
    assert fm["auc"] > 0.6
    w2v = _chapter("4_6_word2vec").train_and_evaluate(epochs=2)
    assert w2v["recall@5"] > 0.005


@pytest.mark.accuracy
def test_gbdt_lr_on_criteo_beats_sanity_floor():
    result = _chapter("4_5_gbdt_lr").train_and_evaluate(epochs=16)
    assert result["dataset"]["train_rows"] == 33_003_326
    assert result["auc"] > 0.70
    assert result["logloss"] < 0.60


@pytest.mark.accuracy
def test_deepfm_on_criteo_beats_lr_baseline():
    result = _chapter("6_2_deepfm").run_deepfm(epochs=2)
    assert result["dataset"]["test_rows"] == 4_587_167
    assert result["auc"] > result["lr_auc"]
    assert result["auc"] > 0.72


@pytest.mark.cuda_accuracy
@pytest.mark.skipif(not torch.cuda.is_available(), reason="generative accuracy requires CUDA")
def test_openonerec_on_recif_official_semantic_id_catalog():
    result = _chapter("8_2_openonerec_practice").run_openonerec(epochs=8)
    # 官方 SID 目录 + trie 约束：合法率必须是硬指标，而不是教学代理。
    assert result["dataset"]["users"] == 162_074
    assert result["dataset"]["semantic_catalog_size"] > 1_000_000
    assert result["invalid_constrained"] == 0.0
    assert result["validation_mode"] == "cuda_accuracy"


@pytest.mark.cuda_accuracy
@pytest.mark.skipif(not torch.cuda.is_available(), reason="generative accuracy requires CUDA")
def test_hstu_on_movielens_20m_beats_popularity():
    result = _chapter("8_3_dlrm_hstu_practice").run_hstu(epochs=6)
    assert result["dataset"]["rows_used"] == 20_000_263
    assert result["hr@5"] > result["popularity_hr@5"]
    assert result["validation_mode"] == "cuda_accuracy"
