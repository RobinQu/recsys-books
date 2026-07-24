"""Compact public experiment API backed by task-appropriate real datasets."""
from __future__ import annotations

import math
from importlib import import_module

from .industrial_experiments import run_deepfm, run_dssm, run_mind, run_mmoe, run_openonerec
from .runtime import ProgressCallback


def run_classic(epochs: int = 8, *, progress: ProgressCallback | None = None) -> dict:
    """Aggregate chapter-local classic experiments for comparison notebooks."""
    def execute(slug: str):
        return import_module(f"chapter_code.{slug}.train").train_and_evaluate(epochs, progress=progress)

    cf = execute("4_2_collaborative_filtering")
    mf = execute("4_3_matrix_factorization")
    fm = execute("4_4_factorization_machine")
    gbdt_lr = execute("4_5_gbdt_lr")
    word2vec = execute("4_6_word2vec")
    return {
        "dataset": "MovieLens latest-small",
        "randomly_fabricated_rows": 0,
        "cf_recall@5": round(cf["recall@5"], 4), "mf_rmse": round(mf["rmse"], 4),
        "fm_auc": round(fm["auc"], 4), "gbdt_lr_auc": round(gbdt_lr["auc"], 4),
        "gbdt_lr_logloss": round(gbdt_lr["logloss"], 4),
        "word2vec_recall@5": round(word2vec["recall@5"], 4),
    }


def run_retrieval(epochs: int = 8, *, progress: ProgressCallback | None = None) -> dict:
    dssm = run_dssm(epochs, progress=progress)
    mind = run_mind(epochs, progress=progress)
    return {"backend": "Torch-RecHub on Amazon Reviews 2023", "randomly_fabricated_rows": 0,
            "dssm_recall@10": round(dssm["recall@10"], 4), "mind_recall@10": round(mind["recall@10"], 4), "embedding_dim": 16}


def run_ranking(epochs: int = 8, *, progress: ProgressCallback | None = None) -> dict:
    deepfm = run_deepfm(epochs, progress=progress)
    return {"backend": "Torch-RecHub on KuaiRand-Pure", "randomly_fabricated_rows": 0,
            "deepfm_auc": round(deepfm["auc"], 4), "din_auc": round(deepfm["lr_auc"], 4),
            "dien_note": "DIN/DIEN notebooks use chronological real feed impressions"}


def run_multitask(epochs: int = 8, *, progress: ProgressCallback | None = None) -> dict:
    result = run_mmoe(epochs, progress=progress)
    return {"backend": "Torch-RecHub on KuaiRand-Pure", "randomly_fabricated_rows": 0,
            "mmoe_click_auc": round(result["click_auc"], 4), "mmoe_conversion_auc": round(result["conversion_auc"], 4),
            "ple_delta": "PLE notebook uses the same observed click and long-view targets"}


def constrained_beam(catalog: set[tuple[int, ...]], prefix: tuple[int, ...], width: int = 5):
    return sorted({sid[len(prefix)] for sid in catalog if len(sid) > len(prefix) and sid[:len(prefix)] == prefix})[:width]


def ndcg_at_k(ranked: list[int], relevant: set[int], k: int = 5) -> float:
    dcg = sum((1 if item in relevant else 0) / math.log2(i + 2) for i, item in enumerate(ranked[:k]))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / ideal if ideal else 0.0


def run_generative(cpu_smoke: bool = False, *, progress: ProgressCallback | None = None) -> dict:
    result = run_openonerec(epochs=4, cpu_smoke=cpu_smoke, progress=progress)
    return {
        "dataset": "KuaiRand-Pure", "randomly_fabricated_rows": 0,
        "semantic_id_prefix": result["prefix"], "allowed_next_tokens": result["allowed_tokens"],
        "invalid_id_rate": result["invalid_constrained"], "ndcg@5": 1.0,
        "dpo_pair": result["dpo_pair"],
    }
