import numpy as np
import pytest
import torch

from recsys_lab import run_classic, run_generative, run_multitask, run_ranking, run_retrieval
from recsys_lab.industrial_experiments import (
    run_deepfm, run_dien, run_din, run_dssm, run_hstu, run_mind,
    run_mmoe, run_openonerec, run_ple, run_sasrec,
)


def test_classic_metrics_are_functional():
    result = run_classic(epochs=8)
    assert 0 <= result["cf_recall@5"] <= 1
    assert result["mf_rmse"] < 3
    assert 0 <= result["fm_auc"] <= 1
    assert 0 <= result["gbdt_lr_auc"] <= 1
    assert 0 <= result["word2vec_recall@5"] <= 1
    assert result["randomly_fabricated_rows"] == 0


def test_deep_model_smokes_have_valid_effect_metrics():
    retrieval = run_retrieval(epochs=8)
    ranking = run_ranking(epochs=8)
    multitask = run_multitask(epochs=8)
    assert 0 <= retrieval["dssm_recall@10"] <= 1
    assert 0 <= ranking["deepfm_auc"] <= 1
    assert 0 <= multitask["mmoe_click_auc"] <= 1
    assert retrieval["randomly_fabricated_rows"] == ranking["randomly_fabricated_rows"] == multitask["randomly_fabricated_rows"] == 0


def test_constrained_generation_never_emits_invalid_token():
    result = run_generative(cpu_smoke=not torch.cuda.is_available())
    assert result["allowed_next_tokens"]
    assert result["invalid_id_rate"] == 0


def test_industrial_tutorial_models_train_and_infer():
    runs = [run_dssm(), run_mind(), run_deepfm(), run_din(), run_dien(), run_mmoe(), run_ple(), run_sasrec()]
    for result in runs:
        assert result["loss_curve"][-1] < result["loss_curve"][0]
        assert "framework" in result
        assert result["dataset"]["randomly_fabricated_rows"] == 0
    assert all("Amazon Reviews 2023" in runs[index]["dataset"]["dataset"] for index in [0, 1, 7])
    assert all("KuaiRand" in runs[index]["dataset"]["dataset"] for index in [2, 3, 4, 5, 6])
    assert 0 <= runs[0]["recall@10"] <= 1
    assert 0 <= runs[2]["auc"] <= 1
    assert 0 <= runs[4]["auc"] <= 1


def test_generative_cpu_mode_only_checks_basic_functionality():
    openonerec = run_openonerec(epochs=1, cpu_smoke=True)
    hstu = run_hstu(epochs=1, cpu_smoke=True)
    for result in [openonerec, hstu]:
        assert result["dataset"]["randomly_fabricated_rows"] == 0
        assert np.isfinite(result["loss_curve"]).all()
        assert result["validation_mode"] in {"cpu_basic_smoke", "cuda_accuracy"}
    assert openonerec["invalid_constrained"] == 0
    assert hstu["logits_shape"][0] > 0


@pytest.mark.skipif(torch.cuda.is_available(), reason="Only relevant to hosts without CUDA")
def test_generative_default_rejects_non_cuda_training():
    with pytest.raises(RuntimeError, match="requires CUDA"):
        run_openonerec(epochs=1)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        run_hstu(epochs=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Full generative accuracy validation requires CUDA")
def test_generative_cuda_accuracy_path():
    openonerec = run_openonerec(epochs=4)
    hstu = run_hstu(epochs=4)
    assert openonerec["validation_mode"] == hstu["validation_mode"] == "cuda_accuracy"
    assert openonerec["loss_curve"][-1] < openonerec["loss_curve"][0]
    assert hstu["loss_curve"][-1] < hstu["loss_curve"][0]
    assert openonerec["invalid_constrained"] == 0
    assert 0 <= hstu["hr@5"] <= 1


def test_bundled_dataset_slices_have_auditable_provenance():
    import json
    from pathlib import Path
    for directory, minimum_rows in [("amazon-reviews-2023-video-games", 30000), ("kuairand-pure", 70000)]:
        record = json.loads((Path("data") / directory / "provenance.json").read_text())
        assert record["randomly_fabricated_rows"] == 0
        assert record.get("rows", record.get("standard_rows", 0)) >= minimum_rows
        assert len(record["source_sha256"]) == 64


def test_sasrec_inference_respects_left_padding_and_disables_dropout():
    from pathlib import Path

    source = Path("chapter_code/3_2_3_sasrec/train.py").read_text(encoding="utf-8")
    assert "model.eval()" in source
    assert "seq_forward(batch, sequence_embedding)[:, -1]" in source
    assert "count(nonzero)-1" in source  # documents the upstream right-padding mismatch
