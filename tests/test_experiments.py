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
    result = run_generative()
    assert result["allowed_next_tokens"]
    assert result["invalid_id_rate"] == 0


def test_industrial_tutorial_models_train_and_infer():
    runs = [run_dssm(), run_mind(), run_deepfm(), run_din(), run_dien(), run_mmoe(), run_ple(), run_sasrec(), run_openonerec(), run_hstu()]
    for result in runs:
        assert result["loss_curve"][-1] < result["loss_curve"][0]
        assert "framework" in result
        assert result["dataset"]["randomly_fabricated_rows"] == 0
        assert "MovieLens latest-small" in result["dataset"]["dataset"]
    assert 0 <= runs[0]["recall@10"] <= 1
    assert 0 <= runs[2]["auc"] <= 1
    assert 0 <= runs[4]["auc"] <= 1
    assert runs[8]["invalid_constrained"] == 0
    assert 0 <= runs[9]["hr@5"] <= 1
