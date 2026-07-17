from pathlib import Path

import nbformat


def test_math_foundations_is_visual_and_high_school_accessible():
    notebook = nbformat.read(Path("notebooks") / "3_0_math_foundations.ipynb", as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)
    for token in ["行为表", "点积", "余弦", "矩阵乘法", "Sigmoid", "LogLoss", "梯度下降", "Recall@K", "matplotlib"]:
        assert token in source
    assert "高中" not in source
    assert source.count("plt.") >= 8
    assert not any(output.output_type == "error" for cell in notebook.cells if cell.cell_type == "code" for output in cell.get("outputs", []))


def test_math_foundations_defines_and_computes_recommendation_metrics():
    notebook = nbformat.read(Path("notebooks") / "3_0_math_foundations.ipynb", as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)
    required = [
        "Precision@K", "Recall@K", "HitRate@K", "F1@K",
        "MRR", "NDCG@K", "MAP@K", "MAE", "RMSE",
        "LogLoss", "AUC", "GAUC", "Coverage@K",
        "macro average", "micro average",
    ]
    assert all(token in source for token in required)
    for expression in ["precision_at_5 =", "ndcg_at_5 =", "ap_at_5 =", "rmse =", "pairwise_auc =", "coverage_at_3 ="]:
        assert expression in source


def test_chapter_31_is_split_by_algorithm_with_summary():
    expected = {
        "3_1_summary.ipynb": ["横向对比", "UserCF", "BiasMF", "FM", "GBDT+LR", "results/chapter_3_1", "来源论文"],
        "3_1_1_collaborative_filtering.ipynb": ["UserCF", "ItemCF", "余弦相似度", "推理", "结果讨论", "toy_R @ toy_R.T", "Sarwar", "直觉版"],
        "3_1_2_matrix_factorization.ipynb": ["BiasMF", "低秩", "训练", "Top-K", "结果讨论", "直觉版"],
        "3_1_3_factorization_machine.ipynb": ["FactorizationMachine", "二阶交互", "AUC", "LogLoss", "结果讨论", "直觉版"],
        "3_1_4_gbdt_lr.ipynb": ["XGBClassifier", "叶节点", "LogisticRegression", "推理", "结果讨论", "直觉版"],
    }
    for filename, required in expected.items():
        notebook = nbformat.read(Path("notebooks") / filename, as_version=4)
        source = "\n".join(cell.source for cell in notebook.cells)
        assert all(token in source for token in required)
        assert "高中" not in source
        assert "## Checks" in source and "## Next Steps" in source
        assert not any(output.output_type == "error" for cell in notebook.cells if cell.cell_type == "code" for output in cell.get("outputs", []))


def test_chapter_31_algorithm_notebooks_export_metrics():
    for filename in [
        "3_1_1_collaborative_filtering.ipynb",
        "3_1_2_matrix_factorization.ipynb",
        "3_1_3_factorization_machine.ipynb",
        "3_1_4_gbdt_lr.ipynb",
    ]:
        notebook = nbformat.read(Path("notebooks") / filename, as_version=4)
        source = "\n".join(cell.source for cell in notebook.cells)
        assert "results" in source and "chapter_3_1" in source and "source_notebook" in source


def test_deep_chapters_are_split_and_use_industrial_framework_classes():
    expected = {
        "3_2_1_dssm.ipynb": ["DSSM", "Math by Hand", "run_dssm", "Train & Inference", "Results Discussion"],
        "3_2_2_mind.ipynb": ["MIND", "CapsuleNetwork", "run_mind", "Train & Inference", "Results Discussion"],
        "3_3_1_deepfm.ipynb": ["DeepFM", "二阶", "run_deepfm", "Train & Inference", "Results Discussion"],
        "3_3_2_din.ipynb": ["DIN", "候选", "run_din", "Train & Inference", "Results Discussion"],
        "3_3_3_dien.ipynb": ["DIEN", "AUGRU", "run_dien", "Train & Inference", "Results Discussion"],
        "3_4_1_mmoe.ipynb": ["MMoE", "gate", "run_mmoe", "Train & Inference", "Results Discussion"],
        "3_4_2_ple.ipynb": ["PLE", "专属专家", "run_ple", "Train & Inference", "Results Discussion"],
        "3_5_1_sasrec.ipynb": ["SASRec", "因果", "run_sasrec", "Train & Inference", "Results Discussion"],
        "4_2_openonerec_practice.ipynb": ["OpenOneRec", "trie", "run_openonerec", "Train & Inference", "Results Discussion"],
        "4_3_dlrm_hstu_practice.ipynb": ["HSTUModel", "因果", "run_hstu", "Train & Inference", "Results Discussion"],
    }
    for filename, required in expected.items():
        notebook = nbformat.read(Path("notebooks") / filename, as_version=4)
        source = "\n".join(cell.source for cell in notebook.cells)
        assert all(token in source for token in required)
        for section in ["## Paper & Context", "## Math by Hand", "## Data", "## Model & Framework", "## Checks", "## Next Steps"]:
            assert section in source
        assert "matplotlib" in source and "save_records" in source
        assert not any(output.output_type == "error" for cell in notebook.cells if cell.cell_type == "code" for output in cell.get("outputs", []))


def test_every_large_chapter_has_a_result_aggregation_notebook():
    expected = {
        "3_1_summary.ipynb": 4,
        "3_2_summary.ipynb": 2,
        "3_3_summary.ipynb": 3,
        "3_4_summary.ipynb": 2,
        "3_5_summary.ipynb": 1,
        "4_1_generative_overview.ipynb": 2,
    }
    for filename, count in expected.items():
        notebook = nbformat.read(Path("notebooks") / filename, as_version=4)
        source = "\n".join(cell.source for cell in notebook.cells)
        assert "results" in source and "不手填" in source
        assert f"len(comparison)=={count}" in source or (filename == "3_1_summary.ipynb" and "len(comparison) == 5" in source)


def test_every_large_chapter_has_a_math_opening_with_python_demo():
    expected = {
        "3_1_0_classic_foundations.ipynb": ["本章布局与选型地图", "共同数学", "余弦", "低秩", "Sigmoid"],
        "3_2_0_retrieval_foundations.ipynb": ["本章布局与选型地图", "Softmax", "Recall", "多兴趣", "temperature"],
        "3_3_0_ranking_foundations.ipynb": ["本章布局与选型地图", "LogLoss", "AUC", "注意力", "GRU"],
        "3_4_0_multitask_foundations.ipynb": ["本章布局与选型地图", "Softmax gate", "梯度", "负迁移", "cosine"],
        "3_5_0_transformer_foundations.ipynb": ["本章布局与选型地图", "SASRec", "BERT4Rec", "Scaled Dot-Product", "因果"],
        "4_0_generative_foundations.ipynb": ["本章布局与选型地图", "自回归", "因果 mask", "trie", "NDCG"],
    }
    for filename, tokens in expected.items():
        notebook = nbformat.read(Path("notebooks") / filename, as_version=4)
        source = "\n".join(cell.source for cell in notebook.cells)
        assert all(token in source for token in tokens)
        assert "来源论文" in source and "matplotlib" in source and "## Checks" in source
        assert sum(cell.cell_type == "code" for cell in notebook.cells) >= 3


def test_every_notebook_uses_the_bundled_real_dataset():
    notebook_paths = sorted(Path("notebooks").glob("*.ipynb"))
    assert len(notebook_paths) == 27
    for path in notebook_paths:
        notebook = nbformat.read(path, as_version=4)
        source = "\n".join(cell.source for cell in notebook.cells)
        assert "MovieLens latest-small" in source, path.name
        assert "REAL_DATASET" in source, path.name
        assert 'REAL_DATASET["randomly_fabricated_rows"] == 0' in source, path.name
