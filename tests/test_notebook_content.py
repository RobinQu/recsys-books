import re
from pathlib import Path

import nbformat

from app.content import NOTEBOOKS


NOTEBOOK_KIND_BY_SLUG = {notebook["slug"]: notebook["kind"] for notebook in NOTEBOOKS}
CURRICULUM_ANCHORS = {
    "3_2_data_ml_basics": {"observation-label", "implicit-feedback", "split-leakage"},
    "3_3_linear_algebra": {
        "tensors-shapes", "elementwise-dot", "matmul-embedding", "low-rank-attention",
    },
    "3_4_calculus": {"functions", "derivative-gradient", "chain-rule"},
    "3_5_probability_statistics": {
        "random-variable", "conditional-chain", "expectation-variance", "likelihood-calibration",
    },
    "3_6_information_theory": {
        "information-entropy", "cross-entropy-kl", "softmax-temperature", "sequence-nll-dpo",
    },
    "3_7_optimization": {"sgd", "learning-rate", "regularization", "gradient-conflict"},
}
TEMPORARY_OBSOLETE_NOTEBOOKS = {
    "3_0_data_pipeline",
    "a_4_1_data_ml_basics", "a_4_2_linear_algebra", "a_4_3_calculus",
    "a_4_4_probability_statistics", "a_4_5_optimization", "a_4_6_information_theory",
}
LEGACY_CURRICULUM_SLUGS = {
    "a_4_1_data_ml_basics": "3_2_data_ml_basics",
    "a_4_2_linear_algebra": "3_3_linear_algebra",
    "a_4_3_calculus": "3_4_calculus",
    "a_4_4_probability_statistics": "3_5_probability_statistics",
    "a_4_5_optimization": "3_7_optimization",
    "a_4_6_information_theory": "3_6_information_theory",
}


def _source(notebook):
    return "\n".join(cell.source for cell in notebook.cells)


def _has_error_output(notebook):
    return any(
        output.output_type == "error"
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
    )


def test_math_foundations_is_visual_and_high_school_accessible():
    notebook = nbformat.read(Path("notebooks") / "3_1_math_foundations.ipynb", as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)
    for token in ["行为表", "点积", "余弦", "矩阵乘法", "Sigmoid", "LogLoss", "Softmax", "temperature", "QK^\\top", "遮罩", "梯度下降", "Recall@K", "matplotlib"]:
        assert token in source
    assert "高中" not in source
    assert source.count("plt.") >= 8
    assert not any(output.output_type == "error" for cell in notebook.cells if cell.cell_type == "code" for output in cell.get("outputs", []))


def test_math_foundations_defines_and_computes_recommendation_metrics():
    notebook = nbformat.read(Path("notebooks") / "3_1_math_foundations.ipynb", as_version=4)
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
        "4_7_classic_summary.ipynb": ["横向对比", "UserCF", "BiasMF", "FM", "GBDT+LR", "word2vec", "results/chapter_4", "来源论文"],
        "4_2_collaborative_filtering.ipynb": ["UserCF", "ItemCF", "余弦相似度", "推理", "结果讨论", "toy_R @ toy_R.T", "Sarwar", "直觉版"],
        "4_3_matrix_factorization.ipynb": ["BiasMF", "低秩", "训练", "Top-K", "结果讨论", "直觉版"],
        "4_4_factorization_machine.ipynb": ["FactorizationMachine", "二阶交互", "AUC", "LogLoss", "结果讨论", "直觉版"],
        "4_5_gbdt_lr.ipynb": ["XGBClassifier", "叶节点", "LogisticRegression", "推理", "结果讨论", "直觉版"],
        "4_6_word2vec.ipynb": ["SkipGram", "Item2Vec", "负采样", "Recall", "结果讨论", "直觉版"],
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
        "4_2_collaborative_filtering.ipynb",
        "4_3_matrix_factorization.ipynb",
        "4_4_factorization_machine.ipynb",
        "4_5_gbdt_lr.ipynb",
        "4_6_word2vec.ipynb",
    ]:
        notebook = nbformat.read(Path("notebooks") / filename, as_version=4)
        source = "\n".join(cell.source for cell in notebook.cells)
        assert "results" in source and "chapter_4" in source and "source_notebook" in source


def test_deep_chapters_are_split_and_use_industrial_framework_classes():
    expected = {
        "5_2_dssm.ipynb": ["DSSM", "Math by Hand", "run_dssm", "Train & Inference", "Results Discussion"],
        "5_3_mind.ipynb": ["MIND", "CapsuleNetwork", "run_mind", "Train & Inference", "Results Discussion"],
        "6_2_deepfm.ipynb": ["DeepFM", "二阶", "run_deepfm", "Train & Inference", "Results Discussion"],
        "6_3_din.ipynb": ["DIN", "候选", "run_din", "Train & Inference", "Results Discussion"],
        "6_4_dien.ipynb": ["DIEN", "AUGRU", "run_dien", "Train & Inference", "Results Discussion"],
        "7_2_mmoe.ipynb": ["MMoE", "gate", "run_mmoe", "Train & Inference", "Results Discussion"],
        "7_3_ple.ipynb": ["PLE", "专属专家", "run_ple", "Train & Inference", "Results Discussion"],
        "5_4_sasrec.ipynb": ["SASRec", "因果", "run_sasrec", "Train & Inference", "Results Discussion"],
        "8_2_openonerec_practice.ipynb": ["OpenOneRec", "trie", "run_openonerec", "Train & Inference", "Results Discussion"],
        "8_3_dlrm_hstu_practice.ipynb": ["HSTUModel", "因果", "run_hstu", "Train & Inference", "Results Discussion"],
    }
    for filename, required in expected.items():
        notebook = nbformat.read(Path("notebooks") / filename, as_version=4)
        source = "\n".join(cell.source for cell in notebook.cells)
        assert all(token in source for token in required)
        for section in ["## Paper & Context", "## Model Structure & Formula Walkthrough", "## Math by Hand", "## Data", "## Model & Framework", "## Checks", "## Next Steps"]:
            assert section in source
        assert "matplotlib" in source and "save_records" in source
        assert not any(output.output_type == "error" for cell in notebook.cells if cell.cell_type == "code" for output in cell.get("outputs", []))


def test_deep_notebooks_derive_model_specific_structure_and_loss():
    expected = {
        "5_2_dssm.ipynb": ["[B,N]", "temperature", "ANN"],
        "5_3_mind.ipynb": ["c_{jk}", "squash", "[B,K,d]"],
        "6_2_deepfm.ipynb": ["O(nd)", "二元交叉熵", "共享 embedding"],
        "6_3_din.ipynb": ["e_j-e_t", "[B,L,d]", "ActivationUnit"],
        "6_4_dien.ipynb": ["重置门", "AUGRU", "辅助损失"],
        "7_2_mmoe.ipynb": ["[B,E,d]", "task gate", "\\lambda_k"],
        "7_3_ple.ipynb": ["CGC", "共享专家", "progressive extraction"],
        "5_4_sasrec.ipynb": ["QK^\\top", "因果 mask", "softplus"],
        "8_2_openonerec_practice.ipynb": ["L_{DPO}", "teacher forcing", "trie"],
        "8_3_dlrm_hstu_practice.ipynb": ["SiLU", "pointwise aggregated attention", "next-item"],
    }
    for filename, tokens in expected.items():
        notebook = nbformat.read(Path("notebooks") / filename, as_version=4)
        source = "\n".join(cell.source for cell in notebook.cells)
        assert all(token in source for token in tokens), filename
        assert "高中" not in source


def test_generative_notebooks_are_cuda_first_with_cpu_basic_fallback():
    names = [
        "8_1_generative_foundations.ipynb",
        "8_4_generative_summary.ipynb",
        "8_2_openonerec_practice.ipynb",
        "8_3_dlrm_hstu_practice.ipynb",
    ]
    for name in names:
        notebook = nbformat.read(Path("notebooks") / name, as_version=4)
        assert notebook.metadata["recsys"]["requires_cuda"] is True
        source = "\n".join(cell.source for cell in notebook.cells)
        assert "默认要求 CUDA" in source
    for name in names[-2:]:
        source = "\n".join(cell.source for cell in nbformat.read(Path("notebooks") / name, as_version=4).cells)
        assert "cpu_smoke=not torch.cuda.is_available()" in source
        assert "validation_mode" in source


def test_pipeline_notebook_opens_imports_and_reimplements_core_steps():
    notebook = nbformat.read(Path("notebooks") / "3_8_data_pipeline.ipynb", as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)
    for token in ["inspect.getsource", "leave_last_out", "逐步重写", "class TinyMF", "loss.backward()", "model.eval()", "RMSE"]:
        assert token in source
    assert "randomly_fabricated_rows" in source and "高中" not in source


def test_every_large_chapter_has_a_result_aggregation_notebook():
    expected = {
        "4_7_classic_summary.ipynb": 6,
        "5_5_retrieval_summary.ipynb": 3,
        "6_5_ranking_summary.ipynb": 3,
        "7_4_multitask_summary.ipynb": 2,
        "8_4_generative_summary.ipynb": 2,
    }
    for filename, count in expected.items():
        notebook = nbformat.read(Path("notebooks") / filename, as_version=4)
        source = "\n".join(cell.source for cell in notebook.cells)
        assert "results" in source and "不手填" in source
        assert f"len(comparison)=={count}" in source or (filename == "4_7_classic_summary.ipynb" and "len(comparison) == 6" in source)


def test_summary_notebooks_explain_paper_comparability_and_do_not_hide_glyph_warnings():
    for filename in ["4_7_classic_summary.ipynb", "5_5_retrieval_summary.ipynb", "6_5_ranking_summary.ipynb"]:
        notebook = nbformat.read(Path("notebooks") / filename, as_version=4)
        source = "\n".join(cell.source for cell in notebook.cells)
        assert "原论文" in source and ("不可直接比较" in source or "不能相减" in source)
    for filename in ["5_5_retrieval_summary.ipynb", "6_5_ranking_summary.ipynb", "7_4_multitask_summary.ipynb", "8_4_generative_summary.ipynb"]:
        notebook = nbformat.read(Path("notebooks") / filename, as_version=4)
        source = "\n".join(cell.source for cell in notebook.cells)
        assert "Noto Sans CJK" in source and "ASCII fallback" in source
        assert "warnings.filterwarnings(" not in source
        output_text = "\n".join(str(output) for cell in notebook.cells for output in cell.get("outputs", []))
        assert "Glyph " not in output_text


def test_curriculum_notebooks_configure_a_cjk_chart_font_and_publish_clean_outputs():
    from app.content import NOTEBOOKS

    curriculum = [n["slug"] for n in NOTEBOOKS if n["kind"] == "curriculum"]
    assert len(curriculum) == 6
    for slug in curriculum:
        notebook = nbformat.read(Path("notebooks") / f"{slug}.ipynb", as_version=4)
        source = "\n".join(cell.source for cell in notebook.cells)
        assert "Noto Sans CJK" in source and "cjk_font" in source
        assert "plt.rcParams['axes.unicode_minus'] = False" in source
        assert "warnings.filterwarnings(" not in source
        output_text = "\n".join(str(output) for cell in notebook.cells for output in cell.get("outputs", []))
        assert "Glyph " not in output_text
        assert "图表字体:" in output_text
        assert "未找到中文字体" not in output_text


def test_every_large_chapter_has_a_math_opening_with_python_demo():
    expected = {
        "4_1_classic_foundations.ipynb": ["本章布局与选型地图", "共同数学", "余弦", "低秩", "Sigmoid"],
        "5_1_retrieval_foundations.ipynb": ["本章布局与选型地图", "Softmax", "Recall", "多兴趣", "temperature", "SASRec", "因果", "QK^\\top"],
        "6_1_ranking_foundations.ipynb": ["本章布局与选型地图", "LogLoss", "AUC", "注意力", "GRU"],
        "7_1_multitask_foundations.ipynb": ["本章布局与选型地图", "Softmax gate", "梯度", "负迁移", "cosine"],
        "8_1_generative_foundations.ipynb": ["本章布局与选型地图", "自回归", "因果 mask", "trie", "NDCG"],
    }
    for filename, tokens in expected.items():
        notebook = nbformat.read(Path("notebooks") / filename, as_version=4)
        source = "\n".join(cell.source for cell in notebook.cells)
        assert all(token in source for token in tokens)
        assert "来源论文" in source and "matplotlib" in source and "## Checks" in source
        assert sum(cell.cell_type == "code" for cell in notebook.cells) >= 3


def test_every_notebook_uses_a_task_appropriate_bundled_real_dataset():
    disk_slugs = {path.stem for path in Path("notebooks").glob("*.ipynb")}
    preview_slugs = {path.stem for path in Path("notebook_previews").glob("*.html")}
    registered_slugs = set(NOTEBOOK_KIND_BY_SLUG)
    assert len(registered_slugs) == 33
    assert disk_slugs == registered_slugs
    assert preview_slugs == registered_slugs
    notebook_paths = [
        Path("notebooks") / f"{slug}.ipynb" for slug in sorted(registered_slugs)
    ]
    for path in notebook_paths:
        if NOTEBOOK_KIND_BY_SLUG[path.stem] == "curriculum":
            continue
        notebook = nbformat.read(path, as_version=4)
        source = "\n".join(cell.source for cell in notebook.cells)
        if path.name.startswith(("3_", "4_")):
            assert "MovieLens latest-small" in source, path.name
        elif path.name.startswith("5_2"):
            assert "Amazon Reviews Books 5-core" in source, path.name
        elif path.name.startswith("5_3"):
            assert "Amazon Books 2014" in source and "6,271,511" in source, path.name
        elif path.name.startswith("5_4"):
            assert "MovieLens 1M" in source, path.name
        elif path.name.startswith("5_"):
            assert any(dataset in source for dataset in ["Amazon Reviews 2023", "Amazon Reviews Books 5-core", "Amazon Books 2014", "MovieLens 1M"]), path.name
        elif path.name.startswith(("6_3", "6_4")):
            assert "Amazon Reviews Electronics 5-core" in source, path.name
        elif path.name.startswith(("7_2", "7_3", "7_4_multitask_summary")):
            assert "Census-Income KDD" in source, path.name
        else:
            assert "KuaiRand" in source, path.name
        assert "REAL_DATASET" in source, path.name
        assert 'REAL_DATASET["randomly_fabricated_rows"] == 0' in source, path.name


def test_notebooks_default_to_formal_full_profile():
    for slug in NOTEBOOK_KIND_BY_SLUG:
        path = Path("notebooks") / f"{slug}.ipynb"
        notebook = nbformat.read(path, as_version=4)
        assert notebook.metadata["recsys"]["profile"] == "full", path.name
        if NOTEBOOK_KIND_BY_SLUG[path.stem] == "curriculum":
            assert notebook.metadata["recsys"]["kind"] == "curriculum", path.name
            continue
        setup = "\n".join(cell.source for cell in notebook.cells[:4])
        assert 'RECSYS_PROFILE", "full"' in setup, path.name


def test_3_math_curriculum_has_complete_teaching_structure():
    registered_curriculum = {
        slug for slug, kind in NOTEBOOK_KIND_BY_SLUG.items() if kind == "curriculum"
    }
    assert registered_curriculum == set(CURRICULUM_ANCHORS)

    algorithm_slugs = {
        slug for slug, kind in NOTEBOOK_KIND_BY_SLUG.items() if kind == "algorithm"
    }
    for slug, required_anchors in CURRICULUM_ANCHORS.items():
        notebook = nbformat.read(Path("notebooks") / f"{slug}.ipynb", as_version=4)
        source = _source(notebook)
        assert notebook.metadata["recsys"]["kind"] == "curriculum", slug
        assert notebook.metadata["recsys"]["source_of_truth"] == "scripts/tutorial_math_specs.py", slug
        assert all(
            heading in source
            for heading in ["## 学习路径", "## 符号表", "## 常见误区", "## 算法回链", "## Checks", "## Next Steps"]
        ), slug
        assert source.count("### 数字代入") >= 2, slug
        assert sum(cell.cell_type == "code" and "Demo" in cell.source for cell in notebook.cells) >= 2, slug
        anchors = set(re.findall(r'<a\s+id=["\']([^"\']+)["\']\s*></a>', source))
        assert required_anchors <= anchors, slug
        backlinks = set(re.findall(r"\]\(/notebooks/([^#/)]+)\)", source))
        assert backlinks and backlinks <= algorithm_slugs, slug
        assert not _has_error_output(notebook), slug

        if slug == "3_2_data_ml_basics":
            assert "load_movielens" in source and "movielens_provenance" in source
            assert "REAL_DATASET" in source
            assert 'REAL_DATASET["randomly_fabricated_rows"] == 0' in source
        else:
            assert "数学教学对象" in source, slug
            assert "REAL_DATASET" not in source, slug


def test_algorithm_prerequisite_links_resolve_to_published_3_0_curriculum_anchors():
    algorithm_slugs = [
        slug for slug, kind in NOTEBOOK_KIND_BY_SLUG.items() if kind == "algorithm"
    ]
    for algorithm_slug in algorithm_slugs:
        notebook = nbformat.read(
            Path("notebooks") / f"{algorithm_slug}.ipynb", as_version=4
        )
        source = _source(notebook)
        assert "通用先修" in source, algorithm_slug
        assert "本论文新增数学" in source, algorithm_slug
        links = re.findall(
            r"\]\(/notebooks/(?P<slug>(?:3_[2-7]_[^#/)]+|a_4_[^#/)]+))#(?P<anchor>[^)]+)\)",
            source,
        )
        assert links, algorithm_slug
        for curriculum_slug, anchor in links:
            curriculum_slug = LEGACY_CURRICULUM_SLUGS.get(curriculum_slug, curriculum_slug)
            assert curriculum_slug in CURRICULUM_ANCHORS, algorithm_slug
            assert anchor in CURRICULUM_ANCHORS[curriculum_slug], (
                algorithm_slug, curriculum_slug, anchor
            )

    generator_sources = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in ["scripts/generate_notebooks.py", "scripts/tutorial_deep_specs.py"]
    )
    assert "/notebooks/a_4_" not in generator_sources


def test_priority_math_audit_keeps_formula_and_data_protocol_boundaries_explicit():
    required = {
        "4_5_gbdt_lr": {
            "formula": ["叶节点", "one-hot", "Sigmoid", "Normalized Entropy"],
            "paper_protocol": ["Facebook", "内部广告数据口径"],
            "tutorial_protocol": ["MovieLens latest-small", "不可平移"],
        },
        "5_3_mind": {
            "formula": ["c_{jk}", "squash", "[B,K,d]", "label-aware"],
            "paper_protocol": ["6,271,511", "19:1", "Amazon Books"],
            "tutorial_protocol": ["时间切分", "smoke", "不能把两套数值直接相减"],
        },
        "6_3_din": {
            "formula": ["e_j-e_t", r"e_j\odot e_t", "不要求权重和为 1", "Dice"],
            "paper_protocol": ["Amazon Electronics", "阿里广告系统口径"],
            "tutorial_protocol": ["KuaiRand", "不能相减", "当前样本的负标签"],
        },
        "8_2_openonerec_practice": {
            "formula": ["P(y_1", r"\prod", "L_{DPO}", "trie", "Semantic ID"],
            "paper_protocol": ["RecIF", "reward", "官方"],
            "tutorial_protocol": ["KuaiRand", "smoke", "不把本地 chosen/rejected 当成官方 reward 数据"],
        },
        "8_3_dlrm_hstu_practice": {
            "formula": ["SiLU", "不做整序列 softmax", "因果 mask", "next-item"],
            "paper_protocol": ["MovieLens-20M", "Amazon Books", "full 协议"],
            "tutorial_protocol": ["KuaiRand", "不会自动切换", "严格时间切分"],
        },
    }
    for slug, sections in required.items():
        notebook = nbformat.read(Path("notebooks") / f"{slug}.ipynb", as_version=4)
        source = _source(notebook)
        for section, tokens in sections.items():
            missing = [token for token in tokens if token not in source]
            assert not missing, f"{slug} missing {section}: {missing}"
