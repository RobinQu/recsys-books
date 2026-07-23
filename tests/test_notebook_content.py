import re
from pathlib import Path

import nbformat

from app.content import NOTEBOOKS


NOTEBOOK_KIND_BY_SLUG = {notebook["slug"]: notebook["kind"] for notebook in NOTEBOOKS}
CURRICULUM_ANCHORS = {
    "3_0_1_data_ml_basics": {"observation-label", "implicit-feedback", "split-leakage"},
    "3_0_2_linear_algebra": {
        "tensors-shapes", "elementwise-dot", "matmul-embedding", "low-rank-attention",
    },
    "3_0_3_calculus": {"functions", "derivative-gradient", "chain-rule"},
    "3_0_4_probability_statistics": {
        "random-variable", "conditional-chain", "expectation-variance", "likelihood-calibration",
    },
    "3_0_5_information_theory": {
        "information-entropy", "cross-entropy-kl", "softmax-temperature", "sequence-nll-dpo",
    },
    "3_0_6_optimization": {"sgd", "learning-rate", "regularization", "gradient-conflict"},
}
TEMPORARY_OBSOLETE_NOTEBOOKS = {
    "3_0_data_pipeline",
    "a_4_1_data_ml_basics", "a_4_2_linear_algebra", "a_4_3_calculus",
    "a_4_4_probability_statistics", "a_4_5_optimization", "a_4_6_information_theory",
}
LEGACY_CURRICULUM_SLUGS = {
    "a_4_1_data_ml_basics": "3_0_1_data_ml_basics",
    "a_4_2_linear_algebra": "3_0_2_linear_algebra",
    "a_4_3_calculus": "3_0_3_calculus",
    "a_4_4_probability_statistics": "3_0_4_probability_statistics",
    "a_4_5_optimization": "3_0_6_optimization",
    "a_4_6_information_theory": "3_0_5_information_theory",
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
    notebook = nbformat.read(Path("notebooks") / "3_0_math_foundations.ipynb", as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)
    for token in ["行为表", "点积", "余弦", "矩阵乘法", "Sigmoid", "LogLoss", "Softmax", "temperature", "QK^\\top", "遮罩", "梯度下降", "Recall@K", "matplotlib"]:
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
        "3_1_summary.ipynb": ["横向对比", "UserCF", "BiasMF", "FM", "GBDT+LR", "word2vec", "results/chapter_3_1", "来源论文"],
        "3_1_1_collaborative_filtering.ipynb": ["UserCF", "ItemCF", "余弦相似度", "推理", "结果讨论", "toy_R @ toy_R.T", "Sarwar", "直觉版"],
        "3_1_2_matrix_factorization.ipynb": ["BiasMF", "低秩", "训练", "Top-K", "结果讨论", "直觉版"],
        "3_1_3_factorization_machine.ipynb": ["FactorizationMachine", "二阶交互", "AUC", "LogLoss", "结果讨论", "直觉版"],
        "3_1_4_gbdt_lr.ipynb": ["XGBClassifier", "叶节点", "LogisticRegression", "推理", "结果讨论", "直觉版"],
        "3_1_5_word2vec.ipynb": ["SkipGram", "Item2Vec", "负采样", "Recall", "结果讨论", "直觉版"],
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
        "3_1_5_word2vec.ipynb",
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
        "3_2_3_sasrec.ipynb": ["SASRec", "因果", "run_sasrec", "Train & Inference", "Results Discussion"],
        "4_1_openonerec_practice.ipynb": ["OpenOneRec", "trie", "run_openonerec", "Train & Inference", "Results Discussion"],
        "4_2_dlrm_hstu_practice.ipynb": ["HSTUModel", "因果", "run_hstu", "Train & Inference", "Results Discussion"],
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
        "3_2_1_dssm.ipynb": ["[B,N]", "temperature", "ANN"],
        "3_2_2_mind.ipynb": ["c_{jk}", "squash", "[B,K,d]"],
        "3_3_1_deepfm.ipynb": ["O(nd)", "二元交叉熵", "共享 embedding"],
        "3_3_2_din.ipynb": ["e_j-e_t", "[B,L,d]", "ActivationUnit"],
        "3_3_3_dien.ipynb": ["重置门", "AUGRU", "辅助损失"],
        "3_4_1_mmoe.ipynb": ["[B,E,d]", "task gate", "\\lambda_k"],
        "3_4_2_ple.ipynb": ["CGC", "共享专家", "progressive extraction"],
        "3_2_3_sasrec.ipynb": ["QK^\\top", "因果 mask", "softplus"],
        "4_1_openonerec_practice.ipynb": ["L_{DPO}", "teacher forcing", "trie"],
        "4_2_dlrm_hstu_practice.ipynb": ["SiLU", "pointwise aggregated attention", "next-item"],
    }
    for filename, tokens in expected.items():
        notebook = nbformat.read(Path("notebooks") / filename, as_version=4)
        source = "\n".join(cell.source for cell in notebook.cells)
        assert all(token in source for token in tokens), filename
        assert "高中" not in source


def test_generative_notebooks_are_cuda_first_with_cpu_basic_fallback():
    names = [
        "4_0_generative_foundations.ipynb",
        "4_3_generative_summary.ipynb",
        "4_1_openonerec_practice.ipynb",
        "4_2_dlrm_hstu_practice.ipynb",
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
    notebook = nbformat.read(Path("notebooks") / "3_0_7_data_pipeline.ipynb", as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)
    for token in ["inspect.getsource", "leave_last_out", "逐步重写", "class TinyMF", "loss.backward()", "model.eval()", "RMSE"]:
        assert token in source
    assert "randomly_fabricated_rows" in source and "高中" not in source


def test_every_large_chapter_has_a_result_aggregation_notebook():
    expected = {
        "3_1_summary.ipynb": 6,
        "3_2_summary.ipynb": 3,
        "3_3_summary.ipynb": 3,
        "3_4_summary.ipynb": 2,
        "4_3_generative_summary.ipynb": 2,
    }
    for filename, count in expected.items():
        notebook = nbformat.read(Path("notebooks") / filename, as_version=4)
        source = "\n".join(cell.source for cell in notebook.cells)
        assert "results" in source and "不手填" in source
        assert f"len(comparison)=={count}" in source or (filename == "3_1_summary.ipynb" and "len(comparison) == 6" in source)


def test_summary_notebooks_explain_paper_comparability_and_do_not_hide_glyph_warnings():
    for filename in ["3_1_summary.ipynb", "3_2_summary.ipynb", "3_3_summary.ipynb"]:
        notebook = nbformat.read(Path("notebooks") / filename, as_version=4)
        source = "\n".join(cell.source for cell in notebook.cells)
        assert "原论文" in source and ("不可直接比较" in source or "不能相减" in source)
    for filename in ["3_2_summary.ipynb", "3_3_summary.ipynb", "3_4_summary.ipynb", "4_3_generative_summary.ipynb"]:
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
        "3_1_0_classic_foundations.ipynb": ["本章布局与选型地图", "共同数学", "余弦", "低秩", "Sigmoid"],
        "3_2_0_retrieval_foundations.ipynb": ["本章布局与选型地图", "Softmax", "Recall", "多兴趣", "temperature", "SASRec", "因果", "QK^\\top"],
        "3_3_0_ranking_foundations.ipynb": ["本章布局与选型地图", "LogLoss", "AUC", "注意力", "GRU"],
        "3_4_0_multitask_foundations.ipynb": ["本章布局与选型地图", "Softmax gate", "梯度", "负迁移", "cosine"],
        "4_0_generative_foundations.ipynb": ["本章布局与选型地图", "自回归", "因果 mask", "trie", "NDCG"],
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
        if path.name.startswith(("3_0", "3_1")):
            assert "MovieLens latest-small" in source, path.name
        elif path.name.startswith("3_2_1"):
            assert "Amazon Reviews Books 5-core" in source, path.name
        elif path.name.startswith("3_2_2"):
            assert "Amazon Books 2014" in source and "6,271,511" in source, path.name
        elif path.name.startswith("3_2_3"):
            assert "MovieLens 1M" in source, path.name
        elif path.name.startswith("3_2"):
            assert any(dataset in source for dataset in ["Amazon Reviews 2023", "Amazon Reviews Books 5-core", "Amazon Books 2014", "MovieLens 1M"]), path.name
        elif path.name.startswith(("3_3_2", "3_3_3")):
            assert "Amazon Reviews Electronics 5-core" in source, path.name
        elif path.name.startswith(("3_4_1", "3_4_2", "3_4_summary")):
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


def test_3_0_math_curriculum_has_complete_teaching_structure():
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

        if slug == "3_0_1_data_ml_basics":
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
            r"\]\(/notebooks/(?P<slug>(?:3_0_[1-6][^#/)]+|a_4_[^#/)]+))#(?P<anchor>[^)]+)\)",
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
        "3_1_4_gbdt_lr": {
            "formula": ["叶节点", "one-hot", "Sigmoid", "Normalized Entropy"],
            "paper_protocol": ["Facebook", "内部广告数据口径"],
            "tutorial_protocol": ["MovieLens latest-small", "不可平移"],
        },
        "3_2_2_mind": {
            "formula": ["c_{jk}", "squash", "[B,K,d]", "label-aware"],
            "paper_protocol": ["6,271,511", "19:1", "Amazon Books"],
            "tutorial_protocol": ["时间切分", "smoke", "不能把两套数值直接相减"],
        },
        "3_3_2_din": {
            "formula": ["e_j-e_t", r"e_j\odot e_t", "不要求权重和为 1", "Dice"],
            "paper_protocol": ["Amazon Electronics", "阿里广告系统口径"],
            "tutorial_protocol": ["KuaiRand", "不能相减", "当前样本的负标签"],
        },
        "4_1_openonerec_practice": {
            "formula": ["P(y_1", r"\prod", "L_{DPO}", "trie", "Semantic ID"],
            "paper_protocol": ["RecIF", "reward", "官方"],
            "tutorial_protocol": ["KuaiRand", "smoke", "不把本地 chosen/rejected 当成官方 reward 数据"],
        },
        "4_2_dlrm_hstu_practice": {
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
