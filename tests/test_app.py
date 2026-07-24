import re
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_and_model_api():
    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["notebooks"] == 33
    response = client.get("/api/models", params={"stage": "召回"})
    assert response.status_code == 200
    assert {m["id"] for m in response.json()["items"]} >= {"dssm", "mind"}


def test_notebook_kind_schema_is_explicit_and_visible_in_catalogue():
    from app.content import CHAPTER_CODE_NOTEBOOK_KINDS, NOTEBOOK_KIND_LABELS, NOTEBOOKS

    assert len(NOTEBOOKS) == 33
    assert {notebook["kind"] for notebook in NOTEBOOKS} == {"foundation", "algorithm", "summary", "curriculum"}
    assert set(CHAPTER_CODE_NOTEBOOK_KINDS) == {"foundation", "algorithm", "summary"}
    assert set(NOTEBOOK_KIND_LABELS) == {"foundation", "algorithm", "summary", "curriculum"}
    assert [notebook["slug"] for notebook in NOTEBOOKS[:8]] == [
        "3_1_math_foundations",
        "3_2_data_ml_basics",
        "3_3_linear_algebra",
        "3_4_calculus",
        "3_5_probability_statistics",
        "3_6_information_theory",
        "3_7_optimization",
        "3_8_data_pipeline",
    ]
    assert [notebook["kind"] for notebook in NOTEBOOKS[:8]] == [
        "foundation", "curriculum", "curriculum", "curriculum",
        "curriculum", "curriculum", "curriculum", "foundation",
    ]
    html = client.get("/").text
    for kind in {notebook["kind"] for notebook in NOTEBOOKS}:
        assert f'data-notebook-kind="{kind}"' in html
        assert NOTEBOOK_KIND_LABELS[kind] in html


def test_curriculum_page_uses_course_source_instead_of_chapter_code():
    from app.content import NOTEBOOKS

    curriculum = next(notebook for notebook in NOTEBOOKS if notebook["kind"] == "curriculum")
    html = client.get(f"/notebooks/{curriculum['slug']}").text
    assert "课程阅读与练习" in html
    assert html.count('role="tab"') == 3
    for mode in ("preview", "execute", "source"):
        assert f'data-mode="{mode}"' in html
    assert "论文导读" not in html and "代码实现" in html
    assert "本课程内容生成源" in html and "generate_notebooks.py" in html
    assert f"chapter_code/{curriculum['slug']}" not in html
    ide = client.get(f'/ide/{curriculum["slug"]}', follow_redirects=False)
    assert ide.status_code == 303
    assert "folder=/home/coder/project" in ide.headers["location"]
    assert "goto=/home/coder/project/scripts/tutorial_math_specs.py:1:1" in ide.headers["location"]
    source = client.get(f'/source/{curriculum["slug"]}', follow_redirects=False)
    assert source.status_code == 307
    assert source.headers["location"] == f'/notebooks/{curriculum["slug"]}#source'

    catalogue = client.get("/").text
    assert 'data-notebook-kind="curriculum"' in catalogue
    assert "数学课程 · pandas / NumPy" in catalogue and ">练习 ↗</a>" in catalogue

    from app.source_browser import CURRICULUM_SOURCE_PATHS

    assert CURRICULUM_SOURCE_PATHS[0][0].name == "tutorial_math_specs.py"


def test_unknown_notebook_hash_checks_the_iframe_before_switching_tabs():
    html = client.get("/notebooks/5_2_dssm").text
    function = html.split("function focusPreviewAnchor(anchor)", 1)[1].split(
        "function applyLocationHash()", 1
    )[0]
    assert function.index("getElementById(anchor)") < function.index("previewButton.click()")
    apply_hash = html.split("function applyLocationHash()", 1)[1].split(
        "previewFrame?.addEventListener", 1
    )[0]
    assert "pendingPreviewAnchor = target" in apply_hash
    assert "previewButton.click()" not in apply_hash


def test_curriculum_kind_never_gets_a_paper_guide():
    from app.content import NOTEBOOKS, notebook_has_paper_guide

    assert all(
        not notebook_has_paper_guide(notebook["slug"])
        for notebook in NOTEBOOKS
        if notebook["kind"] == "curriculum"
    )


def test_home_contains_required_sections():
    html = client.get("/").text
    for token in ["DSSM", "MIND", "DeepFM", "DIN", "DIEN", "MMoE", "PLE", "SASRec", "BERT4Rec", "OpenOneRec", "DLRM HSTU"]:
        assert token in html
    assert "<iframe" not in html
    assert "/notebooks/4_1_classic_foundations" in html
    assert "/notebooks/5_1_retrieval_foundations" in html
    assert "/notebooks/6_1_ranking_foundations" in html
    assert "/notebooks/7_1_multitask_foundations" in html
    assert "/notebooks/5_4_sasrec" in html
    assert "/notebooks/8_1_generative_foundations" in html
    assert "/chapters/" not in html
    assert "/notebooks/4_2_collaborative_filtering" in html
    assert "/notebooks/5_2_dssm" in html
    assert "/notebooks/6_2_deepfm" in html
    assert "/notebooks/7_2_mmoe" in html
    assert "先把公式翻译成可以手算的直觉" in html
    assert 'class="sidebar-subnav"' in html
    assert "/notebooks/8_2_openonerec_practice" in html
    assert "/notebooks/8_3_dlrm_hstu_practice" in html
    assert "查看 33 个 Notebook" in html
    assert "/notebooks/3_8_data_pipeline" in html
    assert "/notebooks/3_2_data_ml_basics" in html
    assert "Amazon Reviews 2023" in html and "KuaiRand" in html
    assert "开源数据集清单" in html
    assert 'href="/papers/deepfm"' in html
    assert "本地 PDF" in html
    assert 'id="appendix"' in html
    assert "A.1 论文清单" in html and "A.2 数据集清单" in html and "A.3 Notebook 清单" in html
    assert html.index('id="sources"') < html.index('id="datasets"') < html.index('id="labs"')
    assert 'id="frameworks"' not in html
    assert "TorchEasyRec vs Torch-RecHub" not in html


def test_full_and_smoke_dataset_protocols_are_explicit_in_catalogue_and_headers():
    from app import content

    affected = {
        "5_2_dssm": ("full：Amazon Books 5-core 迁移", "smoke：Amazon 真实切片"),
        "5_3_mind": ("full：Amazon Books 2014 10-core", "smoke：Amazon 真实切片"),
        "5_4_sasrec": ("full：MovieLens-1M 论文协议", "smoke：Amazon 真实切片"),
        "6_2_deepfm": ("full：Criteo", "smoke：KuaiRand-Pure is_click"),
        "6_3_din": ("full：Amazon Electronics", "smoke：KuaiRand feed 序列"),
        "6_4_dien": ("full：Amazon Electronics", "smoke：KuaiRand feed 序列"),
        "7_2_mmoe": ("full：Census-Income", "smoke：KuaiRand 双目标"),
        "7_3_ple": ("full：Census-Income", "smoke：KuaiRand 双目标"),
        "8_3_dlrm_hstu_practice": ("full：Meta MovieLens-20M + Amazon Books", "smoke：KuaiRand feed 序列"),
    }
    notebooks = {notebook["slug"]: notebook for notebook in content.NOTEBOOKS}
    for slug, phrases in affected.items():
        assert all(phrase in notebooks[slug]["dataset"] for phrase in phrases)
        detail = client.get(f"/notebooks/{slug}").text
        assert all(phrase in detail for phrase in phrases)

    home = client.get("/").text
    assert "数据口径按“算法 × 协议”记录，不把一个数据集套给整章" in home
    assert "两档结果不可直接相减" in home
    assert "深度召回与 SASRec 使用 Amazon Reviews 2023" not in home
    assert "CTR、行为序列、多目标和 HSTU 使用 KuaiRand" not in home
    assert not hasattr(content, "_LEGACY_MATH_PREREQUISITES")


def test_math_map_covers_all_models_with_valid_acyclic_curriculum_links():
    from app.content import MATH_PREREQUISITES, MODELS, NOTEBOOKS

    curriculum = {n["slug"] for n in NOTEBOOKS if n["kind"] == "curriculum"}
    algorithms = {n["slug"] for n in NOTEBOOKS if n["kind"] == "algorithm"}
    assert curriculum == {
        "3_2_data_ml_basics", "3_3_linear_algebra", "3_4_calculus",
        "3_5_probability_statistics", "3_6_information_theory", "3_7_optimization",
    }
    expected_anchors = {
        "3_2_data_ml_basics": {"observation-label", "implicit-feedback", "split-leakage"},
        "3_3_linear_algebra": {"tensors-shapes", "elementwise-dot", "matmul-embedding", "low-rank-attention"},
        "3_4_calculus": {"functions", "derivative-gradient", "chain-rule"},
        "3_5_probability_statistics": {"random-variable", "conditional-chain", "expectation-variance", "likelihood-calibration"},
        "3_6_information_theory": {"information-entropy", "cross-entropy-kl", "softmax-temperature", "sequence-nll-dpo"},
        "3_7_optimization": {"sgd", "learning-rate", "regularization", "gradient-conflict"},
    }
    assert {
        slug: {item["anchor"] for item in MATH_PREREQUISITES if item["notebook"] == slug}
        for slug in curriculum
    } == expected_anchors

    required_fields = {
        "id", "area", "topic", "intuition", "used_by", "notebook", "anchor",
        "prerequisites", "model_ids", "notebook_slugs",
    }
    topic_ids = {item["id"] for item in MATH_PREREQUISITES}
    assert len(topic_ids) == len(MATH_PREREQUISITES)
    model_ids = {model["id"] for model in MODELS}
    model_notebooks = {model["notebook"] for model in MODELS}
    covered_models: set[str] = set()
    covered_notebooks: set[str] = set()
    graph = {}
    for item in MATH_PREREQUISITES:
        assert required_fields <= item.keys()
        assert item["notebook"] in curriculum
        assert item["area"] and item["topic"] and item["intuition"] and item["used_by"]
        assert set(item["prerequisites"]) <= topic_ids
        assert set(item["model_ids"]) <= model_ids
        assert set(item["notebook_slugs"]) <= algorithms
        assert set(item["notebook_slugs"]) == {
            model["notebook"] for model in MODELS if model["id"] in item["model_ids"]
        }
        covered_models.update(item["model_ids"])
        covered_notebooks.update(item["notebook_slugs"])
        graph[item["id"]] = item["prerequisites"]
    assert covered_models == model_ids
    assert covered_notebooks == model_notebooks

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(topic_id: str) -> None:
        assert topic_id not in visiting, f"prerequisite cycle at {topic_id}"
        if topic_id in visited:
            return
        visiting.add(topic_id)
        for prerequisite in graph[topic_id]:
            visit(prerequisite)
        visiting.remove(topic_id)
        visited.add(topic_id)

    for topic_id in graph:
        visit(topic_id)

    html = client.get("/").text
    assert 'id="math-map"' in html
    assert 'id="knowledge-graph-data"' in html
    assert "A.4 · KNOWLEDGE GRAPH" in html
    assert "两跳和有限节点" in html


def test_page_context_exposes_default_and_six_focused_knowledge_graph_views():
    from starlette.requests import Request

    from app.content import CHAPTERS
    from app.main import page_context

    request = Request({
        "type": "http", "method": "GET", "path": "/", "root_path": "",
        "scheme": "http", "query_string": b"", "headers": [],
        "client": ("test", 50000), "server": ("testserver", 80), "app": app,
    })
    graph_payload = page_context(request)["knowledge_graph"]
    assert set(graph_payload) == {"default", "views"}
    assert set(graph_payload["views"]) == {f"chapter:{key}" for key in CHAPTERS}
    assert [node["id"] for node in graph_payload["default"]["nodes"]] == [
        f"chapter:{key}" for key in CHAPTERS
    ]
    for key in CHAPTERS:
        view = graph_payload["views"][f"chapter:{key}"]
        assert view["state"]["focus"] == f"chapter:{key}"
        assert len(view["nodes"]) <= view["limits"]["focus"]

    classic = graph_payload["views"]["chapter:classic"]
    assert len(classic["nodes"]) > len(graph_payload["default"]["nodes"])
    assert {node["type"] for node in classic["nodes"]} == {"chapter", "model", "math"}


def test_legacy_chapter_routes_redirect_to_math_openings():
    expected = {
        "foundations": "3_1_math_foundations", "classic": "4_1_classic_foundations",
        "retrieval": "5_1_retrieval_foundations", "ranking": "6_1_ranking_foundations",
        "multitask": "7_1_multitask_foundations", "transformer": "5_1_retrieval_foundations",
        "generative": "8_1_generative_foundations",
    }
    for chapter, opening in expected.items():
        response = client.get(f"/chapters/{chapter}", follow_redirects=False)
        assert response.status_code == 307
        assert response.headers["location"] == f"/notebooks/{opening}"


def test_legacy_math_routes_redirect_to_published_3_0_curriculum():
    expected = {
        "3_0_data_pipeline": "3_8_data_pipeline",
        "a_4_1_data_ml_basics": "3_2_data_ml_basics",
        "a_4_2_linear_algebra": "3_3_linear_algebra",
        "a_4_3_calculus": "3_4_calculus",
        "a_4_4_probability_statistics": "3_5_probability_statistics",
        "a_4_5_optimization": "3_7_optimization",
        "a_4_6_information_theory": "3_6_information_theory",
    }
    for old, new in expected.items():
        detail = client.get(f"/notebooks/{old}", follow_redirects=False)
        assert detail.status_code == 307
        assert detail.headers["location"] == f"/notebooks/{new}"
        content = client.get(f"/notebooks/{old}/content", follow_redirects=False)
        assert content.status_code == 307
        assert content.headers["location"] == f"/notebooks/{new}/content"
        source = client.get(f"/source/{old}", follow_redirects=False)
        assert source.status_code == 307
        assert source.headers["location"] == f"/notebooks/{new}#source"


def test_notebook_preview_uses_application_shell_and_raw_content_route():
    slug = "4_2_collaborative_filtering"
    response = client.get(f"/notebooks/{slug}")
    assert response.status_code == 200
    html = response.text
    assert "RecSys Atlas" in html
    assert "返回 4.1 导读与数学基础" not in html
    assert "notebook-breadcrumb" not in html
    for label in ["论文导读", "实验预览", "可交互执行", "代码实现", "独立 IDE"]:
        assert label in html
    assert "http://localhost:8889" in html
    assert "http://localhost:8888" not in html
    assert f"/lab/tree/notebooks/{slug}.ipynb?token=recsys" in html
    assert f'src="/notebooks/{slug}/content"' in html
    assert "下一篇" in html

    content = client.get(f"/notebooks/{slug}/content")
    assert content.status_code == 200
    assert "UserCF" in content.text and "ItemCF" in content.text
    assert 'class="jp-Notebook"' in content.text


def test_home_and_detail_share_the_same_global_navigation():
    home = client.get("/").text
    detail = client.get("/notebooks/5_2_dssm").text
    for token in ["搜索模型", "启动实验室 ↗", "导读", "附录", "论文清单", "数据集清单", "Notebook 清单"]:
        assert token in home and token in detail
    assert "框架选型" not in home and "框架选型" not in detail
    assert home.count('<header class="topbar">') == detail.count('<header class="topbar">') == 1
    assert home.count('<aside class="sidebar"') == detail.count('<aside class="sidebar"') == 1
    assert 'href="/notebooks/5_2_dssm" class="active" aria-current="page"' in detail
    assert 'href="/notebooks/5_2_dssm" class="active" aria-current="page"' not in home


def test_dssm_preview_contains_model_structure_and_formula_derivation():
    shell = client.get("/notebooks/5_2_dssm").text
    for mode in ["paper", "preview", "execute", "source"]:
        assert f'data-mode="{mode}"' in shell
    assert "run_dssm" in shell and "_real_amazon" in shell
    content = client.get("/notebooks/5_2_dssm/content")
    assert content.status_code == 200
    for token in ["Model Structure &amp; Formula Walkthrough", "结构：两条独立编码路径", "从相似度到训练目标", "temperature", "[B,N]"]:
        assert token in content.text
    for token in ["论文证据导读", "双塔结构", "排序质量", "/static/paper-figures/dssm.webp", "paper-guide-split", "paper-annotation"]:
        assert token in shell
    assert shell.count('class="paper-annotation') >= 8


def test_detail_pages_render_modes_and_paper_guide_by_role():
    from app.content import NOTEBOOKS, notebook_has_paper_guide

    no_guide = {n["slug"] for n in NOTEBOOKS if not notebook_has_paper_guide(n["slug"])}
    assert no_guide == {
        "3_1_math_foundations", "3_2_data_ml_basics", "3_3_linear_algebra",
        "3_4_calculus", "3_5_probability_statistics", "3_6_information_theory",
        "3_7_optimization", "3_8_data_pipeline",
        "4_1_classic_foundations", "5_1_retrieval_foundations",
        "6_1_ranking_foundations", "7_1_multitask_foundations",
        "8_1_generative_foundations",
        # 总结章节在 notebook 内做跨论文比较，不再显示论文导读 tab
        "4_7_classic_summary", "5_5_retrieval_summary", "6_5_ranking_summary", "7_4_multitask_summary",
        "8_4_generative_summary",
    }
    for notebook in NOTEBOOKS:
        slug = notebook["slug"]
        html = client.get(f"/notebooks/{slug}").text
        has_guide = notebook_has_paper_guide(slug)
        if has_guide:
            assert html.count('role="tab"') == 4
            assert 'data-panel="paper"' in html
            assert 'id="paper-guide-frame"' in html
            assert 'class="paper-guide-split"' in html
            assert html.count('class="paper-annotation') >= 8
            # only the 论文导读 iframe carries data-paper-frame now (实验预览 has none)
            assert len(re.findall(r'<iframe[^>]*data-paper-frame', html)) == 1
        else:
            assert html.count('role="tab"') == 3
            assert 'data-panel="paper"' not in html
            assert 'id="paper-guide-frame"' not in html
            assert 'paper-guide-split' not in html
            assert 'paper-annotation' not in html
            assert len(re.findall(r'<iframe[^>]*data-paper-frame', html)) == 0
        assert 'data-panel="preview"' in html
        assert 'class="experiment-preview-pane"' in html
        assert f'src="/notebooks/{slug}/content"' in html
    css = Path("app/static/notebook.css").read_text(encoding="utf-8")
    assert "grid-template-columns:minmax(0,1fr) minmax(0,1fr)" in css
    assert "grid-template-columns:minmax(360px,1fr) minmax(0,2fr)" in css
    assert ".notebook-reader-head,.notebook-frame-wrap" in css and "width:100%;max-width:none" in css


def test_generative_interactive_mode_follows_cuda_capability(monkeypatch):
    monkeypatch.setenv("RECSYS_CUDA_AVAILABLE", "0")
    html = client.get("/notebooks/8_2_openonerec_practice").text
    assert 'data-mode="execute" disabled aria-disabled="true"' in html
    assert "未检测到 CUDA" in html and "此实验需要 CUDA" in html
    assert "/lab/tree/notebooks/8_2_openonerec_practice.ipynb" not in html

    monkeypatch.setenv("RECSYS_CUDA_AVAILABLE", "1")
    html = client.get("/notebooks/8_2_openonerec_practice").text
    assert 'data-mode="execute" disabled' not in html
    assert "CUDA 已就绪" in html
    assert "/lab/tree/notebooks/8_2_openonerec_practice.ipynb" in html


def test_cuda_compose_override_builds_cuda_image_and_exposes_gpu():
    base = Path("docker-compose.yml").read_text(encoding="utf-8")
    cuda = Path("docker-compose.cuda.yml").read_text(encoding="utf-8")
    assert "RECSYS_CUDA_AVAILABLE: ${RECSYS_CUDA_AVAILABLE:-0}" in base
    assert cuda.count("gpus: all") == 6
    assert cuda.count("dockerfile: Dockerfile.cuda") == 6
    assert cuda.count("image: recsys-atlas-cuda:latest") == 6
    assert 'RECSYS_CUDA_AVAILABLE: "1"' in cuda
    for service in ["web:", "jupyter:", "test:", "test-full:", "test-accuracy:", "test-cuda-accuracy:"]:
        assert service in cuda
    cuda_dockerfile = Path("Dockerfile.cuda").read_text(encoding="utf-8")
    assert "--no-install-package torch" in cuda_dockerfile
    assert "torch.version.cuda" in cuda_dockerfile
    workflow = Path(".github/workflows/publish-images.yml").read_text(encoding="utf-8")
    assert "recsys-atlas-cuda" in workflow and "Dockerfile.cuda" in workflow


def test_local_paper_reader_and_evidence_api():
    reader = client.get("/papers/dssm", params={"page": 3, "evidence": "fig1-dssm"})
    assert reader.status_code == 200
    for token in ["LOCAL PAPER", "Figure 1", "/resources/papers/dssm.pdf", "EmbedPDF", "scrollToPage", "ZoomMode.FitWidth"]:
        assert token in reader.text
    payload = client.get("/api/papers/dssm")
    assert payload.status_code == 200
    assert payload.json()["id"] == "dssm"
    assert payload.json()["anchors"][0]["page"] == 3
    assert client.get("/papers/not-a-paper").status_code == 404


def test_embedded_paper_reader_is_viewer_only():
    embedded = client.get(
        "/papers/dssm",
        params={"page": 3, "evidence": "fig1-dssm", "embedded": 1},
    ).text
    assert "paper-embedded" in embedded and 'id="pdf-viewer"' in embedded
    for chrome in ['class="topbar"', 'class="sidebar"', 'class="paper-head"', 'class="active-citation"']:
        assert chrome not in embedded
    shell = client.get("/notebooks/5_2_dssm").text
    assert "evidence=fig1-dssm&embedded=1" in shell


def test_sidebar_keeps_current_notebook_in_view():
    script = Path("app/static/app.js").read_text(encoding="utf-8")
    for token in ["requestAnimationFrame", "pageLink.getBoundingClientRect", "sidebar.scrollTo", "window.addEventListener('load'", "document.fonts?.ready"]:
        assert token in script


def test_notebook_shell_unknown_slug_returns_404():
    assert client.get("/notebooks/not-a-notebook").status_code == 404
    assert client.get("/notebooks/not-a-notebook/content").status_code == 404


def test_preview_iframe_retargets_internal_links_to_the_top_page():
    """Site-internal links inside the preview must not nest the app shell."""
    shell = client.get("/notebooks/3_1_math_foundations").text
    iframe = re.search(r'<iframe id="notebook-preview-frame"[^>]*>', shell).group(0)
    assert "allow-top-navigation-by-user-activation" in iframe
    assert "allow-popups" in iframe

    content = client.get("/notebooks/3_1_math_foundations/content").text
    assert 'id="recsys-preview-link-targets"' in content
    assert "anchor.target = '_top'" in content
    assert "anchor.target = '_blank'" in content
    # 3.0 overview really does link to the 3.2 curriculum page
    assert '/notebooks/3_2_data_ml_basics#observation-label' in content


def test_embedded_notebook_page_hides_application_chrome():
    embedded = client.get("/notebooks/3_2_data_ml_basics", params={"embedded": 1}).text
    assert 'class="notebook-shell-page embedded"' in embedded
    # default tab degrades to the preview panel
    assert 'data-mode="preview"' in embedded
    assert 'src="/notebooks/3_2_data_ml_basics/content"' in embedded

    full = client.get("/notebooks/3_2_data_ml_basics").text
    assert 'class="notebook-shell-page"' in full

    # iframe self-detection safety net + embedded CSS rules
    assert "window.self !== window.top" in full
    css = Path("app/static/notebook.css").read_text(encoding="utf-8")
    assert ".notebook-shell-page.embedded .topbar" in css
    assert '.notebook-shell-page.embedded .mode-panel[data-panel="preview"]' in css


def test_source_is_a_detail_tab_instead_of_a_separate_page():
    response = client.get("/source/5_2_dssm", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/notebooks/5_2_dssm#source"
    html = client.get("/notebooks/5_2_dssm").text
    for token in ["Notebook 公用代码", "本章节独立目录", "Torch-RecHub 框架源码", "model.py", "train.py", "inference.py", "test_model.py", "dssm.py", "match_trainer.py", "run_dssm", "在 IDE 中打开章节目录"]:
        assert token in html
    assert 'class="source-code"' in html and 'class="kn"' in html
    assert 'class="linenos"' in html and 'class="hll"' in html
    assert client.get("/source/not-a-notebook").status_code == 404


def test_ide_route_opens_the_chapter_directory():
    response = client.get("/ide/5_2_dssm", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "http://localhost:8090/?folder=/home/coder/project/chapter_code/5_2_dssm&goto=/home/coder/project/chapter_code/5_2_dssm/train.py:1:1"
    pipeline = client.get("/ide/3_8_data_pipeline", follow_redirects=False)
    assert pipeline.status_code == 303
    assert "chapter_code/3_8_data_pipeline" in pipeline.headers["location"]
    legacy_pipeline = client.get("/ide/3_0_data_pipeline", follow_redirects=False)
    assert legacy_pipeline.headers["location"] == pipeline.headers["location"]
    assert client.get("/ide/not-a-notebook").status_code == 404


def test_compose_exposes_local_passwordless_browser_ide():
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    for token in ["ide:", "Dockerfile.ide", "recsys-code-server", "--auth none", "127.0.0.1:8090:8080", "IDE_PUBLIC_URL", "generate_chapter_code.py", "frame-ancestors", "http://localhost:8010"]:
        assert token in compose
    assert "PASSWORD:" not in compose
    assert "--auth password" not in compose
    assert "IDE_OPENER_INTERNAL_URL" not in compose


def test_implementation_notebooks_have_independent_python_directories():
    from app.content import CHAPTER_CODE_NOTEBOOK_KINDS, MODELS, NOTEBOOKS
    from app.source_browser import chapter_source_slug

    implementation_notebooks = [
        notebook for notebook in NOTEBOOKS if notebook["kind"] in CHAPTER_CODE_NOTEBOOK_KINDS
    ]
    for notebook in implementation_notebooks:
        directory = Path("chapter_code") / chapter_source_slug(notebook["slug"])
        assert directory.is_dir()
        assert {"model.py", "train.py", "inference.py", "test_model.py"} <= {path.name for path in directory.iterdir()}
        settings = (directory / ".vscode" / "settings.json").read_text(encoding="utf-8")
        assert "basedpyright.analysis.extraPaths" in settings
        assert '"basedpyright.analysis.typeCheckingMode": "off"' in settings
        assert "/opt/venv/lib/python3.11/site-packages" in settings
        assert '"python.defaultInterpreterPath": "/opt/venv/bin/python"' in settings
        assert '"ruff.configuration": "/home/coder/project/pyproject.toml"' in settings
        assert '"ruff.lint.enable": true' in settings
        assert '"workbench.secondarySideBar.defaultVisibility": "hidden"' in settings
        assert '"window.autoDetectColorScheme": true' in settings
        assert (directory / "pyrightconfig.json").is_file()
        assert '"typeCheckingMode": "off"' in (directory / "pyrightconfig.json").read_text(encoding="utf-8")
    for notebook in NOTEBOOKS:
        if notebook["kind"] == "curriculum":
            assert not (Path("chapter_code") / notebook["slug"]).exists()
    # 算法章节的结构与训练文件必须包含面向读者的行内注释，而不只依赖页面叙述。
    for slug in {model["notebook"] for model in MODELS}:
        directory = Path("chapter_code") / slug
        assert "#" in (directory / "model.py").read_text(encoding="utf-8")
        assert "#" in (directory / "train.py").read_text(encoding="utf-8")


def test_widgets_code_wrapping_and_theme_support_are_packaged():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    notebook_css = Path("app/static/notebook.css").read_text(encoding="utf-8")
    app_js = Path("app/static/app.js").read_text(encoding="utf-8")
    ide_image = Path("Dockerfile.ide").read_text(encoding="utf-8")
    app_image = Path("Dockerfile").read_text(encoding="utf-8")
    ide_settings = Path("config/code-server-settings.json").read_text(encoding="utf-8")
    startup_extension = Path("config/recsys-startup-extension/extension.js").read_text(encoding="utf-8")
    assert "ipywidgets==8.1.7" in pyproject
    assert "torch==2.6.0" in pyproject and "ruff==0.11.13" in pyproject
    assert "white-space:pre-wrap" in notebook_css
    assert ".source-code .linenos" in notebook_css
    assert ".source-code .hll" in notebook_css
    assert "prefers-color-scheme: light" in app_js
    assert "localStorage.getItem('theme')" in app_js
    for token in ["uv.lock", "uv sync --locked", "ms-python.python", "detachhead.basedpyright@1.39.9", "charliermarsh.ruff", "COPY --from=python-runtime /usr/local /usr/local", "COPY --from=python-runtime /opt/venv /opt/venv", "disabled-vscode-extensions"]:
        assert token in ide_image
    for token in ["ghcr.io/astral-sh/uv:python3.11", "uv.lock", "uv sync --locked", "UV_PROJECT_ENVIRONMENT=/opt/venv"]:
        assert token in app_image and token in ide_image
    assert '"ruff.configuration": "/home/coder/project/pyproject.toml"' in ide_settings
    assert '"python.defaultInterpreterPath": "/opt/venv/bin/python"' in ide_settings
    assert '"basedpyright.analysis.typeCheckingMode": "off"' in ide_settings
    assert '"workbench.startupEditor": "none"' in ide_settings
    assert '"workbench.secondarySideBar.defaultVisibility": "hidden"' in ide_settings
    assert '"window.autoDetectColorScheme": true' in ide_settings
    assert "openTextDocument(train)" in startup_extension


def test_algorithm_implementations_are_chapter_local():
    local_expectations = {
        "5_3_mind": "def _mind_rows",
        "5_4_sasrec": "def _sequence_windows_from_sequences",
        "6_2_deepfm": "def _ranking_fields",
        "6_3_din": "def _run_sequence_ranker",
        "7_2_mmoe": "def _run_multitask",
        "8_2_openonerec_practice": "def _semantic_catalog",
        "8_3_dlrm_hstu_practice": "def _sequence_windows_from_sequences",
    }
    for slug, token in local_expectations.items():
        source = (Path("chapter_code") / slug / "train.py").read_text(encoding="utf-8")
        assert token in source
        assert "industrial_experiments as base" not in source
    compatibility = Path("recsys_lab/industrial_experiments.py").read_text(encoding="utf-8")
    assert "load_runner" in compatibility
    assert "class TinyListGenerator" not in compatibility
    assert "def _mind_rows" not in compatibility
