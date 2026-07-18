import re
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_and_model_api():
    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["notebooks"] == 26
    response = client.get("/api/models", params={"stage": "召回"})
    assert response.status_code == 200
    assert {m["id"] for m in response.json()["items"]} >= {"dssm", "mind"}


def test_home_contains_required_sections():
    html = client.get("/").text
    for token in ["DSSM", "MIND", "DeepFM", "DIN", "DIEN", "MMoE", "PLE", "SASRec", "BERT4Rec", "OpenOneRec", "DLRM HSTU"]:
        assert token in html
    assert "<iframe" not in html
    assert "/notebooks/3_1_0_classic_foundations" in html
    assert "/notebooks/3_2_0_retrieval_foundations" in html
    assert "/notebooks/3_3_0_ranking_foundations" in html
    assert "/notebooks/3_4_0_multitask_foundations" in html
    assert "/notebooks/3_2_3_sasrec" in html
    assert "/notebooks/3_5_0_transformer_foundations" not in html
    assert "/notebooks/4_0_generative_foundations" in html
    assert "/chapters/" not in html
    assert "/notebooks/3_1_1_collaborative_filtering" in html
    assert "/notebooks/3_2_1_dssm" in html
    assert "/notebooks/3_3_1_deepfm" in html
    assert "/notebooks/3_4_1_mmoe" in html
    assert "先把公式翻译成可以手算的直觉" in html
    assert 'class="sidebar-subnav"' in html
    assert "/notebooks/4_2_openonerec_practice" in html
    assert "/notebooks/4_3_dlrm_hstu_practice" in html
    assert "查看 26 个实验" in html
    assert "/notebooks/3_0_data_pipeline" in html
    assert "3.0.1" not in html
    assert "Amazon Reviews 2023" in html and "KuaiRand" in html
    assert "开源数据集清单" in html
    assert 'href="/papers/deepfm"' in html
    assert "本地 PDF" in html
    assert 'id="appendix"' in html
    assert "A.1 论文清单" in html and "A.2 数据集清单" in html and "A.3 Notebook 清单" in html
    assert html.index('id="sources"') < html.index('id="datasets"') < html.index('id="labs"')
    assert 'id="frameworks"' not in html
    assert "TorchEasyRec vs Torch-RecHub" not in html


def test_legacy_chapter_routes_redirect_to_math_openings():
    expected = {
        "foundations": "3_0_math_foundations", "classic": "3_1_0_classic_foundations",
        "retrieval": "3_2_0_retrieval_foundations", "ranking": "3_3_0_ranking_foundations",
        "multitask": "3_4_0_multitask_foundations", "transformer": "3_2_0_retrieval_foundations",
        "generative": "4_0_generative_foundations",
    }
    for chapter, opening in expected.items():
        response = client.get(f"/chapters/{chapter}", follow_redirects=False)
        assert response.status_code == 307
        assert response.headers["location"] == f"/notebooks/{opening}"


def test_notebook_preview_uses_application_shell_and_raw_content_route():
    slug = "3_1_1_collaborative_filtering"
    response = client.get(f"/notebooks/{slug}")
    assert response.status_code == 200
    html = response.text
    assert "RecSys Atlas" in html
    assert "返回 3.1 导读与数学基础" not in html
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
    detail = client.get("/notebooks/3_2_1_dssm").text
    for token in ["搜索模型", "启动实验室 ↗", "召回 × 排序演进", "发展历程综述", "附录", "论文清单", "数据集清单", "Notebook 清单"]:
        assert token in home and token in detail
    assert "框架选型" not in home and "框架选型" not in detail
    assert home.count('<header class="topbar">') == detail.count('<header class="topbar">') == 1
    assert home.count('<aside class="sidebar"') == detail.count('<aside class="sidebar"') == 1
    assert 'href="/notebooks/3_2_1_dssm" class="active" aria-current="page"' in detail
    assert 'href="/notebooks/3_2_1_dssm" class="active" aria-current="page"' not in home


def test_dssm_preview_contains_model_structure_and_formula_derivation():
    shell = client.get("/notebooks/3_2_1_dssm").text
    for mode in ["paper", "preview", "execute", "source"]:
        assert f'data-mode="{mode}"' in shell
    assert "run_dssm" in shell and "_real_amazon" in shell
    content = client.get("/notebooks/3_2_1_dssm/content")
    assert content.status_code == 200
    for token in ["Model Structure &amp; Formula Walkthrough", "结构：两条独立编码路径", "从相似度到训练目标", "temperature", "[B,N]"]:
        assert token in content.text
    for token in ["论文证据导读", "双塔结构", "NDCG 实验", "/static/paper-figures/dssm.webp", "paper-guide-split", "paper-annotation"]:
        assert token in shell
    assert shell.count('class="paper-annotation') >= 8


def test_detail_pages_render_modes_and_paper_guide_by_role():
    from app.content import NOTEBOOKS, notebook_has_paper_guide

    no_guide = {n["slug"] for n in NOTEBOOKS if not notebook_has_paper_guide(n["slug"])}
    assert no_guide == {
        "3_0_math_foundations", "3_0_data_pipeline",
        "3_1_0_classic_foundations", "3_2_0_retrieval_foundations",
        "3_3_0_ranking_foundations", "3_4_0_multitask_foundations",
        "4_0_generative_foundations",
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
    html = client.get("/notebooks/4_2_openonerec_practice").text
    assert 'data-mode="execute" disabled aria-disabled="true"' in html
    assert "未检测到 CUDA" in html and "此实验需要 CUDA" in html
    assert "/lab/tree/notebooks/4_2_openonerec_practice.ipynb" not in html

    monkeypatch.setenv("RECSYS_CUDA_AVAILABLE", "1")
    html = client.get("/notebooks/4_2_openonerec_practice").text
    assert 'data-mode="execute" disabled' not in html
    assert "CUDA 已就绪" in html
    assert "/lab/tree/notebooks/4_2_openonerec_practice.ipynb" in html


def test_cuda_compose_override_exposes_gpu_to_generative_services():
    base = Path("docker-compose.yml").read_text(encoding="utf-8")
    cuda = Path("docker-compose.cuda.yml").read_text(encoding="utf-8")
    assert "RECSYS_CUDA_AVAILABLE: ${RECSYS_CUDA_AVAILABLE:-0}" in base
    assert cuda.count("gpus: all") == 3
    for service in ["web:", "jupyter:", "test:"]:
        assert service in cuda


def test_local_paper_reader_and_evidence_api():
    reader = client.get("/papers/dssm", params={"page": 3, "evidence": "dssm-architecture"})
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
        params={"page": 3, "evidence": "dssm-architecture", "embedded": 1},
    ).text
    assert "paper-embedded" in embedded and 'id="pdf-viewer"' in embedded
    for chrome in ['class="topbar"', 'class="sidebar"', 'class="paper-head"', 'class="active-citation"']:
        assert chrome not in embedded
    shell = client.get("/notebooks/3_2_1_dssm").text
    assert "evidence=dssm-architecture&embedded=1" in shell


def test_sidebar_keeps_current_notebook_in_view():
    script = Path("app/static/app.js").read_text(encoding="utf-8")
    for token in ["requestAnimationFrame", "pageLink.getBoundingClientRect", "sidebar.scrollTo", "window.addEventListener('load'", "document.fonts?.ready"]:
        assert token in script


def test_notebook_shell_unknown_slug_returns_404():
    assert client.get("/notebooks/not-a-notebook").status_code == 404
    assert client.get("/notebooks/not-a-notebook/content").status_code == 404


def test_source_is_a_detail_tab_instead_of_a_separate_page():
    response = client.get("/source/3_2_1_dssm", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/notebooks/3_2_1_dssm#source"
    html = client.get("/notebooks/3_2_1_dssm").text
    for token in ["Notebook 公用代码", "本章节独立目录", "Torch-RecHub 框架源码", "model.py", "train.py", "inference.py", "test_model.py", "dssm.py", "match_trainer.py", "run_dssm", "在 IDE 中打开章节目录"]:
        assert token in html
    assert 'class="source-code"' in html and 'class="kn"' in html
    assert 'class="linenos"' in html and 'class="hll"' in html
    assert client.get("/source/not-a-notebook").status_code == 404


def test_ide_route_opens_the_chapter_directory():
    response = client.get("/ide/3_2_1_dssm", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "http://localhost:8090/?folder=/home/coder/project/chapter_code/3_2_1_dssm&goto=/home/coder/project/chapter_code/3_2_1_dssm/train.py:1:1"
    assert client.get("/ide/not-a-notebook").status_code == 404


def test_compose_exposes_local_passwordless_browser_ide():
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    for token in ["ide:", "Dockerfile.ide", "recsys-code-server", "--auth none", "127.0.0.1:8090:8080", "IDE_PUBLIC_URL", "generate_chapter_code.py", "frame-ancestors", "http://localhost:8010"]:
        assert token in compose
    assert "PASSWORD:" not in compose
    assert "--auth password" not in compose
    assert "IDE_OPENER_INTERNAL_URL" not in compose


def test_every_notebook_has_an_independent_python_directory():
    from app.content import MODELS, NOTEBOOKS
    for notebook in NOTEBOOKS:
        directory = Path("chapter_code") / notebook["slug"]
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
        "3_2_2_mind": "def _mind_rows",
        "3_2_3_sasrec": "def _sequence_windows_from_sequences",
        "3_3_1_deepfm": "def _ranking_fields",
        "3_3_2_din": "def _run_sequence_ranker",
        "3_4_1_mmoe": "def _run_multitask",
        "4_2_openonerec_practice": "def _semantic_catalog",
        "4_3_dlrm_hstu_practice": "def _sequence_windows_from_sequences",
    }
    for slug, token in local_expectations.items():
        source = (Path("chapter_code") / slug / "train.py").read_text(encoding="utf-8")
        assert token in source
        assert "industrial_experiments as base" not in source
    compatibility = Path("recsys_lab/industrial_experiments.py").read_text(encoding="utf-8")
    assert "load_runner" in compatibility
    assert "class TinyListGenerator" not in compatibility
    assert "def _mind_rows" not in compatibility
