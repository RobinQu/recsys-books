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
    assert "按任务选择真实数据" in html


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
    for label in ["阅读预览", "可交互执行", "代码实现", "独立 IDE"]:
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
    for token in ["搜索模型", "启动实验室 ↗", "召回 × 排序演进", "发展历程综述", "框架选型", "数据集", "证据清单"]:
        assert token in home and token in detail
    assert home.count('<header class="topbar">') == detail.count('<header class="topbar">') == 1
    assert home.count('<aside class="sidebar"') == detail.count('<aside class="sidebar"') == 1
    assert 'href="/notebooks/3_2_1_dssm" class="active" aria-current="page"' in detail
    assert 'href="/notebooks/3_2_1_dssm" class="active" aria-current="page"' not in home


def test_dssm_preview_contains_model_structure_and_formula_derivation():
    shell = client.get("/notebooks/3_2_1_dssm").text
    assert 'data-mode="preview"' in shell and 'data-mode="execute"' in shell and 'data-mode="source"' in shell
    assert "run_dssm" in shell and "_real_amazon" in shell
    content = client.get("/notebooks/3_2_1_dssm/content")
    assert content.status_code == 200
    for token in ["Model Structure &amp; Formula Walkthrough", "结构：两条独立编码路径", "从相似度到训练目标", "temperature", "[B,N]"]:
        assert token in content.text


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
        assert "/usr/local/lib/python3.11/site-packages" in settings
        assert '"python.defaultInterpreterPath": "/usr/local/bin/python"' in settings
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
    requirements = Path("requirements.txt").read_text(encoding="utf-8")
    notebook_css = Path("app/static/notebook.css").read_text(encoding="utf-8")
    app_js = Path("app/static/app.js").read_text(encoding="utf-8")
    ide_image = Path("Dockerfile.ide").read_text(encoding="utf-8")
    app_image = Path("Dockerfile").read_text(encoding="utf-8")
    ide_settings = Path("config/code-server-settings.json").read_text(encoding="utf-8")
    startup_extension = Path("config/recsys-startup-extension/extension.js").read_text(encoding="utf-8")
    assert "ipywidgets==8.1.7" in requirements
    assert "white-space:pre-wrap" in notebook_css
    assert ".source-code .linenos" in notebook_css
    assert ".source-code .hll" in notebook_css
    assert "prefers-color-scheme: light" in app_js
    assert "localStorage.getItem('theme')" in app_js
    for token in ["requirements.txt", "torch==2.6.0", "ms-python.python", "detachhead.basedpyright@1.39.9", "charliermarsh.ruff", "COPY --from=python-runtime /usr/local /usr/local", "disabled-vscode-extensions"]:
        assert token in ide_image
    assert "ruff==0.11.13" in app_image and "ruff==0.11.13" in ide_image
    assert '"ruff.configuration": "/home/coder/project/pyproject.toml"' in ide_settings
    assert '"python.defaultInterpreterPath": "/usr/local/bin/python"' in ide_settings
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
