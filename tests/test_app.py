from pathlib import Path

from fastapi.testclient import TestClient

import app.main as main_module
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
    for label in ["阅读预览", "可交互执行", "实现源码", "独立 IDE"]:
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
    for token in ["_load_amazon_cached", "load_amazon_2023", "leave_last_out", "_train_binary", "_real_amazon", "run_dssm", "在独立 IDE 中编辑"]:
        assert token in html
    assert client.get("/source/not-a-notebook").status_code == 404


def test_ide_route_opens_the_corresponding_whitelisted_source(monkeypatch):
    opened = []
    monkeypatch.setattr(main_module, "request_ide_open", opened.append)
    response = client.get("/ide/3_2_1_dssm")
    assert response.status_code == 200
    assert 'src="http://localhost:8090/?folder=/home/coder/project"' in response.text
    assert "/api/ide/3_2_1_dssm/open" in response.text
    assert opened == []

    response = client.post("/api/ide/3_2_1_dssm/open")
    assert response.status_code == 200
    assert opened == ["industrial"]

    response = client.post("/api/ide/3_1_1_collaborative_filtering/open")
    assert response.status_code == 200
    assert opened[-1] == "classic"
    assert client.get("/ide/not-a-notebook").status_code == 404


def test_compose_exposes_local_passwordless_browser_ide():
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    for token in ["ide:", "codercom/code-server", 'entrypoint: ["/bin/sh", "-c"]', "--auth none", "127.0.0.1:8090:8080", "IDE_PUBLIC_URL", "IDE_OPENER_INTERNAL_URL", "code_server_opener.mjs", "frame-ancestors", "http://localhost:8010"]:
        assert token in compose
    assert "PASSWORD:" not in compose
    assert "--auth password" not in compose
