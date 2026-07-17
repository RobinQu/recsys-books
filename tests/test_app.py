from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_and_model_api():
    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["notebooks"] == 27
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
    assert "/notebooks/3_5_0_transformer_foundations" in html
    assert "/notebooks/4_0_generative_foundations" in html
    assert "/chapters/" not in html
    assert "先把公式翻译成可以手算的直觉" in html
    assert 'class="sidebar-subnav"' in html
    assert "/notebooks/4_2_openonerec_practice" in html
    assert "/notebooks/4_3_dlrm_hstu_practice" in html
    assert "查看 27 个实验" in html


def test_legacy_chapter_routes_redirect_to_math_openings():
    expected = {
        "foundations": "3_0_math_foundations", "classic": "3_1_0_classic_foundations",
        "retrieval": "3_2_0_retrieval_foundations", "ranking": "3_3_0_ranking_foundations",
        "multitask": "3_4_0_multitask_foundations", "transformer": "3_5_0_transformer_foundations",
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
    assert "返回 3.1 导读与数学基础" in html
    assert "切换到可执行模式" in html
    assert "http://localhost:8889" in html
    assert "http://localhost:8888" not in html
    assert f"/lab/tree/notebooks/{slug}.ipynb?token=recsys" in html
    assert f'src="/notebooks/{slug}/content"' in html
    assert "下一篇" in html

    content = client.get(f"/notebooks/{slug}/content")
    assert content.status_code == 200
    assert "UserCF" in content.text and "ItemCF" in content.text
    assert 'class="jp-Notebook"' in content.text


def test_notebook_shell_unknown_slug_returns_404():
    assert client.get("/notebooks/not-a-notebook").status_code == 404
    assert client.get("/notebooks/not-a-notebook/content").status_code == 404
