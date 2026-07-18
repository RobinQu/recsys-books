import json
from pathlib import Path

from recsys_lab.resources import ROOT, ensure_resources, iter_resources, source_url, target_path


def test_resource_manifest_has_safe_unique_paths_and_resolvable_sources():
    rows = list(iter_resources())
    ids = [item["id"] for _, item in rows]
    paths = [item["path"] for _, item in rows]
    assert len(ids) == len(set(ids))
    assert len(paths) == len(set(paths))
    for kind, item in rows:
        target = target_path(item).resolve()
        assert target.is_relative_to((ROOT / "resources").resolve())
        if item["provider"] not in {"huggingface", "npm"}:
            assert source_url(item).startswith("https://")
        if kind == "papers":
            assert item["title"] and item["path"].endswith(".pdf")
    assert next(item for _, item in rows if item["id"] == "ple")["path"] == "papers/ple-recsys2020.pdf"
    kuai = next(item for _, item in rows if item["id"] == "kuairand-pure-full")
    assert kuai["url"].endswith("?download=1")
    assert "video_features_basic_pure.csv" in kuai["expected_members"]
    protocol = json.loads(Path("config/reproduction_protocols.json").read_text(encoding="utf-8"))
    assert protocol["3_2_2_mind"]["resource"] == "amazon-books-2014-ratings"
    assert protocol["3_2_3_sasrec"]["resource"] == "movielens-1m-full"
    assert protocol["3_3_1_deepfm"]["resource"] == "criteo-x1-full"
    dataset_ids = {item["id"] for kind, item in rows if kind == "datasets"}
    assert "aliccp-x1-full" not in dataset_ids
    criteo = next(item for kind, item in rows if kind == "datasets" and item["id"] == "criteo-x1-full")
    assert criteo["expected_files"] == ["Criteo_x1.zip"]
    assert len(criteo["file_sha256"]["Criteo_x1.zip"]) == 64


def test_downloaded_resources_are_ignored_and_initializer_is_root_runnable():
    ignore = Path(".gitignore").read_text(encoding="utf-8")
    script = Path("scripts/init_resources.py").read_text(encoding="utf-8")
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    assert "resources/" in ignore
    assert "sys.path.insert" in script
    assert "scripts/init_resources.py --strict" in dockerfile
    assert "COPY . ." in dockerfile


def test_resource_verification_writes_machine_readable_lock():
    result = ensure_resources(download=False, kinds=["papers"], ids=["dssm"])
    assert result["items"][0]["status"] in {"ready", "missing"}
    lock = json.loads((ROOT / "resources" / "resource-lock.json").read_text(encoding="utf-8"))
    assert lock["schema_version"] == 1


def test_full_profile_dataset_contract_is_explicit():
    source = Path("recsys_lab/data.py").read_text(encoding="utf-8")
    assert 'RECSYS_PROFILE", "smoke"' in source
    assert "Video_Games.csv.gz" in source
    assert "KuaiRand-Pure.tar.gz" in source
    assert "all official standard-policy rows; no user/item truncation" in source
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    assert "RECSYS_PROFILE: full" in compose
