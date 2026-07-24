import gzip
import importlib
import json
import os
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

from recsys_lab.data import (
    CRITEO_COLUMNS,
    amazon_2018_provenance,
    criteo_provenance,
    load_amazon_2018,
    load_criteo,
    load_movielens,
    load_movielens_1m,
    load_movielens_20m,
    load_mind_amazon_books,
    load_recif,
    mind_amazon_provenance,
    movielens_1m_provenance,
    movielens_20m_provenance,
    movielens_provenance,
    positive_sequences,
    recif_provenance,
    load_census_income,
)


pytestmark = [
    pytest.mark.full_data,
    pytest.mark.skipif(os.getenv("RECSYS_PROFILE") != "full", reason="run with RECSYS_PROFILE=full"),
]


def _gzip_rows(path: Path) -> int:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def test_complete_movielens_files_are_not_truncated():
    import zipfile

    latest, _ = load_movielens(max_users=1, max_items=1)
    with zipfile.ZipFile("resources/datasets/movielens/ml-latest.zip") as bundle, bundle.open("ml-latest/ratings.csv") as handle:
        raw_rows = sum(1 for _ in handle) - 1
    assert len(latest) == raw_rows >= 33_000_000
    provenance = movielens_provenance(latest)
    assert provenance["profile"] == "full"
    assert "MovieLens latest" in provenance["dataset"]

    ml1m = load_movielens_1m()
    provenance = movielens_1m_provenance(ml1m)
    assert provenance["raw_rows_read"] == 1_000_209
    assert (len(ml1m), ml1m.user_id.nunique(), ml1m.item_id.nunique()) == (999_611, 6_040, 3_416)
    assert provenance["rows_used"] == len(ml1m) and provenance["profile"] == "full"


def test_complete_movielens_20m_matches_official_statistics():
    frame = load_movielens_20m()
    provenance = movielens_20m_provenance(frame)
    assert (len(frame), frame.user_id.nunique(), frame.item_id.nunique()) == (20_000_263, 138_493, 26_744)
    assert provenance["rows_used"] == len(frame) and provenance["profile"] == "full"


def test_complete_criteo_split_matches_official_statistics():
    train, valid, test = load_criteo()
    provenance = criteo_provenance(train, valid, test)
    assert (len(train), len(valid), len(test)) == (33_003_326, 8_250_124, 4_587_167)
    assert list(train.columns) == CRITEO_COLUMNS
    assert set(np.unique(train.label.to_numpy())) <= {0, 1}
    assert provenance["train_rows"] + provenance["valid_rows"] + provenance["test_rows"] == 45_840_617


def test_recif_release_and_sid_catalog_match_official_statistics():
    users, catalog = load_recif()
    provenance = recif_provenance(users, catalog)
    assert len(users) == 162_074
    assert len(catalog) == 15_885_203
    assert {"uid", "hist_video_pid", "target_video_pid"} <= set(users.columns)
    assert list(catalog.columns) == ["pid", "sid"]
    sample_sid = catalog.iloc[0]["sid"]
    assert len(sample_sid) == 3
    assert provenance["users"] == len(users) and provenance["randomly_fabricated_rows"] == 0


def test_sasrec_full_test_set_contains_every_eligible_user():
    ratings = load_movielens_1m()
    sequences = positive_sequences(ratings, threshold=-float("inf"), min_length=5)
    chapter = importlib.import_module("chapter_code.5_4_sasrec.train")
    user_ids, _, _, eval_input, eval_target = chapter._sequence_windows_from_sequences(sequences, length=200)
    assert len(user_ids) == len(eval_input) == len(eval_target) == 6_040
    assert len(set(user_ids)) == 6_040


def test_multitask_full_test_set_uses_every_official_census_row():
    x_train, y_train, x_test, y_test = load_census_income()
    assert x_train.shape == (199_523, 38)
    assert x_test.shape == (99_762, 38)
    assert y_train.shape == (199_523, 2) and y_test.shape == (99_762, 2)
    assert set(np.unique(y_test[:, 0])) == {0.0, 1.0}
    assert set(np.unique(y_test[:, 1])) == {0.0, 1.0}


@pytest.mark.parametrize("category", ["Books", "Electronics"])
def test_complete_amazon_5core_files_are_not_truncated(category):
    frame = load_amazon_2018(category, min_user_events=5)
    source = Path("resources/datasets/amazon-2018") / f"{category}_5.json.gz"
    # 官方 5-core 文件仍含少量 <5 交互用户；loader 的 min_user_events 过滤只应
    # 移除这部分残差（记录在 attrs），其余行必须完整保留。
    assert _gzip_rows(source) == frame.attrs["raw_rows"]
    assert len(frame) == frame.attrs["raw_rows"] - frame.attrs["dropped_rows"]
    assert frame.attrs["dropped_rows"] / frame.attrs["raw_rows"] < 0.001
    provenance = amazon_2018_provenance(frame)
    assert provenance["profile"] == "full"
    assert provenance["rows_used"] == len(frame)


def test_mind_uses_the_paper_release_and_exact_10_core_statistics():
    frame = load_mind_amazon_books()
    provenance = mind_amazon_provenance(frame)
    # item-first 单遍 10-core 恰好还原论文样本数；users/items 以可复现口径为准。
    assert (len(frame), frame.user_id.nunique(), frame.item_id.nunique()) == (6_271_511, 218_972, 369_114)
    assert provenance["rows_used"] == len(frame)
    assert frame.groupby("user_id").size().min() >= 10


def test_reproduction_protocols_reference_downloadable_resources():
    protocols = json.loads(Path("config/reproduction_protocols.json").read_text(encoding="utf-8"))
    manifest = json.loads(Path("config/resources.json").read_text(encoding="utf-8"))
    resource_ids = {item["id"] for item in manifest["datasets"]} | {"bundled"}
    assert len(protocols) >= 13
    for slug, protocol in protocols.items():
        assert protocol["resource"] in resource_ids, slug
        assert protocol["dataset"] and protocol["split"] and protocol["metrics"]


def test_download_lock_contains_content_hashes_for_formal_datasets():
    lock = json.loads(Path("resources/resource-lock.json").read_text(encoding="utf-8"))
    formal = {"movielens-1m-full", "amazon-books-5core", "amazon-books-2014-ratings", "amazon-electronics-5core", "census-income-kdd"}
    rows = {item["id"]: item for item in lock["items"]}
    assert formal <= rows.keys()
    for resource_id in formal:
        assert rows[resource_id]["status"] == "ready"
        assert len(rows[resource_id]["sha256"]) == 64
