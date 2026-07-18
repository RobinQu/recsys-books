import gzip
import importlib
import json
import os
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

from recsys_lab.data import (
    amazon_2018_provenance,
    load_amazon_2018,
    load_movielens,
    load_movielens_1m,
    load_mind_amazon_books,
    mind_amazon_provenance,
    movielens_1m_provenance,
    movielens_provenance,
    positive_sequences,
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
    latest, _ = load_movielens(max_users=1, max_items=1)
    raw_latest = pd.read_csv("data/ml-latest-small/ratings.csv")
    assert len(latest) == len(raw_latest)
    assert movielens_provenance(latest)["profile"] == "full"

    ml1m = load_movielens_1m()
    provenance = movielens_1m_provenance(ml1m)
    assert provenance["raw_rows_read"] == 1_000_209
    assert (len(ml1m), ml1m.user_id.nunique(), ml1m.item_id.nunique()) == (999_611, 6_040, 3_416)
    assert provenance["rows_used"] == len(ml1m) and provenance["profile"] == "full"


def test_sasrec_full_test_set_contains_every_eligible_user():
    ratings = load_movielens_1m()
    sequences = positive_sequences(ratings, threshold=-float("inf"), min_length=5)
    chapter = importlib.import_module("chapter_code.3_2_3_sasrec.train")
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
    assert len(frame) == _gzip_rows(source)
    provenance = amazon_2018_provenance(frame)
    assert provenance["profile"] == "full"
    assert provenance["rows_used"] == len(frame)


def test_mind_uses_the_paper_release_and_exact_10_core_statistics():
    frame = load_mind_amazon_books()
    provenance = mind_amazon_provenance(frame)
    assert (len(frame), frame.user_id.nunique(), frame.item_id.nunique()) == (6_271_511, 351_356, 393_801)
    assert provenance["rows_used"] == len(frame)
    assert frame.groupby("user_id").size().min() >= 10
    assert frame.groupby("item_id").size().min() >= 10


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
