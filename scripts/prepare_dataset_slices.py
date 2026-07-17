"""Build deterministic, auditable CPU slices from official recommendation datasets.

This script is not used during tests: CI consumes the committed slices.  It is
kept so a reader can reproduce exactly how those slices were selected from the
official archives, without sampling or fabricating interactions.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_most_active(frame: pd.DataFrame, user_column: str, users: int) -> pd.Index:
    return (
        frame.groupby(user_column).size().rename("events").reset_index()
        .sort_values(["events", user_column], ascending=[False, True])
        .head(users)[user_column]
    )


def prepare_amazon(source: Path, users: int = 500) -> None:
    output = ROOT / "data" / "amazon-reviews-2023-video-games"
    output.mkdir(parents=True, exist_ok=True)
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        frame = pd.read_csv(handle)
    chosen = select_most_active(frame, "user_id", users)
    subset = (
        frame[frame.user_id.isin(chosen)]
        .sort_values(["timestamp", "user_id", "parent_asin"])
        .reset_index(drop=True)
    )
    subset.to_csv(output / "interactions.csv", index=False)
    provenance = {
        "dataset": "Amazon Reviews 2023 / Video Games / 5-core",
        "source_url": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Video_Games.csv.gz",
        "source_sha256": sha256(source),
        "selection": f"top {users} users by observed interaction count; ties by user_id; all their rows retained",
        "rows": len(subset),
        "users": subset.user_id.nunique(),
        "items": subset.parent_asin.nunique(),
        "randomly_fabricated_rows": 0,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


def prepare_kuairand(source_dir: Path, users: int = 256) -> None:
    output = ROOT / "data" / "kuairand-pure"
    output.mkdir(parents=True, exist_ok=True)
    data_dir = source_dir / "data"
    standard_paths = [
        data_dir / "log_standard_4_08_to_4_21_pure.csv",
        data_dir / "log_standard_4_22_to_5_08_pure.csv",
    ]
    standard = pd.concat([pd.read_csv(path) for path in standard_paths], ignore_index=True)
    chosen = select_most_active(standard, "user_id", users)
    standard = (
        standard[standard.user_id.isin(chosen)]
        .sort_values(["time_ms", "user_id", "video_id"])
        .reset_index(drop=True)
    )
    random_log = pd.read_csv(data_dir / "log_random_4_22_to_5_08_pure.csv")
    random_log = (
        random_log[random_log.user_id.isin(chosen)]
        .sort_values(["time_ms", "user_id", "video_id"])
        .reset_index(drop=True)
    )
    standard.to_csv(output / "standard_interactions.csv", index=False)
    random_log.to_csv(output / "random_interactions.csv", index=False)

    videos = pd.read_csv(data_dir / "video_features_basic_pure.csv")
    selected_videos = pd.Index(pd.concat([standard.video_id, random_log.video_id]).unique())
    videos[videos.video_id.isin(selected_videos)].sort_values("video_id").to_csv(output / "videos.csv", index=False)
    shutil.copyfile(source_dir / "LICENSE", output / "LICENSE")
    provenance = {
        "dataset": "KuaiRand-Pure",
        "source_url": "https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz",
        "source_sha256": "c814bf6f3624c0cfae83c57de3df26b2ed206e5c57bab4c4dcbfabbabe20cbf0",
        "selection": f"top {users} users by standard-feed interaction count; ties by user_id; all standard and random-policy rows for those users retained",
        "standard_rows": len(standard),
        "random_policy_rows": len(random_log),
        "users": standard.user_id.nunique(),
        "items": selected_videos.size,
        "randomly_fabricated_rows": 0,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--amazon", type=Path, required=True, help="official Video_Games.csv.gz")
    parser.add_argument("--kuairand", type=Path, required=True, help="extracted KuaiRand-Pure directory")
    args = parser.parse_args()
    prepare_amazon(args.amazon)
    prepare_kuairand(args.kuairand)
