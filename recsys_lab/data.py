"""Versioned, deterministic access to the tutorial's real datasets.

MovieLens supports the classic chapters, Amazon Reviews 2023 supports modern
retrieval/sequential models, and KuaiRand supports feed ranking and multitask
learning.  All task views are deterministic transformations of observed rows;
no interaction or behavior sequence is randomly fabricated.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "ml-latest-small"
SOURCE_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATASET_NAME = "MovieLens latest-small (GroupLens, generated 2018-09-26)"
AMAZON_DIR = ROOT / "data" / "amazon-reviews-2023-video-games"
AMAZON_NAME = "Amazon Reviews 2023 / Video Games / 5-core"
KUAI_DIR = ROOT / "data" / "kuairand-pure"
KUAIRAND_NAME = "KuaiRand-Pure (Kuaishou, CIKM 2022)"


@lru_cache(maxsize=8)
def _load_cached(max_users: int, max_items: int, min_user_events: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings_path = DATA_DIR / "ratings.csv"
    movies_path = DATA_DIR / "movies.csv"
    if not ratings_path.exists() or not movies_path.exists():
        raise FileNotFoundError(
            f"缺少真实数据集：{ratings_path}。请从 {SOURCE_URL} 下载并解压到 {DATA_DIR}。"
        )

    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)

    # 固定选取行为最多的用户与这些用户中最常出现的电影，保证 CPU 实验可运行。
    user_order = (
        ratings.groupby("userId").size().rename("events").reset_index()
        .sort_values(["events", "userId"], ascending=[False, True])
        .head(max_users)["userId"]
    )
    subset = ratings[ratings.userId.isin(user_order)].copy()
    item_order = (
        subset.groupby("movieId").size().rename("events").reset_index()
        .sort_values(["events", "movieId"], ascending=[False, True])
        .head(max_items)["movieId"]
    )
    subset = subset[subset.movieId.isin(item_order)].copy()
    valid_users = subset.groupby("userId").size()
    subset = subset[subset.userId.isin(valid_users[valid_users >= min_user_events].index)].copy()

    user_ids = {raw: idx for idx, raw in enumerate(sorted(subset.userId.unique()))}
    item_ids = {raw: idx for idx, raw in enumerate(sorted(subset.movieId.unique()))}
    subset["user_id"] = subset.userId.map(user_ids).astype("int64")
    subset["item_id"] = subset.movieId.map(item_ids).astype("int64")
    selected_movies = movies[movies.movieId.isin(item_ids)].copy()
    subset = subset.merge(selected_movies, on="movieId", how="left", validate="many_to_one")

    subset["primary_genre"] = subset.genres.fillna("(no genres listed)").str.split("|").str[0]
    genre_values = sorted(subset.primary_genre.unique())
    genre_ids = {name: idx for idx, name in enumerate(genre_values)}
    subset["genre_id"] = subset.primary_genre.map(genre_ids).astype("int64")
    subset["release_year"] = pd.to_numeric(subset.title.str.extract(r"\((\d{4})\)$")[0], errors="coerce")
    fallback_year = int(subset.release_year.median())
    subset["release_decade"] = ((subset.release_year.fillna(fallback_year).astype(int) // 10) * 10).astype("int64")
    decade_values = sorted(subset.release_decade.unique())
    decade_ids = {value: idx for idx, value in enumerate(decade_values)}
    subset["decade_id"] = subset.release_decade.map(decade_ids).astype("int64")
    timestamp = pd.to_datetime(subset.timestamp, unit="s", utc=True)
    subset["hour"] = timestamp.dt.hour.astype("int64")
    subset["weekday"] = timestamp.dt.weekday.astype("int64")
    subset["like"] = (subset.rating >= 4.0).astype("float32")
    subset["very_like"] = (subset.rating >= 4.5).astype("float32")
    item_popularity = subset.groupby("item_id").size()
    user_activity = subset.groupby("user_id").size()
    subset["item_popularity"] = subset.item_id.map(item_popularity).astype("float32")
    subset["user_activity"] = subset.user_id.map(user_activity).astype("float32")
    subset = subset.sort_values(["timestamp", "user_id", "item_id"]).reset_index(drop=True)

    selected_movies = (
        selected_movies.assign(item_id=selected_movies.movieId.map(item_ids).astype("int64"))
        .sort_values("item_id").reset_index(drop=True)
    )
    return subset, selected_movies


def load_movielens(max_users: int = 120, max_items: int = 600, min_user_events: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return isolated copies of a deterministic real-data subset."""
    ratings, movies = _load_cached(max_users, max_items, min_user_events)
    return ratings.copy(), movies.copy()


def movielens_provenance(ratings: pd.DataFrame) -> dict:
    return {
        "dataset": DATASET_NAME,
        "source": SOURCE_URL,
        "license_file": str(DATA_DIR / "README.txt"),
        "rows_used": int(len(ratings)),
        "users_used": int(ratings.user_id.nunique()),
        "items_used": int(ratings.item_id.nunique()),
        "time_min_utc": pd.to_datetime(ratings.timestamp.min(), unit="s", utc=True).isoformat(),
        "time_max_utc": pd.to_datetime(ratings.timestamp.max(), unit="s", utc=True).isoformat(),
        "positive_rule": "like := observed rating >= 4.0; very_like := observed rating >= 4.5",
        "randomly_fabricated_rows": 0,
    }


@lru_cache(maxsize=8)
def _load_amazon_cached(max_users: int, min_user_events: int) -> pd.DataFrame:
    path = AMAZON_DIR / "interactions.csv"
    if not path.exists():
        raise FileNotFoundError(f"缺少真实数据集切片：{path}")
    frame = pd.read_csv(path)
    user_order = (
        frame.groupby("user_id").size().rename("events").reset_index()
        .sort_values(["events", "user_id"], ascending=[False, True]).head(max_users)["user_id"]
    )
    frame = frame[frame.user_id.isin(user_order)].copy()
    valid = frame.groupby("user_id").size()
    frame = frame[frame.user_id.isin(valid[valid >= min_user_events].index)].copy()
    user_map = {raw: idx for idx, raw in enumerate(sorted(frame.user_id.unique()))}
    item_map = {raw: idx for idx, raw in enumerate(sorted(frame.parent_asin.unique()))}
    frame["raw_user_id"] = frame.user_id
    frame["user_id"] = frame.user_id.map(user_map).astype("int64")
    frame["item_id"] = frame.parent_asin.map(item_map).astype("int64")
    frame["timestamp_ms"] = frame.timestamp.astype("int64")
    frame["timestamp"] = (frame.timestamp_ms // 1000).astype("int64")
    frame["like"] = (frame.rating >= 4.0).astype("float32")
    frame["very_like"] = (frame.rating >= 5.0).astype("float32")
    frame["hour"] = pd.to_datetime(frame.timestamp_ms, unit="ms", utc=True).dt.hour.astype("int64")
    frame = frame.sort_values(["timestamp", "user_id", "item_id"]).reset_index(drop=True)
    return frame


def load_amazon_2023(max_users: int = 160, min_user_events: int = 12) -> pd.DataFrame:
    return _load_amazon_cached(max_users, min_user_events).copy()


def amazon_provenance(frame: pd.DataFrame) -> dict:
    source = pd.read_json(AMAZON_DIR / "provenance.json", typ="series")
    return {
        "dataset": AMAZON_NAME,
        "source": source["source_url"],
        "source_sha256": source["source_sha256"],
        "slice_rule": source["selection"],
        "rows_used": int(len(frame)),
        "users_used": int(frame.user_id.nunique()),
        "items_used": int(frame.item_id.nunique()),
        "time_min_utc": pd.to_datetime(frame.timestamp.min(), unit="s", utc=True).isoformat(),
        "time_max_utc": pd.to_datetime(frame.timestamp.max(), unit="s", utc=True).isoformat(),
        "positive_rule": "observed Amazon rating >= 4.0",
        "randomly_fabricated_rows": 0,
    }


@lru_cache(maxsize=8)
def _load_kuairand_cached(max_users: int, max_items: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    interactions = pd.read_csv(KUAI_DIR / "standard_interactions.csv")
    videos = pd.read_csv(KUAI_DIR / "videos.csv")
    user_order = (
        interactions.groupby("user_id").size().rename("events").reset_index()
        .sort_values(["events", "user_id"], ascending=[False, True]).head(max_users)["user_id"]
    )
    frame = interactions[interactions.user_id.isin(user_order)].copy()
    item_order = (
        frame.groupby("video_id").size().rename("events").reset_index()
        .sort_values(["events", "video_id"], ascending=[False, True]).head(max_items)["video_id"]
    )
    frame = frame[frame.video_id.isin(item_order)].copy()
    user_map = {raw: idx for idx, raw in enumerate(sorted(frame.user_id.unique()))}
    item_map = {raw: idx for idx, raw in enumerate(sorted(frame.video_id.unique()))}
    frame["raw_user_id"] = frame.user_id
    frame["user_id"] = frame.user_id.map(user_map).astype("int64")
    frame["item_id"] = frame.video_id.map(item_map).astype("int64")
    frame["timestamp"] = (frame.time_ms // 1000).astype("int64")
    frame["hour"] = (frame.hourmin // 100).clip(0, 23).astype("int64")
    frame["duration_bucket"] = pd.cut(
        frame.duration_ms, bins=[-1, 7000, 18000, 60000, np.inf], labels=False
    ).astype("int64")
    item_popularity = frame.groupby("item_id").size()
    user_activity = frame.groupby("user_id").size()
    frame["item_popularity"] = frame.item_id.map(item_popularity).astype("float32")
    frame["user_activity"] = frame.user_id.map(user_activity).astype("float32")
    frame = frame.sort_values(["timestamp", "user_id", "item_id"]).reset_index(drop=True)
    video_view = videos[videos.video_id.isin(item_map)].copy()
    video_view["item_id"] = video_view.video_id.map(item_map).astype("int64")
    video_view = video_view.sort_values("item_id").reset_index(drop=True)
    return frame, video_view


def load_kuairand(max_users: int = 128, max_items: int = 2500) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame, videos = _load_kuairand_cached(max_users, max_items)
    return frame.copy(), videos.copy()


def kuairand_provenance(frame: pd.DataFrame) -> dict:
    source = pd.read_json(KUAI_DIR / "provenance.json", typ="series")
    return {
        "dataset": KUAIRAND_NAME,
        "source": source["source_url"],
        "source_sha256": source["source_sha256"],
        "license_file": str(KUAI_DIR / "LICENSE"),
        "slice_rule": source["selection"],
        "rows_used": int(len(frame)),
        "users_used": int(frame.user_id.nunique()),
        "items_used": int(frame.item_id.nunique()),
        "time_min_utc": pd.to_datetime(frame.timestamp.min(), unit="s", utc=True).isoformat(),
        "time_max_utc": pd.to_datetime(frame.timestamp.max(), unit="s", utc=True).isoformat(),
        "targets": "observed is_click, long_view, is_like and other feed feedback",
        "randomly_fabricated_rows": 0,
    }


def leave_last_out(ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = ratings.sort_values(["user_id", "timestamp", "item_id"]).copy()
    test_index = ordered.groupby("user_id").tail(1).index
    test = ordered.loc[test_index].sort_values("user_id").copy()
    train = ordered.drop(test_index).copy()
    return train, test


def positive_sequences(ratings: pd.DataFrame, threshold: float = 3.5, min_length: int = 6) -> dict[int, list[int]]:
    positive = ratings[ratings.rating >= threshold].sort_values(["user_id", "timestamp", "item_id"])
    sequences = positive.groupby("user_id").item_id.apply(lambda values: [int(v) + 1 for v in values]).to_dict()
    return {int(user): sequence for user, sequence in sequences.items() if len(sequence) >= min_length}


def pad_left(values: list[int], length: int) -> list[int]:
    values = values[-length:]
    return [0] * (length - len(values)) + values


def sequence_classification_rows(ratings: pd.DataFrame, max_len: int = 10, limit: int = 1600) -> dict[str, np.ndarray]:
    """Create DIN/DIEN rows from observed chronological ratings only."""
    rows: list[tuple[int, list[int], list[int], int, float, int]] = []
    for user_id, frame in ratings.sort_values(["user_id", "timestamp", "item_id"]).groupby("user_id"):
        positive_history: list[int] = []
        negative_history: list[int] = []
        for event in frame.itertuples():
            if len(positive_history) >= 2:
                rows.append((
                    int(user_id) + 1,
                    pad_left(positive_history, max_len),
                    pad_left(negative_history, max_len),
                    int(event.item_id) + 1,
                    float(event.rating >= 4.0),
                    int(event.timestamp),
                ))
            if event.rating >= 4.0:
                positive_history.append(int(event.item_id) + 1)
            elif event.rating <= 3.0:
                negative_history.append(int(event.item_id) + 1)
    rows.sort(key=lambda row: (row[5], row[0], row[3]))
    rows = rows[-limit:]
    return {
        "user_id": np.asarray([row[0] for row in rows], dtype=np.int64),
        "history": np.asarray([row[1] for row in rows], dtype=np.int64),
        "negative_history": np.asarray([row[2] for row in rows], dtype=np.int64),
        "item_id": np.asarray([row[3] for row in rows], dtype=np.int64),
        "label": np.asarray([row[4] for row in rows], dtype=np.float32),
        "timestamp": np.asarray([row[5] for row in rows], dtype=np.int64),
    }


def kuairand_sequence_classification_rows(
    interactions: pd.DataFrame, max_len: int = 20, limit: int = 2600
) -> dict[str, np.ndarray]:
    """Build DIN/DIEN examples from observed KuaiRand feed impressions."""
    rows: list[tuple[int, list[int], list[int], int, float, int]] = []
    for user_id, frame in interactions.sort_values(["user_id", "timestamp", "item_id"]).groupby("user_id"):
        clicked: list[int] = []
        skipped: list[int] = []
        for event in frame.itertuples():
            if len(clicked) >= 2:
                rows.append((
                    int(user_id) + 1,
                    pad_left(clicked, max_len),
                    pad_left(skipped, max_len),
                    int(event.item_id) + 1,
                    float(event.is_click),
                    int(event.timestamp),
                ))
            if event.is_click:
                clicked.append(int(event.item_id) + 1)
            else:
                skipped.append(int(event.item_id) + 1)
    rows.sort(key=lambda row: (row[5], row[0], row[3]))
    rows = rows[-limit:]
    return {
        "user_id": np.asarray([row[0] for row in rows], dtype=np.int64),
        "history": np.asarray([row[1] for row in rows], dtype=np.int64),
        "negative_history": np.asarray([row[2] for row in rows], dtype=np.int64),
        "item_id": np.asarray([row[3] for row in rows], dtype=np.int64),
        "label": np.asarray([row[4] for row in rows], dtype=np.float32),
        "timestamp": np.asarray([row[5] for row in rows], dtype=np.int64),
    }


def clicked_sequences(interactions: pd.DataFrame, min_length: int = 8) -> dict[int, list[int]]:
    clicked = interactions[interactions.is_click == 1].sort_values(["user_id", "timestamp", "item_id"])
    sequences = clicked.groupby("user_id").item_id.apply(lambda values: [int(v) + 1 for v in values]).to_dict()
    return {int(user): sequence for user, sequence in sequences.items() if len(sequence) >= min_length}
