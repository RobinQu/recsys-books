"""Versioned, deterministic access to the tutorial's real datasets.

MovieLens supports the classic chapters, Amazon Reviews 2023 supports modern
retrieval/sequential models, and KuaiRand supports feed ranking and multitask
learning.  All task views are deterministic transformations of observed rows;
no interaction or behavior sequence is randomly fabricated.
"""
from __future__ import annotations

from functools import lru_cache
import io
import os
from pathlib import Path
import tarfile
import zipfile

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
RESOURCE_DATA = ROOT / "resources" / "datasets"


def _full_profile() -> bool:
    return os.getenv("RECSYS_PROFILE", "smoke").casefold() == "full"


@lru_cache(maxsize=8)
def _load_cached(max_users: int, max_items: int, min_user_events: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    # full 档读取官方完整 MovieLens latest（约 33M 评分）；smoke 档使用仓库内
    # latest-small 的确定性教学切片。
    if _full_profile():
        path = RESOURCE_DATA / "movielens" / "ml-latest.zip"
        if not path.exists():
            raise FileNotFoundError(
                f"缺少 MovieLens latest 完整版：{path}。请运行 "
                "`python scripts/init_resources.py --include-optional --kind datasets --id movielens-latest-full`"
            )
        with zipfile.ZipFile(path) as bundle:
            with bundle.open("ml-latest/ratings.csv") as handle:
                ratings = pd.read_csv(handle)
            with bundle.open("ml-latest/movies.csv") as handle:
                movies = pd.read_csv(handle)
        # GroupLens 会定期重新生成 latest，因此只校验数量级并以 provenance 记录实际行数。
        if len(ratings) < 33_000_000:
            raise ValueError(f"MovieLens latest 行数异常：预期约 33M 评分，实际 {len(ratings)}")
        subset = ratings.copy()
    else:
        ratings_path = DATA_DIR / "ratings.csv"
        movies_path = DATA_DIR / "movies.csv"
        if not ratings_path.exists() or not movies_path.exists():
            raise FileNotFoundError(
                f"缺少真实数据集：{ratings_path}。请从 {SOURCE_URL} 下载并解压到 {DATA_DIR}。"
            )

        ratings = pd.read_csv(ratings_path)
        movies = pd.read_csv(movies_path)

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
    subset.attrs["resource_profile"] = "full" if _full_profile() else "smoke"

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
    full = ratings.attrs.get("resource_profile") == "full"
    return {
        "dataset": "MovieLens latest (GroupLens, ~33M ratings)" if full else DATASET_NAME,
        "source": "https://files.grouplens.org/datasets/movielens/ml-latest.zip" if full else SOURCE_URL,
        "license_file": str(DATA_DIR / "README.txt"),
        "profile": "full" if full else "smoke",
        "slice_rule": "complete official MovieLens latest ratings file" if full else "deterministic user/item subset",
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
    full_path = RESOURCE_DATA / "amazon-reviews-2023" / "Video_Games.csv.gz"
    path = full_path if _full_profile() else AMAZON_DIR / "interactions.csv"
    if not path.exists():
        command = "python scripts/init_resources.py --include-optional --kind datasets --id amazon-video-games-full"
        raise FileNotFoundError(f"缺少真实数据集：{path}。full 档请运行 `{command}`")
    frame = pd.read_csv(path)
    if not _full_profile():
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
    frame.attrs["resource_profile"] = "full" if _full_profile() else "smoke"
    frame.attrs["resource_path"] = str(path)
    return frame


def load_amazon_2023(max_users: int = 160, min_user_events: int = 12) -> pd.DataFrame:
    return _load_amazon_cached(max_users, min_user_events).copy()


def amazon_provenance(frame: pd.DataFrame) -> dict:
    source = pd.read_json(AMAZON_DIR / "provenance.json", typ="series")
    full = frame.attrs.get("resource_profile") == "full"
    return {
        "dataset": AMAZON_NAME,
        "source": source["source_url"],
        "source_sha256": source["source_sha256"],
        "slice_rule": "complete official 5-core rating file; no user truncation" if full else source["selection"],
        "profile": "full" if full else "smoke",
        "local_resource": frame.attrs.get("resource_path"),
        "rows_used": int(len(frame)),
        "users_used": int(frame.user_id.nunique()),
        "items_used": int(frame.item_id.nunique()),
        "time_min_utc": pd.to_datetime(frame.timestamp.min(), unit="s", utc=True).isoformat(),
        "time_max_utc": pd.to_datetime(frame.timestamp.max(), unit="s", utc=True).isoformat(),
        "positive_rule": "observed Amazon rating >= 4.0",
        "randomly_fabricated_rows": 0,
    }


@lru_cache(maxsize=4)
def _load_amazon_2018_cached(category: str, min_user_events: int) -> pd.DataFrame:
    """Load every row of an official Amazon 5-core category for formal runs."""
    path = RESOURCE_DATA / "amazon-2018" / f"{category}_5.json.gz"
    if not path.exists():
        resource_id = f"amazon-{category.casefold()}-5core"
        raise FileNotFoundError(
            f"缺少 {category} 完整数据：{path}。请运行 `python scripts/init_resources.py --include-optional --kind datasets --id {resource_id}`"
        )
    raw = pd.read_json(path, lines=True, compression="gzip")
    frame = raw.rename(columns={"reviewerID": "raw_user_id", "asin": "raw_item_id", "overall": "rating", "unixReviewTime": "timestamp"})
    valid = frame.groupby("raw_user_id").size()
    filtered = frame[frame.raw_user_id.isin(valid[valid >= min_user_events].index)].copy()
    # 官方 5-core 文件仍含少量 <min_user_events 交互的用户；记录过滤残差供完整性校验。
    raw_rows, dropped_rows = len(frame), len(frame) - len(filtered)
    frame = filtered
    user_map = {raw_id: index for index, raw_id in enumerate(sorted(frame.raw_user_id.unique()))}
    item_map = {raw_id: index for index, raw_id in enumerate(sorted(frame.raw_item_id.unique()))}
    frame["user_id"] = frame.raw_user_id.map(user_map).astype("int64")
    frame["item_id"] = frame.raw_item_id.map(item_map).astype("int64")
    frame["like"] = (frame.rating >= 4.0).astype("float32")
    frame["very_like"] = (frame.rating >= 5.0).astype("float32")
    frame["hour"] = pd.to_datetime(frame.timestamp, unit="s", utc=True).dt.hour.astype("int64")
    frame = frame.sort_values(["timestamp", "user_id", "item_id"]).reset_index(drop=True)
    frame.attrs.update(resource_profile="full", resource_path=str(path), dataset_name=f"Amazon Reviews {category} 5-core (2018)")
    frame.attrs.update(raw_rows=raw_rows, dropped_rows=dropped_rows)
    return frame


def load_amazon_2018(category: str, min_user_events: int = 5) -> pd.DataFrame:
    return _load_amazon_2018_cached(category, min_user_events).copy()


def amazon_2018_provenance(frame: pd.DataFrame) -> dict:
    return {
        "dataset": frame.attrs["dataset_name"], "source": "https://nijianmo.github.io/amazon/index.html",
        "profile": "full", "slice_rule": "complete official category 5-core file; no user/item truncation",
        "local_resource": frame.attrs["resource_path"], "rows_used": int(len(frame)),
        "users_used": int(frame.user_id.nunique()), "items_used": int(frame.item_id.nunique()),
        "randomly_fabricated_rows": 0,
    }


MIND_AMAZON_STATS = {"rows": 6_271_511, "users": 218_972, "items": 369_114}


@lru_cache(maxsize=1)
def _load_mind_amazon_books_cached() -> pd.DataFrame:
    """Rebuild the Amazon Books 10-core view described in the MIND paper."""
    path = RESOURCE_DATA / "amazon-2014" / "ratings_Books.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"缺少 MIND 论文版 Amazon Books：{path}。请运行 "
            "`python scripts/init_resources.py --include-optional --kind datasets --id amazon-books-2014-ratings`"
        )
    frame = pd.read_csv(
        path, names=["raw_user_id", "raw_item_id", "rating", "timestamp"],
        dtype={"raw_user_id": "string", "raw_item_id": "string", "rating": "float32", "timestamp": "int64"},
    )
    # MIND 论文口径（实测还原）：先按 item 过滤（>=10 个用户），再按 user 过滤
    # （>=10 个物品），单遍不迭代。该顺序恰好还原论文的 6,271,511 个样本；
    # 迭代收敛会过度过滤（4.7M），user-first 单遍则得到 5.79M。
    item_count = frame.groupby("raw_item_id", observed=True).raw_user_id.transform("size")
    frame = frame[item_count >= 10]
    user_count = frame.groupby("raw_user_id", observed=True).raw_item_id.transform("size")
    frame = frame[user_count >= 10]
    actual = {"rows": len(frame), "users": frame.raw_user_id.nunique(), "items": frame.raw_item_id.nunique()}
    if actual != MIND_AMAZON_STATS:
        raise ValueError(f"MIND Amazon preprocessing drift: expected {MIND_AMAZON_STATS}, got {actual}")
    frame["user_id"] = pd.factorize(frame.raw_user_id, sort=True)[0].astype("int64")
    frame["item_id"] = pd.factorize(frame.raw_item_id, sort=True)[0].astype("int64")
    frame["like"] = np.ones(len(frame), dtype="float32")
    frame = frame.sort_values(["timestamp", "user_id", "item_id"]).reset_index(drop=True)
    frame.attrs.update(resource_profile="full", resource_path=str(path), dataset_name="Amazon Books 2014 / MIND 10-core")
    return frame


def load_mind_amazon_books() -> pd.DataFrame:
    return _load_mind_amazon_books_cached().copy()


def mind_amazon_provenance(frame: pd.DataFrame) -> dict:
    return {
        "dataset": frame.attrs["dataset_name"],
        "source": "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Books.csv",
        "profile": "full", "slice_rule": "complete 2014 ratings file; item-first single-pass user/item 10-core（论文口径，恰好还原 6,271,511 样本）",
        "paper_expected_stats": MIND_AMAZON_STATS, "local_resource": frame.attrs["resource_path"],
        "paper_table1_note": "论文 Table 1 的 users/items（351,356/393,801）与本可复现口径（218,972/369,114）不同；样本行数 6,271,511 完全一致，以行数为比较锚点",
        "rows_used": int(len(frame)), "users_used": int(frame.user_id.nunique()),
        "items_used": int(frame.item_id.nunique()), "randomly_fabricated_rows": 0,
    }


@lru_cache(maxsize=1)
def _load_movielens_1m_cached() -> pd.DataFrame:
    path = RESOURCE_DATA / "movielens" / "ml-1m.zip"
    if not path.exists():
        raise FileNotFoundError(
            f"缺少 MovieLens 1M：{path}。请运行 `python scripts/init_resources.py --include-optional --kind datasets --id movielens-1m-full`"
        )
    with zipfile.ZipFile(path) as bundle, bundle.open("ml-1m/ratings.dat") as handle:
        frame = pd.read_csv(handle, sep="::", engine="python", names=["raw_user_id", "raw_item_id", "rating", "timestamp"])
    raw_rows = len(frame)
    # SASRec follows the standard sequential-recommendation 5-core protocol.
    # This is algorithm-defined preprocessing, not an arbitrary tutorial crop.
    while True:
        before = len(frame)
        frame = frame[frame.groupby("raw_user_id").raw_item_id.transform("size") >= 5]
        frame = frame[frame.groupby("raw_item_id").raw_user_id.transform("size") >= 5]
        if len(frame) == before:
            break
    if (len(frame), frame.raw_user_id.nunique(), frame.raw_item_id.nunique()) != (999_611, 6_040, 3_416):
        raise ValueError("MovieLens 1M SASRec 5-core statistics do not match paper Table II")
    user_map = {raw_id: index for index, raw_id in enumerate(sorted(frame.raw_user_id.unique()))}
    item_map = {raw_id: index for index, raw_id in enumerate(sorted(frame.raw_item_id.unique()))}
    frame["user_id"] = frame.raw_user_id.map(user_map).astype("int64")
    frame["item_id"] = frame.raw_item_id.map(item_map).astype("int64")
    frame["like"] = (frame.rating >= 4.0).astype("float32")
    frame = frame.sort_values(["timestamp", "user_id", "item_id"]).reset_index(drop=True)
    frame.attrs.update(resource_profile="full", resource_path=str(path), raw_rows=raw_rows)
    return frame


def load_movielens_1m() -> pd.DataFrame:
    return _load_movielens_1m_cached().copy()


def movielens_1m_provenance(frame: pd.DataFrame) -> dict:
    return {
        "dataset": "MovieLens 1M", "source": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "profile": "full", "slice_rule": "read all 1,000,209 official ratings, then apply paper-defined iterative user/item 5-core",
        "local_resource": frame.attrs["resource_path"], "rows_used": int(len(frame)),
        "raw_rows_read": int(frame.attrs["raw_rows"]),
        "users_used": int(frame.user_id.nunique()), "items_used": int(frame.item_id.nunique()),
        "randomly_fabricated_rows": 0,
    }


@lru_cache(maxsize=1)
def _load_movielens_20m_cached() -> pd.DataFrame:
    """MovieLens 20M：HSTU 论文公开基准之一，作为隐式反馈序列使用。"""
    path = RESOURCE_DATA / "movielens" / "ml-20m.zip"
    if not path.exists():
        raise FileNotFoundError(
            f"缺少 MovieLens 20M：{path}。请运行 "
            "`python scripts/init_resources.py --include-optional --kind datasets --id movielens-20m-full`"
        )
    with zipfile.ZipFile(path) as bundle, bundle.open("ml-20m/ratings.csv") as handle:
        frame = pd.read_csv(handle)
    if (len(frame), frame.userId.nunique(), frame.movieId.nunique()) != (20_000_263, 138_493, 26_744):
        raise ValueError("MovieLens 20M 官方统计（20,000,263 / 138,493 / 26,744 有评分电影）不一致")
    frame = frame.rename(columns={"userId": "raw_user_id", "movieId": "raw_item_id"})
    user_map = {raw_id: index for index, raw_id in enumerate(sorted(frame.raw_user_id.unique()))}
    item_map = {raw_id: index for index, raw_id in enumerate(sorted(frame.raw_item_id.unique()))}
    frame["user_id"] = frame.raw_user_id.map(user_map).astype("int64")
    frame["item_id"] = frame.raw_item_id.map(item_map).astype("int64")
    frame = frame.sort_values(["timestamp", "user_id", "item_id"]).reset_index(drop=True)
    frame.attrs.update(resource_profile="full", resource_path=str(path))
    return frame


def load_movielens_20m() -> pd.DataFrame:
    return _load_movielens_20m_cached().copy()


def movielens_20m_provenance(frame: pd.DataFrame) -> dict:
    return {
        "dataset": "MovieLens 20M", "source": "https://files.grouplens.org/datasets/movielens/ml-20m.zip",
        "profile": "full",
        "slice_rule": "complete official 20,000,263 ratings; every observed rating is implicit feedback (generative-recommenders 口径)",
        "local_resource": frame.attrs["resource_path"], "rows_used": int(len(frame)),
        "users_used": int(frame.user_id.nunique()), "items_used": int(frame.item_id.nunique()),
        "randomly_fabricated_rows": 0,
    }


CRITEO_SPLITS = {"train.csv": 33_003_326, "valid.csv": 8_250_124, "test.csv": 4_587_167}
CRITEO_COLUMNS = ["label", *[f"I{index}" for index in range(1, 14)], *[f"C{index}" for index in range(1, 27)]]


@lru_cache(maxsize=1)
def _load_criteo_cached() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the official reczoo Criteo_x1 7:2:1 split (45,840,617 rows total).

    类别列按 category 读入以控制内存；完整训练属于 dispatch 级任务，需要大内存机器。
    """
    path = RESOURCE_DATA / "criteo-x1" / "Criteo_x1.zip"
    if not path.exists():
        raise FileNotFoundError(
            f"缺少 Criteo_x1：{path}。请运行 "
            "`python scripts/init_resources.py --include-optional --kind datasets --id criteo-x1-full`"
        )
    dtype = {"label": "int8", **{f"I{index}": "float32" for index in range(1, 14)}, **{f"C{index}": "category" for index in range(1, 27)}}
    frames = []
    with zipfile.ZipFile(path) as bundle:
        for member, expected_rows in CRITEO_SPLITS.items():
            with bundle.open(member) as probe:
                has_header = probe.readline().decode(errors="replace").strip().split(",")[0] == "label"
            with bundle.open(member) as handle:
                frame = pd.read_csv(handle, header=0 if has_header else None, dtype=dtype)
            if list(frame.columns) != CRITEO_COLUMNS:
                if len(frame.columns) != len(CRITEO_COLUMNS):
                    raise ValueError(f"Criteo_x1 {member} 列数 {len(frame.columns)} 与协议 40 不一致")
                frame.columns = CRITEO_COLUMNS
                frame = frame.astype(dtype)
            if len(frame) != expected_rows:
                raise ValueError(f"Criteo_x1 {member} 行数 {len(frame):,} 与官方 {expected_rows:,} 不一致")
            frame.attrs.update(resource_profile="full", resource_path=str(path), split=member.removesuffix(".csv"))
            frames.append(frame)
    return tuple(frames)


def load_criteo() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return _load_criteo_cached()


def criteo_provenance(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame) -> dict:
    return {
        "dataset": "Criteo Display Ads (reczoo Criteo_x1)", "source": "https://huggingface.co/datasets/reczoo/Criteo_x1",
        "profile": "full", "split": "official 7:2:1 train/valid/test files",
        "train_rows": int(len(train)), "valid_rows": int(len(valid)), "test_rows": int(len(test)),
        "features": "13 dense (I1-I13) + 26 categorical (C1-C26)",
        "randomly_fabricated_rows": 0,
    }


RECIF_DIR = RESOURCE_DATA / "openonerec-recif"


@lru_cache(maxsize=1)
def _load_recif_cached() -> tuple[pd.DataFrame, pd.DataFrame]:
    """OpenOneRec RecIF-Bench：官方用户交互 release 与 video/ads 域 pid→Semantic ID 目录。"""
    release = RECIF_DIR / "onerec_bench_release.parquet"
    catalog_path = RECIF_DIR / "video_ad_pid2sid.parquet"
    if not release.exists() or not catalog_path.exists():
        raise FileNotFoundError(
            f"缺少 RecIF-Bench：{RECIF_DIR}。该数据为 gated 资源，请先在 Hugging Face 接受官方条款，"
            "再运行 `HF_TOKEN=... python scripts/init_resources.py --include-optional --kind datasets --id openonerec-recif`"
        )
    users = pd.read_parquet(release)
    catalog = pd.read_parquet(catalog_path)
    if len(users) != 162_074:
        raise ValueError(f"RecIF-Bench release 用户数 {len(users):,} 与官方 162,074 不一致")
    users.attrs.update(resource_profile="full", resource_path=str(release))
    catalog.attrs.update(resource_path=str(catalog_path))
    return users, catalog


def load_recif() -> tuple[pd.DataFrame, pd.DataFrame]:
    users, catalog = _load_recif_cached()
    return users.copy(), catalog.copy()


def recif_provenance(users: pd.DataFrame, catalog: pd.DataFrame) -> dict:
    return {
        "dataset": "OpenOneRec RecIF-Bench (Apache-2.0, gated 授权副本)",
        "source": "https://huggingface.co/datasets/OpenOneRec/OpenOneRec-RecIF",
        "profile": "full", "license": "Apache-2.0（接受官方条款后下载；不二次分发）",
        "users": int(len(users)), "video_ad_items": int(len(catalog)),
        "split": "official release；benchmark_data 提供 8 个任务的官方测试集",
        "randomly_fabricated_rows": 0,
    }


@lru_cache(maxsize=1)
def _load_census_income_cached() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load all official Census-Income train/test rows for MMoE/PLE."""
    path = RESOURCE_DATA / "census-income" / "census-income-kdd.zip"
    if not path.exists():
        raise FileNotFoundError(
            f"缺少 Census-Income KDD：{path}。请运行 "
            "`python scripts/init_resources.py --include-optional --kind datasets --id census-income-kdd`"
        )
    with zipfile.ZipFile(path) as outer:
        archive = io.BytesIO(outer.read("census.tar.gz"))
    with tarfile.open(fileobj=archive, mode="r:gz") as bundle:
        train = pd.read_csv(bundle.extractfile("census-income.data"), header=None, skipinitialspace=True)
        test = pd.read_csv(bundle.extractfile("census-income.test"), header=None, skipinitialspace=True)
    if (len(train), len(test), train.shape[1]) != (199_523, 99_762, 42):
        raise ValueError("Census-Income official split statistics do not match the MMoE paper")
    combined = pd.concat([train, test], ignore_index=True)
    # Education (4) and marital status (7) are targets in the paper and must
    # not leak into the shared input. Column 24 is the instance weight.
    input_columns = [column for column in range(41) if column not in {4, 7, 24}]
    encoded = np.empty((len(combined), len(input_columns)), dtype=np.float32)
    for output_column, source_column in enumerate(input_columns):
        numeric = pd.to_numeric(combined[source_column], errors="coerce")
        if numeric.notna().all():
            scale = float(numeric.std()) or 1.0
            encoded[:, output_column] = ((numeric - numeric.mean()) / scale).astype("float32")
        else:
            codes, uniques = pd.factorize(combined[source_column].astype("string"), sort=True)
            encoded[:, output_column] = codes.astype("float32") / max(1, len(uniques) - 1)
    income = combined[41].astype("string").str.contains(r"50000\+", regex=True).to_numpy(dtype=np.float32)
    never_married = combined[7].astype("string").str.strip().eq("Never married").to_numpy(dtype=np.float32)
    labels = np.c_[income, never_married].astype(np.float32)
    boundary = len(train)
    return encoded[:boundary], labels[:boundary], encoded[boundary:], labels[boundary:]


def load_census_income() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return tuple(value.copy() for value in _load_census_income_cached())


def census_income_provenance() -> dict:
    return {
        "dataset": "Census-Income KDD", "source": "https://archive.ics.uci.edu/dataset/117/census+income+kdd",
        "profile": "full", "train_rows": 199_523, "test_rows": 99_762,
        "slice_rule": "official train file and complete official test file; no row truncation",
        "tasks": ["income > 50K", "never married"], "randomly_fabricated_rows": 0,
    }


@lru_cache(maxsize=8)
def _load_kuairand_cached(max_users: int, max_items: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    archive = RESOURCE_DATA / "kuairand-pure" / "KuaiRand-Pure.tar.gz"
    if _full_profile():
        if not archive.exists():
            command = "python scripts/init_resources.py --include-optional --kind datasets --id kuairand-pure-full"
            raise FileNotFoundError(f"缺少 KuaiRand-Pure 完整资源：{archive}。请运行 `{command}`")
        with tarfile.open(archive, "r:gz") as bundle:
            members = {Path(member.name).name: member for member in bundle.getmembers() if member.isfile()}
            log_names = [name for name in members if name.startswith("log_standard_") and name.endswith("_pure.csv")]
            if not log_names or "video_features_basic_pure.csv" not in members:
                raise ValueError("KuaiRand-Pure archive layout does not match the official release")
            interactions = pd.concat(
                [pd.read_csv(bundle.extractfile(members[name])) for name in sorted(log_names)],
                ignore_index=True,
            )
            videos = pd.read_csv(bundle.extractfile(members["video_features_basic_pure.csv"]))
        frame = interactions.copy()
    else:
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
    frame.attrs["resource_profile"] = "full" if _full_profile() else "smoke"
    frame.attrs["resource_path"] = str(archive if _full_profile() else KUAI_DIR / "standard_interactions.csv")
    video_view = videos[videos.video_id.isin(item_map)].copy()
    video_view["item_id"] = video_view.video_id.map(item_map).astype("int64")
    video_view = video_view.sort_values("item_id").reset_index(drop=True)
    return frame, video_view


def load_kuairand(max_users: int = 128, max_items: int = 2500) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame, videos = _load_kuairand_cached(max_users, max_items)
    return frame.copy(), videos.copy()


def kuairand_provenance(frame: pd.DataFrame) -> dict:
    source = pd.read_json(KUAI_DIR / "provenance.json", typ="series")
    full = frame.attrs.get("resource_profile") == "full"
    return {
        "dataset": KUAIRAND_NAME,
        "source": source["source_url"],
        "source_sha256": source["source_sha256"],
        "license_file": str(KUAI_DIR / "LICENSE"),
        "slice_rule": "all official standard-policy rows; no user/item truncation" if full else source["selection"],
        "profile": "full" if full else "smoke",
        "local_resource": frame.attrs.get("resource_path"),
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
    if not _full_profile() and limit:
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
    if not _full_profile() and limit:
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
