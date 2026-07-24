"""Cross-chapter runtime utilities.

Only concerns genuinely reused by several algorithms live here: deterministic
seeding, real-dataset adapters, generic binary optimization, metrics and result
serialization. Model construction and algorithm-specific tensor preparation
belong to ``chapter_code/<chapter>/``.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .data import (
    amazon_2018_provenance, amazon_provenance, kuairand_provenance,
    load_amazon_2018, load_amazon_2023, load_kuairand, load_movielens_1m,
    load_mind_amazon_books, mind_amazon_provenance, movielens_1m_provenance,
    load_census_income, census_income_provenance,
    load_criteo, criteo_provenance,
    load_movielens_20m, movielens_20m_provenance,
    load_recif, recif_provenance,
)


ROOT = Path(__file__).resolve().parents[1]
ProgressCallback = Callable[[dict[str, object]], None]


def emit_progress(
    progress: ProgressCallback | None,
    *,
    stage: str,
    current: int | None = None,
    total: int | None = None,
    message: str | None = None,
    metrics: Mapping[str, object] | None = None,
) -> None:
    """Send one deterministic progress event when a callback is configured."""
    if progress is None:
        return
    event: dict[str, object] = {"stage": stage}
    if current is not None:
        event["current"] = current
    if total is not None:
        event["total"] = total
    if message is not None:
        event["message"] = message
    if metrics is not None:
        event["metrics"] = dict(metrics)
    progress(event)


def progress_due(current: int, total: int, max_updates: int = 20) -> bool:
    """Return whether a bounded loop should publish its current progress.

    Boundaries are always reported. Between them, at most roughly
    ``max_updates`` evenly spaced checkpoints are selected.
    """
    if max_updates <= 0:
        raise ValueError("max_updates must be positive")
    if current <= 0 or current >= total:
        return True
    if total <= 0:
        return False
    interval = max(1, (total + max_updates - 1) // max_updates)
    return current % interval == 0


def print_progress(event: dict[str, object]) -> None:
    """Print a stable single-line representation suitable for notebooks."""
    stage = str(event.get("stage", "progress")).replace("\n", " ")
    parts = [f"[{stage}]"]
    current, total = event.get("current"), event.get("total")
    if current is not None and total is not None:
        parts.append(f"{current}/{total}")
    elif current is not None:
        parts.append(str(current))
    message = event.get("message")
    if message is not None:
        parts.append(str(message).replace("\n", " "))
    metrics = event.get("metrics")
    if isinstance(metrics, Mapping):
        parts.extend(f"{key}={_format_progress_value(metrics[key])}" for key in sorted(metrics))
    print(" ".join(parts), flush=True)


def _format_progress_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("\n", " ")


def full_profile() -> bool:
    """True only for paper-comparison runs over complete initialized data."""
    return os.getenv("RECSYS_PROFILE", "smoke").casefold() == "full"


def complete_or_recent(frame, smoke_rows: int):
    """Never truncate formal runs; keep a deterministic recent slice for CI only."""
    return frame if full_profile() else frame.sort_values("timestamp").tail(smoke_rows).copy()


def seed_everything(seed: int = 2026) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if not full_profile():
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        return
    configured = os.getenv("RECSYS_TORCH_THREADS")
    if configured is not None:
        try:
            thread_count = int(configured)
        except ValueError as error:
            raise ValueError("RECSYS_TORCH_THREADS must be a positive integer") from error
        if thread_count <= 0:
            raise ValueError("RECSYS_TORCH_THREADS must be a positive integer")
    else:
        thread_count = min(8, os.cpu_count() or 1)
    torch.set_num_threads(thread_count)


def training_device() -> torch.device:
    """Use CUDA when available; fall back to CPU otherwise."""
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        return torch.device("cuda")
    return torch.device("cpu")


def save_records(chapter: str, name: str, records: list[dict]) -> Path:
    artifact_root = Path(os.environ.get("RECSYS_ARTIFACT_ROOT", ROOT)).expanduser().resolve()
    path = artifact_root / "results" / chapter / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else 0.5


def recall_single_target(
    scores: np.ndarray,
    targets: np.ndarray,
    k: int,
    seen: list[set[int]] | None = None,
) -> float:
    scores = scores.copy()
    if seen:
        for row, items in enumerate(seen):
            if items:
                scores[row, list(items)] = -np.inf
    topk = np.argpartition(-scores, kth=min(k, scores.shape[1]) - 1, axis=1)[:, :k]
    return float(np.mean([target in candidates for target, candidates in zip(targets, topk)]))


def sampled_embedding_rank_metrics(
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    targets: np.ndarray,
    k: int = 10,
    negatives: int = 100,
    seen: list[set[int]] | None = None,
    seed: int = 2026,
) -> dict[str, float | int]:
    """Evaluate every test user against one target and deterministic sampled negatives.

    ``user_embeddings`` may be ``[U,d]`` or ``[U,K,d]``.  The latter uses the
    best matching interest, matching MIND's multi-vector retrieval contract.
    """
    rng = np.random.default_rng(seed)
    hits = 0
    ndcg = 0.0
    item_count = len(item_embeddings)
    for row, target in enumerate(targets.astype(int)):
        excluded = set() if seen is None else set(seen[row])
        excluded.add(target)
        candidates: list[int] = []
        while len(candidates) < min(negatives, item_count - len(excluded)):
            proposal = int(rng.integers(0, item_count))
            if proposal not in excluded:
                excluded.add(proposal)
                candidates.append(proposal)
        candidate_ids = np.asarray([target, *candidates], dtype=np.int64)
        candidate_vectors = item_embeddings[candidate_ids]
        user = user_embeddings[row]
        values = user @ candidate_vectors.T
        if values.ndim == 2:
            values = values.max(axis=0)
        rank = int(np.sum(values[1:] > values[0])) + 1
        if rank <= k:
            hits += 1
            ndcg += 1.0 / np.log2(rank + 1)
    users = len(targets)
    return {f"hr@{k}": hits / users, f"ndcg@{k}": ndcg / users,
            "evaluation_users": users, "sampled_negatives": negatives}


def train_binary(
    model: torch.nn.Module,
    data: dict[str, torch.Tensor],
    labels: torch.Tensor,
    epochs: int,
    lr: float,
    dien: bool = False,
    batch_size: int = 4096,
    *,
    device: torch.device | None = None,
    progress: ProgressCallback | None = None,
) -> list[float]:
    """Mini-batch BCE loop that can consume complete public datasets."""
    if device is not None:
        model.to(device)
        data = {name: value.to(device) for name, value in data.items()}
        labels = labels.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []
    steps_per_epoch = (len(labels) + batch_size - 1) // batch_size
    total_steps = epochs * steps_per_epoch
    emit_progress(progress, stage="train", current=0, total=total_steps, message="starting")
    step = 0
    for epoch in range(epochs):
        weighted_loss = 0.0
        for start in range(0, len(labels), batch_size):
            stop = min(start + batch_size, len(labels))
            output = model({name: value[start:stop] for name, value in data.items()})
            if dien:
                probability, auxiliary_loss = output
                loss = torch.nn.functional.binary_cross_entropy(probability, labels[start:stop]) + auxiliary_loss
            else:
                probability = output
                loss = torch.nn.functional.binary_cross_entropy(probability, labels[start:stop])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            weighted_loss += float(loss.detach()) * (stop - start)
            step += 1
            if step < total_steps and progress_due(step, total_steps):
                emit_progress(
                    progress,
                    stage="train",
                    current=step,
                    total=total_steps,
                    message=f"epoch {epoch + 1}/{epochs}",
                    metrics={"batch_loss": float(loss.detach())},
                )
        losses.append(weighted_loss / len(labels))
    if total_steps:
        emit_progress(
            progress,
            stage="train",
            current=total_steps,
            total=total_steps,
            message=f"epoch {epochs}/{epochs}",
            metrics={"loss": losses[-1]},
        )
    return losses


def real_amazon(max_users: int = 128):
    ratings = load_amazon_2023(max_users=max_users, min_user_events=12)
    return ratings, amazon_provenance(ratings)


def real_amazon_books(max_users: int = 128):
    if full_profile():
        ratings = load_amazon_2018("Books")
        return ratings, amazon_2018_provenance(ratings)
    return real_amazon(max_users=max_users)


def real_mind_amazon_books(max_users: int = 128):
    if full_profile():
        ratings = load_mind_amazon_books()
        return ratings, mind_amazon_provenance(ratings)
    return real_amazon(max_users=max_users)


def real_amazon_electronics(max_users: int = 128):
    if full_profile():
        ratings = load_amazon_2018("Electronics")
        return ratings, amazon_2018_provenance(ratings)
    return real_amazon(max_users=max_users)


def real_sasrec_dataset(max_users: int = 160):
    if full_profile():
        ratings = load_movielens_1m()
        return ratings, movielens_1m_provenance(ratings)
    return real_amazon(max_users=max_users)


def real_kuairand(max_users: int = 96, max_items: int = 2200):
    interactions, videos = load_kuairand(max_users=max_users, max_items=max_items)
    return interactions, videos, kuairand_provenance(interactions)


def real_multitask_dataset(max_users: int = 96, max_items: int = 2200):
    if full_profile():
        x_train, y_train, x_test, y_test = load_census_income()
        return x_train, y_train, x_test, y_test, census_income_provenance()
    interactions, _, provenance = real_kuairand(max_users=max_users, max_items=max_items)
    frame = complete_or_recent(interactions, 8000)
    tabs = np.eye(int(interactions.tab.max()) + 1, dtype=np.float32)[frame.tab.to_numpy()]
    duration = np.eye(int(interactions.duration_bucket.max()) + 1, dtype=np.float32)[frame.duration_bucket.to_numpy()]
    continuous = np.c_[
        np.sin(frame.hour.to_numpy() / 24 * 2 * np.pi), np.cos(frame.hour.to_numpy() / 24 * 2 * np.pi),
        np.log1p(frame.item_popularity.to_numpy()) / np.log1p(interactions.item_popularity.max()),
        np.log1p(frame.user_activity.to_numpy()) / np.log1p(interactions.user_activity.max()),
    ].astype(np.float32)
    x = np.c_[tabs, duration, continuous].astype(np.float32)
    labels = np.c_[frame.is_click, frame.long_view].astype(np.float32)
    split = int(len(x) * .8)
    return x[:split], labels[:split], x[split:], labels[split:], provenance


def real_criteo():
    """CTR 论文同任务公开数据：full 档返回官方 7:2:1 切分与 provenance。"""
    train, valid, test = load_criteo()
    return train, valid, test, criteo_provenance(train, valid, test)


def real_hstu_dataset(max_users: int = 128, max_items: int = 2500):
    """HSTU：full 档切换到论文公开基准 MovieLens 20M；smoke 沿用 KuaiRand 序列。"""
    if full_profile():
        ratings = load_movielens_20m()
        return ratings, None, movielens_20m_provenance(ratings)
    return real_kuairand(max_users=max_users, max_items=max_items)


def real_openonerec_dataset(max_users: int = 96, max_items: int = 2200):
    """OpenOneRec：full 档读取授权的 RecIF-Bench 本地副本；smoke 沿用 KuaiRand 适配器。"""
    if full_profile():
        users, catalog = load_recif()
        return users, catalog, recif_provenance(users, catalog)
    return real_kuairand(max_users=max_users, max_items=max_items)
