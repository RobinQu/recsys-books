"""Cross-chapter runtime utilities.

Only concerns genuinely reused by several algorithms live here: deterministic
seeding, real-dataset adapters, generic binary optimization, metrics and result
serialization. Model construction and algorithm-specific tensor preparation
belong to ``chapter_code/<chapter>/``.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .data import amazon_provenance, kuairand_provenance, load_amazon_2023, load_kuairand


ROOT = Path(__file__).resolve().parents[1]


def seed_everything(seed: int = 2026) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def save_records(chapter: str, name: str, records: list[dict]) -> Path:
    path = ROOT / "results" / chapter / f"{name}.json"
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


def train_binary(
    model: torch.nn.Module,
    data: dict[str, torch.Tensor],
    labels: torch.Tensor,
    epochs: int,
    lr: float,
    dien: bool = False,
) -> list[float]:
    """Generic BCE loop shared by DSSM and several CTR models."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []
    for _ in range(epochs):
        output = model(data)
        if dien:
            probability, auxiliary_loss = output
            loss = torch.nn.functional.binary_cross_entropy(probability, labels) + auxiliary_loss
        else:
            probability = output
            loss = torch.nn.functional.binary_cross_entropy(probability, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def real_amazon(max_users: int = 128):
    ratings = load_amazon_2023(max_users=max_users, min_user_events=12)
    return ratings, amazon_provenance(ratings)


def real_kuairand(max_users: int = 96, max_items: int = 2200):
    interactions, videos = load_kuairand(max_users=max_users, max_items=max_items)
    return interactions, videos, kuairand_provenance(interactions)
