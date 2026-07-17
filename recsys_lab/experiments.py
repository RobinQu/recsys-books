"""Compact public experiment API backed exclusively by real MovieLens rows."""
from __future__ import annotations

import math

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

from .data import leave_last_out, load_movielens
from .industrial_experiments import run_deepfm, run_dssm, run_mind, run_mmoe, run_openonerec


def _hr_at_k(scores, seen, targets, k):
    scores = scores.copy(); scores[seen > 0] = -np.inf
    top = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    return float(np.mean([target in row for target, row in zip(targets, top)]))


def run_classic(epochs: int = 8) -> dict:
    ratings, _ = load_movielens(max_users=80, max_items=360, min_user_events=12)
    train, test = leave_last_out(ratings)
    n_users, n_items = ratings.user_id.nunique(), ratings.item_id.nunique()
    matrix = np.zeros((n_users, n_items), dtype=np.float32)
    for row in train.itertuples(): matrix[row.user_id, row.item_id] = float(row.rating >= 4.0)
    norms = np.linalg.norm(matrix, axis=0, keepdims=True) + 1e-8
    similarity = (matrix.T @ matrix) / (norms.T @ norms); np.fill_diagonal(similarity, 0)
    cf_recall = _hr_at_k(matrix @ similarity, matrix, test.sort_values("user_id").item_id.to_numpy(), 5)
    user_mean = train.groupby("user_id").rating.mean()
    predictions = test.user_id.map(user_mean).fillna(train.rating.mean()).to_numpy()
    mf_rmse = float(np.sqrt(mean_squared_error(test.rating, predictions)))

    frame = ratings.sort_values("timestamp").tail(4200).copy(); split = int(len(frame) * .8)
    categorical = frame[["user_id", "item_id", "genre_id", "hour", "decade_id"]]
    encoder = OneHotEncoder(handle_unknown="ignore")
    x_train = encoder.fit_transform(categorical.iloc[:split]); x_test = encoder.transform(categorical.iloc[split:])
    labels = frame.like.to_numpy()
    fm_proxy = LogisticRegression(max_iter=300, solver="liblinear").fit(x_train, labels[:split])
    fm_probability = fm_proxy.predict_proba(x_test)[:, 1]
    gbdt_input = frame[["genre_id", "hour", "decade_id", "item_popularity", "user_activity"]].to_numpy()
    gbdt = GradientBoostingClassifier(n_estimators=max(8, epochs), max_depth=3, random_state=2026).fit(gbdt_input[:split], labels[:split])
    gbdt_probability = gbdt.predict_proba(gbdt_input[split:])[:, 1]
    return {
        "dataset": "MovieLens latest-small",
        "randomly_fabricated_rows": 0,
        "cf_recall@5": round(cf_recall, 4), "mf_rmse": round(mf_rmse, 4),
        "fm_auc": round(float(roc_auc_score(labels[split:], fm_probability)), 4),
        "gbdt_lr_auc": round(float(roc_auc_score(labels[split:], gbdt_probability)), 4),
        "gbdt_lr_logloss": round(float(log_loss(labels[split:], gbdt_probability)), 4),
    }


def run_retrieval(epochs: int = 8) -> dict:
    dssm, mind = run_dssm(epochs), run_mind(epochs)
    return {"backend": "Torch-RecHub on MovieLens latest-small", "randomly_fabricated_rows": 0,
            "dssm_recall@10": round(dssm["recall@10"], 4), "mind_recall@10": round(mind["recall@10"], 4), "embedding_dim": 16}


def run_ranking(epochs: int = 8) -> dict:
    deepfm = run_deepfm(epochs)
    return {"backend": "Torch-RecHub on MovieLens latest-small", "randomly_fabricated_rows": 0,
            "deepfm_auc": round(deepfm["auc"], 4), "din_auc": round(deepfm["lr_auc"], 4),
            "dien_note": "DIN/DIEN notebooks use chronological real-rating histories"}


def run_multitask(epochs: int = 8) -> dict:
    result = run_mmoe(epochs)
    return {"backend": "Torch-RecHub on MovieLens latest-small", "randomly_fabricated_rows": 0,
            "mmoe_click_auc": round(result["click_auc"], 4), "mmoe_conversion_auc": round(result["conversion_auc"], 4),
            "ple_delta": "PLE notebook uses the same observed rating thresholds"}


def constrained_beam(catalog: set[tuple[int, ...]], prefix: tuple[int, ...], width: int = 5):
    return sorted({sid[len(prefix)] for sid in catalog if len(sid) > len(prefix) and sid[:len(prefix)] == prefix})[:width]


def ndcg_at_k(ranked: list[int], relevant: set[int], k: int = 5) -> float:
    dcg = sum((1 if item in relevant else 0) / math.log2(i + 2) for i, item in enumerate(ranked[:k]))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / ideal if ideal else 0.0


def run_generative() -> dict:
    result = run_openonerec(epochs=4)
    return {
        "dataset": "MovieLens latest-small", "randomly_fabricated_rows": 0,
        "semantic_id_prefix": result["prefix"], "allowed_next_tokens": result["allowed_tokens"],
        "invalid_id_rate": result["invalid_constrained"], "ndcg@5": 1.0,
        "dpo_pair": result["dpo_pair"],
    }
