"""GBDT leaf encoding followed by a separately trained logistic regression."""
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss, roc_auc_score

from .model import build_models
from recsys_lab.data import load_movielens

def train_and_evaluate(epochs: int = 8) -> dict:
    # 严格只用训练段拟合树、叶节点编码器与 LR，避免时间泄漏。
    ratings, _ = load_movielens(max_users=80, max_items=360, min_user_events=12)
    frame = ratings.sort_values("timestamp").tail(4200)
    x = frame[["genre_id", "hour", "decade_id", "item_popularity", "user_activity"]].to_numpy()
    y = frame.like.to_numpy(); split = int(len(frame) * .8)
    gbdt, lr = build_models(); gbdt.set_params(n_estimators=max(8, epochs))
    gbdt.fit(x[:split], y[:split])
    encoder = OneHotEncoder(handle_unknown="ignore")
    train_leaf = encoder.fit_transform(gbdt.apply(x[:split])[:, :, 0])
    test_leaf = encoder.transform(gbdt.apply(x[split:])[:, :, 0])
    lr.fit(train_leaf, y[:split]); probability = lr.predict_proba(test_leaf)[:, 1]
    return {"dataset": "MovieLens latest-small", "randomly_fabricated_rows": 0, "auc": float(roc_auc_score(y[split:], probability)), "logloss": float(log_loss(y[split:], probability))}
