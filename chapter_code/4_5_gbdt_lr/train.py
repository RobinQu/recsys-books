"""GBDT leaf encoding followed by a separately trained logistic regression."""
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss, roc_auc_score

from .model import build_models
from recsys_lab.data import load_movielens
from recsys_lab.runtime import full_profile, real_criteo

CRITEO_DENSE = [f"I{index}" for index in range(1, 14)]
CRITEO_SPARSE = [f"C{index}" for index in range(1, 27)]


def _criteo_features(train, test) -> tuple[np.ndarray, np.ndarray]:
    """数值列缺失填 0；类别列用 train 的类别表编码，测试集未见类别归为 -1。"""
    x_train = np.zeros((len(train), len(CRITEO_DENSE) + len(CRITEO_SPARSE)), dtype=np.float32)
    x_test = np.zeros_like(x_train, shape=(len(test), x_train.shape[1]))
    for offset, name in enumerate(CRITEO_DENSE):
        x_train[:, offset] = train[name].fillna(0).to_numpy(dtype=np.float32)
        x_test[:, offset] = test[name].fillna(0).to_numpy(dtype=np.float32)
    for offset, name in enumerate(CRITEO_SPARSE, start=len(CRITEO_DENSE)):
        mapping = {value: code for code, value in enumerate(train[name].cat.categories)}
        x_train[:, offset] = train[name].cat.codes.to_numpy(dtype=np.float32)
        x_test[:, offset] = test[name].astype("object").map(mapping).fillna(-1).to_numpy(dtype=np.float32)
    return x_train, x_test


def _fit_gbdt_lr(x_train, y_train, x_test, epochs):
    """论文两阶段链：GBDT 学条件规则 → 叶节点 one-hot → LR 概率校准。"""
    gbdt, lr = build_models(); gbdt.set_params(n_estimators=max(8, epochs))
    gbdt.fit(x_train, y_train)
    encoder = OneHotEncoder(handle_unknown="ignore")
    train_leaf = encoder.fit_transform(gbdt.apply(x_train)[:, :, 0])
    test_leaf = encoder.transform(gbdt.apply(x_test)[:, :, 0])
    lr.fit(train_leaf, y_train)
    return lr.predict_proba(test_leaf)[:, 1]


def train_and_evaluate(epochs: int = 8) -> dict:
    # 严格只用训练段拟合树、叶节点编码器与 LR，避免时间泄漏。
    if full_profile():
        train, _, test, provenance = real_criteo()
        x_train, x_test = _criteo_features(train, test)
        y_train = train.label.to_numpy(dtype=np.float32); y_test = test.label.to_numpy(dtype=np.float32)
        probability = _fit_gbdt_lr(x_train, y_train, x_test, epochs)
        return {
            "dataset": provenance | {"note": "官方 7:2:1 切分；valid 留作调参，本报告只评估 train→test"},
            "randomly_fabricated_rows": 0,
            "auc": float(roc_auc_score(y_test, probability)), "logloss": float(log_loss(y_test, probability)),
        }
    ratings, _ = load_movielens(max_users=80, max_items=360, min_user_events=12)
    frame = ratings.sort_values("timestamp").tail(4200)
    x = frame[["genre_id", "hour", "decade_id", "item_popularity", "user_activity"]].to_numpy()
    y = frame.like.to_numpy(); split = int(len(frame) * .8)
    probability = _fit_gbdt_lr(x[:split], y[:split], x[split:], epochs)
    return {"dataset": "MovieLens latest-small", "randomly_fabricated_rows": 0, "auc": float(roc_auc_score(y[split:], probability)), "logloss": float(log_loss(y[split:], probability))}
