"""Training and evaluation orchestration for 3_2_1_dssm.

Shared dataset/metric helpers stay in the common module; the model-specific
tensor construction, loss, optimizer, inference and metrics are visible here.
"""
import numpy as np
import torch
from torch_rechub.basic.features import SparseFeature
from torch_rechub.models.matching import DSSM
from recsys_lab.data import clicked_sequences, positive_sequences, kuairand_sequence_classification_rows
from recsys_lab.runtime import (
    real_amazon as _real_amazon,
    real_kuairand as _real_kuairand,
    recall_single_target as _recall_single_target,
    safe_auc as _safe_auc,
    seed_everything,
    train_binary as _train_binary,
)

def run_dssm(epochs: int = 24) -> dict:
    # 1) 固定参数初始化，并读取本章指定的真实数据切片。
    seed_everything(); ratings, provenance = _real_amazon()
    events = ratings.sort_values("timestamp").tail(5200).reset_index(drop=True)
    split = int(len(events) * .8)
    train, test = events.iloc[:split], events.iloc[split:]
    n_users, n_items = ratings.user_id.nunique(), ratings.item_id.nunique()
    user_features = [SparseFeature("user_id", n_users + 1, 16, padding_idx=0)]
    item_features = [SparseFeature("item_id", n_items + 1, 16, padding_idx=0)]
    tower = {"dims": [32, 16], "activation": "relu", "dropout": 0.0}
    # 2) 按论文结构实例化模型；这里是理解层尺寸和特征契约的入口。
    model = DSSM(user_features, item_features, tower, tower, temperature=.12)

    def fields(frame):
        return {
            "user_id": torch.tensor(frame.user_id.to_numpy() + 1, dtype=torch.long),
            "item_id": torch.tensor(frame.item_id.to_numpy() + 1, dtype=torch.long),
        }

    # 4) 公共训练循环执行 forward、二元交叉熵、backward 和 optimizer.step。
    losses = _train_binary(model, fields(train), torch.tensor(train.like.to_numpy()), epochs, .004)
    # 5) 关闭梯度进入推理阶段，降低显存占用并防止误更新参数。
    with torch.no_grad():
        probability = model(fields(test)).numpy()
        model.mode = "user"
        user_embedding = model({"user_id": torch.arange(1, n_users + 1)}).numpy()
        model.mode = "item"
        item_embedding = model({"item_id": torch.arange(1, n_items + 1)}).numpy()
        model.mode = None
    user_embedding = np.nan_to_num(user_embedding); item_embedding = np.nan_to_num(item_embedding)
    user_embedding /= np.linalg.norm(user_embedding, axis=1, keepdims=True) + 1e-8
    item_embedding /= np.linalg.norm(item_embedding, axis=1, keepdims=True) + 1e-8
    scores = user_embedding @ item_embedding.T
    targets, seen, valid_users = [], [], []
    for user, frame in ratings.sort_values("timestamp").groupby("user_id"):
        positives = frame[frame.rating >= 4.0].item_id.tolist()
        if len(positives) >= 2:
            valid_users.append(int(user)); targets.append(int(positives[-1])); seen.append(set(map(int, positives[:-1])))
    recall = _recall_single_target(scores[valid_users], np.asarray(targets), 10, seen)
    # 6) 返回真实测试切分上的指标和必要诊断信息，供章节总结统一读取。
    return {
        "framework": "torch_rechub.models.matching.DSSM",
        "dataset": provenance | {"train_rows": len(train), "test_rows": len(test), "label": "observed Amazon rating >= 4.0"},
        "loss_curve": losses,
        "test_auc": _safe_auc(test.like.to_numpy(), probability),
        "recall@10": recall,
        "embedding_shape": [list(user_embedding.shape), list(item_embedding.shape)],
        "score_sample": scores[:3, :8].round(3).tolist(),
    }

def train_and_evaluate(epochs: int = 4) -> dict:
    """Small default for learning/CI; increase epochs for the full experiment."""
    return run_dssm(epochs=epochs)
