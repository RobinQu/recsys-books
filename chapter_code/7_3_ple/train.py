"""Training and evaluation orchestration for 7_3_ple.

Shared dataset/metric helpers stay in the common module; the model-specific
tensor construction, loss, optimizer, inference and metrics are visible here.
"""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch_rechub.basic.features import DenseFeature
from torch_rechub.models.multi_task import MMOE, PLE
from recsys_lab.data import clicked_sequences, positive_sequences, kuairand_sequence_classification_rows
from recsys_lab.runtime import (
    real_amazon as _real_amazon,
    real_kuairand as _real_kuairand,
    recall_single_target as _recall_single_target,
    safe_auc as _safe_auc,
    seed_everything,
    train_binary as _train_binary,
    complete_or_recent,
    real_multitask_dataset as _real_multitask,
)

def _multitask_view(interactions):
    frame = complete_or_recent(interactions, 8000)
    tabs = np.eye(int(interactions.tab.max()) + 1, dtype=np.float32)[frame.tab.to_numpy()]
    duration = np.eye(int(interactions.duration_bucket.max()) + 1, dtype=np.float32)[frame.duration_bucket.to_numpy()]
    continuous = np.c_[
        np.sin(frame.hour.to_numpy() / 24 * 2 * np.pi), np.cos(frame.hour.to_numpy() / 24 * 2 * np.pi),
        np.log1p(frame.item_popularity.to_numpy()) / np.log1p(interactions.item_popularity.max()),
        np.log1p(frame.user_activity.to_numpy()) / np.log1p(interactions.user_activity.max()),
    ].astype(np.float32)
    return np.c_[tabs, duration, continuous].astype(np.float32), np.c_[frame.is_click, frame.long_view].astype(np.float32)

def _run_multitask(kind: str, epochs: int) -> dict:
    # 1) 固定参数初始化，并读取本章指定的真实数据切片。
    seed_everything(); x_train, y_train, x_test, y_test, provenance = _real_multitask()
    x = np.r_[x_train, x_test]; labels = np.r_[y_train, y_test]; split = len(x_train)
    features = [DenseFeature(f"x{i}") for i in range(x.shape[1])]
    expert = {"dims": [24, 12], "activation": "relu", "dropout": 0.0}; towers = [{"dims": [12], "activation": "relu", "dropout": 0.0}] * 2
    # 2) 按论文结构实例化模型；这里是理解层尺寸和特征契约的入口。
    model = MMOE(features, ["classification", "classification"], 4, expert, towers) if kind == "mmoe" else PLE(features, ["classification", "classification"], 2, 2, 2, expert, towers)
    data = {f"x{i}": torch.tensor(x[:, i]) for i in range(x.shape[1])}; target = torch.tensor(labels)
    # 3) optimizer 只在训练阶段更新参数；推理阶段不应再调用它。
    optimizer = torch.optim.Adam(model.parameters(), lr=.012); losses = []
    # 4) 一个 epoch：前向计算 → loss → 梯度清零 → 反向传播 → 参数更新。
    batch_size = 4096
    for _ in range(epochs):
        epoch_loss = 0.0
        for start in range(0, split, batch_size):
            stop = min(start + batch_size, split)
            probability = model({name: value[start:stop] for name, value in data.items()})
            loss = sum(torch.nn.functional.binary_cross_entropy(probability[:, task], target[start:stop, task]) for task in range(2))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_loss += float(loss.detach()) * (stop - start)
        losses.append(epoch_loss / split)
    # 5) 关闭梯度进入推理阶段，降低显存占用并防止误更新参数。
    with torch.no_grad(): probability = torch.cat([model({name: value[start:start + batch_size] for name, value in data.items()}) for start in range(split, len(x), batch_size)]).numpy()
    independent_auc = []
    for task in range(2):
        baseline = LogisticRegression(max_iter=300, solver="liblinear").fit(x[:split], labels[:split, task])
        independent_auc.append(_safe_auc(labels[split:, task], baseline.predict_proba(x[split:])[:, 1]))
    # 6) 返回真实测试切分上的指标和必要诊断信息，供章节总结统一读取。
    return {
        "framework": f"torch_rechub.models.multi_task.{kind.upper()}",
        "dataset": provenance | {"rows": len(x), "train_rows": len(x_train), "test_rows": len(x_test), "features": x.shape[1]},
        "loss_curve": losses, "click_auc": _safe_auc(labels[split:, 0], probability[:, 0]),
        "long_view_auc": _safe_auc(labels[split:, 1], probability[:, 1]),
        "conversion_auc": _safe_auc(labels[split:, 1], probability[:, 1]), "independent_lr_auc": independent_auc,
    }

def run_ple(epochs: int = 28) -> dict: return _run_multitask("ple", epochs)

def train_and_evaluate(epochs: int = 4) -> dict:
    """Small default for learning/CI; increase epochs for the full experiment."""
    return run_ple(epochs=epochs)
