"""Training and evaluation orchestration for 6_2_deepfm.

Shared dataset/metric helpers stay in the common module; the model-specific
tensor construction, loss, optimizer, inference and metrics are visible here.
"""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from torch_rechub.basic.features import SparseFeature
from torch_rechub.models.ranking import DeepFM
from recsys_lab.data import clicked_sequences, positive_sequences, kuairand_sequence_classification_rows
from recsys_lab.runtime import (
    real_amazon as _real_amazon,
    real_kuairand as _real_kuairand,
    recall_single_target as _recall_single_target,
    safe_auc as _safe_auc,
    seed_everything,
    train_binary as _train_binary,
    complete_or_recent,
)

def _ranking_fields(interactions):
    frame = complete_or_recent(interactions, 7000)
    # 6) 返回真实测试切分上的指标和必要诊断信息，供章节总结统一读取。
    return {
        "user_id": frame.user_id.to_numpy() + 1,
        "item_id": frame.item_id.to_numpy() + 1,
        "tab": frame.tab.to_numpy() + 1,
        "hour": frame.hour.to_numpy() + 1,
        "duration_bucket": frame.duration_bucket.to_numpy() + 1,
    }, frame.is_click.to_numpy(dtype=np.float32), frame

def run_deepfm(epochs: int = 28) -> dict:
    # 1) 固定参数初始化，并读取本章指定的真实数据切片。
    seed_everything(); interactions, _, provenance = _real_kuairand(); fields, labels, frame = _ranking_fields(interactions); split = int(len(labels) * .8)
    features = [
        SparseFeature("user_id", int(max(fields["user_id"])) + 1, 12),
        SparseFeature("item_id", int(max(fields["item_id"])) + 1, 12),
        SparseFeature("tab", int(max(fields["tab"])) + 1, 12),
        SparseFeature("hour", 25, 12),
        SparseFeature("duration_bucket", int(max(fields["duration_bucket"])) + 1, 12),
    ]
    # 2) 按论文结构实例化模型；这里是理解层尺寸和特征契约的入口。
    model = DeepFM(features, features, {"dims": [48, 24], "activation": "relu", "dropout": 0.0})
    tensors = {name: torch.tensor(value, dtype=torch.long) for name, value in fields.items()}
    # 4) 公共训练循环执行 forward、二元交叉熵、backward 和 optimizer.step。
    losses = _train_binary(model, {k: v[:split] for k, v in tensors.items()}, torch.tensor(labels[:split]), epochs, .015)
    # 5) 关闭梯度进入推理阶段，降低显存占用并防止误更新参数。
    with torch.no_grad(): probability = model({k: v[split:] for k, v in tensors.items()}).numpy()
    baseline_x = np.c_[frame.tab.to_numpy(), frame.hour.to_numpy(), frame.duration_bucket.to_numpy()]
    baseline = LogisticRegression(max_iter=300).fit(baseline_x[:split], labels[:split])
    baseline_probability = baseline.predict_proba(baseline_x[split:])[:, 1]
    return {
        "framework": "torch_rechub.models.ranking.DeepFM",
        "dataset": provenance | {"rows": len(labels), "train_rows": split, "label": "observed KuaiRand is_click"},
        "loss_curve": losses, "auc": _safe_auc(labels[split:], probability),
        "logloss": float(log_loss(labels[split:], np.clip(probability, 1e-6, 1 - 1e-6))),
        "lr_auc": _safe_auc(labels[split:], baseline_probability), "probability_sample": probability[:8].round(3).tolist(),
    }

def train_and_evaluate(epochs: int = 4) -> dict:
    """Small default for learning/CI; increase epochs for the full experiment."""
    return run_deepfm(epochs=epochs)
