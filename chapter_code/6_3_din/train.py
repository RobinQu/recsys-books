"""Training and evaluation orchestration for 6_3_din.

Shared dataset/metric helpers stay in the common module; the model-specific
tensor construction, loss, optimizer, inference and metrics are visible here.
"""
import numpy as np
import torch
from sklearn.metrics import log_loss
from torch_rechub.basic.features import SequenceFeature, SparseFeature
from torch_rechub.models.ranking import DIEN, DIN
from recsys_lab.data import clicked_sequences, positive_sequences, kuairand_sequence_classification_rows, sequence_classification_rows
from recsys_lab.runtime import (
    ProgressCallback,
    emit_progress,
    real_amazon as _real_amazon,
    real_kuairand as _real_kuairand,
    recall_single_target as _recall_single_target,
    safe_auc as _safe_auc,
    seed_everything,
    training_device,
    train_binary as _train_binary,
    full_profile,
    real_amazon_electronics,
    progress_due,
)

def _run_sequence_ranker(
    kind: str, epochs: int, *, progress: ProgressCallback | None = None,
) -> dict:
    # 1) 固定参数初始化，并读取本章指定的真实数据切片。
    emit_progress(progress, stage="data_prepare", current=0, total=1, message=f"加载并构造 {kind.upper()} 序列")
    seed_everything()
    device = training_device()
    if full_profile():
        interactions, provenance = real_amazon_electronics()
        rows = sequence_classification_rows(interactions, max_len=20, limit=0)
    else:
        interactions, _, provenance = _real_kuairand()
        rows = kuairand_sequence_classification_rows(interactions, max_len=20, limit=2600)
    labels = rows.pop("label"); timestamps = rows.pop("timestamp"); split = int(len(labels) * .8)
    n_users, n_items = interactions.user_id.nunique(), interactions.item_id.nunique()
    user = [SparseFeature("user_id", n_users + 1, 12)]
    item = [SparseFeature("item_id", n_items + 1, 12, padding_idx=0)]
    history = [SequenceFeature("history", n_items + 1, 12, pooling="concat", shared_with="item_id", padding_idx=0)]
    tensors = {name: torch.tensor(value, dtype=torch.long) for name, value in rows.items()}
    if kind == "din":
        # 2) 按论文结构实例化模型；这里是理解层尺寸和特征契约的入口。
        model = DIN(user, history, item, {"dims": [40, 20], "dropout": 0.0}, {"dims": [24, 12], "activation": "prelu", "use_softmax": True}); used = ["user_id", "history", "item_id"]
    else:
        negative = [SequenceFeature("negative_history", n_items + 1, 12, pooling="concat", shared_with="item_id", padding_idx=0)]
        model = DIEN(user, history, negative, item, {"dims": [40, 20], "dropout": 0.0}, alpha=.1); used = list(tensors)
    model.to(device)
    train = {name: tensors[name][:split] for name in used}; y_train = torch.tensor(labels[:split])
    emit_progress(progress, stage="data_prepare", current=1, total=1, metrics={"rows": len(labels)})
    # 4) 公共训练循环执行 forward、二元交叉熵、backward 和 optimizer.step。
    losses = _train_binary(
        model, train, y_train, epochs, .012, dien=kind == "dien", device=device, progress=progress,
    )
    # 5) 关闭梯度进入推理阶段，降低显存占用并防止误更新参数。
    inference_batch = 65_536
    inference_rows = len(labels) - split
    inference_chunks = (inference_rows + inference_batch - 1) // inference_batch
    emit_progress(progress, stage="inference", current=0, total=inference_chunks, message="分批生成序列排序概率")
    tensors = {k: v.to(device) for k, v in tensors.items()}
    model.eval()
    with torch.inference_mode():
        chunks = []
        for chunk_index, start in enumerate(range(split, len(labels), inference_batch), start=1):
            output = model({name: tensors[name][start:start + inference_batch] for name in used})
            chunks.append(output[0] if kind == "dien" else output)
            if progress_due(chunk_index, inference_chunks):
                emit_progress(progress, stage="inference", current=chunk_index, total=inference_chunks)
        probability = torch.cat(chunks).cpu().numpy() if chunks else np.empty(0, dtype=np.float32)
    history_values = rows["history"]
    overlap = (history_values == rows["item_id"][:, None]).mean(1)
    # 6) 返回真实测试切分上的指标和必要诊断信息，供章节总结统一读取。
    emit_progress(progress, stage="baseline", current=0, total=1, message="计算静态重合度基线")
    baseline_auc = _safe_auc(labels[split:], overlap[split:])
    emit_progress(progress, stage="baseline", current=1, total=1, metrics={"auc": baseline_auc})
    emit_progress(progress, stage="evaluate", current=0, total=1, message="计算 AUC 与 LogLoss")
    auc = _safe_auc(labels[split:], probability)
    loss = float(log_loss(labels[split:], np.clip(probability, 1e-6, 1 - 1e-6)))
    emit_progress(progress, stage="evaluate", current=1, total=1, metrics={"auc": auc, "logloss": loss})
    return {
        "framework": f"torch_rechub.models.ranking.{kind.upper()}",
        "dataset": provenance | {"rows": len(labels), "sequence_length": 20, "label": "observed KuaiRand is_click", "negative_history": "observed skipped impressions", "time_ordered": True},
        "loss_curve": losses, "auc": auc,
        "logloss": loss,
        "static_overlap_auc": baseline_auc, "probability_sample": probability[:8].round(3).tolist(),
    }

def run_din(epochs: int = 26, *, progress: ProgressCallback | None = None) -> dict:
    return _run_sequence_ranker("din", epochs, progress=progress)

def train_and_evaluate(epochs: int = 4, *, progress: ProgressCallback | None = None) -> dict:
    """Small default for learning/CI; increase epochs for the full experiment."""
    return run_din(epochs=epochs, progress=progress)
