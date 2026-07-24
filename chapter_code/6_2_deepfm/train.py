"""Training and evaluation orchestration for 6_2_deepfm.

Shared dataset/metric helpers stay in the common module; the model-specific
tensor construction, loss, optimizer, inference and metrics are visible here.
"""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.models.ranking import DeepFM
from recsys_lab.data import clicked_sequences, positive_sequences, kuairand_sequence_classification_rows
from recsys_lab.runtime import (
    ProgressCallback,
    emit_progress,
    real_amazon as _real_amazon,
    real_criteo as _real_criteo,
    real_kuairand as _real_kuairand,
    recall_single_target as _recall_single_target,
    safe_auc as _safe_auc,
    seed_everything,
    train_binary as _train_binary,
    complete_or_recent,
    full_profile,
    progress_due,
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

def _run_criteo_deepfm(epochs: int, *, progress: ProgressCallback | None = None) -> dict:
    """full 档论文协议：Criteo_x1 官方 7:2:1 切分，13 数值 + 26 类别特征。

    45.8M 行的完整训练是 dispatch 级任务，需要大内存机器；类别词表只从
    train 构建，测试集未见类别编码为 0。
    """
    emit_progress(progress, stage="data_prepare", current=0, total=1, message="加载并编码 Criteo_x1 官方切分")
    train, _, test, provenance = _real_criteo()
    dense = [f"I{index}" for index in range(1, 14)]
    sparse_names = [f"C{index}" for index in range(1, 27)]
    features = [DenseFeature(name) for name in dense]
    train_x, test_x = {}, {}
    for name in sparse_names:
        categories = train[name].cat.categories
        features.append(SparseFeature(name, len(categories) + 1, 16))
        mapping = {value: code + 1 for code, value in enumerate(categories)}
        train_x[name] = torch.tensor(train[name].cat.codes.to_numpy(dtype=np.int64) + 1, dtype=torch.long)
        test_x[name] = torch.tensor(test[name].astype("object").map(mapping).fillna(0).to_numpy(dtype=np.int64), dtype=torch.long)
    for name in dense:
        train_x[name] = torch.tensor(train[name].fillna(0).to_numpy(dtype=np.float32))
        test_x[name] = torch.tensor(test[name].fillna(0).to_numpy(dtype=np.float32))
    labels_train = torch.tensor(train.label.to_numpy(dtype=np.float32))
    labels_test = test.label.to_numpy(dtype=np.float32)
    emit_progress(progress, stage="data_prepare", current=1, total=1, metrics={"train_rows": len(train), "test_rows": len(test)})
    # 2) 按论文结构实例化模型；这里是理解层尺寸和特征契约的入口。
    model = DeepFM(features, features, {"dims": [64, 32], "activation": "relu", "dropout": 0.0})
    # 4) 公共训练循环执行 forward、二元交叉熵、backward 和 optimizer.step。
    losses = _train_binary(
        model, train_x, labels_train, epochs, .001, batch_size=100_000, progress=progress,
    )
    # 5) 关闭梯度进入推理阶段，降低显存占用并防止误更新参数。
    inference_batch = 500_000
    inference_chunks = (len(labels_test) + inference_batch - 1) // inference_batch
    emit_progress(progress, stage="inference", current=0, total=inference_chunks, message="分批生成 Criteo 测试概率")
    model.eval()
    with torch.inference_mode():
        probability_chunks = []
        for chunk_index, start in enumerate(range(0, len(labels_test), inference_batch), start=1):
            probability_chunks.append(model({name: value[start:start + inference_batch] for name, value in test_x.items()}))
            if progress_due(chunk_index, inference_chunks):
                emit_progress(progress, stage="inference", current=chunk_index, total=inference_chunks)
        probability = torch.cat(probability_chunks).numpy()
    # LR 基线只用前 200 万训练行拟合（口径写入 provenance），与论文的 LR 对照保持同任务。
    emit_progress(progress, stage="baseline", current=0, total=1, message="训练 LR 基线")
    baseline = LogisticRegression(max_iter=100).fit(
        train[dense].fillna(0).to_numpy(dtype=np.float32)[:2_000_000], train.label.to_numpy()[:2_000_000],
    )
    baseline_probability = baseline.predict_proba(test[dense].fillna(0).to_numpy(dtype=np.float32))[:, 1]
    emit_progress(progress, stage="baseline", current=1, total=1)
    emit_progress(progress, stage="evaluate", current=0, total=1, message="计算 AUC 与 LogLoss")
    auc = _safe_auc(labels_test, probability)
    loss = float(log_loss(labels_test, np.clip(probability, 1e-6, 1 - 1e-6)))
    baseline_auc = _safe_auc(labels_test, baseline_probability)
    emit_progress(progress, stage="evaluate", current=1, total=1, metrics={"auc": auc, "logloss": loss})
    return {
        "framework": "torch_rechub.models.ranking.DeepFM",
        "dataset": provenance | {"label": "observed Criteo click label", "lr_baseline": "first 2M train rows"},
        "loss_curve": losses, "auc": auc,
        "logloss": loss,
        "lr_auc": baseline_auc, "probability_sample": probability[:8].round(3).tolist(),
    }


def run_deepfm(epochs: int = 28, *, progress: ProgressCallback | None = None) -> dict:
    # 1) 固定参数初始化，并读取本章指定的真实数据切片。
    seed_everything()
    if full_profile():
        return _run_criteo_deepfm(epochs, progress=progress)
    emit_progress(progress, stage="data_prepare", current=0, total=1, message="加载 KuaiRand 排序数据")
    interactions, _, provenance = _real_kuairand(); fields, labels, frame = _ranking_fields(interactions); split = int(len(labels) * .8)
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
    emit_progress(progress, stage="data_prepare", current=1, total=1, metrics={"rows": len(labels)})
    # 4) 公共训练循环执行 forward、二元交叉熵、backward 和 optimizer.step。
    losses = _train_binary(
        model, {k: v[:split] for k, v in tensors.items()}, torch.tensor(labels[:split]), epochs, .015,
        progress=progress,
    )
    # 5) 关闭梯度进入推理阶段，降低显存占用并防止误更新参数。
    inference_batch = 131_072
    inference_chunks = (len(labels) - split + inference_batch - 1) // inference_batch
    emit_progress(progress, stage="inference", current=0, total=inference_chunks, message="分批生成测试概率")
    model.eval()
    with torch.inference_mode():
        chunks = []
        for chunk_index, start in enumerate(range(split, len(labels), inference_batch), start=1):
            chunks.append(model({k: v[start:start + inference_batch] for k, v in tensors.items()}))
            if progress_due(chunk_index, inference_chunks):
                emit_progress(progress, stage="inference", current=chunk_index, total=inference_chunks)
        probability = torch.cat(chunks).numpy()
    baseline_x = np.c_[frame.tab.to_numpy(), frame.hour.to_numpy(), frame.duration_bucket.to_numpy()]
    emit_progress(progress, stage="baseline", current=0, total=1, message="训练独立 LR 基线")
    baseline = LogisticRegression(max_iter=300).fit(baseline_x[:split], labels[:split])
    baseline_probability = baseline.predict_proba(baseline_x[split:])[:, 1]
    emit_progress(progress, stage="baseline", current=1, total=1)
    emit_progress(progress, stage="evaluate", current=0, total=1, message="计算 AUC 与 LogLoss")
    auc = _safe_auc(labels[split:], probability)
    loss = float(log_loss(labels[split:], np.clip(probability, 1e-6, 1 - 1e-6)))
    baseline_auc = _safe_auc(labels[split:], baseline_probability)
    emit_progress(progress, stage="evaluate", current=1, total=1, metrics={"auc": auc, "logloss": loss})
    return {
        "framework": "torch_rechub.models.ranking.DeepFM",
        "dataset": provenance | {"rows": len(labels), "train_rows": split, "label": "observed KuaiRand is_click"},
        "loss_curve": losses, "auc": auc,
        "logloss": loss,
        "lr_auc": baseline_auc, "probability_sample": probability[:8].round(3).tolist(),
    }

def train_and_evaluate(epochs: int = 4, *, progress: ProgressCallback | None = None) -> dict:
    """Small default for learning/CI; increase epochs for the full experiment."""
    return run_deepfm(epochs=epochs, progress=progress)
