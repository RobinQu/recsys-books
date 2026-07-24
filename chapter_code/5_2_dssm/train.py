"""Training and evaluation orchestration for 5_2_dssm.

Shared dataset/metric helpers stay in the common module; the model-specific
tensor construction, loss, optimizer, inference and metrics are visible here.
"""
import numpy as np
import torch
from torch_rechub.basic.features import SparseFeature
from torch_rechub.models.matching import DSSM
from recsys_lab.data import clicked_sequences, positive_sequences, kuairand_sequence_classification_rows
from recsys_lab.runtime import (
    ProgressCallback,
    emit_progress,
    real_amazon_books as _real_amazon,
    real_kuairand as _real_kuairand,
    recall_single_target as _recall_single_target,
    safe_auc as _safe_auc,
    seed_everything,
    training_device,
    train_binary as _train_binary,
    complete_or_recent,
    full_profile,
    sampled_embedding_rank_metrics,
    progress_due,
)

def run_dssm(epochs: int = 24, *, progress: ProgressCallback | None = None) -> dict:
    # 1) 固定参数初始化，并读取本章指定的真实数据切片。
    emit_progress(progress, stage="data_prepare", current=0, total=1, message="加载并切分 DSSM 数据")
    seed_everything(); device = training_device(); ratings, provenance = _real_amazon()
    events = complete_or_recent(ratings, 5200).reset_index(drop=True)
    split = int(len(events) * .8)
    train, test = events.iloc[:split], events.iloc[split:]
    n_users, n_items = ratings.user_id.nunique(), ratings.item_id.nunique()
    user_features = [SparseFeature("user_id", n_users + 1, 16, padding_idx=0)]
    item_features = [SparseFeature("item_id", n_items + 1, 16, padding_idx=0)]
    tower = {"dims": [32, 16], "activation": "relu", "dropout": 0.0}
    # 2) 按论文结构实例化模型；这里是理解层尺寸和特征契约的入口。
    model = DSSM(user_features, item_features, tower, tower, temperature=.12)
    model.to(device)

    def fields(frame):
        return {
            "user_id": torch.tensor(frame.user_id.to_numpy() + 1, dtype=torch.long),
            "item_id": torch.tensor(frame.item_id.to_numpy() + 1, dtype=torch.long),
        }

    # 4) 公共训练循环执行 forward、二元交叉熵、backward 和 optimizer.step。
    emit_progress(progress, stage="data_prepare", current=1, total=1, metrics={"train_rows": len(train), "test_rows": len(test)})
    losses = _train_binary(
        model, fields(train), torch.tensor(train.like.to_numpy()), epochs, .004, device=device, progress=progress,
    )
    # 5) 关闭梯度进入推理阶段，降低显存占用并防止误更新参数。
    inference_batch = 131_072
    test_fields = fields(test)
    test_fields = {k: v.to(device) for k, v in test_fields.items()}
    test_chunks = (len(test) + inference_batch - 1) // inference_batch
    user_chunks = (n_users + inference_batch - 1) // inference_batch
    item_chunks = (n_items + inference_batch - 1) // inference_batch
    total_chunks = test_chunks + user_chunks + item_chunks
    completed = 0
    emit_progress(progress, stage="inference", current=0, total=total_chunks, message="分批生成概率与塔向量")
    model.eval()
    with torch.inference_mode():
        probability_chunks = []
        for start in range(0, len(test), inference_batch):
            probability_chunks.append(model({name: value[start:start + inference_batch] for name, value in test_fields.items()}))
            completed += 1
            if progress_due(completed, total_chunks):
                emit_progress(progress, stage="inference", current=completed, total=total_chunks)
        probability = torch.cat(probability_chunks).cpu().numpy() if probability_chunks else np.empty(0, dtype=np.float32)
        model.mode = "user"
        user_chunks_out = []
        for start in range(1, n_users + 1, inference_batch):
            user_chunks_out.append(model({"user_id": torch.arange(start, min(start + inference_batch, n_users + 1)).to(device)}))
            completed += 1
            if progress_due(completed, total_chunks):
                emit_progress(progress, stage="inference", current=completed, total=total_chunks)
        user_embedding = torch.cat(user_chunks_out).cpu().numpy() if user_chunks_out else np.empty((0, 16), dtype=np.float32)
        model.mode = "item"
        item_chunks_out = []
        for start in range(1, n_items + 1, inference_batch):
            item_chunks_out.append(model({"item_id": torch.arange(start, min(start + inference_batch, n_items + 1)).to(device)}))
            completed += 1
            if progress_due(completed, total_chunks):
                emit_progress(progress, stage="inference", current=completed, total=total_chunks)
        item_embedding = torch.cat(item_chunks_out).cpu().numpy() if item_chunks_out else np.empty((0, 16), dtype=np.float32)
        model.mode = None
    user_embedding = np.nan_to_num(user_embedding, nan=0.0, posinf=0.0, neginf=0.0)
    item_embedding = np.nan_to_num(item_embedding, nan=0.0, posinf=0.0, neginf=0.0)
    user_embedding /= np.linalg.norm(user_embedding, axis=1, keepdims=True) + 1e-8
    item_embedding /= np.linalg.norm(item_embedding, axis=1, keepdims=True) + 1e-8
    targets, seen, valid_users = [], [], []
    for user, frame in ratings.sort_values("timestamp").groupby("user_id"):
        positives = frame[frame.rating >= 4.0].item_id.tolist()
        if len(positives) >= 2:
            valid_users.append(int(user)); targets.append(int(positives[-1])); seen.append(set(map(int, positives[:-1])))
    target_array = np.asarray(targets)
    emit_progress(progress, stage="evaluate", current=0, total=1, message="计算采样排序指标")
    sampled = sampled_embedding_rank_metrics(user_embedding[valid_users], item_embedding, target_array, 10, 100, seen)
    if full_profile():
        recall = float(sampled["hr@10"])
        score_sample = user_embedding[:3] @ item_embedding[:8].T
    else:
        scores = np.clip(user_embedding, -1.0, 1.0) @ np.clip(item_embedding, -1.0, 1.0).T
        recall = _recall_single_target(scores[valid_users], target_array, 10, seen)
        score_sample = scores[:3, :8]
    emit_progress(progress, stage="evaluate", current=1, total=1, metrics={"recall@10": recall})
    # 6) 返回真实测试切分上的指标和必要诊断信息，供章节总结统一读取。
    return {
        "framework": "torch_rechub.models.matching.DSSM",
        "dataset": provenance | {"train_rows": len(train), "test_rows": len(test), "label": "observed Amazon rating >= 4.0"},
        "loss_curve": losses,
        "test_auc": _safe_auc(test.like.to_numpy(), probability),
        "recall@10": recall,
        "paper_protocol_hr@10": sampled["hr@10"], "paper_protocol_ndcg@10": sampled["ndcg@10"],
        "evaluation_users": sampled["evaluation_users"], "sampled_negatives": sampled["sampled_negatives"],
        "embedding_shape": [list(user_embedding.shape), list(item_embedding.shape)],
        "score_sample": score_sample.round(3).tolist(),
    }

def train_and_evaluate(epochs: int = 4, *, progress: ProgressCallback | None = None) -> dict:
    """Small default for learning/CI; increase epochs for the full experiment."""
    return run_dssm(epochs=epochs, progress=progress)
