"""Training and evaluation orchestration for 5_3_mind.

Shared dataset/metric helpers stay in the common module; the model-specific
tensor construction, loss, optimizer, inference and metrics are visible here.
"""
import numpy as np
import torch
from torch_rechub.basic.features import SequenceFeature, SparseFeature
from torch_rechub.models.matching import MIND
from recsys_lab.data import clicked_sequences, positive_sequences, kuairand_sequence_classification_rows
from recsys_lab.runtime import (
    ProgressCallback,
    emit_progress,
    real_mind_amazon_books as _real_amazon,
    real_kuairand as _real_kuairand,
    recall_single_target as _recall_single_target,
    safe_auc as _safe_auc,
    seed_everything,
    training_device,
    train_binary as _train_binary,
    full_profile,
    sampled_embedding_rank_metrics,
    progress_due,
)

def _mind_rows(ratings, history_length=10, negatives=5):
    """Build deterministic next-item examples without treating ratings as negatives."""
    item_count = int(ratings.item_id.nunique())
    rows = []
    for user, frame in ratings.sort_values("timestamp").groupby("user_id"):
        positives = [int(v) + 1 for v in frame.item_id]
        if len(positives) < 4:
            continue
        seen = set(positives)
        candidate = (int(user) * 104729 + 2026) % item_count + 1
        negative_items = []
        while len(negative_items) < negatives:
            if candidate not in seen:
                negative_items.append(candidate)
            candidate = candidate % item_count + 1
        history = positives[:-2][-history_length:]
        history = [0] * (history_length - len(history)) + history
        inference_history = positives[:-1][-history_length:]
        inference_history = [0] * (history_length - len(inference_history)) + inference_history
        rows.append((int(user) + 1, history, positives[-2], negative_items, inference_history, positives[-1]))
    return rows

def run_mind(epochs: int = 26, *, progress: ProgressCallback | None = None) -> dict:
    # 1) 固定参数初始化，并读取本章指定的真实数据切片。
    emit_progress(progress, stage="data_prepare", current=0, total=1, message="加载数据并构造 MIND 序列")
    seed_everything(); device = training_device(); ratings, provenance = _real_amazon()
    rows = _mind_rows(ratings); n_users, n_items = ratings.user_id.nunique(), ratings.item_id.nunique(); history_length = 10
    embedding_dim = 36 if full_profile() else 12
    interest_num = 3 if full_profile() else 2
    user_feature = [SparseFeature("user_id", n_users + 1, embedding_dim)]
    item_feature = [SparseFeature("item_id", n_items + 1, embedding_dim, padding_idx=0)]
    history_feature = [SequenceFeature("history", n_items + 1, embedding_dim, pooling="concat", shared_with="item_id", padding_idx=0)]
    negative_feature = [SequenceFeature("negative_items", n_items + 1, embedding_dim, pooling="concat", shared_with="item_id", padding_idx=0)]
    # 2) 按论文结构实例化模型；这里是理解层尺寸和特征契约的入口。
    model = MIND(user_feature, history_feature, item_feature, negative_feature, history_length, interest_num=interest_num)
    model.to(device)
    train_data = {
        "user_id": torch.tensor([r[0] for r in rows]),
        "history": torch.tensor([r[1] for r in rows]),
        "item_id": torch.tensor([r[2] for r in rows]),
        "negative_items": torch.tensor([r[3] for r in rows]),
    }
    train_data = {k: v.to(device) for k, v in train_data.items()}
    emit_progress(progress, stage="data_prepare", current=1, total=1, metrics={"sequence_users": len(rows)})
    # 3) optimizer 只在训练阶段更新参数；推理阶段不应再调用它。
    optimizer = torch.optim.Adam(model.parameters(), lr=.005); losses = []
    # 4) 一个 epoch：前向计算 → loss → 梯度清零 → 反向传播 → 参数更新。
    batch_size = 2048
    total_batches = epochs * ((len(rows) + batch_size - 1) // batch_size)
    completed = 0
    emit_progress(progress, stage="train", current=0, total=total_batches, message="训练多兴趣网络")
    for _ in range(epochs):
        epoch_loss = 0.0
        for start in range(0, len(rows), batch_size):
            stop = min(start + batch_size, len(rows))
            batch = {name: value[start:stop] for name, value in train_data.items()}
            logits = model(batch)
            loss = torch.nn.functional.cross_entropy(logits, torch.zeros(stop - start, dtype=torch.long))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_loss += float(loss.detach()) * (stop - start)
            completed += 1
            if progress_due(completed, total_batches):
                emit_progress(progress, stage="train", current=completed, total=total_batches)
        losses.append(epoch_loss / len(rows))
    inference = {"user_id": train_data["user_id"], "history": torch.tensor([r[4] for r in rows]).to(device)}
    # 5) 关闭梯度进入推理阶段，降低显存占用并防止误更新参数。
    item_batch_size = 131_072
    user_chunks = (len(rows) + batch_size - 1) // batch_size
    item_chunks = (n_items + item_batch_size - 1) // item_batch_size
    total_inference = 1 + user_chunks + item_chunks
    completed = 0
    emit_progress(progress, stage="inference", current=0, total=total_inference, message="分批提取用户兴趣与物品向量")
    model.eval()
    with torch.inference_mode():
        train_top1 = float((model({name: value[:min(batch_size, len(rows))] for name, value in train_data.items()}).argmax(1) == 0).float().mean())
        completed += 1
        model.mode = "user"
        interest_chunks = []
        for start in range(0, len(rows), batch_size):
            interest_chunks.append(model({name: value[start:start + batch_size] for name, value in inference.items()}))
            completed += 1
            if progress_due(completed, total_inference):
                emit_progress(progress, stage="inference", current=completed, total=total_inference)
        user_interests = torch.cat(interest_chunks).cpu().numpy()
        model.mode = "item"
        item_chunks_out = []
        for start in range(1, n_items + 1, item_batch_size):
            item_chunks_out.append(model({"item_id": torch.arange(start, min(start + item_batch_size, n_items + 1)).to(device)}))
            completed += 1
            if progress_due(completed, total_inference):
                emit_progress(progress, stage="inference", current=completed, total=total_inference)
        item_embeddings = torch.cat(item_chunks_out).cpu().numpy()
        model.mode = None
    seen = [set(np.asarray(r[4])[np.asarray(r[4]) > 0] - 1) for r in rows]
    targets = np.asarray([r[5] - 1 for r in rows])
    emit_progress(progress, stage="evaluate", current=0, total=1, message="计算采样排序指标")
    sampled = sampled_embedding_rank_metrics(user_interests, item_embeddings, targets, 10, 100, seen)
    if full_profile():
        recall = float(sampled["hr@10"])
    else:
        scores = np.einsum("ukd,id->uki", user_interests, item_embeddings).max(1)
        recall = _recall_single_target(scores, targets, 10, seen)
    emit_progress(progress, stage="evaluate", current=1, total=1, metrics={"recall@10": recall})
    # 6) 返回真实测试切分上的指标和必要诊断信息，供章节总结统一读取。
    return {
        "framework": "torch_rechub.models.matching.MIND",
        "dataset": provenance | {"sequence_users": len(rows), "history_length": history_length, "negative_source": "sampled unobserved catalog items"},
        "loss_curve": losses, "positive_top1": train_top1,
        "recall@10": recall, "paper_protocol_hr@10": sampled["hr@10"],
        "paper_protocol_ndcg@10": sampled["ndcg@10"], "evaluation_users": sampled["evaluation_users"],
        "sampled_negatives": sampled["sampled_negatives"],
        "interest_shape": list(user_interests.shape),
        "interest_cosine": float(np.mean(np.sum(user_interests[:, 0] * user_interests[:, 1], axis=1))),
    }

def train_and_evaluate(epochs: int = 4, *, progress: ProgressCallback | None = None) -> dict:
    """Small default for learning/CI; increase epochs for the full experiment."""
    return run_mind(epochs=epochs, progress=progress)
