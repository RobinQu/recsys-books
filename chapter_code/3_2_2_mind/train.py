"""Training and evaluation orchestration for 3_2_2_mind.

Shared dataset/metric helpers stay in the common module; the model-specific
tensor construction, loss, optimizer, inference and metrics are visible here.
"""
import numpy as np
import torch
from torch_rechub.basic.features import SequenceFeature, SparseFeature
from torch_rechub.models.matching import MIND
from recsys_lab.data import clicked_sequences, positive_sequences, kuairand_sequence_classification_rows
from recsys_lab.runtime import (
    real_amazon as _real_amazon,
    real_kuairand as _real_kuairand,
    recall_single_target as _recall_single_target,
    safe_auc as _safe_auc,
    seed_everything,
    train_binary as _train_binary,
)

def _mind_rows(ratings, history_length=10, negatives=5):
    global_disliked = [int(v) + 1 for v in ratings[ratings.rating <= 2.5].item_id.tolist()]
    rows = []
    for user, frame in ratings.sort_values("timestamp").groupby("user_id"):
        positives = [int(v) + 1 for v in frame[frame.rating >= 4.0].item_id]
        disliked = [int(v) + 1 for v in frame[frame.rating <= 3.0].item_id]
        if len(positives) < 4:
            continue
        pool = disliked or global_disliked
        negative_items = [pool[(int(user) + offset) % len(pool)] for offset in range(negatives)]
        history = positives[:-2][-history_length:]
        history = [0] * (history_length - len(history)) + history
        inference_history = positives[:-1][-history_length:]
        inference_history = [0] * (history_length - len(inference_history)) + inference_history
        rows.append((int(user) + 1, history, positives[-2], negative_items, inference_history, positives[-1]))
    return rows

def run_mind(epochs: int = 26) -> dict:
    # 1) 固定参数初始化，并读取本章指定的真实数据切片。
    seed_everything(); ratings, provenance = _real_amazon()
    rows = _mind_rows(ratings); n_users, n_items = ratings.user_id.nunique(), ratings.item_id.nunique(); history_length = 10
    user_feature = [SparseFeature("user_id", n_users + 1, 12)]
    item_feature = [SparseFeature("item_id", n_items + 1, 12, padding_idx=0)]
    history_feature = [SequenceFeature("history", n_items + 1, 12, pooling="concat", shared_with="item_id", padding_idx=0)]
    negative_feature = [SequenceFeature("negative_items", n_items + 1, 12, pooling="concat", shared_with="item_id", padding_idx=0)]
    # 2) 按论文结构实例化模型；这里是理解层尺寸和特征契约的入口。
    model = MIND(user_feature, history_feature, item_feature, negative_feature, history_length, interest_num=2)
    train_data = {
        "user_id": torch.tensor([r[0] for r in rows]),
        "history": torch.tensor([r[1] for r in rows]),
        "item_id": torch.tensor([r[2] for r in rows]),
        "negative_items": torch.tensor([r[3] for r in rows]),
    }
    # 3) optimizer 只在训练阶段更新参数；推理阶段不应再调用它。
    optimizer = torch.optim.Adam(model.parameters(), lr=.005); losses = []
    # 4) 一个 epoch：前向计算 → loss → 梯度清零 → 反向传播 → 参数更新。
    for _ in range(epochs):
        logits = model(train_data); loss = torch.nn.functional.cross_entropy(logits, torch.zeros(len(rows), dtype=torch.long))
        optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
    inference = {"user_id": train_data["user_id"], "history": torch.tensor([r[4] for r in rows])}
    # 5) 关闭梯度进入推理阶段，降低显存占用并防止误更新参数。
    with torch.no_grad():
        train_top1 = float((model(train_data).argmax(1) == 0).float().mean())
        model.mode = "user"; user_interests = model(inference).numpy()
        model.mode = "item"; item_embeddings = model({"item_id": torch.arange(1, n_items + 1)}).numpy(); model.mode = None
    scores = np.einsum("ukd,id->uki", user_interests, item_embeddings).max(1)
    seen = [set(np.asarray(r[4])[np.asarray(r[4]) > 0] - 1) for r in rows]
    targets = np.asarray([r[5] - 1 for r in rows])
    # 6) 返回真实测试切分上的指标和必要诊断信息，供章节总结统一读取。
    return {
        "framework": "torch_rechub.models.matching.MIND",
        "dataset": provenance | {"sequence_users": len(rows), "history_length": history_length, "negative_source": "observed Amazon rating <= 3.0"},
        "loss_curve": losses, "positive_top1": train_top1,
        "recall@10": _recall_single_target(scores, targets, 10, seen),
        "interest_shape": list(user_interests.shape),
        "interest_cosine": float(np.mean(np.sum(user_interests[:, 0] * user_interests[:, 1], axis=1))),
    }

def train_and_evaluate(epochs: int = 4) -> dict:
    """Small default for learning/CI; increase epochs for the full experiment."""
    return run_mind(epochs=epochs)
