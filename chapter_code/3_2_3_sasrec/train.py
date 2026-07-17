"""Training and evaluation orchestration for 3_2_3_sasrec.

Shared dataset/metric helpers stay in the common module; the model-specific
tensor construction, loss, optimizer, inference and metrics are visible here.
"""
import numpy as np
import torch
from torch_rechub.basic.features import SequenceFeature
from torch_rechub.models.matching import SASRec
from recsys_lab.data import clicked_sequences, positive_sequences, kuairand_sequence_classification_rows
from recsys_lab.runtime import (
    real_amazon as _real_amazon,
    real_kuairand as _real_kuairand,
    recall_single_target as _recall_single_target,
    safe_auc as _safe_auc,
    seed_everything,
    train_binary as _train_binary,
)

def _sequence_windows_from_sequences(sequences, length=12):
    train_input, train_target, eval_input, eval_target = [], [], [], []
    for sequence in sequences.values():
        prior = sequence[:-1]
        train = prior[-(length + 1):]
        if len(train) < length + 1: continue
        train_input.append(train[:-1]); train_target.append(train[1:])
        eval_input.append(sequence[-(length + 1):-1]); eval_target.append(sequence[-1])
    return np.asarray(train_input), np.asarray(train_target), np.asarray(eval_input), np.asarray(eval_target)

def run_sasrec(epochs: int = 30) -> dict:
    # 1) 固定参数初始化，并读取本章指定的真实数据切片。
    seed_everything(); ratings, provenance = _real_amazon(max_users=160); length = 20
    sequences = positive_sequences(ratings, threshold=4.0, min_length=length + 3)
    train_input, train_target, eval_input, eval_target = _sequence_windows_from_sequences(sequences, length)
    low_items = [int(v) + 1 for v in ratings[ratings.rating <= 2.5].item_id]
    negative = np.asarray([[low_items[(row + col) % len(low_items)] for col in range(length)] for row in range(len(train_input))])
    vocab = ratings.item_id.nunique() + 1
    seq = SequenceFeature("seq", vocab, 24, pooling="concat", padding_idx=0)
    pos = SequenceFeature("pos", vocab, 24, pooling="concat", shared_with="seq", padding_idx=0)
    neg = SequenceFeature("neg", vocab, 24, pooling="concat", shared_with="seq", padding_idx=0)
    # 2) 按论文结构实例化模型；这里是理解层尺寸和特征契约的入口。
    model = SASRec([seq, pos, neg], max_len=length, dropout_rate=.1, num_blocks=1, num_heads=2)
    data = {"seq": torch.tensor(train_input), "pos": torch.tensor(train_target), "neg": torch.tensor(negative)}
    # 3) optimizer 只在训练阶段更新参数；推理阶段不应再调用它。
    optimizer = torch.optim.Adam(model.parameters(), lr=.008); losses=[]
    # 4) 一个 epoch：前向计算 → loss → 梯度清零 → 反向传播 → 参数更新。
    for _ in range(epochs):
        pos_logits, neg_logits = model(data)
        loss = torch.nn.functional.softplus(-pos_logits).mean() + torch.nn.functional.softplus(neg_logits).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
    # 5) 关闭梯度进入推理阶段，降低显存占用并防止误更新参数。
    with torch.no_grad():
        user_vectors = model.user_tower({"seq": torch.tensor(eval_input)}).squeeze(1)
        item_vectors = model.item_emb.embed_dict["seq"].weight[1:]
        scores = (user_vectors @ item_vectors.T).numpy()
    seen = [set(row[row > 0] - 1) for row in eval_input]
    popularity = np.bincount(train_input.ravel(), minlength=vocab)[1:]
    popularity_top10 = np.argsort(-popularity)[:10]
    popularity_hr = float(np.mean([target - 1 in popularity_top10 for target in eval_target]))
    # 6) 返回真实测试切分上的指标和必要诊断信息，供章节总结统一读取。
    return {
        "framework": "torch_rechub.models.matching.SASRec",
        "dataset": provenance | {"sequence_users": len(eval_target), "sequence_length": length, "negative_source": "observed Amazon rating <= 2.5"},
        "loss_curve": losses, "hr@10": _recall_single_target(scores, eval_target - 1, 10, seen),
        "popularity_hr@10": popularity_hr, "embedding_dim": 24,
    }

def train_and_evaluate(epochs: int = 4) -> dict:
    """Small default for learning/CI; increase epochs for the full experiment."""
    return run_sasrec(epochs=epochs)
