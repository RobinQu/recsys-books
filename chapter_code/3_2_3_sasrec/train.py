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
    real_sasrec_dataset as _real_amazon,
    real_kuairand as _real_kuairand,
    recall_single_target as _recall_single_target,
    safe_auc as _safe_auc,
    seed_everything,
    train_binary as _train_binary,
    full_profile,
    sampled_embedding_rank_metrics,
)

def _sequence_windows_from_sequences(sequences, length=12):
    """Leave the final event for test and the penultimate event for validation."""
    user_ids, train_input, train_target, eval_input, eval_target = [], [], [], [], []
    for user_id, sequence in sequences.items():
        # Paper protocol: training ends before the validation and test items.
        prior = sequence[:-2]
        if len(prior) < 2:
            continue
        source = prior[-(length + 1):]
        source = [0] * (length + 1 - len(source)) + source
        user_ids.append(int(user_id)); train_input.append(source[:-1]); train_target.append(source[1:])
        history = sequence[:-1][-length:]
        eval_input.append([0] * (length - len(history)) + history); eval_target.append(sequence[-1])
    return user_ids, np.asarray(train_input), np.asarray(train_target), np.asarray(eval_input), np.asarray(eval_target)


def _sample_unobserved_negatives(user_ids, sequences, rows, vocab):
    """Deterministically sample only items that the user never interacted with."""
    negatives = np.empty_like(rows)
    for row_index, user_id in enumerate(user_ids):
        seen = set(sequences[user_id])
        candidate = (user_id * 104729 + 2026) % (vocab - 1) + 1
        for column in range(rows.shape[1]):
            while candidate in seen:
                candidate = candidate % (vocab - 1) + 1
            negatives[row_index, column] = candidate
            candidate = candidate % (vocab - 1) + 1
    return negatives

def run_sasrec(epochs: int | None = None) -> dict:
    # 1) 固定参数初始化，并读取本章指定的真实数据切片。
    seed_everything(); ratings, provenance = _real_amazon(max_users=160)
    epochs = (201 if full_profile() else 30) if epochs is None else epochs
    length = 200 if full_profile() else 20
    # SASRec treats every observed action as implicit positive feedback. A low
    # rating is still an interaction and must not be recycled as a negative.
    sequences = positive_sequences(ratings, threshold=-np.inf, min_length=5)
    user_ids, train_input, train_target, eval_input, eval_target = _sequence_windows_from_sequences(sequences, length)
    vocab = ratings.item_id.nunique() + 1
    negative = _sample_unobserved_negatives(user_ids, sequences, train_target, vocab)
    embedding_dim = 50 if full_profile() else 24
    seq = SequenceFeature("seq", vocab, embedding_dim, pooling="concat", padding_idx=0)
    pos = SequenceFeature("pos", vocab, embedding_dim, pooling="concat", shared_with="seq", padding_idx=0)
    neg = SequenceFeature("neg", vocab, embedding_dim, pooling="concat", shared_with="seq", padding_idx=0)
    # 2) 按论文结构实例化模型；这里是理解层尺寸和特征契约的入口。
    model = SASRec([seq, pos, neg], max_len=length, dropout_rate=.2 if full_profile() else .1, num_blocks=2 if full_profile() else 1, num_heads=1 if full_profile() else 2)
    data = {"seq": torch.tensor(train_input), "pos": torch.tensor(train_target), "neg": torch.tensor(negative)}
    # 3) optimizer 只在训练阶段更新参数；推理阶段不应再调用它。
    optimizer = torch.optim.Adam(model.parameters(), lr=.001 if full_profile() else .008); losses=[]
    # 4) 一个 epoch：前向计算 → loss → 梯度清零 → 反向传播 → 参数更新。
    batch_size = 128 if full_profile() else 2048
    for _ in range(epochs):
        epoch_loss = 0.0
        for start in range(0, len(train_input), batch_size):
            stop = min(start + batch_size, len(train_input))
            pos_logits, neg_logits = model({name: value[start:stop] for name, value in data.items()})
            valid = data["pos"][start:stop].ne(0)
            loss = torch.nn.functional.softplus(-pos_logits[valid]).mean() + torch.nn.functional.softplus(neg_logits[valid]).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_loss += float(loss.detach()) * (stop - start)
        losses.append(epoch_loss / len(train_input))
    # 5) 关闭梯度进入推理阶段，降低显存占用并防止误更新参数。
    model.eval()
    with torch.no_grad():
        user_batches = []
        for start in range(0, len(eval_input), batch_size):
            batch = {"seq": torch.tensor(eval_input[start:start + batch_size])}
            sequence_embedding = model.item_emb(batch, model.features[:1])[:, 0]
            # Inputs use left padding, so the newest valid event is always at
            # the final column. Torch-RecHub's user_tower currently computes
            # ``count(nonzero)-1`` (right-padding semantics), which selects a
            # wrong position for short users and materially depresses HR/NDCG.
            user_batches.append(model.seq_forward(batch, sequence_embedding)[:, -1])
        user_vectors = torch.cat(user_batches)
        item_vectors = model.item_emb.embed_dict["seq"].weight[1:]
        user_vectors = np.nan_to_num(user_vectors.numpy(), nan=0.0, posinf=0.0, neginf=0.0)
        item_vectors = np.nan_to_num(item_vectors.numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    seen = [set(row[row > 0] - 1) for row in eval_input]
    popularity = np.bincount(train_input.ravel(), minlength=vocab)[1:]
    popularity_top10 = np.argsort(-popularity)[:10]
    popularity_hr = float(np.mean([target - 1 in popularity_top10 for target in eval_target]))
    sampled = sampled_embedding_rank_metrics(user_vectors, item_vectors, eval_target - 1, 10, 100, seen)
    if full_profile():
        hr = float(sampled["hr@10"])
    else:
        scores = user_vectors @ item_vectors.T
        hr = _recall_single_target(scores, eval_target - 1, 10, seen)
    # 6) 返回真实测试切分上的指标和必要诊断信息，供章节总结统一读取。
    return {
        "framework": "torch_rechub.models.matching.SASRec",
        "dataset": provenance | {"sequence_users": len(eval_target), "sequence_length": length, "negative_source": "sampled unobserved catalog items"},
        "loss_curve": losses, "hr@10": hr, "paper_protocol_hr@10": sampled["hr@10"],
        "paper_protocol_ndcg@10": sampled["ndcg@10"], "evaluation_users": sampled["evaluation_users"],
        "sampled_negatives": sampled["sampled_negatives"],
        "popularity_hr@10": popularity_hr, "embedding_dim": embedding_dim,
    }

def train_and_evaluate(epochs: int = 4) -> dict:
    """Small default for learning/CI; increase epochs for the full experiment."""
    return run_sasrec(epochs=epochs)
