"""Training and evaluation orchestration for 8_3_dlrm_hstu_practice.

Shared dataset/metric helpers stay in the common module; the model-specific
tensor construction, loss, optimizer, inference and metrics are visible here.
"""
import numpy as np
import torch
from torch_rechub.models.generative import HSTUModel
from recsys_lab.data import clicked_sequences, positive_sequences, kuairand_sequence_classification_rows
from recsys_lab.runtime import (
    real_amazon as _real_amazon,
    real_hstu_dataset as _real_hstu_dataset,
    real_kuairand as _real_kuairand,
    recall_single_target as _recall_single_target,
    safe_auc as _safe_auc,
    seed_everything,
    train_binary as _train_binary,
    full_profile,
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

def _training_device(cpu_smoke: bool) -> torch.device:
    """Use CUDA for the real experiment and reserve CPU for basic CI checks."""
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        return torch.device("cuda")
    if not cpu_smoke:
        raise RuntimeError("HSTU training requires CUDA; pass cpu_smoke=True only for basic CI checks")
    return torch.device("cpu")


def run_hstu(epochs: int = 26, cpu_smoke: bool = False) -> dict:
    # 1) 固定参数初始化，并读取本章指定的真实数据切片。
    seed_everything(); device = _training_device(cpu_smoke)
    max_users, max_items, length = (32, 800, 8) if device.type == "cpu" else (128, 2500, 24)
    if full_profile():
        # 论文公开基准：MovieLens 20M 全部评分按时间序组成隐式序列。
        ratings, _, provenance = _real_hstu_dataset(max_users=max_users, max_items=max_items)
        interactions = ratings.assign(is_click=np.float32(1.0))
    else:
        interactions, _, provenance = _real_kuairand(max_users=max_users, max_items=max_items)
    sequences = clicked_sequences(interactions, min_length=length + 3)
    train_input, train_target, eval_input, eval_target = _sequence_windows_from_sequences(sequences, length)
    vocab = interactions.item_id.nunique() + 1
    inputs = torch.tensor(train_input, device=device); targets = torch.tensor(train_target, device=device)
    # 2) 按论文结构实例化模型；这里是理解层尺寸和特征契约的入口。
    model = HSTUModel(vocab, d_model=32, n_heads=2, n_layers=1, dqk=8, dv=8, max_seq_len=length, dropout=0.0, use_time_embedding=False).to(device)
    # 3) optimizer 只在训练阶段更新参数；推理阶段不应再调用它。
    optimizer = torch.optim.AdamW(model.parameters(), lr=.008); losses=[]
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    effective_epochs = min(epochs, 2) if device.type == "cpu" else epochs
    # full 档词表达 26k+，整条目前向的 [B,L,V] logits 会爆显存/内存：按用户 mini-batch
    # 训练与评估（CPU 档 512、CUDA 档 4096），论文工业实现的对应手段是 sampled softmax。
    batch = 512 if device.type == "cpu" else 4096
    # 4) 一个 epoch：前向计算 → loss → 梯度清零 → 反向传播 → 参数更新。
    for _ in range(effective_epochs):
        weighted = 0.0
        for start in range(0, len(inputs), batch):
            stop = min(start + batch, len(inputs))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(inputs[start:stop])
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab), targets[start:stop].reshape(-1))
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            weighted += float(loss.detach().cpu()) * (stop - start)
        losses.append(weighted / len(inputs))
    # 5) 关闭梯度进入推理阶段，降低显存占用并防止误更新参数。
    top5_chunks = []
    with torch.inference_mode():
        for start in range(0, len(eval_input), batch):
            final_logits = model(torch.tensor(eval_input[start:start + batch], device=device))[:, -1]
            top5_chunks.append(final_logits.topk(5, dim=1).indices)
    top5 = torch.cat(top5_chunks)
    hit = (top5 == torch.tensor(eval_target, device=device)[:, None]).any(1).float().mean().item()
    popularity = np.bincount(train_input.ravel(), minlength=vocab); popular_top5 = np.argsort(-popularity[1:])[:5] + 1
    baseline = float(np.isin(eval_target, popular_top5).mean())
    # 6) 返回真实测试切分上的指标和必要诊断信息，供章节总结统一读取。
    return {
        "framework": "torch_rechub.models.generative.HSTUModel",
        "device": str(device), "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "validation_mode": "cuda_accuracy" if device.type == "cuda" else "cpu_basic_smoke", "mixed_precision": amp_enabled,
        "dataset": provenance | {"sequence_users": len(eval_target), "sequence_length": length, "vocab": vocab, "strict_time_split": True},
        "loss_curve": losses, "hr@5": hit, "popularity_hr@5": baseline, "logits_shape": list(top5.shape),
    }

def train_and_evaluate(epochs: int = 4, cpu_smoke: bool = False) -> dict:
    """CUDA-first tutorial entry; CPU smoke only validates the basic contract."""
    return run_hstu(epochs=epochs, cpu_smoke=cpu_smoke)
