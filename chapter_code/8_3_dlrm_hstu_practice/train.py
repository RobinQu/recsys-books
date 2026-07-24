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
    # 4) 一个 epoch：前向计算 → loss → 梯度清零 → 反向传播 → 参数更新。
    for _ in range(effective_epochs):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1))
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach().cpu()))
    # 5) 关闭梯度进入推理阶段，降低显存占用并防止误更新参数。
    with torch.inference_mode():
        final_logits = model(torch.tensor(eval_input, device=device))[:, -1]
        top5 = final_logits.topk(5, dim=1).indices
        hit = (top5 == torch.tensor(eval_target, device=device)[:, None]).any(1).float().mean().item()
    popularity = np.bincount(train_input.ravel(), minlength=vocab); popular_top5 = np.argsort(-popularity[1:])[:5] + 1
    baseline = float(np.isin(eval_target, popular_top5).mean())
    # 6) 返回真实测试切分上的指标和必要诊断信息，供章节总结统一读取。
    return {
        "framework": "torch_rechub.models.generative.HSTUModel",
        "device": str(device), "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "validation_mode": "cuda_accuracy" if device.type == "cuda" else "cpu_basic_smoke", "mixed_precision": amp_enabled,
        "dataset": provenance | {"sequence_users": len(eval_target), "sequence_length": length, "vocab": vocab, "strict_time_split": True},
        "loss_curve": losses, "hr@5": hit, "popularity_hr@5": baseline, "logits_shape": list(final_logits.shape),
    }

def train_and_evaluate(epochs: int = 4, cpu_smoke: bool = False) -> dict:
    """CUDA-first tutorial entry; CPU smoke only validates the basic contract."""
    return run_hstu(epochs=epochs, cpu_smoke=cpu_smoke)
