"""Training and evaluation orchestration for 4_2_openonerec_practice.

Shared dataset/metric helpers stay in the common module; the model-specific
tensor construction, loss, optimizer, inference and metrics are visible here.
"""
import numpy as np
import torch
from .model import TinyListGenerator
from recsys_lab.data import clicked_sequences, positive_sequences, kuairand_sequence_classification_rows
from recsys_lab.runtime import (
    real_amazon as _real_amazon,
    real_kuairand as _real_kuairand,
    recall_single_target as _recall_single_target,
    safe_auc as _safe_auc,
    seed_everything,
    train_binary as _train_binary,
)

def _semantic_catalog(videos):
    def first_tag(value) -> int:
        raw = str(value).split(",")[0]
        try: return int(float(raw))
        except ValueError: return 0
    # 6) 返回真实测试切分上的指标和必要诊断信息，供章节总结统一读取。
    return {
        int(row.item_id): (
            first_tag(row.tag) % 128 + 1,
            int(0 if np.isnan(row.music_type) else row.music_type) % 32 + 1,
            int(row.item_id % 64) + 1,
        )
        for row in videos.itertuples()
    }

def run_openonerec(epochs: int = 32) -> dict:
    # 1) 固定参数初始化，并读取本章指定的真实数据切片。
    seed_everything(); interactions, videos, provenance = _real_kuairand(); item_to_code = _semantic_catalog(videos); catalog = set(item_to_code.values())
    positive = interactions[interactions.is_click == 1].sort_values("timestamp").tail(2000)
    sequences = np.asarray([item_to_code[int(item)] for item in positive.item_id], dtype=np.int64)
    vocabulary_size = int(sequences.max()) + 2
    model_input = torch.tensor(np.c_[np.zeros((len(sequences), 1), dtype=np.int64), sequences[:, :-1]])
    # 2) 按论文结构实例化模型；这里是理解层尺寸和特征契约的入口。
    target = torch.tensor(sequences); model = TinyListGenerator(vocabulary_size); optimizer = torch.optim.Adam(model.parameters(), lr=.02); losses=[]
    # 4) 一个 epoch：前向计算 → loss → 梯度清零 → 反向传播 → 参数更新。
    for _ in range(epochs):
        logits = model(model_input); loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
        optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
    prefix_counts = {}
    for code in catalog: prefix_counts.setdefault(code[:2], set()).add(code[2])
    prefix = sorted(prefix_counts, key=lambda value: (-len(prefix_counts[value]), value))[0]
    allowed = sorted(prefix_counts[prefix])
    # 5) 关闭梯度进入推理阶段，降低显存占用并防止误更新参数。
    with torch.no_grad(): next_logits = model(torch.tensor([[0, *prefix]]))[0, -1].numpy()
    stress_logits = next_logits.copy(); invalid_token = vocabulary_size - 1
    while prefix + (invalid_token,) in catalog: invalid_token -= 1
    stress_logits[invalid_token] = float(stress_logits.max() + .5)
    unconstrained = int(stress_logits.argmax()); constrained = int(max(allowed, key=lambda token: stress_logits[token]))
    chosen_row = interactions[(interactions.is_click == 1) & (interactions.long_view == 1)].iloc[0]
    rejected_row = interactions[interactions.is_click == 0].iloc[0]
    return {
        "framework": "OpenOneRec contract + PyTorch executable proxy",
        "dataset": provenance | {
            "semantic_catalog_size": len(catalog), "training_codes": len(sequences),
            "code_source": "observed KuaiRand video tag + music type + item id partition",
            "recif_bench_access": "official OpenOneRec-RecIF repository is gated; full profile requires authenticated acceptance",
        },
        "loss_curve": losses, "catalog_size": len(catalog), "prefix": prefix, "allowed_tokens": allowed,
        "unconstrained_token": unconstrained, "constrained_token": constrained,
        "invalid_unconstrained": float(prefix + (unconstrained,) not in catalog), "invalid_constrained": float(prefix + (constrained,) not in catalog),
        "dpo_pair": {"chosen": item_to_code[int(chosen_row.item_id)], "rejected": item_to_code[int(rejected_row.item_id)]},
    }

def train_and_evaluate(epochs: int = 4) -> dict:
    """Small default for learning/CI; increase epochs for the full experiment."""
    return run_openonerec(epochs=epochs)
