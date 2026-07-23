"""Training and evaluation orchestration for 4_1_openonerec_practice.

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

def _training_device(cpu_smoke: bool) -> torch.device:
    """Use CUDA by default; CPU is an explicit contract-only test mode."""
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        return torch.device("cuda")
    if not cpu_smoke:
        raise RuntimeError("OpenOneRec training requires CUDA; pass cpu_smoke=True only for basic CI checks")
    return torch.device("cpu")


def run_openonerec(epochs: int = 32, cpu_smoke: bool = False) -> dict:
    # 1) 固定参数初始化，并读取本章指定的真实数据切片。
    seed_everything(); device = _training_device(cpu_smoke)
    interactions, videos, provenance = _real_kuairand(); item_to_code = _semantic_catalog(videos); catalog = set(item_to_code.values())
    sample_rows = 256 if device.type == "cpu" else 2000
    positive = interactions[interactions.is_click == 1].sort_values("timestamp").tail(sample_rows)
    sequences = np.asarray([item_to_code[int(item)] for item in positive.item_id], dtype=np.int64)
    vocabulary_size = int(sequences.max()) + 2
    model_input = torch.tensor(np.c_[np.zeros((len(sequences), 1), dtype=np.int64), sequences[:, :-1]], device=device)
    # 2) 按论文结构实例化模型；这里是理解层尺寸和特征契约的入口。
    target = torch.tensor(sequences, device=device); model = TinyListGenerator(vocabulary_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.02); losses=[]
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    effective_epochs = min(epochs, 2) if device.type == "cpu" else epochs
    # 4) 一个 epoch：前向计算 → loss → 梯度清零 → 反向传播 → 参数更新。
    for _ in range(effective_epochs):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(model_input)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach().cpu()))
    prefix_counts = {}
    for code in catalog: prefix_counts.setdefault(code[:2], set()).add(code[2])
    prefix = sorted(prefix_counts, key=lambda value: (-len(prefix_counts[value]), value))[0]
    allowed = sorted(prefix_counts[prefix])
    # 5) 关闭梯度进入推理阶段，降低显存占用并防止误更新参数。
    with torch.inference_mode():
        next_logits = model(torch.tensor([[0, *prefix]], device=device))[0, -1].float().cpu().numpy()
    stress_logits = next_logits.copy(); invalid_token = vocabulary_size - 1
    while prefix + (invalid_token,) in catalog: invalid_token -= 1
    stress_logits[invalid_token] = float(stress_logits.max() + .5)
    unconstrained = int(stress_logits.argmax()); constrained = int(max(allowed, key=lambda token: stress_logits[token]))
    chosen_row = interactions[(interactions.is_click == 1) & (interactions.long_view == 1)].iloc[0]
    rejected_row = interactions[interactions.is_click == 0].iloc[0]
    return {
        "framework": "OpenOneRec contract + PyTorch executable proxy",
        "device": str(device), "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "validation_mode": "cuda_accuracy" if device.type == "cuda" else "cpu_basic_smoke", "mixed_precision": amp_enabled,
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

def train_and_evaluate(epochs: int = 4, cpu_smoke: bool = False) -> dict:
    """CUDA-first tutorial entry; CPU smoke only validates the basic contract."""
    return run_openonerec(epochs=epochs, cpu_smoke=cpu_smoke)
