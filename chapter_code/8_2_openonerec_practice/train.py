"""Training and evaluation orchestration for 8_2_openonerec_practice.

Shared dataset/metric helpers stay in the common module; the model-specific
tensor construction, loss, optimizer, inference and metrics are visible here.
"""
import numpy as np
import torch
from .model import TinyListGenerator
from recsys_lab.data import clicked_sequences, positive_sequences, kuairand_sequence_classification_rows
from recsys_lab.runtime import (
    ProgressCallback,
    emit_progress,
    real_amazon as _real_amazon,
    real_kuairand as _real_kuairand,
    real_openonerec_dataset as _real_openonerec_dataset,
    recall_single_target as _recall_single_target,
    safe_auc as _safe_auc,
    seed_everything,
    train_binary as _train_binary,
    full_profile,
    progress_due,
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


def run_openonerec(
    epochs: int = 32,
    cpu_smoke: bool = False,
    *,
    progress: ProgressCallback | None = None,
) -> dict:
    # 1) 固定参数初始化，并读取本章指定的真实数据切片。
    emit_progress(progress, stage="data_prepare", current=0, total=1, message="加载数据并构造 Semantic ID 训练序列")
    seed_everything(); device = _training_device(cpu_smoke)
    sample_rows = 256 if device.type == "cpu" else 2000
    if full_profile():
        users, catalog_frame, provenance = _real_openonerec_dataset()
        # 官方 Semantic ID：pid → 三层整数编码，+1 偏移把 0 留给起始符。
        item_to_code = {int(row.pid): tuple(int(value) + 1 for value in row.sid) for row in catalog_frame.itertuples()}
        catalog = set(item_to_code.values())
        # 训练序列来自 release 中真实 hist_video_pid（只保留有官方 SID 映射的物品）。
        sequences = []
        for row in users.itertuples():
            for pid in list(row.hist_video_pid)[-9:]:
                code = item_to_code.get(int(pid))
                if code is not None:
                    sequences.append(code)
            if len(sequences) >= sample_rows:
                break
        sequences = np.asarray(sequences[:sample_rows], dtype=np.int64)
        code_source = "official RecIF-Bench pid→Semantic ID mapping"
        first = users.iloc[0]
        chosen_pid = int(np.asarray(first["target_video_longview"]).ravel()[0]) if len(np.asarray(first["target_video_longview"]).ravel()) else None
        rejected_pid = int(np.asarray(first["target_video_not_interested"]).ravel()[0]) if len(np.asarray(first["target_video_not_interested"]).ravel()) else None
        dpo_pair = {
            "chosen": item_to_code.get(chosen_pid, sequences[0].tolist()),
            "rejected": item_to_code.get(rejected_pid, sequences[-1].tolist()),
        }
    else:
        interactions, videos, provenance = _real_kuairand(); item_to_code = _semantic_catalog(videos); catalog = set(item_to_code.values())
        positive = interactions[interactions.is_click == 1].sort_values("timestamp").tail(sample_rows)
        sequences = np.asarray([item_to_code[int(item)] for item in positive.item_id], dtype=np.int64)
        code_source = "observed KuaiRand video tag + music type + item id partition"
        chosen_row = interactions[(interactions.is_click == 1) & (interactions.long_view == 1)].iloc[0]
        rejected_row = interactions[interactions.is_click == 0].iloc[0]
        dpo_pair = {"chosen": item_to_code[int(chosen_row.item_id)], "rejected": item_to_code[int(rejected_row.item_id)]}
    emit_progress(progress, stage="data_prepare", current=1, total=1, metrics={"training_codes": len(sequences)})
    vocabulary_size = int(sequences.max()) + 2
    model_input = torch.tensor(np.c_[np.zeros((len(sequences), 1), dtype=np.int64), sequences[:, :-1]], device=device)
    # 2) 按论文结构实例化模型；这里是理解层尺寸和特征契约的入口。
    target = torch.tensor(sequences, device=device); model = TinyListGenerator(vocabulary_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.02); losses=[]
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    effective_epochs = min(epochs, 2) if device.type == "cpu" else epochs
    # 4) 一个 epoch：前向计算 → loss → 梯度清零 → 反向传播 → 参数更新。
    emit_progress(progress, stage="train", current=0, total=effective_epochs, message="训练生成式列表代理模型")
    for epoch in range(effective_epochs):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(model_input)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach().cpu()))
        if progress_due(epoch + 1, effective_epochs):
            emit_progress(progress, stage="train", current=epoch + 1, total=effective_epochs, metrics={"loss": losses[-1]})
    prefix_counts = {}
    for code in catalog: prefix_counts.setdefault(code[:2], set()).add(code[2])
    prefix = sorted(prefix_counts, key=lambda value: (-len(prefix_counts[value]), value))[0]
    allowed = sorted(prefix_counts[prefix])
    # 5) 关闭梯度进入推理阶段，降低显存占用并防止误更新参数。
    emit_progress(progress, stage="inference", current=0, total=1, message="生成下一层 Semantic ID logits")
    model.eval()
    with torch.inference_mode():
        next_logits = model(torch.tensor([[0, *prefix]], device=device))[0, -1].float().cpu().numpy()
    emit_progress(progress, stage="inference", current=1, total=1)
    emit_progress(progress, stage="baseline", current=0, total=1, message="构造无约束 argmax 对照")
    stress_logits = next_logits.copy(); invalid_token = vocabulary_size - 1
    while prefix + (invalid_token,) in catalog: invalid_token -= 1
    stress_logits[invalid_token] = float(stress_logits.max() + .5)
    unconstrained = int(stress_logits.argmax()); constrained = int(max(allowed, key=lambda token: stress_logits[token]))
    emit_progress(progress, stage="baseline", current=1, total=1)
    emit_progress(progress, stage="evaluate", current=0, total=1, message="校验目录约束")
    invalid_unconstrained = float(prefix + (unconstrained,) not in catalog)
    invalid_constrained = float(prefix + (constrained,) not in catalog)
    emit_progress(progress, stage="evaluate", current=1, total=1, metrics={"invalid_constrained": invalid_constrained})
    return {
        "framework": "OpenOneRec contract + PyTorch executable proxy",
        "device": str(device), "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "validation_mode": "cuda_accuracy" if device.type == "cuda" else "cpu_basic_smoke", "mixed_precision": amp_enabled,
        "dataset": provenance | {
            "semantic_catalog_size": len(catalog), "training_codes": len(sequences),
            "code_source": code_source,
        },
        "loss_curve": losses, "catalog_size": len(catalog), "prefix": prefix, "allowed_tokens": allowed,
        "unconstrained_token": unconstrained, "constrained_token": constrained,
        "invalid_unconstrained": invalid_unconstrained, "invalid_constrained": invalid_constrained,
        "dpo_pair": dpo_pair,
    }

def train_and_evaluate(
    epochs: int = 4,
    cpu_smoke: bool = False,
    *,
    progress: ProgressCallback | None = None,
) -> dict:
    """CUDA-first tutorial entry; CPU smoke only validates the basic contract."""
    return run_openonerec(epochs=epochs, cpu_smoke=cpu_smoke, progress=progress)
