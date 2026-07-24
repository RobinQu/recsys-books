"""Train the chapter's FM implementation on real categorical movie features."""
import numpy as np
import torch

from .model import FactorizationMachine
from recsys_lab.data import load_movielens
from recsys_lab.runtime import ProgressCallback, emit_progress, full_profile, progress_due, training_device

BATCH = 2_000_000  # full 档 mini-batch 上限


def train_and_evaluate(epochs: int = 8, *, progress: ProgressCallback | None = None) -> dict:
    # field offset 让不同字段的相同整数值映射到不同 embedding 行。
    emit_progress(progress, stage="data_prepare", current=0, total=1, message="加载并编码 MovieLens 特征")
    ratings, _ = load_movielens(max_users=80, max_items=360, min_user_events=12)
    # full 档使用完整评分流；smoke 档只取最近 4200 行控制 CI 时间。
    frame = ratings.sort_values("timestamp") if full_profile() else ratings.sort_values("timestamp").tail(4200)
    frame = frame.copy()
    raw = frame[["user_id", "item_id", "genre_id", "hour", "decade_id"]].to_numpy()
    offsets = np.cumsum([0, *(raw[:, column].max() + 1 for column in range(raw.shape[1] - 1))])
    feature_ids = torch.tensor(raw + offsets, dtype=torch.long)
    labels = torch.tensor(frame.like.to_numpy(), dtype=torch.float32)
    emit_progress(progress, stage="data_prepare", current=1, total=1, metrics={"rows": len(frame)})
    split = int(len(frame) * .8); model = FactorizationMachine(int(feature_ids.max()) + 1)
    device = training_device()
    model.to(device)
    feature_ids = feature_ids.to(device)
    labels = labels.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.02); losses = []
    emit_progress(progress, stage="train", current=0, total=epochs, message="训练 Factorization Machine")
    for epoch in range(epochs):
        if split <= BATCH:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(feature_ids[:split]), labels[:split])
            optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        else:
            weighted = 0.0
            for start in range(0, split, BATCH):
                stop = min(start + BATCH, split)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(model(feature_ids[start:stop]), labels[start:stop])
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                weighted += float(loss.detach()) * (stop - start)
            losses.append(weighted / split)
        if progress_due(epoch + 1, epochs):
            emit_progress(progress, stage="train", current=epoch + 1, total=epochs, metrics={"loss": losses[-1]})
    emit_progress(progress, stage="inference", current=0, total=1, message="分批生成测试集概率")
    with torch.inference_mode():
        probability = torch.cat([torch.sigmoid(model(feature_ids[start:start + BATCH])) for start in range(split, len(feature_ids), BATCH)]).cpu().numpy()
    emit_progress(progress, stage="inference", current=1, total=1)
    from sklearn.metrics import log_loss, roc_auc_score
    emit_progress(progress, stage="evaluate", current=0, total=1, message="计算 AUC 与 LogLoss")
    auc = float(roc_auc_score(labels[split:].cpu().numpy(), probability))
    test_logloss = float(log_loss(labels[split:].cpu().numpy(), np.clip(probability, 1e-6, 1 - 1e-6)))
    emit_progress(progress, stage="evaluate", current=1, total=1, metrics={"auc": auc, "logloss": test_logloss})
    dataset = "MovieLens latest (33M)" if full_profile() else "MovieLens latest-small"
    return {
        "dataset": dataset,
        "randomly_fabricated_rows": 0,
        "loss_curve": losses,
        "auc": auc,
        "logloss": test_logloss,
    }
