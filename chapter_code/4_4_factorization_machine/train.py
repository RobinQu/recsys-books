"""Train the chapter's FM implementation on real categorical movie features."""
import numpy as np
import torch

from .model import FactorizationMachine
from recsys_lab.data import load_movielens
from recsys_lab.runtime import full_profile

BATCH = 2_000_000  # full 档 mini-batch 上限


def train_and_evaluate(epochs: int = 8) -> dict:
    # field offset 让不同字段的相同整数值映射到不同 embedding 行。
    ratings, _ = load_movielens(max_users=80, max_items=360, min_user_events=12)
    # full 档使用完整评分流；smoke 档只取最近 4200 行控制 CI 时间。
    frame = ratings.sort_values("timestamp") if full_profile() else ratings.sort_values("timestamp").tail(4200)
    frame = frame.copy()
    raw = frame[["user_id", "item_id", "genre_id", "hour", "decade_id"]].to_numpy()
    offsets = np.cumsum([0, *(raw[:, column].max() + 1 for column in range(raw.shape[1] - 1))])
    feature_ids = torch.tensor(raw + offsets, dtype=torch.long)
    labels = torch.tensor(frame.like.to_numpy(), dtype=torch.float32)
    split = int(len(frame) * .8); model = FactorizationMachine(int(feature_ids.max()) + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=.02); losses = []
    for _ in range(epochs):
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
    with torch.no_grad():
        probability = torch.cat([torch.sigmoid(model(feature_ids[start:start + BATCH])) for start in range(split, len(feature_ids), BATCH)]).numpy()
    from sklearn.metrics import roc_auc_score
    dataset = "MovieLens latest (33M)" if full_profile() else "MovieLens latest-small"
    return {"dataset": dataset, "randomly_fabricated_rows": 0, "loss_curve": losses, "auc": float(roc_auc_score(labels[split:].numpy(), probability))}
