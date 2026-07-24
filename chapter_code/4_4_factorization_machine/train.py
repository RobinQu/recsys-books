"""Train the chapter's FM implementation on real categorical movie features."""
import numpy as np
import torch

from .model import FactorizationMachine
from recsys_lab.data import load_movielens

def train_and_evaluate(epochs: int = 8) -> dict:
    # field offset 让不同字段的相同整数值映射到不同 embedding 行。
    ratings, _ = load_movielens(max_users=80, max_items=360, min_user_events=12)
    frame = ratings.sort_values("timestamp").tail(4200).copy()
    raw = frame[["user_id", "item_id", "genre_id", "hour", "decade_id"]].to_numpy()
    offsets = np.cumsum([0, *(raw[:, column].max() + 1 for column in range(raw.shape[1] - 1))])
    feature_ids = torch.tensor(raw + offsets, dtype=torch.long)
    labels = torch.tensor(frame.like.to_numpy(), dtype=torch.float32)
    split = int(len(frame) * .8); model = FactorizationMachine(int(feature_ids.max()) + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=.02); losses = []
    for _ in range(epochs):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(feature_ids[:split]), labels[:split])
        optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
    with torch.no_grad(): probability = torch.sigmoid(model(feature_ids[split:])).numpy()
    from sklearn.metrics import roc_auc_score
    return {"dataset": "MovieLens latest-small", "randomly_fabricated_rows": 0, "loss_curve": losses, "auc": float(roc_auc_score(labels[split:].numpy(), probability))}
