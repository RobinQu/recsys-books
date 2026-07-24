"""Optimize BiasMF explicitly on MovieLens ratings."""
import numpy as np
import torch

from .model import BiasMF
from recsys_lab.data import leave_last_out, load_movielens

def train_and_evaluate(epochs: int = 8) -> dict:
    # 每次梯度更新同时学习用户/物品隐向量与两侧偏置。
    ratings, _ = load_movielens(max_users=80, max_items=360, min_user_events=12)
    train, test = leave_last_out(ratings)
    model = BiasMF(ratings.user_id.nunique(), ratings.item_id.nunique(), factors=16)
    model.global_mean.data.fill_(float(train.rating.mean()))
    optimizer = torch.optim.Adam(model.parameters(), lr=.03)
    users = torch.tensor(train.user_id.to_numpy()); items = torch.tensor(train.item_id.to_numpy())
    target = torch.tensor(train.rating.to_numpy(), dtype=torch.float32); losses = []
    for _ in range(epochs):
        loss = torch.nn.functional.mse_loss(model(users, items), target)
        optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
    with torch.no_grad():
        prediction = model(torch.tensor(test.user_id.to_numpy()), torch.tensor(test.item_id.to_numpy())).numpy()
    rmse = float(np.sqrt(np.mean((prediction - test.rating.to_numpy()) ** 2)))
    return {"dataset": "MovieLens latest-small", "randomly_fabricated_rows": 0, "loss_curve": losses, "rmse": rmse}
