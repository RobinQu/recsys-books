"""Optimize BiasMF explicitly on MovieLens ratings."""
import numpy as np
import torch

from .model import BiasMF
from recsys_lab.data import leave_last_out, load_movielens
from recsys_lab.runtime import ProgressCallback, emit_progress, full_profile, progress_due, training_device

BATCH = 2_000_000  # full 档 33M 评分的 mini-batch 上限，控制单次前向内存


def train_and_evaluate(epochs: int = 8, *, progress: ProgressCallback | None = None) -> dict:
    # 每次梯度更新同时学习用户/物品隐向量与两侧偏置。
    emit_progress(progress, stage="data_prepare", current=0, total=1, message="加载 MovieLens 并构造时间切分")
    ratings, _ = load_movielens(max_users=80, max_items=360, min_user_events=12)
    train, test = leave_last_out(ratings)
    emit_progress(progress, stage="data_prepare", current=1, total=1, metrics={"rows": len(ratings)})
    device = training_device()
    model = BiasMF(ratings.user_id.nunique(), ratings.item_id.nunique(), factors=16)
    model.to(device)
    model.global_mean.data.fill_(float(train.rating.mean()))
    optimizer = torch.optim.Adam(model.parameters(), lr=.03)
    users = torch.tensor(train.user_id.to_numpy()).to(device); items = torch.tensor(train.item_id.to_numpy()).to(device)
    target = torch.tensor(train.rating.to_numpy(), dtype=torch.float32).to(device); losses = []
    emit_progress(progress, stage="train", current=0, total=epochs, message="训练 BiasMF")
    for epoch in range(epochs):
        if len(users) <= BATCH:
            loss = torch.nn.functional.mse_loss(model(users, items), target)
            optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        else:
            # full 档按 mini-batch 累计一个 epoch 的加权平均损失。
            weighted = 0.0
            for start in range(0, len(users), BATCH):
                stop = min(start + BATCH, len(users))
                loss = torch.nn.functional.mse_loss(model(users[start:stop], items[start:stop]), target[start:stop])
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                weighted += float(loss.detach()) * (stop - start)
            losses.append(weighted / len(users))
        if progress_due(epoch + 1, epochs):
            emit_progress(progress, stage="train", current=epoch + 1, total=epochs, metrics={"loss": losses[-1]})
    emit_progress(progress, stage="inference", current=0, total=1, message="预测留出评分")
    with torch.inference_mode():
        prediction = model(torch.tensor(test.user_id.to_numpy()).to(device), torch.tensor(test.item_id.to_numpy()).to(device)).cpu().numpy()
    emit_progress(progress, stage="inference", current=1, total=1)
    emit_progress(progress, stage="evaluate", current=0, total=1, message="计算 RMSE")
    rmse = float(np.sqrt(np.mean((prediction - test.rating.to_numpy()) ** 2)))
    emit_progress(progress, stage="evaluate", current=1, total=1, metrics={"rmse": rmse})
    dataset = "MovieLens latest (33M)" if full_profile() else "MovieLens latest-small"
    return {"dataset": dataset, "randomly_fabricated_rows": 0, "loss_curve": losses, "rmse": rmse}
