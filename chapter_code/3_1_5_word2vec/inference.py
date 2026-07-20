"""Reusable inference path, deliberately separated from optimization."""
from __future__ import annotations

import torch


def predict(model, center_ids, context_ids):
    """Disable dropout/gradients and return similarity scores for prepared item pairs."""
    # eval() 关闭 dropout 等训练期行为；no_grad() 表明只产生预测，不保留计算图。
    model.eval()
    with torch.no_grad():
        output = model(center_ids, context_ids)
    # 移到 CPU 再交给指标或在线服务，避免设备不一致。
    return output.detach().cpu()
