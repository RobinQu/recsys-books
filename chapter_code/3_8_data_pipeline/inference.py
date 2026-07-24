"""Reusable inference path, deliberately separated from optimization."""
from __future__ import annotations
import torch

def predict(model, batch):
    """Disable dropout/gradients and return model output for a prepared batch."""
    # eval() 会关闭 dropout，并让 batch normalization 使用已学习的统计量。
    model.eval()
    # no_grad() 表明这里只产生预测，不保存反向传播所需的计算图。
    with torch.no_grad():
        output = model(batch)
    # 移到 CPU 后再交给指标代码或在线服务序列化，避免设备不一致。
    return tuple(value.detach().cpu() for value in output) if isinstance(output, tuple) else output.detach().cpu()
