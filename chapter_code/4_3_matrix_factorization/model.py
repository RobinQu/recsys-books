"""Model structure for 4.3 矩阵分解：BiasMF."""

import torch

class BiasMF(torch.nn.Module):
    """Global mean + user/item bias + latent-vector inner product."""
    def __init__(self, users, items, factors=16):
        super().__init__()
        self.user_embedding = torch.nn.Embedding(users, factors)
        self.item_embedding = torch.nn.Embedding(items, factors)
        self.user_bias = torch.nn.Embedding(users, 1)
        self.item_bias = torch.nn.Embedding(items, 1)
        # 全局均值描述总体评分尺度；两个 bias 修正用户/物品的系统性偏差。
        self.global_mean = torch.nn.Parameter(torch.zeros(()))
        # 小尺度初始化避免训练初期内积远离 1—5 分的评分范围。
        torch.nn.init.normal_(self.user_embedding.weight, std=0.05)
        torch.nn.init.normal_(self.item_embedding.weight, std=0.05)
        torch.nn.init.zeros_(self.user_bias.weight)
        torch.nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_id, item_id):
        # 对应位置逐元素相乘再求和，就是用户向量与物品向量的内积。
        interaction = (self.user_embedding(user_id) * self.item_embedding(item_id)).sum(-1)
        return self.global_mean + self.user_bias(user_id).squeeze(-1) + self.item_bias(item_id).squeeze(-1) + interaction
