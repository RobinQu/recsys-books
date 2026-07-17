"""Model structure for 3.1.3 因子分解机：FM."""

import torch

class FactorizationMachine(torch.nn.Module):
    """Linear term plus all second-order feature interactions in O(kd)."""
    def __init__(self, fields, factors=16):
        super().__init__()
        self.linear = torch.nn.Embedding(fields, 1)
        self.embedding = torch.nn.Embedding(fields, factors)

    def forward(self, feature_ids):
        # vectors 的形状为 [batch, field, factor]，每个稀疏特征都有一个隐向量。
        vectors = self.embedding(feature_ids)
        linear = self.linear(feature_ids).sum(1).squeeze(-1)
        # 恒等式将两两内积从 O(field²) 化为 O(field)，避免显式双重循环。
        pairwise = 0.5 * ((vectors.sum(1) ** 2 - (vectors ** 2).sum(1)).sum(1))
        return linear + pairwise
