"""Model structure for 4.6 word2vec / Item2Vec：行为序列嵌入与召回."""
import torch


class SkipGram(torch.nn.Module):
    """Skip-gram with negative sampling: a center item predicts its context items.

    经典 word2vec 把“中心词预测上下文词”作为自监督任务；item2vec 把每个用户的
    正反馈物品序列当作一句话、物品当作词，于是同一序列里共现的物品会在嵌入空间
    被拉近，这正是用行为序列学召回向量的最小原型。
    """

    def __init__(self, num_items: int, dim: int = 32):
        super().__init__()
        # 中心嵌入与上下文嵌入分离，是 word2vec 的常规做法；负采样时只更新相关行。
        self.center = torch.nn.Embedding(num_items, dim)
        self.context = torch.nn.Embedding(num_items, dim)
        torch.nn.init.uniform_(self.center.weight, -0.01, 0.01)
        torch.nn.init.zeros_(self.context.weight)

    def forward(self, center_ids: torch.Tensor, context_ids: torch.Tensor) -> torch.Tensor:
        # 中心向量与上下文向量的内积越大，说明两物品越可能在同一序列共现。
        return (self.center(center_ids) * self.context(context_ids)).sum(-1)


def negative_sampling_loss(score_pos: torch.Tensor, score_neg: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy on logits: pull positives together, push negatives away."""
    # 正样本希望 σ(score)→1，负样本希望 σ(score)→0；BCE 等价于负采样目标函数。
    pos = torch.nn.functional.binary_cross_entropy_with_logits(score_pos, torch.ones_like(score_pos))
    neg = torch.nn.functional.binary_cross_entropy_with_logits(score_neg, torch.zeros_like(score_neg))
    return pos + neg
