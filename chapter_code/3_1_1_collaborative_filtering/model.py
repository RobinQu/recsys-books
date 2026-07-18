"""Model structure for 3.1.1 协同过滤：UserCF / ItemCF."""
import numpy as np

class ItemCF:
    """Item-based collaborative filtering using cosine similarity."""
    def fit(self, interactions):
        values = np.asarray(interactions, dtype=np.float64)
        # 每一列是一件物品；列向量记录哪些用户与它交互过。
        norms = np.linalg.norm(values, axis=0, keepdims=True)
        # 矩阵乘法一次得到所有物品对的共同用户数，再除以长度得到余弦相似度。
        denominator = np.maximum(norms.T @ norms, 1e-12)
        self.similarity = np.nan_to_num((values.T @ values) / denominator)
        # 不允许物品把自己当作邻居，否则自身相似度 1 会淹没其他候选。
        np.fill_diagonal(self.similarity, 0.0)
        return self

    def score(self, interactions):
        # 用户历史 × 物品相似度表，得到该用户对所有候选物品的分数。
        return np.asarray(interactions, dtype=np.float64) @ self.similarity
