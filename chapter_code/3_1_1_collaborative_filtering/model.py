"""Model structure for 3.1.1 协同过滤：UserCF / ItemCF."""

class ItemCF:
    """Item-based collaborative filtering using cosine similarity."""
    def fit(self, interactions):
        import numpy as np
        # 每一列是一件物品；列向量记录哪些用户与它交互过。
        norms = np.linalg.norm(interactions, axis=0, keepdims=True) + 1e-8
        # 矩阵乘法一次得到所有物品对的共同用户数，再除以长度得到余弦相似度。
        self.similarity = (interactions.T @ interactions) / (norms.T @ norms)
        # 不允许物品把自己当作邻居，否则自身相似度 1 会淹没其他候选。
        np.fill_diagonal(self.similarity, 0.0)
        return self

    def score(self, interactions):
        # 用户历史 × 物品相似度表，得到该用户对所有候选物品的分数。
        return interactions @ self.similarity
