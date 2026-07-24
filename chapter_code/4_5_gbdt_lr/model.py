"""Model structure for 4.5 GBDT+LR."""

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

def build_models(seed=2026):
    """The tree creates leaf features; LR calibrates their joint score."""
    # GBDT 负责非线性分桶/组合；训练后应把叶节点编号 one-hot 化再输入 LR。
    return GradientBoostingClassifier(random_state=seed), LogisticRegression(max_iter=300)
