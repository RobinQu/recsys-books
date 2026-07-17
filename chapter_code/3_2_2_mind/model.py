"""Chapter adapter for the production-grade framework model.

The complete third-party implementation is displayed beside this file in the
tutorial source browser; construction parameters live in the training entry.
"""
from torch_rechub.models.matching import MIND

# 阅读结构定义时先抓住以下三点：
# - 行为序列先经动态路由聚成多个兴趣向量
# - 每个兴趣向量独立召回，最后合并候选
# - label-aware attention 只用于训练时选择与目标物品最相关的兴趣
# 完整 forward/layer 实现在页面的“框架源码”分组中，避免复制后与上游版本漂移。
MODEL_CLASS = MIND

def model_class():
    return MODEL_CLASS
