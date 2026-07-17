"""Chapter adapter for the production-grade framework model.

The complete third-party implementation is displayed beside this file in the
tutorial source browser; construction parameters live in the training entry.
"""
from torch_rechub.models.generative import HSTUModel

# 阅读结构定义时先抓住以下三点：
# - item embedding 将高基数 ID 映射为稠密向量
# - HSTU block 针对非平稳长行为流建模
# - 每个位置预测下一物品，最后位置用于在线推荐
# 完整 forward/layer 实现在页面的“框架源码”分组中，避免复制后与上游版本漂移。
MODEL_CLASS = HSTUModel

def model_class():
    return MODEL_CLASS
