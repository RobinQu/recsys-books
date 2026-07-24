"""Chapter adapter for the production-grade framework model.

The complete third-party implementation is displayed beside this file in the
tutorial source browser; construction parameters live in the training entry.
"""
from torch_rechub.models.matching import SASRec

# 阅读结构定义时先抓住以下三点：
# - 因果 mask 保证位置 t 只能读取过去行为
# - 位置编码保存行为顺序，self-attention 选择相关历史
# - 最后位置的隐藏状态作为 next-item 召回用户向量
# 完整 forward/layer 实现在页面的“框架源码”分组中，避免复制后与上游版本漂移。
MODEL_CLASS = SASRec

def model_class():
    return MODEL_CLASS
