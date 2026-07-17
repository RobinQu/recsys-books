"""Chapter adapter for the production-grade framework model.

The complete third-party implementation is displayed beside this file in the
tutorial source browser; construction parameters live in the training entry.
"""
from torch_rechub.models.ranking import DeepFM

# 阅读结构定义时先抓住以下三点：
# - FM 分支显式计算一阶与二阶交叉
# - DNN 分支学习更高阶非线性交互
# - 两条分支共享 embedding，避免同一特征学习两套不一致表示
# 完整 forward/layer 实现在页面的“框架源码”分组中，避免复制后与上游版本漂移。
MODEL_CLASS = DeepFM

def model_class():
    return MODEL_CLASS
