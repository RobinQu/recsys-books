"""Chapter adapter for the production-grade framework model.

The complete third-party implementation is displayed beside this file in the
tutorial source browser; construction parameters live in the training entry.
"""
from torch_rechub.models.ranking import DIN

# 阅读结构定义时先抓住以下三点：
# - 候选物品作为 query，对历史行为逐条计算相关性
# - 加权历史表示会随候选变化
# - 拼接用户、候选与兴趣表示后预测点击概率
# 完整 forward/layer 实现在页面的“框架源码”分组中，避免复制后与上游版本漂移。
MODEL_CLASS = DIN

def model_class():
    return MODEL_CLASS
