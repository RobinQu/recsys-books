"""Chapter adapter for the production-grade framework model.

The complete third-party implementation is displayed beside this file in the
tutorial source browser; construction parameters live in the training entry.
"""
from torch_rechub.models.ranking import DIEN

# 阅读结构定义时先抓住以下三点：
# - GRU 从行为序列提取逐时刻兴趣状态
# - 辅助损失要求相邻兴趣能够区分真实下一行为和负样本
# - AUGRU 用候选相关性控制兴趣状态更新
# 完整 forward/layer 实现在页面的“框架源码”分组中，避免复制后与上游版本漂移。
MODEL_CLASS = DIEN

def model_class():
    return MODEL_CLASS
