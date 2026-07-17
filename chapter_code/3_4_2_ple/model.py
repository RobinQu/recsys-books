"""Chapter adapter for the production-grade framework model.

The complete third-party implementation is displayed beside this file in the
tutorial source browser; construction parameters live in the training entry.
"""
from torch_rechub.models.multi_task import PLE

# 阅读结构定义时先抓住以下三点：
# - 共享 expert 与任务专属 expert 明确分开
# - 每层 gate 控制信息流向，减少不相关任务的负迁移
# - 多层渐进抽取后再进入各任务 tower
# 完整 forward/layer 实现在页面的“框架源码”分组中，避免复制后与上游版本漂移。
MODEL_CLASS = PLE

def model_class():
    return MODEL_CLASS
