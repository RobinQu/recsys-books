"""Chapter adapter for the production-grade framework model.

The complete third-party implementation is displayed beside this file in the
tutorial source browser; construction parameters live in the training entry.
"""
from torch_rechub.models.multi_task import MMOE

# 阅读结构定义时先抓住以下三点：
# - 多个 expert 学习不同共享模式
# - 每个任务的 gate 生成独立专家权重
# - 任务 tower 只读取自己的混合结果并输出对应目标
# 完整 forward/layer 实现在页面的“框架源码”分组中，避免复制后与上游版本漂移。
MODEL_CLASS = MMOE

def model_class():
    return MODEL_CLASS
