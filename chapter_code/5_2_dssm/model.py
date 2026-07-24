"""Chapter adapter for the production-grade framework model.

The complete third-party implementation is displayed beside this file in the
tutorial source browser; construction parameters live in the training entry.
"""
from torch_rechub.models.matching import DSSM

# 阅读结构定义时先抓住以下三点：
# - user tower 与 item tower 分别编码，才能离线预计算物品向量
# - 两个塔输出同维向量，余弦/内积越大表示偏好越强
# - 训练使用点击标签；线上召回用用户向量查询 ANN 索引
# 完整 forward/layer 实现在页面的“框架源码”分组中，避免复制后与上游版本漂移。
MODEL_CLASS = DSSM

def model_class():
    return MODEL_CLASS
