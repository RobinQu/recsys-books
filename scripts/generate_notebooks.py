from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

from tutorial_deep_specs import build_deep_specs
from tutorial_math_specs import build_math_specs
from tutorial_opening_specs import build_opening_specs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"


def md(text: str): return nbf.v4.new_markdown_cell(text.strip())
def code(text: str): return nbf.v4.new_code_cell(text.strip())


SETUP = """
from pathlib import Path
import os, sys, json
import torch
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
ARTIFACT_ROOT = Path(os.environ.get("RECSYS_ARTIFACT_ROOT", PROJECT_ROOT)).expanduser().resolve()
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("RECSYS_PROFILE", "full")
PROFILE = os.environ["RECSYS_PROFILE"]
CUDA_AVAILABLE = torch.cuda.is_available()
from recsys_lab.data import (load_movielens, movielens_provenance, load_amazon_2023,
                             amazon_provenance, load_kuairand, kuairand_provenance,
                             load_amazon_2018, amazon_2018_provenance,
                             load_movielens_1m, movielens_1m_provenance,
                             load_mind_amazon_books, mind_amazon_provenance,
                             load_census_income, census_income_provenance)
DATASET_KEY = "__DATASET_KEY__"
if DATASET_KEY == "movielens":
    real_ratings, real_movies = load_movielens()
    real_interactions = real_ratings
    REAL_DATASET = movielens_provenance(real_ratings)
elif DATASET_KEY == "amazon-books":
    real_ratings = load_amazon_2018("Books") if PROFILE == "full" else load_amazon_2023()
    real_interactions, real_movies = real_ratings, None
    REAL_DATASET = amazon_2018_provenance(real_ratings) if PROFILE == "full" else amazon_provenance(real_ratings)
elif DATASET_KEY == "mind-amazon-books":
    real_ratings = load_mind_amazon_books() if PROFILE == "full" else load_amazon_2023()
    real_interactions, real_movies = real_ratings, None
    REAL_DATASET = mind_amazon_provenance(real_ratings) if PROFILE == "full" else amazon_provenance(real_ratings)
elif DATASET_KEY == "amazon-electronics":
    real_ratings = load_amazon_2018("Electronics") if PROFILE == "full" else load_amazon_2023()
    real_interactions, real_movies = real_ratings, None
    REAL_DATASET = amazon_2018_provenance(real_ratings) if PROFILE == "full" else amazon_provenance(real_ratings)
elif DATASET_KEY == "movielens-1m":
    real_ratings = load_movielens_1m() if PROFILE == "full" else load_amazon_2023()
    real_interactions, real_movies = real_ratings, None
    REAL_DATASET = movielens_1m_provenance(real_ratings) if PROFILE == "full" else amazon_provenance(real_ratings)
elif DATASET_KEY == "census-income" and PROFILE == "full":
    census_train_x, census_train_y, census_test_x, census_test_y = load_census_income()
    real_interactions, real_movies, real_ratings = census_train_x, None, census_train_x
    REAL_DATASET = census_income_provenance()
elif DATASET_KEY == "amazon-2023":
    real_ratings = load_amazon_2023()
    real_interactions, real_movies = real_ratings, None
    REAL_DATASET = amazon_provenance(real_ratings)
else:
    real_interactions, real_movies = load_kuairand()
    real_ratings = real_interactions
    REAL_DATASET = kuairand_provenance(real_interactions)
print({"profile": PROFILE, "project_root": str(PROJECT_ROOT), "artifact_root": str(ARTIFACT_ROOT), "real_dataset": REAL_DATASET,
       "cuda_available": CUDA_AVAILABLE,
       "cuda_device": torch.cuda.get_device_name(0) if CUDA_AVAILABLE else None})
assert REAL_DATASET["randomly_fabricated_rows"] == 0
"""


def dataset_for_title(title: str) -> tuple[str, str]:
    if title.startswith(("3.", "4.")):
        return "movielens", "GroupLens MovieLens latest-small：经典评分与邻域任务"
    if title.startswith("5.5 总结"):
        return "amazon-2023", "Amazon Reviews 2023 Video Games 5-core：召回章节结果汇总（不重新训练）"
    if "MIND" in title:
        return "mind-amazon-books", "Amazon Books 2014：按 MIND 论文执行迭代 10-core，并核验 6,271,511 行"
    if "DSSM" in title:
        return "amazon-books", "Amazon Reviews Books 5-core：DSSM 搜索日志不可公开时的完整电商迁移实验"
    if "SASRec" in title:
        return "movielens-1m", "MovieLens 1M：按 SASRec 论文协议执行 leave-two-out 与 100 负例评测"
    if "DIN" in title or "DIEN" in title:
        return "amazon-electronics", "Amazon Reviews Electronics 5-core：DIN/DIEN 公开复现实验数据"
    if "MMoE" in title or "PLE" in title:
        return "census-income", "Census-Income KDD：MMoE/PLE 论文公开多任务实验的完整官方 train/test"
    if title.startswith("5."):
        return "amazon-2023", "Amazon Reviews 2023 Video Games 5-core：召回章节导读与汇总"
    return "kuairand", "KuaiRand-Pure：真实短视频曝光、点击、长播与多反馈序列"


def notebook(title: str, goal: str, source: str, cells: list):
    dataset_key, dataset_description = dataset_for_title(title)
    requires_cuda = title.startswith("8.")
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "recsys": {"profile": "full", "requires_cuda": requires_cuda, "source_of_truth": "HTML and notebook share app/content.py + recsys_lab"},
    }
    nb["cells"] = [
        md(f"# {title}\n\n> 阅读版与 Web 应用内容一致；实验数值来自本 Notebook 的已执行输出。\n\n## Goal\n\n{goal}"),
        md(f"## Setup\n\n本 Notebook 的默认真实数据是 **{dataset_description}**。`smoke` 档读取仓库内可审计的确定性切片，`full` 档扩大到官方完整文件；两档都不制造交互、曝光、标签或行为序列。切片规则、源地址、哈希与许可记录在 `data/README.md` 及对应数据目录。{' **生成式章节默认要求 CUDA；无 CUDA 时只允许自动化测试执行 CPU basic smoke，不进行完整精度验证。**' if requires_cuda else ''}\n\n**主要资料：** {source}"),
        code(SETUP.replace("__DATASET_KEY__", dataset_key)),
        *cells,
    ]
    return nb


CURRICULUM_FONT_SETUP = r"""
from matplotlib import font_manager

# Matplotlib 默认的 DejaVu Sans 不包含中文字形。优先选择容器中的 Noto CJK
# （镜像已安装 fonts-noto-cjk），其次是宿主机常见中文字体；从字体根源解决，
# 而不是用 warnings.filterwarnings 掩盖 missing glyph 警告。
cjk_candidates = ('Noto Sans CJK SC', 'Noto Sans CJK JP', 'PingFang SC', 'Hiragino Sans GB',
                  'Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'Songti SC')
installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
cjk_font = next((name for name in cjk_candidates if name in installed_fonts), None)
if cjk_font:
    plt.rcParams['font.sans-serif'] = [cjk_font, 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
print('图表字体:', cjk_font or '未找到中文字体（请安装 fonts-noto-cjk）')
"""


CURRICULUM_SETUP = r"""
from pathlib import Path
import os, sys
import numpy as np
import matplotlib.pyplot as plt
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
ARTIFACT_ROOT = Path(os.environ.get("RECSYS_ARTIFACT_ROOT", PROJECT_ROOT)).expanduser().resolve()
sys.path.insert(0, str(PROJECT_ROOT))
TEACHING_OBJECTS_ONLY = True
print({"project_root": str(PROJECT_ROOT), "artifact_root": str(ARTIFACT_ROOT), "kind": "curriculum", "teaching_objects_only": True})
""" + CURRICULUM_FONT_SETUP


CURRICULUM_DATA_SETUP = r"""
from pathlib import Path
import os, sys
import numpy as np
import matplotlib.pyplot as plt
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
ARTIFACT_ROOT = Path(os.environ.get("RECSYS_ARTIFACT_ROOT", PROJECT_ROOT)).expanduser().resolve()
sys.path.insert(0, str(PROJECT_ROOT))
from recsys_lab.data import load_movielens, movielens_provenance
real_ratings, real_movies = load_movielens()
REAL_DATASET = movielens_provenance(real_ratings)
print({"project_root": str(PROJECT_ROOT), "artifact_root": str(ARTIFACT_ROOT), "kind": "curriculum", "real_dataset": REAL_DATASET})
assert len(real_ratings) > 0
assert REAL_DATASET["randomly_fabricated_rows"] == 0
""" + CURRICULUM_FONT_SETUP


def curriculum_notebook(title: str, goal: str, cells: list, *, uses_movielens: bool = False):
    """Build a reusable 3.0 course without applying algorithm title/dataset inference."""
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "recsys": {
            "kind": "curriculum",
            "profile": "full",
            "requires_cuda": False,
            "source_of_truth": "scripts/tutorial_math_specs.py",
        },
    }
    data_note = (
        "本课读取仓库中的 **真实 MovieLens** 评分行来辨认样本、实体、特征与标签；来源和切片均可审计，绝不补造交互。"
        if uses_movielens
        else "本课不加载用户行为数据。代码中的小数组都是带标签的 **数学教学对象**，只用于验证公式，绝不冒充交互、曝光、标签或行为序列。"
    )
    nb["cells"] = [
        md(f"# {title}\n\n> 3.0 基础课程：先直觉，再符号，再数字代入，再用代码和图形核对。\n\n## Goal\n\n{goal}"),
        md(f"## Setup 与数据边界\n\n{data_note}"),
        code(CURRICULUM_DATA_SETUP if uses_movielens else CURRICULUM_SETUP),
        *cells,
    ]
    return nb


def main():
    specs = {}
    specs["3_8_data_pipeline"] = notebook(
        "3.8 数据与实验基础：从 import 到训练循环",
        "拆开教程中常见的 recsys_lab.data 与 industrial_experiments：先阅读函数源码，再用真实 MovieLens 行为逐步重写时间切分、张量化、训练、推理和测试，建立从 Notebook 到工程模块的完整调用地图。",
        "[Python inspect 官方文档](https://docs.python.org/3/library/inspect.html) · [PyTorch Optimizing Model Parameters](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)",
        [
            md(r"""
## 1. 为什么可以 import，但不能把理解也藏起来

工程代码把反复使用的数据读取、切分、训练和落盘封装成函数，优点是每个算法不用复制几百行相同代码，修复泄漏或指标错误时也只改一处。教程需要再加一层约束：

1. Notebook 正文先解释输入、输出和关键步骤；
2. 用一个小例子逐步重写核心逻辑；
3. 给出函数源码与公式—代码映射；
4. 完整工程文件可在“查看实现源码”或浏览器 IDE 中继续阅读。

调用链不是黑盒：

```text
official CSV -> load_* -> deterministic subset -> time split
             -> task tensors -> model.forward -> loss.backward
             -> inference -> metrics -> results/*.json
```

`import` 只是告诉 Python“函数定义在另一个文件”，并不代表它来自不可见的库。下面让 Python 自己打印函数定义和文件位置。
"""),
            code(r"""
import inspect
from recsys_lab import data as data_tools
from recsys_lab import industrial_experiments as experiments

for fn in [data_tools.load_movielens, data_tools.leave_last_out, experiments.run_dssm]:
    print(f"\n{fn.__name__}{inspect.signature(fn)}")
    print("defined in:", inspect.getsourcefile(fn))
print("\nleave_last_out source:\n", inspect.getsource(data_tools.leave_last_out))
loader_lines = inspect.getsource(data_tools._load_cached).splitlines()
print("\nMovieLens loader 的前 36 行（完整函数可在源码讲解页查看）：\n", "\n".join(loader_lines[:36]))
"""),
            md(r"""
## 2. 数据加载：确定性切片不是随机造数据

`load_movielens()` 做四件事：读官方 CSV；按行为数稳定选择用户与物品；把原始 ID 映射为连续整数；派生时间、类型和标签列。模型需要连续 ID 来查 embedding，但原始 `userId/movieId` 仍保留以便审计。

下面直接查看真实行和来源记录。`randomly_fabricated_rows=0` 是机器可测试的数据承诺；它不表示数据没有抽样，而表示所有保留行都能追溯到官方文件。
"""),
            code(r"""
from recsys_lab.data import load_movielens, movielens_provenance

ratings, movies = load_movielens(max_users=12, max_items=80, min_user_events=8)
display(ratings[["userId", "movieId", "user_id", "item_id", "rating", "timestamp"]].head())
print(movielens_provenance(ratings))
assert len(ratings) > 0 and ratings.userId.notna().all()
"""),
            md(r"""
## 3. 时间切分：先手写，再与工具函数核对

leave-last-out 对每位用户保留时间最晚的一条作为测试目标，其余进入训练。排序键中加入 `item_id`，是为了在时间戳相同时仍得到稳定结果。关键原则是用过去预测未来；如果随机切分，未来行为可能泄漏到用户历史。

公式不复杂。对用户 $u$ 的事件时间集合 $T_u$，测试事件索引为 $t_u^*=\arg\max_{t\in T_u}t$，训练集为其余事件。
"""),
            code(r"""
# 逐步重写工具函数
ordered = ratings.sort_values(["user_id", "timestamp", "item_id"]).copy()
last_indices = ordered.groupby("user_id").tail(1).index
manual_test = ordered.loc[last_indices].sort_values("user_id")
manual_train = ordered.drop(last_indices)

helper_train, helper_test = data_tools.leave_last_out(ratings)
assert manual_test[["user_id", "item_id"]].reset_index(drop=True).equals(
    helper_test[["user_id", "item_id"]].reset_index(drop=True)
)
print({"train_rows": len(manual_train), "test_rows": len(manual_test),
       "one_test_row_per_user": bool((manual_test.groupby("user_id").size() == 1).all())})
"""),
            md(r"""
## 4. 从 DataFrame 到张量

DataFrame 适合检查列和时间；模型计算需要张量。下面把 user/item ID 变成整数张量，把评分变成浮点目标。若有 $B$ 行样本、embedding 维数为 $d$，查表后 user/item 形状都是 $[B,d]$，逐维相乘再求和得到 $[B]$ 个预测。
"""),
            code(r"""
import torch

train_users = torch.tensor(manual_train.user_id.to_numpy(), dtype=torch.long)
train_items = torch.tensor(manual_train.item_id.to_numpy(), dtype=torch.long)
train_targets = torch.tensor(manual_train.rating.to_numpy(), dtype=torch.float32)
print({"users": tuple(train_users.shape), "items": tuple(train_items.shape),
       "targets": tuple(train_targets.shape), "dtypes": [str(train_users.dtype), str(train_targets.dtype)]})
"""),
            md(r"""
## 5. 训练循环：forward、loss、backward、step

训练循环的五个动作适用于 DSSM、DeepFM、DIN 和 HSTU，只是 `forward` 与损失定义不同：

1. `model(...)` 计算预测；
2. `loss_fn` 把预测错误压成一个数；
3. `zero_grad()` 清除上一轮梯度；
4. `backward()` 按链式法则计算每个参数对损失的影响；
5. `step()` 沿降低损失的方向更新参数。

这里用最小矩阵分解示范。预测 $\hat r_{ui}=p_u^\top q_i$，均方误差 $L=\frac1B\sum(\hat r-r)^2$。代码使用的每一行都来自真实评分。
"""),
            code(r"""
class TinyMF(torch.nn.Module):
    def __init__(self, users, items, dim=8):
        super().__init__()
        self.user = torch.nn.Embedding(users, dim)
        self.item = torch.nn.Embedding(items, dim)
    def forward(self, user_id, item_id):
        return (self.user(user_id) * self.item(item_id)).sum(dim=1)

torch.manual_seed(7)
model = TinyMF(int(ratings.user_id.max()) + 1, int(ratings.item_id.max()) + 1)
optimizer = torch.optim.Adam(model.parameters(), lr=.03)
loss_fn = torch.nn.MSELoss()
losses = []
for epoch in range(20):
    prediction = model(train_users, train_items)       # forward
    loss = loss_fn(prediction, train_targets)          # score the error
    optimizer.zero_grad()                              # clear old gradients
    loss.backward()                                    # chain rule
    optimizer.step()                                   # update parameters
    losses.append(float(loss.detach()))
print({"loss_start": round(losses[0], 4), "loss_end": round(losses[-1], 4)})
assert losses[-1] < losses[0]
"""),
            md(r"""
## 6. 推理与测试为什么必须分开

推理只做前向计算，不修改参数，因此使用 `model.eval()` 和 `torch.no_grad()`。测试集 RMSE 为

$$\mathrm{RMSE}=\sqrt{\frac1N\sum_{n=1}^N(\hat r_n-r_n)^2}.$$

它回答“评分数值差多少”，不能替代 Top-K 召回指标。深度算法 Notebook 也遵循同样边界：训练函数可以封装，但测试目标、候选集和指标公式必须在正文中说清。
"""),
            code(r"""
test_users = torch.tensor(helper_test.user_id.to_numpy(), dtype=torch.long)
test_items = torch.tensor(helper_test.item_id.to_numpy(), dtype=torch.long)
test_targets = torch.tensor(helper_test.rating.to_numpy(), dtype=torch.float32)
model.eval()
with torch.no_grad():
    test_prediction = model(test_users, test_items)
    rmse = torch.sqrt(torch.mean((test_prediction - test_targets) ** 2)).item()
print({"test_rows": len(test_targets), "RMSE": round(rmse, 4)})
assert torch.isfinite(test_prediction).all()
"""),
            md(r"""
## 7. 怎样阅读 `run_dssm` 这类完整实验

不要从第一行读到最后一行。按以下顺序定位：

1. `_real_amazon`：数据从哪里来；
2. 时间切分与 `fields`：一行表如何变成张量；
3. `DSSM(...)`：模型结构和超参数；
4. `_train_binary`：损失与反向传播；
5. `model.mode = user/item`：两座塔如何分开推理；
6. `_recall_single_target`：全库分数如何变成 Recall@K。

网页顶部的“查看实现源码”会按这些函数分段显示；“在 IDE 中打开”适合跨文件搜索、跳转定义和临时修改。教程正文解释设计，源码页解释实现，IDE 负责自由探索，三者各司其职。
"""),
            md("## Checks\n\n本章已经用真实数据验证加载、时间切分、张量化、训练、推理和测试；也证明手写切分与公共工具一致。"),
            code("assert REAL_DATASET['randomly_fabricated_rows'] == 0\nassert len(helper_test) == ratings.user_id.nunique()\nassert losses[-1] < losses[0]\nprint('PASS：公共管线已逐步展开并完成端到端验证。')"),
            md("## Next Steps\n\n回到任一深度模型 Notebook，先读 Model Structure & Formula Walkthrough，再用源码讲解页把公式对应到实际函数；需要修改实现时进入浏览器 IDE。"),
        ],
    )
    specs["3_1_math_foundations"] = notebook(
        "3.1 基础课程总览：从数据语义到模型训练",
        "用一张循序递进的路线图串起六门数学与机器学习基础课，以及最后的数据实验管线。这里建立共同语言和最小手算直觉；完整教学沿 3.2—3.8 逐课展开。",
        "[MIT OCW Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) · [Google ML Crash Course: Logistic Regression](https://developers.google.com/machine-learning/crash-course/logistic-regression) · [scikit-learn Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)",
        [
            md(r"""
## Steps

## 0. 3.0 渐进学习路线与算法依赖图

3.0 负责建立所有算法共用的语言；算法页只展开论文新引入的结构与公式。建议按表格从上到下学习，遇到熟悉内容也可直接跳到对应的精确小节；六门概念课之后，再进入 [3.8 数据与实验基础](/notebooks/3_8_data_pipeline)：

| 共同领域 | 先掌握什么 | 后续主要用在 |
|---|---|---|
| 数据与机器学习 | [一行数据、特征与标签](/notebooks/3_2_data_ml_basics#observation-label)；[显式/隐式反馈、曝光、未知与假负例](/notebooks/3_2_data_ml_basics#implicit-feedback)；[切分、泄漏与泛化](/notebooks/3_2_data_ml_basics#split-leakage) | 全部章节的数据协议与结果边界 |
| 线性代数 | [张量形状与轴](/notebooks/3_3_linear_algebra#tensors-shapes)；[逐元素运算、点积与余弦](/notebooks/3_3_linear_algebra#elementwise-dot)；[矩阵乘法与 embedding](/notebooks/3_3_linear_algebra#matmul-embedding) | CF、MF、FM、双塔、注意力 |
| 微积分 | [函数与复合](/notebooks/3_4_calculus#functions)；[偏导与梯度](/notebooks/3_4_calculus#derivative-gradient)；[链式法则与反向传播](/notebooks/3_4_calculus#chain-rule) | 所有需要训练参数的模型 |
| 概率与统计 | [条件概率](/notebooks/3_5_probability_statistics#conditional-chain)；[期望与方差](/notebooks/3_5_probability_statistics#expectation-variance)；[odds、logit 与校准](/notebooks/3_5_probability_statistics#likelihood-calibration) | CTR、多任务、序列生成与实验解读 |
| 信息论 | [交叉熵与 KL](/notebooks/3_6_information_theory#cross-entropy-kl)；[Softmax、温度与 NE](/notebooks/3_6_information_theory#softmax-temperature)；[序列 NLL](/notebooks/3_6_information_theory#sequence-nll-dpo) | LR/GBDT、word2vec、DSSM、Transformer |
| 优化 | [SGD 与 mini-batch](/notebooks/3_7_optimization#sgd)；[学习率](/notebooks/3_7_optimization#learning-rate)；[正则化、过拟合与早停](/notebooks/3_7_optimization#regularization) | MF、深度模型、生成式模型 |

依赖顺序可以压成一句话：**先说明一行数据代表什么 → 写出数组形状与运算 → 把模型看成复合函数 → 用概率定义预测与损失 → 用梯度和小批次优化 → 用指标检查泛化。**

## 1. 从“行为表”到矩阵

推荐系统最原始的数据不是公式，而是一张行为表：谁在什么时间看了什么。为了同时观察许多用户与物品，我们把它整理成矩阵 $R$：

- 每一行是一位用户；
- 每一列是一个物品；
- 数字 1 表示发生过喜欢/点击，0 表示没有观察到；
- $R\in\mathbb R^{3\times4}$ 只是在说“这张表有 3 行、4 列”。

注意数据语义不能混用：星级是用户主动给出的**显式反馈**；点击/观看是行为留下的**隐式反馈**；“曝光后未点击”才是有明确分母的负标签；从未曝光或日志未记录通常只能叫**未知**。从未知项抽作负样本会产生**假负例**，必须记录采样规则。完整例子见 [3.2 显式、隐式反馈与未知项](/notebooks/3_2_data_ml_basics#implicit-feedback)。
"""),
            code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

users = ["u0", "u1", "u2"]
items = ["i0", "i1", "i2", "i3"]
R = np.array([
    [1, 1, 0, 0],
    [1, 0, 1, 0],
    [0, 1, 1, 1],
], dtype=float)

display(pd.DataFrame(R, index=users, columns=items))
fig, ax = plt.subplots(figsize=(6.4, 3.1))
image = ax.imshow(R, cmap="YlGn", vmin=0, vmax=1)
ax.set_xticks(range(len(items)), items); ax.set_yticks(range(len(users)), users)
ax.set_xlabel("items (columns)"); ax.set_ylabel("users (rows)")
for row in range(R.shape[0]):
    for col in range(R.shape[1]): ax.text(col, row, int(R[row, col]), ha="center", va="center", fontsize=13)
ax.set_title("User-item interaction matrix R")
plt.colorbar(image, ax=ax, ticks=[0, 1], label="observed interaction")
plt.tight_layout(); plt.show()
"""),
            md(r"""
### 1.1 向量只是“一行数字”

用户 `u0` 的行为向量是 $[1,1,0,0]$。四个位置像四个问题的答案：“看过 i0 吗？看过 i1 吗？……”因此向量不是抽象符号，而是把一个对象沿多个维度描述成一行数字。

向量的**长度**使用勾股定理：$\|a\|_2=\sqrt{a_1^2+\cdots+a_n^2}$。`u0` 有两个 1，所以长度是 $\sqrt2$。
"""),
            code(r"""
u0 = R[0]
u1 = R[1]
print({"u0 vector": u0.tolist(), "u0 length": np.linalg.norm(u0), "expected": np.sqrt(2)})
assert np.isclose(np.linalg.norm(u0), np.sqrt(2))
"""),
            md(r"""
## 2. 点积与余弦：把“共同选择”变成相似度

两个等长向量逐位置相乘再求和，叫作**点积**：

$$a\cdot b=a_1b_1+a_2b_2+\cdots+a_nb_n$$

对 0/1 行为来说，只有“两个人都为 1”的位置会贡献 1，所以点积正好数出共同选择。`u0·u1 = 1`，因为两人共同看过 i0。

但活跃用户可能仅仅因为看得多而获得更大点积。余弦相似度再除以两人的向量长度，把答案压到 0～1：

$$\cos(a,b)=\frac{a\cdot b}{\|a\|_2\|b\|_2}$$

可以把它理解为“方向有多一致”：1 表示选择比例完全一致，0 表示没有共同选择。
"""),
            code(r"""
dot = float(u0 @ u1)
cosine = dot / (np.linalg.norm(u0) * np.linalg.norm(u1))
print({"common choices / dot product": dot, "cosine similarity": round(cosine, 3)})

fig, ax = plt.subplots(figsize=(5.4, 5.0))
a2, b2 = np.array([1, 1]), np.array([1, 0])
ax.quiver(0, 0, *a2, angles="xy", scale_units="xy", scale=1, color="#4f8f00", label="u0=(1,1)")
ax.quiver(0, 0, *b2, angles="xy", scale_units="xy", scale=1, color="#e36b2c", label="u1=(1,0)")
ax.set_xlim(-.1, 1.4); ax.set_ylim(-.1, 1.4); ax.set_aspect("equal")
ax.grid(alpha=.25); ax.legend(); ax.set_title(f"Same direction means larger cosine = {cosine:.3f}")
ax.set_xlabel("interaction dimension 1"); ax.set_ylabel("interaction dimension 2")
plt.tight_layout(); plt.show()
"""),
            md(r"""
## 3. 矩阵乘法：一次完成许多次点积

矩阵乘法没有增加新的神秘规则，它只是批量点积：

- $R R^\top$：每个用户与每个用户做点积，得到用户共现矩阵；
- $R^\top R$：每个物品与每个物品做点积，得到物品共现矩阵。

乘法能否进行，只需检查“中间尺寸相同”。$R$ 是 $3\times4$，$R^\top$ 是 $4\times3$，所以 $(3\times4)(4\times3)$ 得到 $3\times3$。

先看**轴与形状**再看公式：批次张量 $(B,L,d)$ 的三个轴通常是样本、序列位置、特征。`a * b` 的逐元素乘法保留每个位置，结果仍是向量；`a @ b` 的点积会把对应位置乘积求和，结果是一个数；矩阵乘法则批量做点积。程序允许广播不代表语义正确，完整手算见 [3.3 张量形状与轴](/notebooks/3_3_linear_algebra#tensors-shapes) 和 [逐元素运算与点积](/notebooks/3_3_linear_algebra#elementwise-dot)。
"""),
            code(r"""
user_cooccurrence = R @ R.T
item_cooccurrence = R.T @ R
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
for ax, matrix, title, xlabels, ylabels in [
    (axes[0], R, "R: behavior", items, users),
    (axes[1], user_cooccurrence, "R @ R.T: user-user", users, users),
    (axes[2], item_cooccurrence, "R.T @ R: item-item", items, items),
]:
    ax.imshow(matrix, cmap="YlGn")
    ax.set_xticks(range(len(xlabels)), xlabels); ax.set_yticks(range(len(ylabels)), ylabels)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]): ax.text(col, row, int(matrix[row, col]), ha="center", va="center")
    ax.set_title(title)
plt.tight_layout(); plt.show()
print({"R": R.shape, "R @ R.T": user_cooccurrence.shape, "R.T @ R": item_cooccurrence.shape})
"""),
            md(r"""
## 4. 概率、Sigmoid 与 LogLoss

排序模型常先输出任意实数 $z$，叫 **logit**。它可能是 -3、0 或 5，不能直接当概率。Sigmoid 把整条实数轴平滑压到 0～1：

$$p=\sigma(z)=\frac{1}{1+e^{-z}}$$

- $z=0$ 时 $p=0.5$，表示模型拿不准；
- $z$ 很大时概率接近 1；$z$ 很小时接近 0。

LogLoss 衡量概率预测有多糟。真实点击 $y=1$ 时，预测 0.9 只受很小惩罚，预测 0.01 会受到很大惩罚，因为模型“非常自信但答错了”：

$$L=-[y\log p+(1-y)\log(1-p)]$$

这里的指数与对数并不神秘：$e^z$ 会把加法尺度变成正的倍数尺度，$\log$ 是它的反函数，并把连乘变回相加。概率 $p$ 对应的 **odds** 是 $p/(1-p)$；取对数得到 $\log\frac{p}{1-p}=z$，所以 Sigmoid 正是在把 log-odds 还原成概率。若已知用户属于某人群 $B$，点击概率写成条件概率 $P(Y=1\mid B)$；许多样本的平均结果由期望 $E[Y]$ 描述。详见 [3.5 条件概率](/notebooks/3_5_probability_statistics#conditional-chain)、[期望与方差](/notebooks/3_5_probability_statistics#expectation-variance) 和 [odds、logit 与校准](/notebooks/3_5_probability_statistics#likelihood-calibration)。

最后要区分**排序**与**校准**：AUC 高只说明正例通常排在负例前；若模型给出 0.8 的一组样本长期只有约 50% 为正，它仍未校准。LogLoss 会惩罚这种概率误差，但上线前仍应画可靠性图。
"""),
            code(r"""
logits = np.linspace(-6, 6, 300)
probabilities = 1 / (1 + np.exp(-logits))
p_grid = np.linspace(.01, .99, 300)
loss_when_clicked = -np.log(p_grid)
loss_when_not_clicked = -np.log(1 - p_grid)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].plot(logits, probabilities, color="#4f8f00", lw=3); axes[0].axhline(.5, ls="--", color="gray")
axes[0].set(title="Sigmoid: score to probability", xlabel="logit z", ylabel="p(click)"); axes[0].grid(alpha=.2)
axes[1].plot(p_grid, loss_when_clicked, label="actual y=1", lw=2); axes[1].plot(p_grid, loss_when_not_clicked, label="actual y=0", lw=2)
axes[1].set(title="LogLoss punishes confident mistakes", xlabel="predicted p(click)", ylabel="loss"); axes[1].legend(); axes[1].grid(alpha=.2)
plt.tight_layout(); plt.show()
print({"sigmoid(0)": .5, "loss if y=1,p=.9": round(float(-np.log(.9)), 3), "loss if y=1,p=.01": round(float(-np.log(.01)), 3)})
"""),
            md(r"""
## 5. Softmax、温度、加权汇总与遮罩

后续的 DSSM、MIND、DIN、SASRec、MMoE 会反复用到同一组操作：先算若干个分数，再把它们转成权重，最后汇总若干个向量。

Softmax 把任意实数 $z_1,\ldots,z_n$ 转成和为 1 的正数：

$$
\alpha_j=\frac{\exp(z_j/\tau)}{\sum_k\exp(z_k/\tau)},
\qquad \sum_j\alpha_j=1.
$$

$\tau$ 是温度：较小时权重更集中在最高分，较大时更平均。有了权重后，加权汇总就是

$$
v=\sum_j\alpha_j e_j.
$$

它可以理解为“每个历史向量 $e_j$ 贡献多少”。注意力只是让分数 $z_j$ 由当前 query 与 key 的匹配产生。对矩阵 $Q,K$ 通常写为

$$
S=\frac{QK^\top}{\sqrt d},\qquad A=\operatorname{softmax}(S+M),\qquad H=AV.
$$

$QK^\top$ 是批量点积；$\sqrt d$ 防止维数增大时分数过大；遮罩 $M$ 把不允许读取的位置设为 $-\infty$，Softmax 后它们的权重就是 0。序列模型用它阻止偷看未来，变长序列用它忽略 padding。
"""),
            code(r"""
scores = np.array([1.0, 2.0, 4.0])
values = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

def softmax_with_temperature(x, temperature=1.0):
    stable = x / temperature - np.max(x / temperature)
    weights = np.exp(stable)
    return weights / weights.sum()

for temperature in [0.5, 1.0, 2.0]:
    weights = softmax_with_temperature(scores, temperature)
    summary = weights @ values
    print({"temperature": temperature, "weights": weights.round(3).tolist(),
           "weighted vector": summary.round(3).tolist()})

masked_scores = np.array([1.0, 2.0, -np.inf])
masked_weights = softmax_with_temperature(masked_scores)
assert np.isclose(masked_weights.sum(), 1.0) and masked_weights[-1] == 0
"""),
            md(r"""
### 5.1 逐元素运算、残差/LayerNorm 与常见激活

拿到两个同形状向量 $a,b$，**逐元素乘积**（Hadamard 乘积，记作 $\odot$）就是对应位置分别相乘、不再求和：

$$
(a\odot b)_i=a_i b_i.
$$

DIN/DIEN/HSTU 用它表达“两个向量在每一维上共同激活多少”，和第 2 节的点积（最后求和成一个数）形成对照。把输入跳层加回输出，$H\leftarrow H+X$，叫**残差连接**，为深层网络保留一条直接信息与梯度路径；之后常做 **LayerNorm**，沿特征维标准化再缩放平移，让不同样本的数值尺度更稳定。

模型本质上是函数复合：$x\rightarrow z\rightarrow h(z)\rightarrow\hat y\rightarrow L$。激活函数就是其中的 $h$：

- **ReLU**：$\max(0,z)$，负数截为 0；
- **Sigmoid**：$1/(1+e^{-z})$，输出 0～1，可作概率或门；
- **Tanh**：输出 -1～1，常与循环网络的门配合；
- **Softplus**：$\log(1+e^z)$，是 ReLU 的平滑版本；
- **SiLU / Swish**：$z\sigma(z)$，让数值给自己加一道软门。

这些激活通常逐元素作用，不改变张量形状；Softmax 则跨一组位置归一化并让它们竞争权重。完整的函数复合见 [3.4 函数与复合](/notebooks/3_4_calculus#functions)，逐元素/点积对照见 [3.3](/notebooks/3_3_linear_algebra#elementwise-dot)。
"""),
            md(r"""
## 6. 梯度下降：沿着下坡方向一点点改参数

把模型参数想成山坡上的横坐标 $w$，损失 $L(w)$ 是高度。**梯度**就是当前位置的坡度：

- 坡度为正，向左走能下降；
- 坡度为负，向右走能下降；
- 接近 0，说明来到谷底附近。

多个参数时，“暂时固定其他参数，只让一个参数变化”的坡度叫**偏导**；把所有偏导排在一起就是梯度。模型是复合函数，所以反向传播从损失端按**链式法则**逐层相乘，并在分叉处相加；它不是另一套数学。详见 [3.4 偏导与梯度](/notebooks/3_4_calculus#derivative-gradient) 和 [链式法则与反向传播](/notebooks/3_4_calculus#chain-rule)。

更新公式 $w\leftarrow w-\eta\nabla L$ 中，$\eta$ 是学习率，像每一步的步长。过大会跨过谷底甚至发散，过小则走得很慢。实际训练常用 **mini-batch**：每次用一小批样本估计梯度，在内存、速度和噪声间折中。训练误差继续下降而验证误差上升叫**过拟合**；L1/L2 正则化、早停与更多可靠数据都是约束方法。详见 [3.7 SGD 与 mini-batch](/notebooks/3_7_optimization#sgd) 和 [正则化与过拟合](/notebooks/3_7_optimization#regularization)。下面在一元二次函数 $L(w)=(w-3)^2+1$ 上亲手走 12 步。
"""),
            code(r"""
learning_rate = 0.18
w = -2.0
path = [w]
for _ in range(12):
    gradient = 2 * (w - 3)
    w = w - learning_rate * gradient
    path.append(w)

x = np.linspace(-3, 6, 300); loss_curve = (x - 3) ** 2 + 1
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, loss_curve, color="#66736b", label="L(w)=(w-3)^2+1")
ax.scatter(path, [(point-3)**2+1 for point in path], c=np.arange(len(path)), cmap="YlGn", s=52, zorder=3)
for step, point in enumerate(path[::3]): ax.annotate(f"step {step*3}", (point, (point-3)**2+1), xytext=(4, 7), textcoords="offset points", fontsize=8)
ax.set(title="Gradient descent walks downhill", xlabel="parameter w", ylabel="loss"); ax.grid(alpha=.2); ax.legend()
plt.tight_layout(); plt.show()
print({"start": path[0], "after 12 steps": round(path[-1], 4), "optimal": 3.0})
"""),
            md(r"""
## 7. 训练集、测试集与推荐指标

模型在做过的题上得高分，不代表会做新题。因此必须：

1. **训练集**用来修改参数；
2. **验证集**用来选择超参数和停止时机；
3. **测试集**只在最后评估一次。

推荐系统还必须按时间切分：用过去预测未来，不能让未来行为泄漏进训练。下面用统一符号讲清指标：

- $u$：一位用户，$U$：参与评测的用户集合；
- $K$：只看推荐列表的前 $K$ 个位置；
- $R_u@K$：模型给用户 $u$ 的前 $K$ 个推荐；
- $G_u$：测试集中用户 $u$ 真正喜欢或点击的物品（ground truth）；
- $H_u@K=R_u@K\cap G_u$：推荐列表中的命中集合；
- $y_i$：真实标签或评分，$\hat y_i$：模型预测，$N$：样本数。

### 7.1 Top-K 集合指标：找得准、找得全、是否命中

**Precision@K（准确率）**问：“推荐出来的 $K$ 个物品中，有多少是对的？”

$$
\mathrm{Precision@K}(u)=\frac{|H_u@K|}{K}
$$

**Recall@K（召回率）**问：“用户真正喜欢的物品中，有多少被找回来了？”

$$
\mathrm{Recall@K}(u)=\frac{|H_u@K|}{|G_u|}
$$

**HitRate@K（命中率）**只关心有没有至少命中一个，不关心命中了几个：

$$
\mathrm{HitRate@K}=\frac{1}{|U|}\sum_{u\in U}\mathbb{I}(|H_u@K|>0)
$$

其中指示函数 $\mathbb I(\text{条件})$ 在条件成立时等于 1，否则等于 0。若每位用户测试集只有一个目标物品（leave-one-out），Recall@K 和 HitRate@K 数值相同；当每人有多个目标时二者不同。

**F1@K**是 Precision 与 Recall 的调和平均，只有两者都高时才会高：

$$
\mathrm{F1@K}=\frac{2\,\mathrm{Precision@K}\,\mathrm{Recall@K}}
{\mathrm{Precision@K}+\mathrm{Recall@K}}
$$

> 直觉：扩大 $K$ 通常会提高 Recall，却可能降低 Precision。因此比较模型时必须使用相同的 $K$。
"""),
            code(r"""
# 一位用户的 Top-K 手算：5 个推荐里命中 2 个；真正相关物品有 3 个
recommended = ["i3", "i7", "i2", "i9", "i1"]
relevant = {"i2", "i3", "i8"}
hits = set(recommended) & relevant

precision_at_5 = len(hits) / len(recommended)
recall_at_5 = len(hits) / len(relevant)
hit_at_5 = float(len(hits) > 0)
f1_at_5 = 2 * precision_at_5 * recall_at_5 / (precision_at_5 + recall_at_5)

print({
    "hits": sorted(hits),
    "Precision@5": round(precision_at_5, 3),  # 2/5
    "Recall@5": round(recall_at_5, 3),        # 2/3
    "Hit@5": hit_at_5,                        # 至少命中一次
    "F1@5": round(f1_at_5, 3),
})
"""),
            md(r"""
### 7.2 顺序敏感的列表指标：好东西是否排在前面

Precision/Recall 只看集合，不看命中的位置。推荐页面首屏尤其在意顺序。

**MRR（Mean Reciprocal Rank）**只看第一个相关结果的位置。若第一个命中在第 $r_u$ 位，则倒数排名为 $1/r_u$：

$$
\mathrm{MRR@K}=\frac{1}{|U|}\sum_{u\in U}
\begin{cases}
1/r_u,&r_u\le K\\
0,&\text{前 K 位无命中}
\end{cases}
$$

**DCG@K**允许每个结果有不同相关程度 $rel_{u,r}$，并用 $\log_2(r+1)$ 惩罚靠后的位置：

$$
\mathrm{DCG@K}(u)=\sum_{r=1}^{K}\frac{2^{rel_{u,r}}-1}{\log_2(r+1)}
$$

把相同物品按真实相关性从高到低排列可得到理想值 $\mathrm{IDCG@K}$。归一化以后，不同用户才方便平均：

$$
\mathrm{NDCG@K}(u)=\frac{\mathrm{DCG@K}(u)}{\mathrm{IDCG@K}(u)}\in[0,1]
$$

**AP@K（Average Precision）**在每个相关结果出现的位置计算 Precision，再取平均；**MAP@K**是所有用户 AP 的平均：

$$
\mathrm{AP@K}(u)=\frac{1}{\min(|G_u|,K)}
\sum_{r=1}^{K}\mathrm{Precision@r}(u)\cdot rel_{u,r},\qquad
\mathrm{MAP@K}=\frac{1}{|U|}\sum_u\mathrm{AP@K}(u)
$$

二值相关时 $rel_{u,r}\in\{0,1\}$。MRR 强调“第一次命中”，NDCG 能处理多档相关性，MAP 奖励把多个相关结果都提前。
"""),
            code(r"""
# 同样命中两个物品，放在更靠前的位置会得到更高的 MRR / NDCG / AP
binary_relevance = np.array([1, 0, 1, 0, 0])
graded_relevance = np.array([3, 0, 1, 0, 0])

first_hit_rank = np.flatnonzero(binary_relevance)[0] + 1
mrr_at_5 = 1 / first_hit_rank

ranks = np.arange(1, len(graded_relevance) + 1)
discounts = np.log2(ranks + 1)
dcg_at_5 = np.sum((2**graded_relevance - 1) / discounts)
ideal = np.sort(graded_relevance)[::-1]
idcg_at_5 = np.sum((2**ideal - 1) / discounts)
ndcg_at_5 = dcg_at_5 / idcg_at_5

precision_at_each_rank = np.cumsum(binary_relevance) / ranks
ap_at_5 = np.sum(precision_at_each_rank * binary_relevance) / min(binary_relevance.sum(), 5)
print({"MRR@5": round(mrr_at_5, 3), "NDCG@5": round(ndcg_at_5, 3), "AP@5": round(ap_at_5, 3)})
"""),
            md(r"""
### 7.3 评分误差：预测值离真实值多远

**MAE（平均绝对误差）**直接平均距离，单位与原评分相同：

$$
\mathrm{MAE}=\frac{1}{N}\sum_{i=1}^{N}|y_i-\hat y_i|
$$

**RMSE（均方根误差）**先平方再平均，因此对少数大错误惩罚更重：

$$
\mathrm{RMSE}=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat y_i)^2}
$$

二者都是越小越好。RMSE/MAE 适合 1～5 星等评分预测，却不保证 Top-K 排序更好。

### 7.4 概率与二分类排序：LogLoss、AUC 与 GAUC

点击率模型对样本 $i$ 输出概率 $p_i\in(0,1)$，真实标签 $y_i\in\{0,1\}$。

**LogLoss（二元交叉熵）**衡量概率预测本身是否可信，越小越好：

$$
\mathrm{LogLoss}=-\frac{1}{N}\sum_{i=1}^{N}
\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right]
$$

预测正确但不够自信仍会有小损失；非常自信地答错会受到巨大惩罚。因此 LogLoss 同时关心区分能力和概率质量。

**AUC**等价于随机抽一个正样本 $+$ 和一个负样本 $-$，正样本得分更高的概率：

$$
\mathrm{AUC}=P(s^+>s^-)+\frac{1}{2}P(s^+=s^-)
$$

AUC 为 0.5 表示接近随机排序，1 表示所有正样本都在负样本前。它只关心顺序：对所有分数做保持顺序的变换，AUC 不变，但 LogLoss 可能明显变化。

工业推荐常计算每位用户内部的 AUC，再按该用户曝光数 $n_u$ 加权，称为 **GAUC**：

$$
\mathrm{GAUC}=\frac{\sum_{u\in U'}n_u\,\mathrm{AUC}_u}{\sum_{u\in U'}n_u}
$$

$U'$ 只包含同时具有正、负样本的用户。GAUC 避免“用户之间本来就有不同点击率”虚增全局 AUC，更接近同一用户内部的排序质量。
"""),
            code(r"""
# 用 NumPy 手算评分、概率和排序指标
true_ratings = np.array([5.0, 3.0, 1.0, 4.0])
pred_ratings = np.array([4.5, 2.0, 2.0, 4.0])
mae = np.mean(np.abs(true_ratings - pred_ratings))
rmse = np.sqrt(np.mean((true_ratings - pred_ratings) ** 2))

labels = np.array([1, 0, 1, 0, 1, 0])
probabilities_for_metric = np.array([.9, .8, .7, .4, .6, .2])
eps = 1e-12
logloss = -np.mean(labels*np.log(probabilities_for_metric+eps) + (1-labels)*np.log(1-probabilities_for_metric+eps))
positive_scores = probabilities_for_metric[labels == 1]
negative_scores = probabilities_for_metric[labels == 0]
pairwise_auc = np.mean(
    (positive_scores[:, None] > negative_scores[None, :])
    + 0.5 * (positive_scores[:, None] == negative_scores[None, :])
)

print({"MAE": round(mae, 3), "RMSE": round(rmse, 3), "LogLoss": round(logloss, 3), "AUC": round(float(pairwise_auc), 3)})
"""),
            md(r"""
### 7.5 覆盖率与指标选择

准确率高不代表所有用户都在看到丰富的目录。**Catalog Coverage@K** 衡量全部用户的 Top-K 列表一共触达多少种物品：

$$
\mathrm{Coverage@K}=\frac{|\bigcup_{u\in U}R_u@K|}{|I|}
$$

其中 $I$ 是可推荐物品全集。Coverage 越高说明系统不只反复推荐少数热门物品，但覆盖率不能单独代表相关性或用户满意度。

| 任务问题 | 优先指标 | 必须搭配观察 |
|---|---|---|
| 百万物品召回是否漏掉目标 | Recall@K、HitRate@K | Coverage、延迟、分人群 Recall |
| Top-K 列表是否把好内容提前 | NDCG@K、MAP@K、MRR | Recall@K、多样性 |
| 星级评分是否接近真实值 | RMSE、MAE | Top-K 指标 |
| CTR/CVR 排序是否正确 | AUC、GAUC | LogLoss、校准、分桶稳定性 |

用户平均通常采用 **macro average**：先算每位用户的指标，再对用户平均，让每位用户权重相同；把所有命中数和分母先汇总再相除是 **micro average**，活跃用户会占更大权重。论文或报表必须明确平均方式、$K$、负样本策略和时间切分，否则数值不可复现。
"""),
            code(r"""
catalog = {"i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8", "i9", "i10"}
recommendations_by_user = {
    "u1": ["i3", "i7", "i2"],
    "u2": ["i1", "i7", "i5"],
    "u3": ["i3", "i9", "i5"],
}
exposed_items = set().union(*map(set, recommendations_by_user.values()))
coverage_at_3 = len(exposed_items) / len(catalog)
print({"exposed_items": sorted(exposed_items), "Catalog Coverage@3": round(coverage_at_3, 3)})
"""),
            md(r"""
## Checks

这些检查不是为了“证明数学”，而是训练把公式翻译成可验证事实的习惯：矩阵尺寸正确、相似度在合法范围、Sigmoid 输出是概率、梯度下降确实降低损失、指标位于 0～1。
"""),
            code(r"""
assert user_cooccurrence.shape == (3, 3)
assert item_cooccurrence.shape == (4, 4)
assert 0 <= cosine <= 1
assert np.all((probabilities > 0) & (probabilities < 1))
assert (path[-1] - 3) ** 2 < (path[0] - 3) ** 2
assert 0 <= precision_at_5 <= 1 and 0 <= recall_at_5 <= 1 and 0 <= f1_at_5 <= 1
assert 0 <= mrr_at_5 <= 1 and 0 <= ndcg_at_5 <= 1 and 0 <= ap_at_5 <= 1
assert mae <= rmse and 0 <= pairwise_auc <= 1 and logloss >= 0
assert 0 <= coverage_at_3 <= 1
print("PASS：矩阵、相似度、概率、优化和指标示例全部通过。")
"""),
            md(r"""
## Next Steps

现在可以进入 3.1：

- UserCF / ItemCF 会复用矩阵乘法和余弦相似度；
- MF 会把一个大矩阵近似拆成两个较小矩阵；
- FM 会复用点积来共享稀疏特征交互；
- GBDT+LR 会复用 Sigmoid、概率和 LogLoss。

遇到公式时，按四步阅读：**每个符号代表什么 → 数组形状是什么 → 用小数字手算一遍 → 用代码检查**。不需要先记住公式。
"""),
        ],
    )
    specs["3_1_classic_models"] = notebook(
        "3.1 经典推荐算法：从邻域、低秩分解到特征交叉",
        "从零实现并理解 UserCF / ItemCF、矩阵分解（MF）、因子分解机（FM）与 GBDT+LR。读完后应能回答：每种算法压缩了什么信息、训练目标是什么、线上如何推理、应该用什么指标，以及何时不该使用它。",
        "[GroupLens MovieLens](https://grouplens.org/datasets/movielens/) · [Matrix Factorization Techniques](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf) · [FM](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf) · [Facebook GBDT+LR](https://research.facebook.com/publications/practical-lessons-from-predicting-clicks-on-ads-at-facebook/)",
        [
         md(r"""
## 学习路线

四种算法其实对应三种不同的“信息压缩”方式：

| 方法 | 压缩对象 | 学到的核心信息 | 常见位置 |
|---|---|---|---|
| UserCF / ItemCF | 共现邻域 | 谁和谁相似 | 召回、相关推荐 |
| MF | user–item 矩阵 | 用户与物品的低维隐语义 | 召回、评分预测 |
| FM | 稀疏特征交叉 | 任意两个特征的低秩交互 | CTR 排序 |
| GBDT+LR | 表格特征与树规则 | 非线性分桶、条件组合与概率校准 | CTR 排序 |

建议按顺序运行。前两类使用 user–item 行为；后两类使用“曝光—点击”样本。二者的数据生成机制和评价指标不同，不能把 RMSE、Recall 与 AUC 混为一谈。
"""),
         md(r"""
## Steps

## 1. 数据集与评测协议

### 1.1 MovieLens 是什么？

[MovieLens](https://grouplens.org/datasets/movielens/) 是推荐系统最常用的公开教学数据之一。MovieLens 100K 包含用户对电影的 1–5 星评分，核心字段为：

- `user_id`：用户标识；
- `item_id`：电影标识；
- `rating`：显式偏好强度；
- `timestamp`：行为发生时间；
- 电影标题、类型等 metadata。

本 Notebook 使用仓库内固定版本的 **MovieLens latest-small 真实评分**。为控制 CPU 时间，只确定性选取活跃用户和高频电影；所有行仍来自 GroupLens 原始文件，不制造评分或时间戳。这里的数值用于教学回归，不作为统一公开 benchmark 成绩。

### 1.2 为什么按时间切分？

真实推荐只能用过去预测未来。我们把每个用户最后一次行为作为测试目标、其余行为作为训练历史。随机切分会让“未来行为”进入训练集，从而高估效果。
"""),
         code(r"""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, roc_auc_score, log_loss

SEED = 2026
torch.manual_seed(SEED)
from recsys_lab.data import load_movielens, leave_last_out, movielens_provenance

ratings, movies = load_movielens(max_users=48, max_items=360, min_user_events=12)
train_ratings, test_ratings = leave_last_out(ratings)
n_users, n_items = ratings.user_id.nunique(), ratings.item_id.nunique()

print({
    "rows": len(ratings), "users": ratings.user_id.nunique(), "items": ratings.item_id.nunique(),
    "train_rows": len(train_ratings), "test_rows": len(test_ratings),
    "sparsity": round(1 - len(train_ratings) / (n_users * n_items), 3),
    "source": movielens_provenance(ratings)["source"], "fabricated_rows": 0,
})
display(ratings[["userId", "movieId", "rating", "timestamp", "title", "genres"]].head(8))
"""),
         md(r"""
### 1.3 两组任务、四类指标

1. **Top-K 推荐**：对每位用户生成 K 个未见物品，检查测试物品是否命中。使用 HitRate@K、Recall@K、Coverage。
2. **评分预测**：预测 1–5 星，使用 RMSE。RMSE 越低越好，但它不直接等价于 Top-K 体验。
3. **点击率预估**：预测曝光后是否点击，使用 AUC 与 LogLoss。AUC 衡量排序，LogLoss 同时惩罚错误且过度自信的概率。

下面先定义几个透明、可复用的评测函数。
"""),
         code(r"""
def topk_items(score_matrix, seen_matrix, k=10):
    scores = score_matrix.copy()
    scores[seen_matrix > 0] = -np.inf
    return np.argsort(-scores, axis=1)[:, :k]

def hit_rate_at_k(topk, targets):
    return float(np.mean([target in row for row, target in zip(topk, targets)]))

def catalog_coverage(topk, catalog_size):
    return float(len(np.unique(topk)) / catalog_size)

test_targets = test_ratings.sort_values("user_id").item_id.to_numpy()
"""),
         md(r"""
---

## 2. 协同过滤：直接利用邻域

### 2.1 直觉

- **UserCF**：如果 Alice 与 Bob 过去喜欢的电影相似，那么 Bob 喜欢、Alice 没看过的电影可以推荐给 Alice。
- **ItemCF**：如果《A》和《B》经常被同一批用户喜欢，那么看过《A》的用户可能也喜欢《B》。

UserCF 的邻居会随用户兴趣变化，适合用户关系明显的场景；ItemCF 的物品相似度通常更稳定，且“因为你看过 A”更容易解释，因此电商、视频相关推荐中更常见。这一思想可以追溯到 GroupLens（1994）：论文第 7 页手算过 Ken 与 Lee 的相关系数为 -0.8、与 Meg 为 +1，于是同一篇文章对 Ken 预测 4.56 分、对 Nan 只有 3.75 分——推荐权重完全由“过去是否一致”决定。

### 2.2 数学：余弦相似度与邻域加权

**通用先修：** [3.2 隐式反馈、未知项与假负例](/notebooks/3_2_data_ml_basics#implicit-feedback) · [3.3 逐元素、点积与余弦](/notebooks/3_3_linear_algebra#elementwise-dot) · [3.3 矩阵乘法](/notebooks/3_3_linear_algebra#matmul-embedding)<br>
**本论文新增数学：** GroupLens 的 Pearson 用户相关与均值中心化、Sarwar ItemCF 的 item 相似度/加权预测，以及本教程二值余弦实现之间的对应边界。

先把符号说清楚：

- $U$、$I$ 是用户集合与物品集合；$R \in \{0,1\}^{|U|\times|I|}$ 是二值交互矩阵，$R_{ui}=1$ 表示用户 $u$ 喜欢过物品 $i$。
- $R_u$ 是矩阵的第 $u$ 行，即用户 $u$ 的“喜欢清单”，长度等于物品总数 $|I|$。
- $R_u\cdot R_v$ 是点积：两行逐项相乘再相加。元素只有 0/1，所以结果恰好等于两人**共同喜欢**的物品数。
- $\|R_u\|_2=\sqrt{\sum_i R_{ui}^2}$ 是向量的“勾股长度”，这里等于 $\sqrt{\text{喜欢数}}$；除以它是为了避免“行为多的人天然和谁都很像”。

UserCF 的相似度为：

$$
s(u,v)=\frac{R_u\cdot R_v}{\|R_u\|_2\|R_v\|_2}
$$

对用户 $u$ 和候选物品 $i$，UserCF 分数为：

$$
\hat y_{ui}=\sum_{v\in N_k(u)}s(u,v)R_{vi}
$$

其中 $N_k(u)$ 是与 $u$ 最相似的 $k$ 个邻居；公式的意思是：每个邻居投票“我喜欢 $i$”，票数按相似度加权。

**手算一遍（与 2.3 的玩具矩阵相同）。** 取 3 用户 × 4 物品：u0 = [1,1,0,0]，u1 = [1,0,1,0]，u2 = [0,1,1,1]。

- $s(u_0,u_1)=1/(\sqrt2\cdot\sqrt2)=0.5$（共同喜欢只有 i0）；
- $s(u_0,u_2)=1/(\sqrt2\cdot\sqrt3)\approx0.408$（共同喜欢只有 i1）。

给 u0 的候选打分：i2 得 $0.5\times1+0.408\times1=0.908$，i3 得 $0.5\times0+0.408\times1=0.408$，所以 i2 排在 i3 前面。ItemCF 只是把矩阵转置，在物品邻域中做同样的加权：$s(i,j)$ 用 $R$ 的列（物品的“被喜欢清单”）计算，$\hat y_{ui}=\sum_{j\in N_k(i)}s(i,j)R_{uj}$。实际系统还会加入热门度惩罚、时间衰减和相似度截断。

#### 论文公式与本教程实现怎样对应

本教程为了突出“共同喜欢数”，先把 `rating >= 3` 变成 0/1，再算普通余弦；它不会表达“用户 A 总比用户 B 打分严格”。GroupLens 的早期 UserCF 使用 **Pearson 相关**，先减去每位用户自己的平均评分：

$$
s_{\text{Pearson}}(u,v)=
\frac{\sum_{i\in I_{uv}}(r_{ui}-\bar r_u)(r_{vi}-\bar r_v)}
{\sqrt{\sum_{i\in I_{uv}}(r_{ui}-\bar r_u)^2}\sqrt{\sum_{i\in I_{uv}}(r_{vi}-\bar r_v)^2}}.
$$

$I_{uv}$ 是两人都评分的物品；正相关表示“相对各自习惯，两人会一起打高或一起打低”。预测评分时还要把邻居的偏差加回目标用户均值。Sarwar 的 ItemCF 比较了普通余弦、相关性与 **adjusted cosine**；后者把物品列中的每个评分减去该评分用户的均值，再计算 item–item 余弦，从而消除用户评分尺度差异。三者不能混称：**二值余弦适合本教程的隐式 Top-K 演示；Pearson/均值中心化与 adjusted cosine 属于论文评分预测口径。**
"""),
         md("### 2.3 训练：计算 UserCF / ItemCF 相似度"),
         code(r"""
# 教学中把 rating >= 3 视为正向隐式行为。
train_matrix = np.zeros((n_users, n_items), dtype=np.float32)
for row in train_ratings.itertuples():
    train_matrix[row.user_id, row.item_id] = float(row.rating >= 3)

def cosine_similarity_rows(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
    normalized = matrix / norms
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, 0.0)
    return similarity

user_similarity = cosine_similarity_rows(train_matrix)
item_similarity = cosine_similarity_rows(train_matrix.T)

usercf_scores = user_similarity @ train_matrix
itemcf_scores = train_matrix @ item_similarity
print({"user_similarity": user_similarity.shape, "item_similarity": item_similarity.shape})
"""),
         md("### 2.4 推理：屏蔽已看物品并取 Top-K"),
         code(r"""
usercf_top10 = topk_items(usercf_scores, train_matrix, k=10)
itemcf_top10 = topk_items(itemcf_scores, train_matrix, k=10)

cf_metrics = {
    "UserCF HR@10": hit_rate_at_k(usercf_top10, test_targets),
    "ItemCF HR@10": hit_rate_at_k(itemcf_top10, test_targets),
    "UserCF Coverage": catalog_coverage(usercf_top10, n_items),
    "ItemCF Coverage": catalog_coverage(itemcf_top10, n_items),
}
display(pd.Series(cf_metrics, name="value").to_frame())

example_user = 0
print("user 0 的历史 item:", np.where(train_matrix[example_user] > 0)[0].tolist())
print("UserCF 推荐:", usercf_top10[example_user].tolist())
print("ItemCF 推荐:", itemcf_top10[example_user].tolist())
print("真实下一 item:", int(test_targets[example_user]))
"""),
         md(r"""
### 2.5 结果讨论与边界

观察上表时不要只看命中率：Coverage 低可能表示模型只反复推荐热门物品。本小样本中每位用户历史很短，相似度容易受单个共现影响，这正是 CF 在稀疏数据上的典型弱点。

**优点**：无需训练神经网络；解释直接；增量更新容易；ItemCF 可作为强兜底召回。  
**缺点**：新用户、新物品无邻居；相似度矩阵可能很大；共现继承曝光偏差与热门偏差；无法自然使用文本、图片等内容。

**推理复杂度**：离线相似度可截断为每个实体 Top-N 邻居；线上只聚合用户历史物品的邻居倒排表，避免扫描全库。
"""),
         md(r"""
---

## 3. 矩阵分解（MF）：把用户与物品投影到同一隐空间

### 3.1 从相似度到隐语义

CF 依赖显式邻域；MF 假设评分矩阵近似低秩：一张 $|U|\times|I|$ 的大表，可以近似写成“用户坐标表 $P\in\mathbb R^{|U|\times d}$ 与物品坐标表 $Q\in\mathbb R^{|I|\times d}$ 的乘积”，其中 $d\ll\min(|U|,|I|)$。每个用户用向量 $p_u\in\mathbb R^d$（$P$ 的一行）表示，每个物品用 $q_i\in\mathbb R^d$ 表示，内积 $p_u^\top q_i=\sum_{f=1}^{d}p_{uf}q_{if}$ 代表匹配程度。某一维可能与“动作片偏好”相关，但隐维度是学出来的，通常不可直接解释。

### 3.2 数学：带偏置的评分预测

**通用先修：** [3.3 低秩分解与形状](/notebooks/3_3_linear_algebra#low-rank-attention) · [3.4 偏导与梯度](/notebooks/3_4_calculus#derivative-gradient) · [3.7 SGD](/notebooks/3_7_optimization#sgd) · [3.7 L2 与过拟合](/notebooks/3_7_optimization#regularization)<br>
**本论文新增数学：** 只在观察集合 $\Omega$ 上拟合 BiasMF，并把评分误差逐步传给用户向量、物品向量与两类偏置。

$$
\hat r_{ui}=\mu+b_u+b_i+p_u^\top q_i
$$

四个符号各司其职：$\mu$ 是全体评分的均值（“大家平均打几颗星”）；$b_u$ 是用户偏置（“这个人手松还是手紧”）；$b_i$ 是物品偏置（“这部作品是否普遍受欢迎”）；$p_u^\top q_i$ 才是个性化部分。

**手算一遍（借用 Koren 论文的数字）。** 设全体均值 $\mu=3.7$ 星；《Titanic》普遍被认为好于平均，$b_i=+0.5$；Joe 打分偏严，$b_u=-0.3$。不做任何个性化时预测为 $3.7+0.5-0.3=3.9$。再设 $d=2$，Joe 的坐标 $p_u=[0.6,-0.2]$（偏动作、不偏浪漫），《Titanic》的坐标 $q_i=[0.5,0.1]$，内积 $=0.6\times0.5+(-0.2)\times0.1=0.28$，最终 $\hat r_{ui}=3.9+0.28=4.18$。

训练时只在已观察评分集合 $\Omega$ 上最小化正则化平方误差。为让单条梯度写得整齐，把每项乘 $\tfrac12$（这只会整体缩放梯度，可由学习率吸收）：

$$
\mathcal L=\frac12\sum_{(u,i)\in\Omega}(r_{ui}-\hat r_{ui})^2
+\frac\lambda2(\|p_u\|_2^2+\|q_i\|_2^2+b_u^2+b_i^2)
$$

$\Omega$ 是“矩阵里有数字的格子”，未观察的格子不参与（它们是缺失，不是 0 分）；$\lambda$ 控制正则强度。对一条评分先缓存旧的 $p_u,q_i$，再按四步算：

1. 预测并求误差：$e_{ui}=r_{ui}-(\mu+b_u+b_i+p_u^\top q_i)$；
2. 求偏导：$\partial L/\partial p_u=-e_{ui}q_i+\lambda p_u$，$\partial L/\partial q_i=-e_{ui}p_u+\lambda q_i$；
3. 偏置偏导：$\partial L/\partial b_u=-e_{ui}+\lambda b_u$，$\partial L/\partial b_i=-e_{ui}+\lambda b_i$；
4. 减去梯度：$p_u\leftarrow p_u+\eta(e_{ui}q_i-\lambda p_u)$，$q_i\leftarrow q_i+\eta(e_{ui}p_u-\lambda q_i)$，$b_u\leftarrow b_u+\eta(e_{ui}-\lambda b_u)$，$b_i$ 同理。

若真实评分高于预测，$e_{ui}>0$，两张坐标卡会被推得更匹配，偏置也会上调；L2 项同时把参数往 0 拉。这里的“低秩”边界也要说清：只有交互矩阵 $PQ^\top$ 的秩至多为 $d$，偏置是额外的行/列平移；模型只近似观察评分，不会恢复唯一的“真实兴趣”。$d$ 太小会欠拟合，太大又可能记住稀疏评分。
"""),
         md("### 3.3 训练：用 PyTorch 实现 BiasMF"),
         code(r"""
class BiasMF(torch.nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=12, global_mean=3.0):
        super().__init__()
        self.user_embedding = torch.nn.Embedding(n_users, embedding_dim)
        self.item_embedding = torch.nn.Embedding(n_items, embedding_dim)
        self.user_bias = torch.nn.Embedding(n_users, 1)
        self.item_bias = torch.nn.Embedding(n_items, 1)
        self.register_buffer("global_mean", torch.tensor(float(global_mean)))
        torch.nn.init.normal_(self.user_embedding.weight, std=0.08)
        torch.nn.init.normal_(self.item_embedding.weight, std=0.08)
        torch.nn.init.zeros_(self.user_bias.weight)
        torch.nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_ids, item_ids):
        interaction = (self.user_embedding(user_ids) * self.item_embedding(item_ids)).sum(dim=1)
        return self.global_mean + self.user_bias(user_ids).squeeze(1) + self.item_bias(item_ids).squeeze(1) + interaction

train_users = torch.tensor(train_ratings.user_id.to_numpy(), dtype=torch.long)
train_items = torch.tensor(train_ratings.item_id.to_numpy(), dtype=torch.long)
train_targets = torch.tensor(train_ratings.rating.to_numpy(), dtype=torch.float32)
test_users = torch.tensor(test_ratings.user_id.to_numpy(), dtype=torch.long)
test_items = torch.tensor(test_ratings.item_id.to_numpy(), dtype=torch.long)
test_targets_rating = test_ratings.rating.to_numpy()

mf_model = BiasMF(n_users, n_items, embedding_dim=12, global_mean=train_targets.mean())
optimizer = torch.optim.Adam(mf_model.parameters(), lr=0.03, weight_decay=1e-4)
loss_curve = []
for epoch in range(160):
    prediction = mf_model(train_users, train_items)
    loss = torch.nn.functional.mse_loss(prediction, train_targets)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    loss_curve.append(float(loss.detach()))

print({"first_loss": round(loss_curve[0], 4), "last_loss": round(loss_curve[-1], 4)})
"""),
         md("### 3.4 测试与推理：RMSE 和全库 Top-K"),
         code(r"""
mf_model.eval()
with torch.no_grad():
    test_prediction = mf_model(test_users, test_items).numpy()
    mf_rmse = float(np.sqrt(mean_squared_error(test_targets_rating, test_prediction)))
    all_users = torch.arange(n_users).repeat_interleave(n_items)
    all_items = torch.arange(n_items).repeat(n_users)
    mf_score_matrix = mf_model(all_users, all_items).reshape(n_users, n_items).numpy()

mf_top10 = topk_items(mf_score_matrix, train_matrix, k=10)
mf_hr10 = hit_rate_at_k(mf_top10, test_targets)
print({"MF test RMSE": round(mf_rmse, 4), "MF HR@10": round(mf_hr10, 4)})
print("user 0 MF 推荐:", mf_top10[0].tolist())
"""),
         md(r"""
### 3.5 结果讨论与边界

训练损失下降只说明模型拟合了已观察评分；测试 RMSE 才反映未见行为的泛化。Top-K 推理时，MF 与双塔相似：物品向量可以预计算，并用 ANN 搜索内积最大项。

**优点**：比邻域法更紧凑；可平滑稀疏共现；向量检索友好；偏置明确。  
**缺点**：只看 ID 时仍无法解决新实体；平方误差把“未观察”当缺失而非负样本；内积表达能力有限。  
**常见升级**：隐式反馈使用 BPR / sampled-softmax；加入时间偏置、内容特征，或发展为 DSSM 双塔。
"""),
         md(r"""
---

## 4. 因子分解机（FM）：为稀疏特征学习二阶交互

### 4.1 为什么 MF 不够？

**通用先修：** [3.2 样本、特征与标签](/notebooks/3_2_data_ml_basics#observation-label) · [3.3 逐元素与点积](/notebooks/3_3_linear_algebra#elementwise-dot) · [3.6 二元交叉熵](/notebooks/3_6_information_theory#cross-entropy-kl)<br>
**本论文新增数学：** 用共享隐向量参数化任意稀疏特征的二阶交互，并把 $O(n^2k)$ 的逐对计算化成 $O(nk)$ 恒等式。

CTR 排序不只有 `user_id` 和 `item_id`，还会有设备、小时、地域、品类、价格等上下文。直接为每一对特征做 one-hot 交叉会导致参数爆炸；普通 LR 又只能做加法，无法表达“某用户群在晚间更喜欢某品类”。

FM 的输入是一个超长稀疏向量 $x\in\mathbb R^n$：每个特征占一格，one-hot 类别特征取 0/1，连续特征取实数值。模型为每个特征 $i$ 学一个 $k$ 维隐向量 $v_i\in\mathbb R^k$，用内积 $\langle v_i,v_j\rangle=\sum_{f=1}^{k}v_{if}v_{jf}$ 共享所有二阶交叉统计：

$$
\hat y(x)=w_0+\sum_i w_ix_i+\sum_{i<j}\langle v_i,v_j\rangle x_ix_j
$$

三项分别是：$w_0$ 全局偏置；$\sum_i w_ix_i$ 一阶线性项（与 LR 相同）；$\sum_{i<j}\langle v_i,v_j\rangle x_ix_j$ 二阶交叉项——只有同时“打开”（$x_i,x_j$ 都非零）的特征对才参与。关键在于稀疏性：即使某对特征从未在训练集共同出现，它们各自与其他特征的共现也已把 $v_i,v_j$ 学好，交叉强度仍能由内积给出。

朴素计算二阶项要枚举全部 $\binom{n}{2}$ 对特征、每对做一次 $k$ 维内积，复杂度 $O(n^2k)$。利用“和的平方减去平方的和”恒等式可化为 $O(nk)$：

$$
\sum_{i<j}\langle v_i,v_j\rangle x_ix_j
=\frac12\sum_f\left[\left(\sum_i v_{if}x_i\right)^2-\sum_i v_{if}^2x_i^2\right]
$$

推导只需一步代数：把 $(\sum_i v_{if}x_i)^2$ 展开会得到所有有序对 $(i,j)$ 的乘积——包含 $i=j$ 的对角项，且每对 $i<j$ 出现两次；减去对角项 $\sum_i v_{if}^2x_i^2$ 再乘 $\tfrac12$ 去重，正好回到左边。右侧只需对每个因子维度 $f$ 各扫一遍特征。

**手算一遍。** 取 $k=2$、两个特征 $v_1=[1,2]$、$v_2=[0.5,-1]$，且 $x_1=x_2=1$。左边：$\langle v_1,v_2\rangle=1\times0.5+2\times(-1)=-1.5$。右边逐维算：$f=1$ 时 $(1+0.5)^2-(1^2+0.5^2)=2.25-1.25=1$；$f=2$ 时 $(2+(-1))^2-(2^2+(-1)^2)=1-5=-4$；合起来 $\tfrac12[1+(-4)]=-1.5$，与左边一致。

当特征只有 user one-hot 和 item one-hot 时，FM 的交叉项就退化为 MF；因此 FM 可以理解为 MF 对通用稀疏特征的扩展。
"""),
         md(r"""
### 4.2 数据：评分阈值派生“高偏好”标签（不是曝光—点击日志）

MovieLens 记录的是用户主动提交的星级，**没有完整曝光分母**。这里把真实评分 `rating >= 4.0` 派生为高偏好标签、较低评分派生为 0，只用于演示 FM 的二分类、AUC 与 LogLoss 计算；变量名 `click` 是代码兼容别名，不代表真实广告点击。没有评分的 user–item 仍是未知，不能补成曝光未点击。真实 CTR 结论必须改用包含曝光与点击的日志。
"""),
         code(r"""
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ctr = ratings.sort_values("timestamp").tail(5000).copy().reset_index(drop=True)
# `click` 是任务别名；标签由真实评分 rating >= 4.0 确定，不使用随机采样。
ctr["click"] = ctr["like"].astype(int)
split = int(len(ctr) * .8)  # 严格按真实 timestamp 切分
categorical = ["user_id", "item_id", "genre_id", "hour", "decade_id"]
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
train_sparse = encoder.fit_transform(ctr.loc[:split-1, categorical])
test_sparse = encoder.transform(ctr.loc[split:, categorical])
scaler = StandardScaler()
train_numeric = scaler.fit_transform(ctr.loc[:split-1, ["item_popularity", "user_activity"]])
test_numeric = scaler.transform(ctr.loc[split:, ["item_popularity", "user_activity"]])
fm_train_x = np.c_[train_sparse, train_numeric].astype("float32")
fm_test_x = np.c_[test_sparse, test_numeric].astype("float32")
ctr_train_y = ctr.loc[:split-1, "click"].to_numpy()
ctr_test_y = ctr.loc[split:, "click"].to_numpy()
print({"train": fm_train_x.shape, "test": fm_test_x.shape, "train_positive_rate": round(ctr_train_y.mean(), 3)})
"""),
         md("### 4.3 训练：显式实现 FM 的线性时间交叉"),
         code(r"""
class FactorizationMachine(torch.nn.Module):
    def __init__(self, n_features, factor_dim=10):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1)
        self.factors = torch.nn.Parameter(torch.randn(n_features, factor_dim) * 0.03)

    def forward(self, x):
        linear_logit = self.linear(x).squeeze(1)
        summed = x @ self.factors
        squared_sum = summed.pow(2)
        sum_squared = x.pow(2) @ self.factors.pow(2)
        pairwise_logit = 0.5 * (squared_sum - sum_squared).sum(dim=1)
        return linear_logit + pairwise_logit

fm_model = FactorizationMachine(fm_train_x.shape[1], factor_dim=10)
fm_optimizer = torch.optim.Adam(fm_model.parameters(), lr=.025, weight_decay=1e-5)
x_train_tensor = torch.tensor(fm_train_x)
y_train_tensor = torch.tensor(ctr_train_y, dtype=torch.float32)

for epoch in range(100):
    fm_logit = fm_model(x_train_tensor)
    fm_loss = torch.nn.functional.binary_cross_entropy_with_logits(fm_logit, y_train_tensor)
    fm_optimizer.zero_grad(); fm_loss.backward(); fm_optimizer.step()

print({"FM final BCE": round(float(fm_loss.detach()), 4)})
"""),
         md("### 4.4 推理与测试：输出概率、AUC 与 LogLoss"),
         code(r"""
fm_model.eval()
with torch.no_grad():
    fm_probability = torch.sigmoid(fm_model(torch.tensor(fm_test_x))).numpy()
fm_auc = float(roc_auc_score(ctr_test_y, fm_probability))
fm_logloss = float(log_loss(ctr_test_y, fm_probability))
display(pd.DataFrame({"label": ctr_test_y[:8], "p_click": fm_probability[:8].round(4)}))
print({"FM AUC": round(fm_auc, 4), "FM LogLoss": round(fm_logloss, 4)})
"""),
         md(r"""
### 4.5 结果讨论与边界

AUC 关注正样本是否排在负样本前，LogLoss 关注概率本身是否可信。FM 能通过隐向量共享低频交叉的统计，但所有交互仍是二阶且形式相同。

**优点**：适合超稀疏 one-hot；无需枚举交叉；参数能跨组合共享；计算复杂度线性。  
**缺点**：主要建模二阶；不理解行为顺序；所有 field 共用同一种内积。  
**常见升级**：FFM 为不同 field 学不同向量；DeepFM 共享 embedding，同时加入 DNN 学高阶非线性。
"""),
         md(r"""
---

## 5. GBDT+LR：树负责发现规则，LR 负责组合与校准

### 5.1 核心思想

**通用先修：** [3.2 标签、曝光与校准边界](/notebooks/3_2_data_ml_basics#implicit-feedback) · [3.5 odds、logit 与校准](/notebooks/3_5_probability_statistics#likelihood-calibration) · [3.6 交叉熵与 Normalized Entropy](/notebooks/3_6_information_theory#softmax-temperature)<br>
**本论文新增数学：** GBDT 逐轮拟合损失的负梯度、用切分形成条件规则，再把叶节点当稀疏特征交给 LR；并按 Facebook 论文口径解释 NE。

Facebook 经典 CTR 工作把 GBDT 的每棵树视为一个自动特征生成器。样本落到某棵树的哪个叶子，代表它满足了一组条件，例如：

> `hour > 18 AND device = mobile AND price < 20`

将每棵树的叶节点编号 one-hot 后输入 LR：

$$
P(y=1\mid z)=\sigma(w_0+w^\top z)
$$

符号逐个看：$z$ 是所有树叶节点的稀疏指示向量——若有 $T$ 棵树、第 $t$ 棵有 $L_t$ 个叶子，$z$ 的长度就是 $\sum_t L_t$，且每棵树恰好贡献一个 1；$w$ 是 LR 为每个叶子学到的权重；$\sigma(a)=1/(1+e^{-a})$ 把任意实数压到 0～1，变成点击概率。

**手算一遍。** 设 2 棵树，第 1 棵 3 个叶子、第 2 棵 2 个叶子，$z$ 长度为 5。某样本落入第 1 棵的 1 号叶、第 2 棵的 2 号叶，则 $z=[1,0,0,0,1]$。若 $w=[0.8,-0.2,0.1,-0.4,0.9]$、$w_0=-0.5$，线性得分 $=-0.5+0.8+0.9=1.2$，$\sigma(1.2)\approx0.77$——模型预测该样本约有 77% 的点击概率。

GBDT 的“Boosting”是逐轮纠错。第 $m$ 轮已有总分 $F_{m-1}(x)$，新树拟合每个样本让损失下降最快的方向（负梯度，也叫伪残差）：

$$
g_i^{(m)}=-\left.\frac{\partial\ell(y_i,F(x_i))}{\partial F(x_i)}\right|_{F=F_{m-1}},
\qquad F_m(x)=F_{m-1}(x)+\eta h_m(x).
$$

平方误差时 $g_i=y_i-F_{m-1}(x_i)$，就是熟悉的“真实值减当前预测”；二分类 LogLoss 时仍是残差直觉，但准确说法是负梯度。建树时尝试“特征 $j$ 是否小于阈值 $t$”等问题，把样本分到左右两堆；选择能让两边负梯度更一致、目标下降最多的切分。连续多次切分后，一个叶子便代表一串 AND 条件，LR 再学习这些规则的稳定权重。

原论文的 **Normalized Entropy** 用常数基线熵归一化平均 LogLoss。令样本正率 $\bar y=\frac1N\sum_i y_i$：

$$
\mathrm{NE}=
\frac{-\frac1N\sum_i[y_i\log p_i+(1-y_i)\log(1-p_i)]}
{-\bar y\log\bar y-(1-\bar y)\log(1-\bar y)}.
$$

分母是所有样本都预测总体正率 $\bar y$ 时的 LogLoss，因此 NE=1 约等于只会报基准率，越低越好。论文报告树+LR 组合相对单模型的 NE 改进超过 3%，但这是 Facebook 内部广告数据口径下的曝光日志；本教程用 MovieLens **评分阈值派生标签**验证链路，不是曝光点击，该收益不可平移。两阶段训练还意味着树、叶编码器和 LR 不是端到端联合优化。
"""),
         md("### 5.2 训练阶段一：XGBoost 学习树规则"),
         code(r"""
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

tree_features = ["user_id", "item_id", "genre_id", "hour", "decade_id", "item_popularity", "user_activity"]
# GBDT 使用与 FM 相同的 one-hot 类别语义，避免把类别 ID 错当连续数值。
gbdt_train_x = fm_train_x
gbdt_test_x = fm_test_x

gbdt = XGBClassifier(
    n_estimators=80, max_depth=4, learning_rate=.05, subsample=.9,
    colsample_bytree=.9, eval_metric="logloss", random_state=SEED, n_jobs=1
)
gbdt.fit(gbdt_train_x, ctr_train_y)
train_leaf = gbdt.apply(gbdt_train_x)
test_leaf = gbdt.apply(gbdt_test_x)
print({"trees": train_leaf.shape[1], "train_leaf_matrix": train_leaf.shape})
"""),
         md("### 5.3 训练阶段二：叶节点 one-hot 后训练 LR"),
         code(r"""
leaf_encoder = OneHotEncoder(handle_unknown="ignore")
train_leaf_onehot = leaf_encoder.fit_transform(train_leaf)
test_leaf_onehot = leaf_encoder.transform(test_leaf)

leaf_lr = LogisticRegression(max_iter=500, C=.7)
leaf_lr.fit(train_leaf_onehot, ctr_train_y)
gbdt_lr_probability = leaf_lr.predict_proba(test_leaf_onehot)[:, 1]

gbdt_lr_auc = float(roc_auc_score(ctr_test_y, gbdt_lr_probability))
gbdt_lr_logloss = float(log_loss(ctr_test_y, gbdt_lr_probability))
print({"leaf_onehot_dim": train_leaf_onehot.shape[1], "GBDT+LR AUC": round(gbdt_lr_auc, 4), "GBDT+LR LogLoss": round(gbdt_lr_logloss, 4)})
"""),
         md("### 5.4 推理：同一条样本必须依次经过两阶段"),
         code(r"""
def predict_gbdt_lr(frame):
    sparse_part = encoder.transform(frame[categorical])
    numeric_part = scaler.transform(frame[["item_popularity", "user_activity"]])
    encoded_features = np.c_[sparse_part, numeric_part].astype("float32")
    leaf = gbdt.apply(encoded_features)
    leaf_onehot = leaf_encoder.transform(leaf)
    return leaf_lr.predict_proba(leaf_onehot)[:, 1]

online_batch = ctr.loc[split:split+4].copy()
online_batch["p_click"] = predict_gbdt_lr(online_batch)
display(online_batch[tree_features + ["click", "p_click"]])
"""),
         md(r"""
### 5.5 结果讨论与边界

**优点**：连续变量无需手工分桶；树能发现条件组合；LR 服务成熟、输出易校准；对中等规模表格特征很强。  
**缺点**：两阶段可能失配；叶节点空间随树数膨胀；高基数 ID 容易过拟合；树规则会随分布漂移而陈旧；难以表达长行为序列。  
**工业注意**：训练与服务必须共享完全相同的树版本、叶编码器和缺失值处理；监控 unseen leaf、特征漂移与概率校准。

与 FM 的关键区别：FM 假设交互可由低秩内积共享；GBDT+LR 假设有效模式可由一组树规则离散化。两者没有绝对胜负，取决于稀疏度、特征类型、数据量和延迟预算。
"""),
         md(r"""
## Checks

下面把不同任务的指标放在一张“不可横向混比”的检查表中。它的用途是确认每条代码路径确实学习到信号，而不是宣布某算法全面胜出。
"""),
         code(r"""
summary = pd.DataFrame([
    {"algorithm": "UserCF", "task": "Top-K", "primary_metric": "HR@10", "value": cf_metrics["UserCF HR@10"]},
    {"algorithm": "ItemCF", "task": "Top-K", "primary_metric": "HR@10", "value": cf_metrics["ItemCF HR@10"]},
    {"algorithm": "BiasMF", "task": "rating prediction", "primary_metric": "RMSE (lower)", "value": mf_rmse},
    {"algorithm": "FM", "task": "CTR", "primary_metric": "AUC", "value": fm_auc},
    {"algorithm": "GBDT+LR", "task": "CTR", "primary_metric": "AUC", "value": gbdt_lr_auc},
])
display(summary.round(4))

assert 0 <= cf_metrics["UserCF HR@10"] <= 1
assert 0 <= cf_metrics["ItemCF HR@10"] <= 1
assert np.isfinite(mf_rmse) and mf_rmse < 3.0
assert fm_auc > .55 and gbdt_lr_auc > .55
assert np.all((fm_probability >= 0) & (fm_probability <= 1))
assert np.all((gbdt_lr_probability >= 0) & (gbdt_lr_probability <= 1))
print("PASS：四类算法的训练、推理和测试路径均有效。")
print(
    f"本次固定样本：UserCF HR@10={cf_metrics['UserCF HR@10']:.3f}，"
    f"ItemCF HR@10={cf_metrics['ItemCF HR@10']:.3f}；"
    f"MF RMSE={mf_rmse:.3f}；FM AUC={fm_auc:.3f}；GBDT+LR AUC={gbdt_lr_auc:.3f}。"
)
print("注意：前三者与后两者属于不同任务，只能在各自指标和基线内解释。")
"""),
         md(r"""
## 结果讨论：如何读这组实验

1. **CF 的 HR@10 与 Coverage 要一起看。** 命中率相近时，覆盖更多目录的方案通常更有探索价值；但离线覆盖也不能替代线上多样性指标。
2. **MF 的 RMSE 与 HR@10回答不同问题。** 前者衡量星级拟合，后者衡量下一物品是否进入候选。优化平方误差不保证最佳 Top-K。
3. **FM 与 GBDT+LR 才能在同一二分类测试集上比较 AUC/LogLoss。** 两者都使用真实评分派生的 `rating >= 4.0` 标签和相同的时间切分；MovieLens 不是曝光日志，因此这里验证的是稀疏特征建模链路，不能把结果解读为真实 CTR benchmark。
4. **小样本结果首先用于查错。** 真正选型还要比较时间外推、冷启动、Coverage、校准、P99 延迟、内存和特征新鲜度。

### 一句话选型

- 需要简单、可解释的相关推荐：先用 ItemCF。
- 只有稳定 ID 共现且目录很大：MF 是双塔召回的最小原型。
- 大量稀疏类别特征且二阶交互重要：FM。
- 表格连续特征丰富、需要自动分桶与条件规则：GBDT+LR。
"""),
         md(r"""
## Next Steps

1. 将 smoke 子集扩大到完整 MovieLens latest-small 或 MovieLens 1M，并保持逐用户时间外推评测。
2. 为 CF 加入热门度惩罚、时间衰减和邻居 Top-N 截断。
3. 将 MF 的 MSE 换为 BPR pairwise loss，直接优化正负物品排序。
4. 比较 LR、FM、GBDT+LR 的 AUC、LogLoss 与校准曲线，再进入 DeepFM。
5. 在下一版 Notebook 中加入可视化：相似度热力图、MF embedding 投影、FM 交互强度和树叶覆盖率。

> 教学结论：模型演进不是“新模型替代旧模型”，而是从邻域统计、低维表示到上下文交叉，逐步扩大可表达的信息范围，同时付出更多训练、服务与治理成本。
""")
        ])

    specs["3_2_retrieval_dssm_mind"] = notebook(
        "3.2 召回：DSSM 双塔与 MIND",
        "用 Torch-RecHub 的工业化模型契约理解双塔的可索引性、in-batch negatives 与 MIND 多兴趣合并，并用 Recall@K 检查召回链路。",
        "[DSSM](https://www.microsoft.com/en-us/research/publication/learning-deep-structured-semantic-models-for-web-search-using-clickthrough-data/) · [MIND](https://arxiv.org/abs/1904.08030) · [Torch-RecHub](https://github.com/datawhalechina/torch-rechub)",
        [md("## Steps\n\n### 1. 验证工业框架\n\nTorch-RecHub 0.8 提供 `DSSM`、`MIND`、`MatchTrainer`、双塔分开 ONNX 导出和 ANN 插件；TorchEasyRec 工业档增加分片 embedding、Parquet/Kafka 与分布式训练。"),
         code("import torch_rechub\nfrom torch_rechub.models.matching import DSSM, MIND\nfrom torch_rechub.trainers import MatchTrainer\nprint({'torch_rechub': getattr(torch_rechub, '__version__', 'installed'), 'models': [DSSM.__name__, MIND.__name__], 'trainer': MatchTrainer.__name__})"),
         md("### 2. 执行双塔与多兴趣小实验\n\n训练使用归一化 embedding、温度缩放和 in-batch softmax；item embedding 可离线预计算。MIND smoke 以两个兴趣簇演示多路向量检索与 max merge 的效果。"),
         code("from recsys_lab import run_retrieval\nmetrics = run_retrieval(epochs=55 if PROFILE == 'smoke' else 160)\nmetrics"),
         md("### 3. TorchEasyRec 工业配置映射\n\n完整档将 `user_id/history`、`item_id/category` 定义为 feature group，DSSM/MIND 输出塔分别导出；训练数据使用时间切分 Parquet，item 塔批量物化到向量库。"),
         code("torcheasyrec_profile = {'model': 'DSSM or MIND', 'input': 'Parquet/MaxCompute/Kafka', 'distributed': 'TorchRec sharding', 'serving': 'separate user/item tower + ANN', 'monitor': ['Recall@K','coverage','index freshness','P99']}\ntorcheasyrec_profile"),
         md("## Checks\n\n确认 embedding 维度固定、召回指标在 [0,1]，并单独监控热门/长尾/新 item。"),
         code("assert metrics['embedding_dim'] == 16\nassert 0 <= metrics['dssm_recall@10'] <= 1 and 0 <= metrics['mind_recall@10'] <= 1\nprint('PASS: retrieval contract and metrics are valid')"),
         md("## Next Steps\n\n切换 Amazon Reviews 2023 五核子集；用 FAISS/Milvus 替代精确内积；加入 sampled-softmax 频率校正、增量索引版本一致性和多兴趣去重。")])

    specs["3_3_ranking_deepfm_din_dien"] = notebook(
        "3.3 排序：DeepFM、DIN 与 DIEN",
        "在统一 Torch-RecHub 排序管线中比较静态特征交叉、候选感知兴趣激活与兴趣演化，建立 AUC/GAUC、LogLoss 和服务成本的共同视角。",
        "[DeepFM](https://arxiv.org/abs/1703.04247) · [DIN](https://arxiv.org/abs/1706.06978) · [DIEN](https://arxiv.org/abs/1809.03672)",
        [md("## Steps\n\n### 1. 验证框架模型与数据契约\n\nDeepFM 接收 sparse/dense feature columns；DIN/DIEN 额外接收 behavior sequence、candidate item 与 padding mask。"),
         code("import torch_rechub\nfrom torch_rechub.models.ranking import DeepFM, DIN, DIEN\nfrom torch_rechub.trainers import CTRTrainer\nprint({'models':[DeepFM.__name__,DIN.__name__,DIEN.__name__], 'trainer':CTRTrainer.__name__})"),
         md("### 2. 训练与比较\n\nsmoke 档的 DeepFM 真实训练 FM+DNN 共享输入；DIN 对照加入候选—历史相关信号。DIEN 的工业实现应包含兴趣抽取 GRU、逐步辅助损失和目标感知演化 GRU。"),
         code("from recsys_lab import run_ranking\nmetrics = run_ranking(epochs=45 if PROFILE == 'smoke' else 140)\nmetrics"),
         md("## Checks\n\nAUC 只验证模型能学习固定信号。线上还要测 GAUC、校准、分桶稳定性、序列截断损失与 P99。"),
         code("assert metrics['deepfm_auc'] > .55\nassert metrics['din_auc'] > .55\nprint('PASS: ranking models learn non-random affinity')"),
         md("## Next Steps\n\nDeepFM 用 Criteo；DIN/DIEN 用 Amazon 2023 时间序列。导出 ONNX 前固定 padding/mask 语义，并对 20/50/100/200 长度做质量—延迟曲线。")])

    specs["3_4_multitask_mmoe_ple"] = notebook(
        "3.4 多目标：MMoE 与 PLE",
        "用共享专家与任务门控联合学习点击/转化，理解 PLE 如何进一步拆分共享和任务专属知识，并检查每任务指标而非只看总损失。",
        "[MMoE](https://dl.acm.org/doi/10.1145/3219819.3220007) · [PLE](https://dl.acm.org/doi/10.1145/3383313.3412236)",
        [md("## Steps\n\n### 1. 验证 Torch-RecHub 多任务组件\n\n统一 `MTLTrainer` 支持每任务 loss、metric 与 ONNX 导出；完整数据可用 Ali-CCP 或 MerRec 多动作 schema。"),
         code("import torch_rechub\nfrom torch_rechub.models.multi_task import MMOE, PLE\nfrom torch_rechub.trainers import MTLTrainer\nprint({'models':[MMOE.__name__,PLE.__name__], 'trainer':MTLTrainer.__name__})"),
         md("### 2. MMoE 小实验\n\n四个专家输出共享表示；click 与 conversion 各自通过 gate 选择专家，再进入任务塔。PLE 会逐层加入共享/专属专家，避免所有知识被迫共享。"),
         code("from recsys_lab import run_multitask\nmetrics = run_multitask(epochs=55 if PROFILE == 'smoke' else 160)\nmetrics"),
         md("## Checks\n\n分别检查任务 AUC；conversion 正样本只来自 click 正样本时，要警惕样本选择偏差。工业验收还需监控专家利用率、梯度余弦和任务跷跷板。"),
         code("assert metrics['mmoe_click_auc'] > .55\nassert metrics['mmoe_conversion_auc'] > .55\nprint('PASS: both task heads learn signal')"),
         md("## Next Steps\n\n用 PLE 对照 MMoE；引入不确定性加权/GradNorm；业务层将 pCTR、pCVR、时长和负反馈校准后进入显式价值函数，而非直接相加原始概率。")])

    specs["8_2_openonerec_practice"] = notebook(
        "8.2 OpenOneRec 实战：从 RecIF-Bench 到约束列表生成",
        "复现 OpenOneRec 的关键数据与生成接口：行为序列、Semantic ID、合法前缀约束、列表 NDCG、无效 ID 率和 DPO 偏好样本；提供官方大模型运行入口。",
        "[OpenOneRec](https://github.com/Kuaishou-OneRec/OpenOneRec) · [OneRec](https://arxiv.org/abs/2502.18965)",
        [md("## Steps\n\n### 1. 运行边界\n\nOpenOneRec 公开 1.7B/8B Qwen3 基座与 RecIF-Bench（100M 交互、200K 用户）。CI 不下载权重；smoke 档验证与官方接口一致的数据/约束/指标，full 档按官方 README 在多 GPU 环境运行。"),
         code("openonerec_runtime = {'smoke':'local constrained decoder + MovieLens-derived RecIF schema','full':'clone OpenOneRec; prepare RecIF-Bench; launch official SFT/DPO scripts','models':['OneRec-1.7B','OneRec-8B'],'guardrails':['legal item trie','dedup','inventory filter','fallback']}\nopenonerec_runtime"),
         md("### 2. Semantic ID 与约束解码\n\nitem 由多级码组成；每一步只允许目录 trie 中仍可完成为合法 item 的 token。这把 invalid ID rate 从开放生成的风险变成可测试的不变量。"),
         code("from recsys_lab import run_generative\nmetrics = run_generative()\nmetrics"),
         md("### 3. 列表级训练与对齐\n\nOneRec 不是逐 item CTR 的简单替代：session-wise generation 优化整张列表；奖励模型评估观看/转化/多样性，DPO 用 chosen/rejected 列表对齐偏好。"),
         code("chosen, rejected = metrics['dpo_pair']['chosen'], metrics['dpo_pair']['rejected']\nprint({'prompt':'user history + context','chosen':chosen,'rejected':rejected,'training':'DPO preference pair'})"),
         md("## Checks\n\n同时报告 NDCG/Recall、覆盖、新 item、重复率、invalid ID、P99、tokens/s 与 GPU 成本。离线提升不能替代线上 session 级 A/B。"),
         code("assert metrics['invalid_id_rate'] == 0\nassert metrics['allowed_next_tokens'] == [2,3]\nassert 0 <= metrics['ndcg@5'] <= 1\nprint('PASS: constrained generation only emits catalog-valid continuations')"),
         md("## Next Steps\n\n按 OpenOneRec 官方许可获取 RecIF-Bench；建立 item 目录版本与 Semantic ID 码本版本绑定；先旁路生成候选，再灰度生成排序，最后评估端到端召排融合。")])

    specs["8_3_dlrm_hstu_practice"] = notebook(
        "8.3 DLRM HSTU 实战：行为序列生成式推荐",
        "理解 HSTU 的序列建模接口、next-item 目标和 DLRM-v3 工业运行边界；CPU 档验证 schema 与指标，完整档映射 Meta 官方 generative-recommenders。",
        "[HSTU paper](https://arxiv.org/abs/2402.17152) · [Meta official code](https://github.com/meta-recsys/generative-recommenders) · [Torch-RecHub HSTU](https://datawhalechina.github.io/torch-rechub/models/generative.html)",
        [md("## Steps\n\n### 1. 验证轻量工业框架入口\n\nTorch-RecHub `generative` extra 提供 HSTU；输入包括 item sequence、time difference、padding mask，目标是 next-item。Meta 官方仓库面向 Ubuntu/CUDA，MovieLens 1M/20M/Amazon Books，DLRM-v3 示例需要多 GPU。"),
         code("import torch_rechub\ntry:\n    from torch_rechub.models.generative import HSTUModel\n    hstu_entry = HSTUModel.__name__\nexcept (ImportError, ModuleNotFoundError) as exc:\n    hstu_entry = f'install torch-rechub[generative]: {exc.__class__.__name__}'\nprint({'torch_rechub':getattr(torch_rechub,'__version__','installed'),'hstu_entry':hstu_entry})"),
         md("### 2. 序列数据契约与指标 smoke\n\n本实验复用确定性目录约束与 NDCG 检查；完整训练需把每个用户按时间排序，最后一项测试、倒数第二项验证，训练窗口只看过去。"),
         code("from recsys_lab import run_generative\nmetrics = run_generative()\nsequence_contract = {'item_seq':'int64[B,L]','time_delta':'float32[B,L]','mask':'bool[B,L]','target':'int64[B]'}\nsequence_contract, metrics"),
         md("### 3. Meta DLRM-v3 工业档\n\n官方栈含 TorchRec/FBGEMM、HSTU 与 M-FALCON。先跑 MovieLens debug 配置，再逐步放大序列长度、层数、embedding 与负样本；不要在 CPU smoke 数值上推断工业收益。"),
         code("meta_profile = {'repository':'meta-recsys/generative-recommenders','environment':'Ubuntu 22.04, CUDA 12.4, Python 3.10','data':['MovieLens-1M','MovieLens-20M','Amazon Books'],'hardware':'official DLRM-v3 debug documents 4 GPUs; >=24GB each','evaluate':['HR@10','NDCG@10','throughput','P99','peak memory']}\nmeta_profile"),
         md("## Checks\n\n检查严格时间切分、padding 不参与 attention、候选目录合法、NDCG 范围和可重复 seed。"),
         code("assert 0 <= metrics['ndcg@5'] <= 1\nassert metrics['invalid_id_rate'] == 0\nprint('PASS: HSTU data/metric contract is valid; full GPU training is intentionally separate')"),
         md("## Next Steps\n\n执行 Meta 官方 MovieLens-1M debug；记录相同召回集下的 SASRec/HSTU 增益与成本；验证 M-FALCON 的增量推断，并在迁移生产前做目录更新、回滚和线上新鲜度实验。")])

    # 3.1 的长稿是编辑母版；交付时按“一种算法一个 Notebook”拆分，并新增章节总结。
    chapter31_source = specs.pop("3_1_classic_models")
    source_cells = chapter31_source["cells"]
    chapter31_sources = "[MovieLens](https://grouplens.org/datasets/movielens/) · [MF](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf) · [FM](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf) · [GBDT+LR](https://research.facebook.com/publications/practical-lessons-from-predicting-clicks-on-ads-at-facebook/)"

    specs["3_1_overview"] = notebook(
        "4.7 经典推荐算法总结与横向对比",
        "在完成五个独立实验后，从任务、表示、训练目标、指标、推理方式和工业边界六个角度进行横向比较。本文不重复完整实现，只提供章节地图、结果快照和选型评论。",
        chapter31_sources,
        [
            source_cells[3],
            md(r"""
## 来源论文与本章解读

本章不是按发表年份罗列名词，而是沿着“表示什么、怎样推理”阅读五条原始工作：

- **Resnick et al. (1994), GroupLens**：早期系统展示了如何依据用户间评分相似度形成邻域预测；关键遗产是“从群体行为借信号”，而不是某个固定相似度公式。
- **Sarwar et al. (2001), Item-based CF**：把邻域移到更稳定、可离线物化的 item 侧，直接影响了后来的相关推荐和倒排召回。
- **Koren, Bell & Volinsky (2009), Matrix Factorization Techniques**：以全局/用户/物品偏置加低秩内积解释评分；其 user/item 向量结构也是双塔召回的最小原型。
- **Rendle (2010), Factorization Machines**：用特征隐向量共享稀疏二阶交叉统计，并用恒等式把计算从 $O(n^2k)$ 降为 $O(nk)$。
- **He et al. (2014), Practical Lessons from Predicting Clicks on Ads at Facebook**：树负责产生非线性叶规则，LR 负责稀疏组合与概率输出，代表经典的两阶段 CTR 工程路线。
- **Mikolov et al. (2013), Efficient Estimation of Word Representations**：用中心词预测上下文高效学词向量；推荐系统把行为序列当句子学物品向量（Item2Vec），是向量召回的起点。

阅读时请区分两组问题：CF/MF/word2vec 面向 user–item 评分或 Top-K 召回；FM/GBDT+LR 面向曝光后的点击概率。不同任务的指标不能直接排名。
"""),
            md(r"""
## 论文关联关系：后一篇在修补前一篇的哪块短板？

六篇论文不是同一赛道上的名词罗列，而是一条“发现问题 → 换表示 → 暴露新问题”的接力。下表从左到右读：每一行先看它从哪个问题出发，再看它把什么难题交给了下一位。
"""),
            code(r"""
import pandas as pd

paper_relationships = pd.DataFrame([
    {"论文": "Resnick 1994 GroupLens", "出发点问题": "netnews 信息过载，编辑过滤跟不上，让读者互评互助",
     "用户/物品表示": "用户评分向量 + 皮尔逊近邻", "训练信号": "1–5 星显式评分",
     "交给下一步的问题": "邻域计算随规模变贵；评分稀疏时找不到邻居"},
    {"论文": "Sarwar 2001 ItemCF", "出发点问题": "用户兴趣多变、用户邻域不稳定且难扩展",
     "用户/物品表示": "物品列向量 + 物品间相似度", "训练信号": "同一批用户在物品上的共现评分",
     "交给下一步的问题": "仍是显式邻域，无法概括潜在口味结构"},
    {"论文": "Koren 2009 MF", "出发点问题": "邻域无法表达潜在结构，Netflix 规模上难以扩展",
     "用户/物品表示": "用户/物品各一个 d 维隐向量，内积打分", "训练信号": "已观察评分上的平方误差",
     "交给下一步的问题": "只认 ID，新实体冷启动；无法直接利用上下文特征"},
    {"论文": "Rendle 2010 FM", "出发点问题": "MF 只认评分矩阵格式；稀疏特征交叉的独立参数学不到",
     "用户/物品表示": "每个特征一个 k 维隐向量，内积共享交叉", "训练信号": "任意监督标签（评分或点击）",
     "交给下一步的问题": "只有二阶交叉，高阶模式需要更深的结构"},
    {"论文": "He 2014 GBDT+LR", "出发点问题": "工业 CTR 同时要非线性规则、校准概率与快速更新",
     "用户/物品表示": "树叶子 one-hot（离散规则）", "训练信号": "曝光—点击日志的 log loss / NE",
     "交给下一步的问题": "两阶段非端到端，特征与树版本靠人工治理"},
    {"论文": "Mikolov 2013 word2vec", "出发点问题": "词（物品）只是编号，需要可计算、可检索的相似度",
     "用户/物品表示": "每个物品一个嵌入向量", "训练信号": "序列窗口内共现（负采样）",
     "交给下一步的问题": "单向量平均多兴趣、序列建模弱 → 3.2 双塔/多兴趣/序列召回"},
])
display(paper_relationships)
"""),
            md(r"""
## 未来发展：从经典方法走向深度章节

经典方法并没有被“淘汰”，而是各自指出了下一步要解除的约束。下表按四个维度对照 3.1 与后续 3.2/3.3 章的做法；最后一列是仍未回答的问题，不是收益承诺。
"""),
            code(r"""
future = pd.DataFrame([
    {"维度": "表示", "3.1 经典做法": "邻域、单隐向量、特征向量",
     "3.2/3.3 深度做法": "双塔 Embedding（DSSM）、多兴趣（MIND）、序列表示（SASRec）",
     "仍待回答": "表示如何随上下文实时变化，同时保持训练/服务一致？"},
    {"维度": "特征交互", "3.1 经典做法": "内积、FM 二阶交叉、树规则离散组合",
     "3.2/3.3 深度做法": "DeepFM 共享 embedding 端到端学高阶交叉；DIN/DIEN 候选感知兴趣激活与演化",
     "仍待回答": "高阶交互的可解释性、计算成本与过拟合控制？"},
    {"维度": "训练信号", "3.1 经典做法": "显式评分平方误差、点击 log loss、序列共现负采样",
     "3.2/3.3 深度做法": "in-batch softmax 召回目标、辅助损失、next-item 序列目标",
     "仍待回答": "曝光/位置偏差纠正、多目标（点击/转化/时长）如何对齐？"},
    {"维度": "服务", "3.1 经典做法": "离线近邻表、物品向量 ANN、两阶段树+LR",
     "3.2/3.3 深度做法": "用户/物品塔分离 + ANN 召回，统一排序模型在线打分",
     "仍待回答": "索引新鲜度、级联一致性与 P99 成本如何工程化？"},
])
display(future)
"""),
            md(r"""
## Steps

## 1. 五个独立实验

| Notebook | 核心问题 | 主要指标 |
|---|---|---|
| 4.2 协同过滤 | 谁与谁相似，如何用邻域产生候选？ | HR@K、Recall@K、Coverage |
| 4.3 矩阵分解（BiasMF） | 如何把稀疏矩阵压缩成用户/物品隐向量？ | RMSE、HR@K |
| 4.4 FM | 如何在线性复杂度内学习稀疏二阶交互？ | AUC、LogLoss |
| 4.5 GBDT+LR | 如何用树叶规则生成特征，再由 LR 校准概率？ | AUC、LogLoss |
| 4.6 word2vec / Item2Vec | 如何把行为序列学成物品向量做召回？ | Recall@K、Coverage |

这五篇均可独立从头执行，不依赖其他 Notebook 的内存状态。
"""),
            md(r"""
## 2. 结果快照

以下数值由五个算法 Notebook 在执行末尾写入 `results/chapter_4/*.json`，本总结只读取产物，不手填实验值。CF/MF/word2vec 与 FM/GBDT+LR 属于不同任务，表格用于确认代码路径和理解指标，不能按数值大小直接排名。
"""),
            code(r"""
import json
import pandas as pd

result_dir = ARTIFACT_ROOT / "results" / "chapter_4"
result_files = sorted(result_dir.glob("*.json"))
assert len(result_files) == 5, f"需要先执行五个算法 Notebook；当前产物：{[p.name for p in result_files]}"
records = []
for path in result_files:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records.extend(payload["records"])
comparison = pd.DataFrame(records)
display(comparison.round(4))
print("指标来源：", [p.name for p in result_files])
"""),
            md(r"""
## 3. 与原论文的可比性审计

本章算法来自不同任务，原论文也没有一个统一 benchmark。下表的目的不是制造“复现率”，而是明确当前教学实验与论文实验之间的边界。

| 算法 | 当前教程产物 | 原论文代表性结果/规模 | 审计结论 |
|---|---|---|---|
| UserCF / ItemCF | MovieLens latest-small 子集，HR@10=0.125 | GroupLens 与 ItemCF 论文采用更早的数据与 MAE/预测质量等口径 | 实现验证，不是原论文复现；指标定义不同，不能相减。 |
| BiasMF | RMSE=1.1905 | Koren 等报告 Netflix 系统 RMSE=0.9514、大奖目标 0.8563；Netflix 含约 1 亿评分 | 当前小样本结果明显更弱，但数据、切分、维数与正则均未对齐。 |
| FM | AUC=0.5964，LogLoss=2.6061 | FM 原论文重点验证稀疏因子化的通用性，包含 Netflix 1 亿样本等回归/排序任务，并未给出本教程 CTR 口径 | AUC 略高于随机，但 LogLoss 显示严重过度自信；只能作为公式实现练习。 |
| GBDT+LR | AUC=0.6376，LogLoss=0.7414 | Facebook 论文使用工业广告日志与 Normalized Entropy、校准等内部口径 | 当前仅验证“树叶编码 → LR”链路，不能宣称复现工业收益。 |
| word2vec / Item2Vec | Recall@5=0.05，Coverage≈0.18 | Mikolov 在文本词向量任务上评测句法/语义类比；Item2Vec 论文用协同过滤数据 | 小序列 smoke 仅验证 Skip-gram 链路与向量近邻，不能对标论文词向量质量。 |

因此 3.1 的可信结论是代码链路、数学结构和同一教程内的基线差异；不能把这些 smoke 数字当成公开 benchmark。
"""),
            md(r"""
## 4. 横向评论

| 维度 | CF | MF | FM | GBDT+LR | word2vec |
|---|---|---|---|---|---|
| 输入 | user–item 行为 | user–item 评分/行为 | 任意稀疏与连续特征 | 表格特征 | 行为序列 |
| 表示 | 邻域/相似度 | 低维 embedding | 特征 embedding | 树叶规则 | 物品 embedding |
| 交互能力 | 共现传播 | user-item 内积 | 全部二阶交互 | 非线性条件组合 | 序列共现 |
| 冷启动 | 弱 | 弱 | 可加入内容特征 | 可加入内容特征 | 弱（新物品无向量） |
| 序列能力 | 无 | 无 | 无 | 无 | 局部次序 |
| 典型位置 | 召回/相关推荐 | 召回/评分 | CTR 排序 | CTR 排序 | 向量召回 |
| 主要风险 | 稀疏、热门偏差 | 未观察样本、内积限制 | 仅二阶 | 两阶段失配、规则漂移 | 短序列、热门偏置 |

技术演进不是简单替换关系：ItemCF 常作为覆盖与兜底通道长期保留；MF 是双塔向量召回的最小原型；FM 和 GBDT+LR 则代表排序阶段两条经典特征交叉路线；word2vec / Item2Vec 把序列嵌入做成可 ANN 的召回向量，衔接后续 DSSM、SASRec。
"""),
            md(r"""
## Checks

- 比较 Top-K 模型时同时看命中、覆盖和新颖度。
- 比较 CTR 模型时至少同时看 AUC、LogLoss 与校准。
- 只在相同数据、相同时间切分、相同负样本下横向比较。
- smoke 指标用于代码与理解验证，不当作公开 benchmark。
"""),
            code("assert len(comparison) == 6\nassert set(comparison.task) == {'Top-K', '评分预测', 'CTR'}\nassert comparison.source_notebook.nunique() == 5\nprint('PASS：章节总结从五个独立实验聚合了全部经典算法与三类任务。')"),
            md(r"""
## Next Steps

按顺序阅读五篇算法 Notebook。若目标是构建系统基线，建议先落地 ItemCF 与 GBDT+LR/FＭ，再根据目录规模和特征类型引入 MF/双塔与 DeepFM。
"""),
        ],
    )

    specs["4_2_collaborative_filtering"] = notebook(
        "4.2 协同过滤：UserCF 与 ItemCF",
        "单独掌握邻域推荐：从真实 MovieLens 评分数据、余弦相似度，到 UserCF/ItemCF 训练、Top-K 推理、HR@10 与 Coverage 评测。",
        "[GroupLens MovieLens](https://grouplens.org/datasets/movielens/) · [Collaborative Filtering](https://dl.acm.org/doi/10.1145/192844.192905)",
        [md(r"""
## 来源论文与阅读提示

**Resnick et al. (1994), GroupLens** 将“相似用户的评分加权”落成了可运行系统，是 UserCF 的代表性早期来源；**Sarwar et al. (2001), Item-based Collaborative Filtering Recommendation Algorithms** 则系统比较 item 相似度与预测方式。后者的重要工业含义是：item 关系通常比用户兴趣稳定，可以离线计算 Top-N 近邻表，线上只聚合用户历史。

本实验用二值矩阵突出共现传播。真实评分 UserCF 还可做均值中心化；隐式反馈则常加入热门度惩罚、时间衰减和置信度权重。

### 公式怎么读（直觉版）

若向量、点积或矩阵乘法还陌生，请先运行 **3.0 推荐算法数学基础**。这里的 $R_u\cdot R_v$ 只是把两位用户在每个物品上的 0/1 逐项相乘再相加，因此等于共同喜欢数；分母是两行的“勾股长度”，用于避免行为多的人天然得分更高。后面的 `similarity @ train_matrix` 则是一次批量完成所有邻居加权。
""")] + [source_cells[i] for i in [4, 5, 6, 7, 8]] + [
            md(r"""
### 2.3 小矩阵演示：共现矩阵怎样变成推荐分数

先不用完整数据，手工构造 3 个用户 × 4 个物品。`R @ R.T` 统计用户共同喜欢多少物品，`R.T @ R` 统计物品被多少用户共同喜欢；余弦归一化消除活跃度/热门度的量纲。最后的矩阵乘法正对应邻域加权公式。
"""),
            code(r"""
toy_R = np.array([
    [1, 1, 0, 0],  # u0 看过 i0、i1
    [1, 0, 1, 0],  # u1 看过 i0、i2
    [0, 1, 1, 1],  # u2 看过 i1、i2、i3
], dtype=float)

toy_user_cooccurrence = toy_R @ toy_R.T
toy_item_cooccurrence = toy_R.T @ toy_R
display(pd.DataFrame(toy_R, index=["u0","u1","u2"], columns=["i0","i1","i2","i3"]))
display(pd.DataFrame(toy_user_cooccurrence, index=["u0","u1","u2"], columns=["u0","u1","u2"]))
display(pd.DataFrame(toy_item_cooccurrence, index=["i0","i1","i2","i3"], columns=["i0","i1","i2","i3"]))

def toy_cosine(matrix):
    normalized = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, 0)
    return similarity

toy_user_similarity = toy_cosine(toy_R)
toy_item_similarity = toy_cosine(toy_R.T)
toy_usercf_scores = toy_user_similarity @ toy_R
toy_itemcf_scores = toy_R @ toy_item_similarity
toy_usercf_scores[toy_R > 0] = -np.inf
toy_itemcf_scores[toy_R > 0] = -np.inf

print("u0 UserCF 未见物品分数:", {f"i{i}": round(v, 3) for i, v in enumerate(toy_usercf_scores[0]) if np.isfinite(v)})
print("u0 ItemCF 未见物品分数:", {f"i{i}": round(v, 3) for i, v in enumerate(toy_itemcf_scores[0]) if np.isfinite(v)})
assert np.allclose(toy_user_cooccurrence, toy_R @ toy_R.T)
assert np.allclose(toy_item_cooccurrence, toy_R.T @ toy_R)
""")
        ] + [source_cells[i] for i in [9, 10, 11, 12, 13]] + [
            md("## Checks\n\n检查相似度矩阵对角线、已见物品屏蔽、Top-K 范围以及 Coverage。"),
            code("assert np.allclose(np.diag(user_similarity), 0)\nassert np.allclose(np.diag(item_similarity), 0)\nassert 0 <= cf_metrics['UserCF HR@10'] <= 1\nassert 0 <= cf_metrics['ItemCF Coverage'] <= 1\nprint('PASS：UserCF / ItemCF 训练、推理与评测均有效。')"),
            code(r"""
result_dir = ARTIFACT_ROOT / "results" / "chapter_4"
result_dir.mkdir(parents=True, exist_ok=True)
payload = {"records": [
    {"algorithm":"UserCF", "task":"Top-K", "metric":"HR@10 ↑", "value":cf_metrics["UserCF HR@10"], "secondary_metric":"Coverage", "secondary_value":cf_metrics["UserCF Coverage"], "online_inference":"聚合相似用户的历史", "source_notebook":"4_2_collaborative_filtering"},
    {"algorithm":"ItemCF", "task":"Top-K", "metric":"HR@10 ↑", "value":cf_metrics["ItemCF HR@10"], "secondary_metric":"Coverage", "secondary_value":cf_metrics["ItemCF Coverage"], "online_inference":"聚合历史物品的近邻", "source_notebook":"4_2_collaborative_filtering"},
]}
(result_dir / "collaborative_filtering.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print("已写出章节指标：collaborative_filtering.json")
"""),
            md("## Next Steps\n\n加入热门度惩罚、时间衰减、邻居 Top-N 截断，并在 MovieLens 100K 上比较 UserCF 与 ItemCF 的 HR、Coverage 和长尾表现。"),
        ],
    )

    specs["4_3_matrix_factorization"] = notebook(
        "4.3 矩阵分解：BiasMF",
        "单独掌握低秩推荐：理解偏置与 embedding，使用 PyTorch 训练 BiasMF，并分别完成评分预测和全库 Top-K 推理。",
        "[Matrix Factorization Techniques for Recommender Systems](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)",
        [md(r"""
## 来源论文与阅读提示

**Koren, Bell & Volinsky (2009)** 总结了 Netflix Prize 时代的矩阵分解实践。论文最值得关注的不是“做一次 SVD”，而是：只在观察集合 $\Omega$ 上优化、显式建模全局/用户/物品偏置，并用正则化控制 user/item 隐向量。本文的 BiasMF 正对应这一结构。

### 公式怎么读（直觉版）

若矩阵或点积陌生，请先阅读 **3.0 推荐算法数学基础**。MF 可以想成给每位用户和每部电影各发一张“兴趣坐标卡”：例如 `[动作偏好, 喜剧偏好]`。两张卡逐项相乘再相加，方向越一致，预测喜欢程度越高。$R\approx PQ^\top$ 只是说：用“用户坐标表 × 物品坐标表”近似原本巨大且稀疏的评分表。
""")] + [source_cells[i] for i in [4, 5, 6, 7]] + [
            code("train_matrix = np.zeros((n_users, n_items), dtype=np.float32)\nfor row in train_ratings.itertuples():\n    train_matrix[row.user_id, row.item_id] = float(row.rating >= 3)\nprint({'interaction_matrix': train_matrix.shape, 'observed_positive': int(train_matrix.sum())})"),
        ] + [source_cells[i] for i in [14, 15, 16, 17, 18, 19]] + [
            md("## Checks\n\n确认训练损失下降、测试 RMSE 有限、推荐结果不包含已见物品。"),
            code("assert loss_curve[-1] < loss_curve[0]\nassert np.isfinite(mf_rmse) and mf_rmse < 3\nassert not set(mf_top10[0]).intersection(np.where(train_matrix[0] > 0)[0])\nprint('PASS：BiasMF 训练、评分推理和 Top-K 推理均有效。')"),
            code(r"""
result_dir = ARTIFACT_ROOT / "results" / "chapter_4"; result_dir.mkdir(parents=True, exist_ok=True)
payload = {"records": [{"algorithm":"BiasMF", "task":"评分预测", "metric":"RMSE ↓", "value":mf_rmse, "secondary_metric":"HR@10", "secondary_value":mf_hr10, "online_inference":"user/item 向量内积", "source_notebook":"4_3_matrix_factorization"}]}
(result_dir / "matrix_factorization.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print("已写出章节指标：matrix_factorization.json")
"""),
            md("## Next Steps\n\n把平方误差替换为 BPR pairwise loss；比较 embedding 维数与正则强度；再将 item 向量放入 ANN 索引。"),
        ],
    )

    specs["4_4_factorization_machine"] = notebook(
        "4.4 因子分解机：FM",
        "单独掌握稀疏二阶交互：从 MovieLens 评分阈值派生的高偏好标签、FM 数学恒等式，到 PyTorch 训练、概率推理、AUC 与 LogLoss；该教学任务不冒充曝光—点击日志。",
        "[Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)",
        [md(r"""
## 来源论文与阅读提示

**Rendle (2010), Factorization Machines** 的核心贡献是把矩阵分解的隐向量交互推广到任意稀疏特征，并通过代数恒等式在线性时间内计算全部二阶项。阅读时应特别看稀疏条件下“未直接观察过的特征组合”如何借各自隐向量共享统计。

### 公式怎么读（直觉版）

先把 one-hot 理解成一排开关：当前用户、物品、设备对应的开关为 1，其余为 0。FM 为每个开关配置一张小坐标卡；任意两个打开的开关都做一次点积。恒等式不是新模型，只是把“逐对计算”重新整理为“先求和再平方”，避免重复工作。相关向量、点积和概率知识见 **3.0**。
"""), source_cells[5]] + [source_cells[i] for i in [20, 21, 22, 23, 24, 25, 26, 27]] + [
            md("## Checks\n\n确认概率合法、AUC 高于随机水平，并同时观察 LogLoss，避免只看排序不看校准。"),
            code("assert np.all((fm_probability >= 0) & (fm_probability <= 1))\nassert fm_auc > .55\nassert np.isfinite(fm_logloss)\nprint('PASS：FM 训练、概率推理和 CTR 指标均有效。')"),
            code(r"""
result_dir = ARTIFACT_ROOT / "results" / "chapter_4"; result_dir.mkdir(parents=True, exist_ok=True)
payload = {"records": [{"algorithm":"FM", "task":"CTR", "metric":"AUC ↑", "value":fm_auc, "secondary_metric":"LogLoss ↓", "secondary_value":fm_logloss, "online_inference":"线性项 + 二阶隐向量交互", "source_notebook":"4_4_factorization_machine"}]}
(result_dir / "factorization_machine.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print("已写出章节指标：factorization_machine.json")
"""),
            md("## Next Steps\n\n加入 LR 基线；拆解不同 field 的交互；继续学习 FFM、DeepFM 与 DCN。"),
        ],
    )

    specs["4_5_gbdt_lr"] = notebook(
        "4.5 GBDT+LR：树叶特征与概率校准",
        "单独掌握经典两阶段 CTR 模型：用 XGBoost 学条件规则，将叶节点 one-hot 后训练 LR，并复现完整在线推理链。",
        "[Practical Lessons from Predicting Clicks on Ads at Facebook](https://research.facebook.com/publications/practical-lessons-from-predicting-clicks-on-ads-at-facebook/)",
        [md(r"""
## 来源论文与阅读提示

**He et al. (2014), Practical Lessons from Predicting Clicks on Ads at Facebook** 的关键工程判断是把 GBDT 当作监督式特征变换：每棵树的叶节点代表一组条件规则，叶 one-hot 再进入 LR。它不是端到端模型，因此训练/服务必须同时版本化树、叶编码器和 LR。

### 公式怎么读（直觉版）

一棵决策树像连续做选择题：“是否晚间？”“是否移动设备？”最终落到一个叶子。把每棵树落到的叶子变成 0/1 开关，再由 LR 做加权求和。Sigmoid 最后把任意实数分数压到 0～1，变成点击概率；为什么要用 LogLoss 训练，可先看 **3.0 推荐算法数学基础** 的概率图。
"""), source_cells[5]] + [source_cells[i] for i in [20, 21, 22, 28, 29, 30, 31, 32, 33, 34, 35]] + [
            md("## Checks\n\n确认树数、叶特征维数、概率范围和 AUC；推理必须复用同一编码器、树模型与 LR。"),
            code("assert train_leaf.shape[1] == gbdt.n_estimators\nassert train_leaf_onehot.shape[1] > train_leaf.shape[1]\nassert np.all((gbdt_lr_probability >= 0) & (gbdt_lr_probability <= 1))\nassert gbdt_lr_auc > .55\nprint('PASS：GBDT、叶编码、LR 与在线推理链均有效。')"),
            code(r"""
result_dir = ARTIFACT_ROOT / "results" / "chapter_4"; result_dir.mkdir(parents=True, exist_ok=True)
payload = {"records": [{"algorithm":"GBDT+LR", "task":"CTR", "metric":"AUC ↑", "value":gbdt_lr_auc, "secondary_metric":"LogLoss ↓", "secondary_value":gbdt_lr_logloss, "online_inference":"树叶编码 → LR 概率", "source_notebook":"4_5_gbdt_lr"}]}
(result_dir / "gbdt_lr.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print("已写出章节指标：gbdt_lr.json")
"""),
            md("## Next Steps\n\n与原始特征 LR、FM 对照；加入概率校准曲线；模拟数据漂移并监控叶节点覆盖变化。"),
        ],
    )

    specs["4_6_word2vec"] = notebook(
        "4.6 word2vec / Item2Vec：行为序列嵌入与召回",
        "单独掌握用 word2vec 思想把行为序列学成物品向量：从 Skip-gram 与负采样、真实 MovieLens 序列训练，到向量召回与 Recall@K 评测。",
        "[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)",
        [
            md(r"""
## 来源论文与阅读提示

**Mikolov et al. (2013), Efficient Estimation of Word Representations in Vector Space** 提出 Skip-gram 与 CBOW：用一个词预测其上下文，在海量文本上高效学到词向量。推荐系统借这条思路形成 **Item2Vec**——把每个用户的正反馈物品序列当成一句话、物品当成词，于是同一序列里共现的物品会在嵌入空间被拉近，得到可做全库召回的物品向量。本实验用 MovieLens 正反馈序列训练 Skip-gram，再用向量近邻做下一物品 Recall@K。

### 公式怎么读（直觉版）

Skip-gram 的训练样本是 (中心, 上下文) 物品对。中心向量 $v_c$ 与上下文向量 $v_w$ 做内积，越共现越希望内积大；负采样再随机抽几个不共现的物品，希望它们的内积小。用 Sigmoid 把内积压到 0～1，配 LogLoss，就是“拉正、推负”。向量、点积与 Sigmoid 见 **3.0 推荐算法数学基础**。
"""),
            md(r"""
## Steps

## 1. 数据集与序列构造

### 1.1 从真实评分到行为序列

用 MovieLens latest-small 的真实评分，按时间排序、保留 `rating >= 4` 的正反馈，得到每个用户的物品序列。`smoke` 档读取仓库内确定性切片，`full` 档扩大到官方完整文件；两档都不制造交互或序列。
"""),
            code(r"""
import numpy as np
import pandas as pd
import torch

SEED = 2026
torch.manual_seed(SEED)
from recsys_lab.data import load_movielens, leave_last_out, movielens_provenance

ratings, movies = load_movielens(max_users=80, max_items=360, min_user_events=12)
train_ratings, test_ratings = leave_last_out(ratings)
print({"rows": len(ratings), "users": ratings.user_id.nunique(), "items": ratings.item_id.nunique(),
       "source": movielens_provenance(ratings)["source"], "fabricated_rows": 0})
ratings[["userId", "movieId", "rating", "timestamp", "title"]].head(6)
"""),
            md(r"""
### 1.2 为什么把序列当一句话？

协同过滤只看“谁和谁共现”，丢了次序；而用户的观看/购买往往有顺序迁移（看完 A 更可能看 B）。把序列当句子，Skip-gram 用窗口内的邻居作上下文，既利用共现也保留局部次序，学到的向量还能离线预计算、线上做 ANN 召回。
"""),
            md(r"""
## 2. word2vec：Skip-gram 与负采样

### 2.1 直觉

对序列 `[i0, i1, i2, i3]`、窗口 1，生成正对 `(i0,i1),(i1,i0),(i1,i2),(i2,i1),(i2,i3),(i3,i2)`。每对训练“中心预测上下文”；再为每对配几个随机负物品。训练后取中心嵌入作为物品向量。

### 2.2 数学：负采样目标

**通用先修：** [3.2 隐式反馈与假负例](/notebooks/3_2_data_ml_basics#implicit-feedback) · [3.3 点积与 embedding](/notebooks/3_3_linear_algebra#matmul-embedding) · [3.6 二元交叉熵](/notebooks/3_6_information_theory#cross-entropy-kl)<br>
**本论文新增数学：** Skip-gram 的中心/上下文双 embedding、负采样目标，以及按出现次数的 $0.75$ 次幂构造噪声分布。

先说符号：每个物品有两套向量——作中心词时用 $v_c$，作上下文时用 $v_w$（分开学习更稳定，推理时只保留中心向量）；$\sigma(a)=1/(1+e^{-a})$ 把内积压成“会共现”的概率。对正对 $(c,w)$ 和负样本集合 $\mathcal{N}$：

$$
\mathcal{L}=-\log\sigma(v_c\cdot v_w)-\sum_{n\in\mathcal{N}}\log\sigma(-v_c\cdot v_n)
$$

第一项希望正对内积大（$\sigma(v_c\cdot v_w)$ 接近 1）；第二项希望负样本内积小（$\sigma(-v_c\cdot v_n)$ 接近 1，等价于 $\sigma(v_c\cdot v_n)$ 接近 0）。为什么不直接对整个物品表做 softmax？因为分母要遍历全部物品，代价随目录线性增长；负采样把它近似为几个二分类，训练代价与物品总数无关。

**手算一遍。** 取 $v_c=[1,0]$，正样本 $v_w=[0.8,0.2]$：内积 $=0.8$，$\sigma(0.8)\approx0.69$，$-\log0.69\approx0.37$。负样本 $v_n=[-1,1]$：内积 $=-1$，$\sigma(1)\approx0.73$，$-\log0.73\approx0.31$。合计 $\mathcal L\approx0.68$；训练会把 $v_c$ 推向 $v_w$、推离 $v_n$，两项随之变小。

内积越大、Sigmoid 越接近 1；负样本希望内积小、Sigmoid 接近 0。它等价于对正负样本做带 Logits 的二元交叉熵。

负物品也不是随便均匀抽。word2vec 论文实现常用 unigram 分布的 $0.75$ 次幂：若物品 $i$ 在训练上下文中出现 $c_i$ 次，

$$
P_n(i)=\frac{c_i^{0.75}}{\sum_j c_j^{0.75}}.
$$

$0.75<1$ 会压平头部：热门物品仍比冷门物品更常成为负例，但不会完全支配训练。被采为负例只表示“本次窗口没有把它当正上下文”，不证明用户讨厌它，因此仍可能是假负例。

还要标清迁移边界：Mikolov 论文研究的是文本词与语言上下文；**Item2Vec** 才把词换成物品、句子换成会话/行为篮子。可迁移的是“局部共现可学习 embedding”的结构，不能把词语的句法/语义结论直接宣称为用户偏好，也不能把未共现解释成真实负反馈。
"""),
            md(r"""
### 2.3 小序列演示：玩具序列怎样学到共现

先用 3 个用户的小序列手算 Skip-gram 对，确认“同一序列共现的物品对”如何进入训练集。
"""),
            code(r"""
toy_sequences = [[1, 2, 3, 2], [2, 3, 4], [1, 3, 4, 5]]

def skip_gram_pairs(sequences, window=1):
    pairs = []
    for seq in sequences:
        for i, c in enumerate(seq):
            for j in range(max(0, i - window), min(len(seq), i + window + 1)):
                if j != i:
                    pairs.append((c, seq[j]))
    return pairs

toy_pairs = skip_gram_pairs(toy_sequences, window=1)
print("toy 正对数:", len(toy_pairs), "示例:", toy_pairs[:6])
assert (2, 3) in toy_pairs and (3, 2) in toy_pairs
"""),
            md("### 2.4 训练：在真实 MovieLens 序列上训练 Skip-gram"),
            code(r"""
class SkipGram(torch.nn.Module):
    def __init__(self, num_items, dim=32):
        super().__init__()
        # 中心嵌入与上下文嵌入分离，是 word2vec 的常规做法；负采样时只更新相关行。
        self.center = torch.nn.Embedding(num_items, dim)
        self.context = torch.nn.Embedding(num_items, dim)
        torch.nn.init.uniform_(self.center.weight, -0.01, 0.01)
        torch.nn.init.zeros_(self.context.weight)

    def forward(self, c, w):
        # 中心向量与上下文向量的内积越大，说明两物品越可能在同一序列共现。
        return (self.center(c) * self.context(w)).sum(-1)


# 物品 ID +1（0 留作 padding）；按时间构造每个用户的训练序列。
positive = train_ratings[train_ratings.rating >= 4.0].sort_values(["user_id", "timestamp", "item_id"])
sequences = positive.groupby("user_id").item_id.apply(lambda v: [int(x) + 1 for x in v]).to_dict()
sequences = {int(u): s for u, s in sequences.items() if len(s) >= 4}
pairs = skip_gram_pairs(list(sequences.values()), window=3)
num_items = int(ratings.item_id.max()) + 2
center = torch.tensor([p[0] for p in pairs], dtype=torch.long)
context = torch.tensor([p[1] for p in pairs], dtype=torch.long)
rng = np.random.default_rng(SEED)

# 论文式噪声分布：上下文出现次数的 0.75 次幂，而不是均匀采样。
context_counts = np.bincount(context.numpy(), minlength=num_items)[1:].astype(float)
negative_sampling_probability = context_counts ** 0.75
negative_sampling_probability /= negative_sampling_probability.sum()

model = SkipGram(num_items, dim=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
losses = []
NUM_NEG = 5
for _ in range(8):
    neg_items = torch.tensor(
        rng.choice(np.arange(1, num_items), size=(len(pairs), NUM_NEG), p=negative_sampling_probability),
        dtype=torch.long,
    )
    score_pos = model(center, context)
    score_neg = model(center[:, None].expand(-1, NUM_NEG).reshape(-1), neg_items.reshape(-1))
    pos = torch.nn.functional.binary_cross_entropy_with_logits(score_pos, torch.ones_like(score_pos))
    neg = torch.nn.functional.binary_cross_entropy_with_logits(score_neg, torch.zeros_like(score_neg))
    loss = pos + neg
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    losses.append(float(loss.detach()))
print({"pairs": len(pairs), "num_items": num_items, "loss_first": round(losses[0], 4), "loss_last": round(losses[-1], 4)})
"""),
            md(r"""
### 2.5 推理与召回：物品向量 -> 用户向量 -> Top-K

取中心嵌入作为物品向量，归一化后用余弦相似度。用户向量 = 其训练历史物品向量的均值；屏蔽已见物品后取 Top-K，检查留出的下一物品是否命中。
"""),
            code(r"""
with torch.no_grad():
    embeddings = model.center.weight.detach().numpy()
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
normed = embeddings / np.maximum(norms, 1e-12)

test_frame = test_ratings.sort_values("user_id")
targets = test_frame.item_id.to_numpy() + 1
users = test_frame.user_id.to_numpy()
seen = {u: set(s) for u, s in sequences.items()}
hits, all_recs = 0, []
with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
    for user, target in zip(users, targets):
        history = sequences.get(int(user), [])
        if not history:
            continue
        uv = normed[history].mean(axis=0)
        uv = uv / max(float(np.linalg.norm(uv)), 1e-12)
        scores = normed @ uv
        scores[list(seen.get(int(user), []))] = -1.0
        k = min(5, len(scores) - 1)
        top5 = np.argpartition(-scores, kth=k)[:5]
        all_recs.extend(top5.tolist())
        hits += int(target in top5)
word2vec_recall_at_5 = hits / max(len(targets), 1)
word2vec_coverage = len(set(all_recs)) / num_items
print({"word2vec_recall@5": round(word2vec_recall_at_5, 4), "coverage": round(word2vec_coverage, 4), "num_test_users": len(targets)})

# 相似物品示例：任取一个热门物品，看它的向量最近邻。
mode_item = int(ratings.item_id.mode().iloc[0])
sims = normed @ normed[mode_item + 1]
sims[mode_item + 1] = -1.0
neighbors = np.argsort(-sims)[:5]
movie_lookup = ratings.drop_duplicates("item_id").set_index("item_id")["title"]
print("anchor:", movie_lookup.get(mode_item))
print("近邻:", [movie_lookup.get(i - 1) for i in neighbors])
"""),
            md(r"""
### 2.6 结果讨论与边界

Recall@5 在小切片上可能不高：序列短、物品多、负采样随机性大。它仍是合理的召回基线，且物品向量可离线预计算、线上做 ANN。与 ItemCF 相比，向量召回更稠密、可泛化到未直接共现的物品；但冷启动物品无向量，窗口与负样本数对效果影响明显。

**优点**：向量可预计算、ANN 全库召回；序列保留局部次序；易扩展到多模态侧信息。
**缺点**：新物品无向量；短序列学不稳；共现继承曝光与热门偏置；窗口/负采样需调参。

**推理复杂度**：训练只更新被采样到的 embedding 行；线上物品向量离线预计算，用户向量由少量历史向量均值得到，全库检索交给 ANN。
"""),
            md("## Checks\n\n确认序列非空、loss 下降、Recall 范围合法，并比较 word2vec 与 ItemCF 的召回。"),
            code(r"""
assert len(pairs) > 0
assert losses[-1] < losses[0]
assert 0 <= word2vec_recall_at_5 <= 1
assert 0 <= word2vec_coverage <= 1
print('PASS：word2vec / Item2Vec 序列构造、Skip-gram 训练与向量召回均有效。')
"""),
            code(r"""
result_dir = ARTIFACT_ROOT / "results" / "chapter_4"; result_dir.mkdir(parents=True, exist_ok=True)
payload = {"records": [
    {"algorithm": "word2vec / Item2Vec", "task": "Top-K", "metric": "Recall@5 ↑", "value": word2vec_recall_at_5,
     "secondary_metric": "Coverage", "secondary_value": word2vec_coverage,
     "online_inference": "物品向量 ANN 召回", "source_notebook": "4_6_word2vec"},
]}
(result_dir / "word2vec.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print("已写出章节指标：word2vec.json")
"""),
            md("## Next Steps\n\n扩大到 Amazon 完整序列、调窗口/负采样/维度，加入时间衰减与热门度惩罚，并与 MF 向量、SASRec 序列召回对比 Recall 与 Coverage。"),
        ],
    )

    # 统一重写 3.2 之后的教程：每种算法独立成篇，并新增读取实验产物的总结。
    for legacy_slug in [
        "3_2_retrieval_dssm_mind",
        "3_3_ranking_deepfm_din_dien",
        "3_4_multitask_mmoe_ple",
        "8_2_openonerec_practice",
        "8_3_dlrm_hstu_practice",
    ]:
        specs.pop(legacy_slug, None)
    specs["4_7_classic_summary"] = specs.pop("3_1_overview")
    specs.update(build_opening_specs(md, code, notebook))
    specs.update(build_deep_specs(md, code, notebook))
    specs.update(build_math_specs(md, code, curriculum_notebook))
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=sorted(specs), help="只生成一个 Notebook，保留其他已执行输出")
    parser.add_argument("--output-dir", type=Path, default=OUT, help="Notebook 输出目录；默认写入仓库 notebooks/")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = {args.only: specs[args.only]} if args.only else specs
    if not args.only:
        for stale in output_dir.glob("*.ipynb"):
            stale.unlink()
    for slug, nb in selected.items():
        nbf.write(nb, output_dir / f"{slug}.ipynb")
    print(f"generated {len(selected)} notebooks in {output_dir}")


if __name__ == "__main__":
    main()
