#!/usr/bin/env python3
"""Legacy migration helper for the former monolithic 3.0 notebook.

The canonical curriculum now lives in ``scripts/tutorial_math_specs.py`` and
``scripts/tutorial_deep_specs.py``. This retained script is not part of the
publication path; do not run it against generated artifacts.
"""
from __future__ import annotations

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

NEW_SUBSECTION = """### 5.1 逐元素运算、残差/LayerNorm 与常见激活

拿到两个同形状向量 $a,b$，**逐元素乘积**（Hadamard 乘积，记作 $\\odot$）就是对应位置分别相乘、不再求和：

$$
(a\\odot b)_i=a_i b_i.
$$

DIN/DIEN/HSTU 用它表达“两个向量在每一维上共同激活多少”，和第 2 节的点积（求和成一个数）形成对照。把一个向量跳层加回后面，$H\\leftarrow H+X$，叫作**残差连接**，它让梯度在深层网络中仍能传到底；之后常做 **LayerNorm**（沿特征维把向量标准化到均值 0、方差 1 再缩放平移），让训练更稳定。

推荐里反复出现的激活函数：

- **ReLU**：$\\operatorname{ReLU}(z)=\\max(0,z)$，负数直接截 0，计算便宜。
- **Sigmoid**：$\\sigma(z)=1/(1+e^{-z})$，第 4 节已讲，把数压到 0～1 当概率或门。
- **Tanh**：$\\tanh(z)=(e^z-e^{-z})/(e^z+e^{-z})$，输出 $-1\\sim1$，常与门配合。
- **Softplus**：$\\operatorname{softplus}(z)=\\log(1+e^z)$，是 ReLU 的平滑版本，SASRec 用它写 pairwise logistic 损失。
- **SiLU / Swish**：$\\operatorname{SiLU}(z)=z\\sigma(z)$，让向量自己做自己的门，HSTU、Modern 推荐模型常用。

这些激活都是**逐元素**作用，不会改变张量形状；这与 Softmax（跨位置归一化、混合多个位置）用途完全不同。
"""

# Each entry: the prereq paragraph + the original hand-math explanation body (kept after it).
MATH_BY_HAND = {
    "3_2_1_dssm": (
        "**通用先修：** 3.0.2 余弦相似度、3.0.4 Sigmoid 与 3.0.5 Softmax/温度。"
        " **本节新增：** 同一批内共享负样本（in-batch negatives）和物品塔向量离线建索引。",
        "用户塔输出 $u=f_\\theta(x_u)\\in\\mathbb R^d$，物品塔输出 $v=g_\\phi(x_i)\\in\\mathbb R^d$。"
        "归一化后 $s(u,i)=u^\\top v$ 就是余弦相似度。张量形状是用户批次 $[B,d]$ 乘物品库 $[N,d]^\\top$，"
        "一次得到 $[B,N]$ 分数。线上只需算一次用户向量，再从 ANN 索引找 Top-K。",
    ),
    "3_2_2_mind": (
        "**通用先修：** 3.0.2 加权汇总、向量模长与张量形状，3.0.5 Softmax。"
        " **本节新增：** 动态路由（squash 压缩长度 + 按投票对数更新 logit）和 Label-aware Attention（候选物品做 query 选择兴趣）。",
        "设用户有 $K$ 个兴趣向量 $V_u=[v_1,\\ldots,v_K]$。候选物品 $e_i$ 的分数取 $\\max_k v_k^\\top e_i$。"
        "先把历史点分成两团、各自平均得到质心；若强行平均成一个点，它会落在两团之间，反而不像任何真实兴趣。",
    ),
    "3_2_3_sasrec": (
        "**通用先修：** 3.0.2 点积注意力与形状，3.0.5 Softmax 温度。"
        " **本节新增：** 位置编码（告诉模型“行为发生在何时”）与 softplus pairwise 损失（只要求正样本打分高于负样本）。",
        "把 item embedding 与位置 embedding 相加得到 $X$。单头注意力为 $\\operatorname{softmax}(QK^\\top/\\sqrt d\\,+M)\\,V$，"
        "因果 mask 把未来位置设为 $-\\infty$。位置 $t$ 的表示只能读取 $i_1,\\ldots,i_t$，再与正/负物品向量做点积并优化 pairwise logistic 损失。",
    ),
    "3_3_1_deepfm": (
        "**通用先修：** 3.0.2 点积、3.0.4 Sigmoid 与 3.0.5 二元交叉熵。"
        " **本节新增：** FM 二阶项的 $O(nd)$ 展开（避免枚举所有特征对），以及 embedding 在 wide/deep 两路共享。",
        "FM 二阶项是 $\\sum_{i<j}\\langle v_i,v_j\\rangle x_i x_j$；展开成 $(\\sum_i v_{i,f}x_i)^2-\\sum_i(v_{i,f}x_i)^2$ 后，"
        "每个隐维度 $f$ 只需两次求和。Deep 分支把同一组 embedding 展平送入 MLP。"
        "最终 logit 为 linear + FM + DNN，再经 Sigmoid 得点击概率；当前样本只有非零 field 参与交互。",
    ),
    "3_3_2_din": (
        "**通用先修：** 3.0.2 加权汇总与逐元素乘积。"
        " **本节新增：** 候选感知注意力（同一用户面对不同候选得到不同兴趣向量），激活单元输入包含 $e_j\\odot e_t$ 与 $e_j-e_t$。",
        "注意力分数 $a_j=g(e_j,e_t,e_j-e_t,e_j\\odot e_t)$，用户兴趣为 $v=\\sum_j a_j e_j$。"
        "候选像问题、历史像资料：先给相关资料更高权重，再做加权汇总；权重是针对当前候选临时计算的。"
        "DIN 的权重不必做 softmax，保留“注意力强弱”而非“概率分配”的语义。",
    ),
    "3_3_3_dien": (
        "**通用先修：** 3.0.3 函数复合，3.0.2 加权汇总与逐元素乘积。"
        " **本节新增：** GRU 门控（重置/更新门控制写入多少历史）、AUGRU（候选注意力缩放更新门）、为中间隐状态加辅助损失。",
        "GRU 是带记忆的递推 $h_t=\\operatorname{GRU}(e_t,h_{t-1})$：重置门决定看多少过去算候选内容，更新门决定写入多少。"
        "辅助损失要求 $h_t$ 更像下一次真实行为、远离负样本。AUGRU 用候选相关权重控制每步写入多少，"
        "因此同样集合的不同排列会得到不同末状态。",
    ),
    "3_4_1_mmoe": (
        "**通用先修：** 3.0.5 Softmax 与 3.0.2 加权汇总。"
        " **本节新增：** 专家混合（MoE：多个子网络被每个任务用自己的 gate 加权）。",
        "第 $k$ 个任务表示为 $z_k=\\sum_e g_{k,e}(x)\\,f_e(x)$，gate 权重经 softmax 后和为 1。"
        "专家像擅长不同题型的老师：各任务听同一组老师，但自行决定每位老师占多少；点击任务可以偏向专家 1，长播任务偏向专家 3。",
    ),
    "3_4_2_ple": (
        "**通用先修：** 3.0.5 Softmax、3.0.6 梯度冲突，以及上节 MMoE。"
        " **本节新增：** CGC 路由约束（任务 gate 只看共享专家 + 自己的任务专家）和多层 progressive separation。",
        "任务 gate 只从共享专家和自己的专属专家中选择；共享 gate 汇总所有专家供下一层公共表示。层层重复后，共享信息逐步提纯，"
        "同时各任务保留不愿共享的私有特征。直觉上像公共课 + 专业课：都听公共课，但不被迫共享全部专业细节。",
    ),
    "4_1_openonerec_practice": (
        "**通用先修：** 3.0.4 条件概率与 3.0.5 Softmax/序列 NLL。"
        " **本节新增：** 自回归分解（链式概率 $\\prod_t P(y_t|y_{<t},x)$）、Semantic ID（RQ-VAE 把物品变成离散 token）、DPO 偏好对齐损失。",
        "自回归分解 $P(y_1,\\ldots,y_T|x)=\\prod_t P(y_t|y_{<t},x)$ 表示：每一步根据上下文和已生成 token 预测下一 token。"
        "Trie 像目录树：前缀 $(1,2)$ 下只允许真实后继 $\\{3,4\\}$，保证不生成非法物品。"
        "DPO 不训练显式 reward，而是让偏好列表 $y^+$ 相对 $y^-$ 的 log-ratio 超过参考模型一个间隔。",
    ),
    "4_2_dlrm_hstu_practice": (
        "**通用先修：** 3.0.2 注意力/逐元素乘积与 3.0.3 SiLU。"
        " **本节新增：** Pointwise Aggregated Attention（跨位置不归一，多个历史可以同时高分）和相对时间偏置 $r_{t,j}$。",
        "给定序列 $[i_1,\\ldots,i_t]$，HSTU 计算 $a_{t,j}=\\operatorname{SiLU}(q_t\\odot k_j+r_{t,j})$，"
        "$h_t=\\sum_{j\\le t} a_{t,j}\\odot v_j$。因果约束 $j\\le t$ 保证不偷看未来；"
        "与 softmax 注意力不同，SiLU 后不需要除以总和，所以多个强相关事件可以同时贡献大权重，不必互相竞争。",
    ),
}

DEEP_SLUGS = list(MATH_BY_HAND.keys())


def patch_math_foundations() -> None:
    path = NB_DIR / "3_0_math_foundations.ipynb"
    nb = nbformat.read(path, as_version=4)
    if any("### 5.1 逐元素运算" in c.source for c in nb.cells if c.cell_type == "markdown"):
        print("3_0_math_foundations already has §5.1, skipping")
        return
    idx = next(i for i, c in enumerate(nb.cells) if c.cell_type == "markdown" and c.source.startswith("## 6. 梯度下降"))
    nb.cells.insert(idx, nbformat.v4.new_markdown_cell(NEW_SUBSECTION.strip() + "\n"))
    nbformat.write(nb, path)
    print(f"inserted §5.1 at index {idx} in 3_0_math_foundations")


def patch_deep_prereqs() -> None:
    for slug in DEEP_SLUGS:
        prereq, body = MATH_BY_HAND[slug]
        path = NB_DIR / f"{slug}.ipynb"
        nb = nbformat.read(path, as_version=4)
        target_idx = next(
            (i for i, c in enumerate(nb.cells) if c.cell_type == "markdown" and c.source.startswith("## Math by Hand")),
            None,
        )
        if target_idx is None:
            print(f"!! {slug}: no Math by Hand cell")
            continue
        nb.cells[target_idx].source = (
            "## Math by Hand\n\n"
            f"{prereq}\n\n"
            f"{body}\n\n"
            "下面用 NumPy/Matplotlib 验证直觉。二维图只是教学投影，工业 embedding 虽有更多维，计算规则相同。\n"
        )
        nbformat.write(nb, path)
        print(f"{slug}: refreshed Math by Hand")


if __name__ == "__main__":
    patch_math_foundations()
    patch_deep_prereqs()
