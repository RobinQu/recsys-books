#!/usr/bin/env python3
"""Deepen the 公式到代码 sections of the ten deep-learning notebooks and replace
ASCII flow diagrams with SVG figures.

For every notebook this rewrites the tail of the ``## Model Structure & Formula
Walkthrough`` cell (everything from ``### 公式到代码`` onward) with a numbered
walkthrough of what one forward pass actually computes -- written for readers
who have never seen the model or the Torch-RecHub framework. Idempotent: the
section is fully regenerated from this file on each run.
"""
from __future__ import annotations

import re
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

CLOSING = "阅读源码时按“张量形状 → 前向计算 → score → loss → metric”五步追踪，不需要一次读完整个工程文件。"

# slug -> (optional svg figure replacing the ```text block, new 公式到代码 body)
SECTIONS: dict[str, tuple[str | None, str]] = {
    "3_2_1_dssm": (
        "![DSSM 双塔数据流：两塔独立编码，最后才计算相似度](/static/diagrams/flow-dssm.svg)",
        """### 公式到代码：一次前向传播发生了什么

以一条训练样本（用户 $u$ 与其点击过的正物品 $i^+$）为例，前向传播按下面的顺序进行：

1. **查表得到稠密向量**：用户侧和物品侧的特征都是稀疏 id（用户 id、物品 id、类别等）。每个 id 在各自的 embedding 表中查出一行，拼接后得到输入向量 $x_u$ 与 $x_i$。
2. **两塔各自压缩**：$x_u$ 经过用户塔的多层全连接（每层 = 线性变换 $Wx+b$ 加 tanh 非线性），维度逐层降到 $d$；物品塔以同样方式把 $x_i$ 压成 $v$。两座塔参数互不共享。
3. **归一化后点积**：$u,v$ 分别除以自身长度，再做点积，即余弦相似度 $s\\in[-1,1]$。
4. **同批负样本 + softmax**：一个 batch 中其他用户的正物品充当 $u$ 的负样本。把所有候选的 $s/\\tau$ 做 softmax，正例概率越大，交叉熵损失 $L=-\\log P(i^+|u)$ 越小。
5. **反向传播**：损失对两塔参数与 embedding 表同时求梯度并更新。

推理时不再需要损失：物品塔离线对全库计算 $v$ 并建立 ANN 索引；线上只运行用户塔一次，再查询最近邻。Notebook 中 `run_dssm` 用 Torch-RecHub 的 `DSSM` 模型类承载第 1–4 步（它内部就是“embedding 表 + 双塔 MLP”），训练循环、评估与索引导出由本章节代码完成。""",
    ),
    "3_2_2_mind": (
        "![MIND 多兴趣数据流：动态路由把行为序列聚成 K 个兴趣向量](/static/diagrams/flow-mind.svg)",
        """### 公式到代码：一次前向传播发生了什么

1. **行为向量化**：用户最近 $L$ 次行为中的每个物品查 embedding 表并做 pooling，得到历史矩阵 $H\\in\\mathbb R^{L\\times d}$。
2. **动态路由（迭代约 3 轮）**：路由 logit $b_{jk}$ 初始为 0。每一轮：对 $b$ 按胶囊维做 softmax 得到权重 $c_{jk}$；加权求和 $s_k=\\sum_j c_{jk}h_j$；squash 压缩得到胶囊 $v_k$；再用 $b_{jk}\\leftarrow b_{jk}+h_j^\\top v_k$ 更新投票。相似行为逐轮聚到同一个胶囊。
3. **得到 K 个兴趣**：路由结束后 $V_u=[v_1,\\ldots,v_K]$，每个胶囊代表一类兴趣，而不是把用户压成一个点。
4. **训练时的兴趣选择**：用目标物品 embedding $e_i$ 对 K 个兴趣做 label-aware attention，加权成一个用户向量，再与 $e_i$ 计算 sampled softmax 损失。
5. **推理时的多路召回**：线上没有目标物品，因此 K 个兴趣向量分别做 ANN 检索，合并去重后得到候选集。

`run_mind` 中 Torch-RecHub 的 `MIND`（内部为 `CapsuleNetwork`）实现第 2–4 步；序列截断、mask 与评估在本章节代码中。""",
    ),
    "3_2_3_sasrec": (
        None,
        """### 公式到代码：一次前向传播发生了什么

1. **序列嵌入**：位置 $t$ 的输入是物品 embedding 加位置 embedding，$x_t=e_{i_t}+p_t$；位置向量让模型区分“第 1 次点击”与“第 10 次点击”。
2. **因果自注意力**：$X$ 经三组线性投影得到 $Q,K,V$。注意力分数 $QK^\\top/\\sqrt d$ 加上因果遮罩（$j>t$ 处为 $-\\infty$），softmax 后未来位置权重为 0，再加权汇总 $V$。
3. **残差 + 前馈**：注意力输出与输入做残差连接、LayerNorm，再经过逐位置的两层前馈网络；整个 block 可堆叠多次。
4. **逐位置预测**：位置 $t$ 的输出 $h_t$ 与“下一个真实物品” $e^+$ 和随机负物品 $e^-$ 分别点积，softplus 损失要求 $h_t^\\top e^+ > h_t^\\top e^-$。所有位置并行训练。
5. **推理**：只取最后有效位置的 $h_T$，与全库物品 embedding 点积，Top-K 即召回结果。

`run_sasrec` 用 Torch-RecHub 的 `SASRec` 实现第 1–4 步；序列窗口生成、负采样与指标计算在本章节代码中。""",
    ),
    "3_3_1_deepfm": (
        None,
        """### 公式到代码：一次前向传播发生了什么

1. **共享 embedding**：每个 field（用户 id、物品 id、类别、时段……）的稀疏取值在同一张 embedding 表中查向量 $v_i$；FM 分支与 DNN 分支共用这张表。
2. **一阶项**：每个非零特征乘以一个标量权重再求和，$\\sum_i w_ix_i$，相当于逻辑回归的记忆能力。
3. **FM 二阶项**：所有非零 field 两两计算 $\\langle v_i,v_j\\rangle$；用平方展开技巧把 $O(n^2d)$ 降为 $O(nd)$。
4. **DNN 高阶项**：所有 $v_i$ 拼接后经过 2–3 层全连接 + ReLU，学习三阶以上的组合模式。
5. **合并输出**：三路 logit 相加得 $z$，sigmoid 得点击概率 $p=\\sigma(z)$，用二元交叉熵训练，梯度同时流回 embedding 表与两个分支。

`run_deepfm` 用 Torch-RecHub 的 `DeepFM` 实现第 1–5 步；field 构造与标签生成在本章节代码中。""",
    ),
    "3_3_2_din": (
        None,
        """### 公式到代码：一次前向传播发生了什么

1. **特征嵌入**：候选广告、历史行为序列、用户与上下文特征分别查 embedding 表。
2. **候选感知打分**：对每条历史 $e_j$，与候选 $e_a$ 拼成 $[e_j, e_a, e_j-e_a, e_j\\odot e_a]$，送入一个小型 MLP（激活单元）得到相关分 $a_j$。差向量与逐元素积让网络容易学到“相似”与“差异”模式。
3. **加权求和兴趣**：$v_U=\\sum_j a_j e_j$。DIN 故意不做 softmax 归一——权重的大小本身携带“兴趣强度”信息；同一用户面对不同候选得到不同 $v_U$。
4. **CTR 预测**：$v_U$ 与候选、用户、上下文 embedding 拼接，过 MLP + sigmoid 得到点击概率，二元交叉熵训练。

`run_din` 用 Torch-RecHub 的 `DIN` 实现第 1–4 步；历史截断、mask 与“只用当前曝光之前的行为”的防泄漏逻辑在本章节代码中。""",
    ),
    "3_3_3_dien": (
        None,
        """### 公式到代码：一次前向传播发生了什么

1. **兴趣抽取（GRU）**：行为 embedding 按时间顺序送入 GRU，每步用重置门/更新门决定保留多少过去，得到隐状态序列 $h_1,\\ldots,h_T$。
2. **辅助损失**：每个 $h_t$ 都要能区分“下一步真实行为”与随机负行为（一个小型二分类），让中间状态携带兴趣信息，而不只依赖最终 CTR 信号。
3. **兴趣演化（AUGRU）**：候选广告与每个 $h_t$ 算注意力分 $a_t$；AUGRU 把 GRU 的更新门乘以 $a_t$——与候选相关的历史大步写入，无关历史几乎不改状态。
4. **CTR 预测**：演化后的最终状态与候选、用户、上下文拼接，过 MLP + sigmoid；总损失 = 主任务 BCE + $\\alpha\\times$ 辅助 BCE。

`run_dien` 用 Torch-RecHub 的 `DIEN` 实现第 1–4 步；正/负历史序列构造与 mask 在本章节代码中。""",
    ),
    "3_4_1_mmoe": (
        None,
        """### 公式到代码：一次前向传播发生了什么

1. **共享专家**：输入 $x$ 同时送入 $E$ 个独立的前馈网络（专家），各输出一个 $d$ 维向量 $f_e(x)$；专家之间参数不共享。
2. **任务门控**：每个任务 $k$ 有自己的 gate——一个线性层 + softmax，输出 $E$ 个和为 1 的权重 $g_k(x)$。
3. **加权混合**：任务表示 $z_k=\\sum_e g_{k,e}(x)f_e(x)$。同一条样本上，点击任务可以主要听专家 1，完播任务主要听专家 3。
4. **任务塔与损失**：$z_k$ 进入各任务的小 MLP（任务塔）+ sigmoid；总损失是各任务 BCE 的加权和 $\\sum_k\\lambda_k L_k$，梯度同时更新专家、gate 与任务塔。

`run_mmoe` 用 Torch-RecHub 的 `MMOE` 实现第 1–4 步；同一条曝光生成 click / long-view 两个真实标签的逻辑在本章节代码中。""",
    ),
    "3_4_2_ple": (
        None,
        """### 公式到代码：一次前向传播发生了什么

1. **两类专家**：每层包含共享专家（所有任务可见）与任务专属专家（只有本任务可见），各自是独立前馈网络。
2. **受限门控（CGC）**：任务 $k$ 的 gate 只对“共享专家 + 自己的专家”做 softmax 加权，看不到其他任务的专属专家；共享 gate 则读取全部专家，产出下一层的公共表示。
3. **逐层提纯**：多层堆叠后，公共信息与任务特有信息被逐步分离（progressive separation），缓解 MMoE 中所有任务抢同一组专家的问题。
4. **任务塔与损失**：与 MMoE 相同——每任务 tower + sigmoid + 加权 BCE。

`run_ple` 用 Torch-RecHub 的 `PLE` 实现第 1–4 步；数据与标签与 MMoE 章节完全一致，便于对照结构差异。""",
    ),
    "4_1_openonerec_practice": (
        None,
        """### 公式到代码：一次前向传播发生了什么

1. **物品 token 化**：每个物品的 embedding 经残差 K-Means 量化成多级 Semantic ID $(c_1,c_2,c_3)$，物品库因此变成一棵前缀树（trie）。
2. **编码器**：用户行为序列的物品 token 经 fully-visible self-attention + 前馈，得到上下文表示 $H$。
3. **解码器自回归生成**：解码器堆叠 causal self-attention（只看已生成前缀）、cross-attention（读取 $H$）与 MoE 前馈层；逐步预测 session 中下一个物品的每一级 token，训练用 teacher forcing + 交叉熵。
4. **约束解码**：推理时 beam search 沿 trie 扩展，非法前缀直接剪枝，保证生成的 Semantic ID 对应真实物品。
5. **偏好对齐（IPA）**：用 reward 模型给 beam 候选列表打分，最高分作 chosen、低分作 rejected，DPO 损失让模型拉开两者差距；迭代多轮自我提升。

`run_openonerec` 在教程规模下实现第 1–4 步的接口契约；官方完整训练配置与 IPA 流程在 OpenOneRec 框架中。""",
    ),
    "4_2_dlrm_hstu_practice": (
        None,
        """### 公式到代码：一次前向传播发生了什么

1. **统一序列化**：item、动作类型、位置/时间信息各自嵌入后相加，得到事件序列 $X$。
2. **逐点投影**：每个位置经同一个线性层 + SiLU，拆出门控 $U$、query $Q$、key $K$、value $V$ 四组向量。
3. **点积聚合注意力**：$a_{t,j}=\\mathrm{SiLU}(q_t\\odot k_j+r_{t,j})$，其中 $r_{t,j}$ 是相对时间偏置；不做 softmax 归一，多个强相关历史可以同时保留大权重。$h_t=\\sum_{j\\le t}a_{t,j}\\odot v_j$ 完成因果范围内的汇总。
4. **门控变换**：$\\mathrm{Norm}(A(X)V(X))\\odot U(X)$ 后经 $f_2$ 与残差连接输出；block 堆叠多层。
5. **生成式训练**：每个位置预测下一物品，sampled softmax / 交叉熵；推理取最后位置做 Top-K。

`run_hstu` 用 Torch-RecHub 的教学实现验证第 1–5 步的张量契约；Meta generative-recommenders 仓库提供完整工业 HSTU。""",
    ),
}

TEXT_BLOCK_RE = re.compile(r"```text\n.*?```\n?", re.DOTALL)
FORMULA_SECTION_RE = re.compile(r"### 公式到代码.*\Z", re.DOTALL)


def patch(slug: str, svg: str | None, body: str) -> str:
    path = NB_DIR / f"{slug}.ipynb"
    nb = nbformat.read(path, as_version=4)
    for cell in nb.cells:
        if cell.cell_type != "markdown" or not cell.source.startswith("## Model Structure & Formula Walkthrough"):
            continue
        if svg is not None:
            if svg not in cell.source:
                cell.source, n = TEXT_BLOCK_RE.subn(svg + "\n", cell.source, count=1)
                if n == 0:
                    return f"!! {slug}: no ```text block to replace"
        replacement = body.strip() + "\n\n" + CLOSING + "\n"
        cell.source = FORMULA_SECTION_RE.sub(lambda _m: replacement, cell.source, count=1)
        nbformat.write(nb, path)
        return f"patched: {slug}"
    return f"!! {slug}: no Model Structure cell"


def main() -> None:
    for slug, (svg, body) in SECTIONS.items():
        print(patch(slug, svg, body))


if __name__ == "__main__":
    main()
