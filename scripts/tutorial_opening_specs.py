from __future__ import annotations


def build_opening_specs(md, code, notebook):
    """Create a mathematics-and-roadmap opening notebook for every large chapter."""

    chapters = [
        {
            "slug": "4_1_classic_foundations",
            "title": "4.1 导读与数学基础：经典算法",
            "goal": "在进入协同过滤、MF、FM、GBDT+LR 与 word2vec 前，建立邻域、低秩、特征交互、二分类概率和序列嵌入的共同数学地图。",
            "source": "[GroupLens](https://dl.acm.org/doi/10.1145/192844.192905) · [ItemCF](https://dl.acm.org/doi/10.1145/371920.372071) · [MF](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf) · [FM](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf) · [GBDT+LR](https://research.facebook.com/publications/practical-lessons-from-predicting-clicks-on-ads-at-facebook/) · [word2vec](https://arxiv.org/abs/1301.3781)",
            "layout": """## 本章布局与选型地图

| 子章节 | 解决的问题 | 共同数学 | 典型位置 |
|---|---|---|---|
| 4.2 UserCF / ItemCF | 借相似用户或物品的行为推荐 | 矩阵、点积、余弦、加权和 | 召回、相关推荐 |
| 4.3 BiasMF | 用低维坐标压缩稀疏评分矩阵 | 矩阵乘法、内积、均方误差 | 召回、评分预测 |
| 4.4 FM | 学习稀疏 field 的二阶交互 | one-hot、向量内积、LogLoss | CTR 排序 |
| 4.5 GBDT+LR | 用树规则产生特征并预测点击概率 | 条件分支、one-hot、Sigmoid | CTR 排序 |
| 4.6 word2vec / Item2Vec | 把行为序列学成物品向量做召回 | 中心-上下文内积、负采样、Sigmoid | 向量召回 |
| 4.7 总结 | 从实际 JSON 汇总指标 | 指标方向与公平比较 | 选型 |

应用场景并非互斥：ItemCF 常作为可解释兜底通道；MF 是向量召回的最小原型；FM 与 GBDT+LR 是成熟 CTR 基线；word2vec 把序列共现压成可 ANN 的物品向量，衔接后续 DSSM、SASRec。学习顺序建议从矩阵共现开始，再进入可训练表示、概率模型和序列嵌入。""",
            "papers": """## 来源论文解读

- **Resnick et al. (1994)** 把相似用户评分加权落成可运行系统，核心遗产是“从群体行为借信号”。
- **Sarwar et al. (2001)** 把邻域转到更稳定的 item 侧，使相似表可以离线物化。
- **Koren et al. (2009)** 强调只在已观察集合上训练、加入用户/物品偏置并正则化低秩向量。
- **Rendle (2010)** 用共享隐向量估计从未共同出现过的稀疏特征组合。
- **He et al. (2014)** 把树当监督式特征变换，再由 LR 输出可校准概率。
- **Mikolov et al. (2013)** 用中心词预测上下文高效学词向量；推荐系统把行为序列当句子学物品向量（Item2Vec），是向量召回的起点。

这些论文对应两种数据生成机制：CF/MF/word2vec 学 user–item 行为或行为序列；FM/GBDT+LR 学曝光后的点击。两类指标不能直接排名。""",
            "math": r"""## 共同数学：从矩阵到序列

设 $R\in\mathbb R^{|U|\times|I|}$ 是用户—物品矩阵。

1. **邻域：** $RR^\top$ 批量计算用户共现，$R^\top R$ 批量计算物品共现；余弦再消除长度影响。
2. **低秩：** $R\approx PQ^\top$，把每位用户和物品映射到 $d$ 维坐标。$P_u^\top Q_i$ 是两张坐标卡逐项相乘求和。
3. **稀疏交互：** FM 为 one-hot 特征配置向量，使用 $\langle v_i,v_j\rangle x_ix_j$ 表示任意两项交互。
4. **CTR 概率：** LR/GBDT+LR/FM 排序都可输出 logit $z$，再用 $\sigma(z)=1/(1+e^{-z})$ 得到概率，并以 LogLoss 训练。
5. **序列嵌入：** word2vec 把每个用户的正反馈序列当一句话，用中心向量 $v_c$ 与上下文向量 $v_w$ 的内积衡量共现；负采样再随机抽不共现的物品配对，同样套用 $\sigma$ 与 LogLoss 拉正推负，训练后取中心向量做全库 ANN 召回。

先看形状，再看每个数字代表什么，比背公式更重要。""",
            "demo": """import numpy as np, matplotlib.pyplot as plt
R=np.array([[1,1,0,0],[1,0,1,0],[0,1,1,1]],dtype=float)
user_common=R@R.T; item_common=R.T@R
u,s,vt=np.linalg.svd(R,full_matrices=False); rank2=(u[:,:2]*s[:2])@vt[:2]
fig,axes=plt.subplots(1,3,figsize=(11,3.2))
for ax,matrix,title in [(axes[0],R,'behavior R'),(axes[1],item_common,'R.T @ R: ItemCF'),(axes[2],rank2,'rank-2 approximation: MF')]:
    ax.imshow(matrix,cmap='YlGn'); ax.set_title(title)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]): ax.text(col,row,f'{matrix[row,col]:.1f}',ha='center',va='center',fontsize=8)
plt.tight_layout(); plt.show()
z=np.array([-2.,0.,2.]); print({'logit':z.tolist(),'sigmoid':(1/(1+np.exp(-z))).round(3).tolist()})
# word2vec：把序列当一句话，窗口内的(中心,上下文)物品对就是 Skip-gram 的正样本。
seq=[1,2,3,2]; w2v_pairs=[(seq[i],seq[j]) for i in range(len(seq)) for j in range(max(0,i-1),min(len(seq),i+2)) if i!=j]
print({'skip_gram_pairs':w2v_pairs})""",
            "checks": """assert user_common.shape==(3,3)
assert item_common.shape==(4,4)
assert rank2.shape==R.shape
assert (2,3) in w2v_pairs and (3,2) in w2v_pairs
print('PASS：共现、低秩近似、概率变换与序列共现对的基础形状正确。')""",
        },
        {
            "slug": "5_1_retrieval_foundations",
            "title": "5.1 导读与数学基础：向量、多兴趣与序列召回",
            "goal": "在进入 DSSM、MIND 与 SASRec 前，理解向量空间、对比学习、ANN、多兴趣、位置编码、因果注意力和 Top-K 指标。",
            "source": "[DSSM](https://www.microsoft.com/en-us/research/publication/learning-deep-structured-semantic-models-for-web-search-using-clickthrough-data/) · [MIND](https://arxiv.org/abs/1904.08030) · [SASRec](https://arxiv.org/abs/1808.09781) · [BERT4Rec](https://arxiv.org/abs/1904.06690) · [HSTU](https://arxiv.org/abs/2402.17152) · [YouTube DNN](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/)",
            "layout": """## 本章布局与选型地图

| 子章节 | 用户表示 | 检索次数 | 优势 | 主要代价 |
|---|---:|---:|---|---|
| 5.2 DSSM | 1 个向量 | 1 | 简单、低延迟、ANN 友好 | 多兴趣被平均 |
| 5.3 MIND | K 个兴趣向量 | K | 覆盖多意图与长尾 | 路由、合并、去重成本 |
| 5.4 SASRec | 1 个时序向量 | 1 | 捕捉近期转移与顺序依赖 | Attention 成本与序列延迟 |
| 5.5 总结 | 读取三个实验结果 | — | 同口径看 Recall 与成本 | smoke 不是 benchmark |

DSSM 适合通用亿级目录；MIND 适合同时存在多个意图；SASRec 适合下一行为受先后顺序和近期行为影响明显的场景。三者最终都能产生可用于全库检索的用户表示。""",
            "papers": """## 来源论文解读

- **Huang et al. (2013)** 用搜索点击数据让 query/document 表示进入同一语义空间，并以 NDCG 验证。原文没有评测推荐或 ANN；后来推荐系统才把这一结构迁移为 user/item 双塔。
- **YouTube DNN (2016)** 明确区分 candidate generation 与 ranking，并讨论 sampled softmax 和训练/服务分布。
- **MIND (2019)** 用动态路由抽取多个兴趣胶囊，训练时根据目标选择相关兴趣。
- **SASRec (2018)** 用因果自注意力在每个位置预测下一物品。服务时取最后有效位置的表示，与全库 item embedding 点积，因此主要落在序列召回，也可把序列表征提供给排序。
- **BERT4Rec (2019)** 改用双向上下文与 masked-item 训练，更适合离线表征和补全式预训练，但与在线 next-item 的信息边界不同。
- **HSTU (2024)** 面向高基数、非平稳、超长行为流重新设计注意力和系统实现；它在 4.3 继续展开，不与 SASRec 的轻量实验混为同一规模结论。

论文中的离线提升不等于任意目录上的线上增益；负采样、索引新鲜度和候选合并常比换一层 MLP 更重要。""",
            "math": r"""## 共同数学：先修清单

| 知识点 | 最低需要会什么 | 本章在哪里使用 |
|---|---|---|
| 向量与长度 | 会算 $\sqrt{x_1^2+\cdots+x_d^2}$ | 归一化、余弦、squash |
| 点积与矩阵乘法 | 点积是对应坐标相乘再相加；矩阵乘法是批量点积 | 双塔全库分数、QKᵀ |
| 指数、对数 | 指数放大差异；对数把连乘变连加 | Softmax、负对数损失 |
| 条件概率 | “已知用户后，目标物品出现的概率” | DSSM/MIND sampled softmax |
| 加权和 | 权重越大，该向量对结果贡献越大 | MIND 路由、SASRec attention |
| 序列下标 | 第 t 个位置只能用 1…t | SASRec 因果 mask |
| Top-K 指标 | 区分“是否找回”和“排得多靠前” | Recall/HitRate/NDCG |

不需要先修完整微积分课程。阅读顺序固定为：先看输入形状，再算分数，再把分数变成概率/权重，最后理解损失为什么会推动正确答案上升。

### 1. 点积、余弦与批量计算

点积 $u^\top v=\sum_ru_rv_r$ 是“对应坐标相乘再相加”。归一化向量的内积等于余弦相似度：

$$\cos(u,v)=\frac{u^\top v}{\|u\|_2\|v\|_2}.$$

若一批用户是 $U\in\mathbb R^{B\times d}$，物品库是 $V\in\mathbb R^{N\times d}$，那么 $UV^\top\in\mathbb R^{B\times N}$ 的每个格子就是一对 user-item 分数。

### 2. 从分数到概率：Softmax、温度与负对数

对一个正物品和若干负物品，可用

$$P(i^+|u)=\frac{\exp(s(u,i^+)/\tau)}{\sum_j\exp(s(u,i_j)/\tau)}$$

训练，其中温度 $\tau$ 控制分布尖锐程度。$\tau$ 小会放大分数差，$\tau$ 大会让概率更平均。损失 $-\log P(i^+|u)$ 会重罚“正确物品概率很低”的情况。in-batch negative 把同一批次其他正物品当负样本，效率高，但其中可能有用户其实喜欢的假负样本。

### 3. 多兴趣：软分组与最大匹配

召回阶段常看 $\mathrm{Recall@K}=|TopK\cap Relevant|/|Relevant|$。DSSM 只有一个 $u$；MIND 使用 $K$ 个 $u_k$，候选分数为 $\max_k u_k^\top v_i$。

MIND 的路由权重对每条行为经 softmax 后和为 1，所以它是“可以同时属于多组、但总票数固定”的软分组。squash 函数保留兴趣方向并把长度压到 0～1。

### 4. 自注意力：问题、标签、内容与时间边界

SASRec 先令 $x_t=e(i_t)+p_t$，再计算

$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V.$$

Q、K、V 可读作“问题、标签、内容”。$QK^\top$ 先得到每个位置对每段历史的相关分；除以 $\sqrt{d_k}$ 防止维度增大后分数过大；softmax 再把一行变成和为 1 的阅读权重。因果遮罩在未来位置填入 $-\infty$，使预测位置只能读取过去。最后有效位置向量就是随顺序变化的用户召回向量。

### 5. 指标边界

单真值时 HitRate@K 与 Recall@K 数值相同；NDCG@K 还会奖励更靠前的位置。论文若使用不同数据切分、候选集或负样本，即使指标名称相同也不能直接排名。""",
            "demo": """import numpy as np, matplotlib.pyplot as plt
scores=np.array([2.4,1.2,.3,-.2]); temperatures=[1.0,.3]
fig,axes=plt.subplots(1,3,figsize=(12,3.2))
for ax,tau in zip(axes,temperatures):
    probability=np.exp(scores/tau); probability/=probability.sum()
    ax.bar(range(len(scores)),probability,color='#7ca832'); ax.set(title=f'softmax temperature={tau}',xlabel='1 positive + 3 negatives',ylabel='probability',ylim=(0,1))
causal=np.tril(np.ones((5,5))); axes[2].imshow(causal,cmap='YlGn'); axes[2].set(title='SASRec causal visibility',xlabel='history',ylabel='prediction')
plt.tight_layout(); plt.show()
single=np.array([.5,.5]); interests=np.array([[1.,0.],[0.,1.]]); candidates=np.array([[1.,0.],[0.,1.],[-1.,0.]])
print({'single_scores':(candidates@single).tolist(),'multi_interest_scores':(interests@candidates.T).max(0).tolist()})""",
            "checks": """def recall_at_k(ranked,relevant,k):
    return len(set(ranked[:k])&set(relevant))/len(relevant)
assert recall_at_k([2,5,1,7],[1,2,8],3)==2/3
print('PASS：Softmax 概率、单/多兴趣分数与 Recall 示例正确。')""",
        },
        {
            "slug": "6_1_ranking_foundations",
            "title": "6.1 导读与数学基础：CTR 排序",
            "goal": "在进入 DeepFM、DIN 与 DIEN 前，理解 logit、概率、交叉熵、AUC、特征交互、注意力和时序状态。",
            "source": "[DeepFM](https://arxiv.org/abs/1703.04247) · [DIN](https://arxiv.org/abs/1706.06978) · [DIEN](https://arxiv.org/abs/1809.03672)",
            "layout": """## 本章布局与选型地图

| 子章节 | 主要信号 | 关键机制 | 适用场景 |
|---|---|---|---|
| 6.2 DeepFM | 静态 sparse/dense fields | FM 二阶 + DNN 高阶 | 通用 CTR 基线 |
| 6.3 DIN | 候选 + 行为集合 | target-aware attention | 强历史、候选相关性 |
| 6.4 DIEN | 候选 + 有序行为 | GRU、辅助损失、AUGRU | 兴趣变化明显 |
| 6.5 总结 | 各实验 JSON | AUC、LogLoss、baseline | 效果—成本选型 |

由 DeepFM 到 DIN/DIEN，不是简单增加网络深度，而是逐步加入“当前候选”和“行为顺序”这两类结构先验。""",
            "papers": """## 来源论文解读

- **DeepFM (2017)** 共享 embedding 联合训练 FM 与 DNN，减少 Wide&Deep 宽侧的人工交叉。
- **DIN (2018)** 让同一用户面对不同候选时激活不同历史，固定用户向量变成候选感知表示。
- **DIEN (2019)** 用辅助下一行为监督兴趣抽取，再用 AUGRU 表达兴趣演化。

三者都在曝光样本上学习点击概率。若没有曝光日志，不能把“未点击”随意解释成真正负反馈。""",
            "math": r"""## 数据选择：为什么使用 KuaiRand

本章使用 KuaiRand-Pure 的真实短视频曝光日志，而不是 MovieLens 评分。每条记录同时给出候选视频、场景、时间、是否点击和观看时长，因此 `is_click=0` 是一次真实未点击曝光，而不是从目录中人工抽出的负样本。这与 DeepFM、DIN、DIEN 的 CTR/序列排序问题更一致。

## 共同数学：概率、交叉熵、注意力与状态
排序模型先输出 logit $z$，Sigmoid 得 $p=\sigma(z)$。单样本 LogLoss 为

$$L=-[y\log p+(1-y)\log(1-p)]$$

AUC 可理解为随机取一对正负样本，正样本分数更高的概率。它看顺序，不保证概率校准。

DeepFM 关注 field 间交互；DIN 计算 $v=\sum_j a(e_j,e_t)e_j$；DIEN 再递推 $h_t=\mathrm{GRU}(e_t,h_{t-1})$。前者是加权和，后者让顺序改变最终状态。""",
            "demo": """import numpy as np, matplotlib.pyplot as plt
z=np.linspace(-6,6,300); p=1/(1+np.exp(-z))
grid=np.linspace(.01,.99,300)
fig,axes=plt.subplots(1,2,figsize=(10,3.4))
axes[0].plot(z,p,color='#4f8f00',lw=2); axes[0].set(title='Sigmoid: logit to click probability',xlabel='logit',ylabel='p(click)'); axes[0].grid(alpha=.2)
axes[1].plot(grid,-np.log(grid),label='actual click y=1'); axes[1].plot(grid,-np.log(1-grid),label='actual no-click y=0')
axes[1].set(title='LogLoss',xlabel='predicted probability',ylabel='loss'); axes[1].legend(); axes[1].grid(alpha=.2)
plt.tight_layout(); plt.show()
history=np.array([[1.,0.],[.8,.2],[0.,1.]]); target=np.array([1.,0.]); raw=history@target; weight=np.exp(raw)/np.exp(raw).sum()
print({'attention_weights':weight.round(3).tolist(),'weighted_interest':(weight@history).round(3).tolist()})""",
            "checks": """positive=np.array([.9,.7,.6]); negative=np.array([.8,.4,.2])
auc=(positive[:,None]>negative[None,:]).mean()
assert 0<=auc<=1 and np.isclose(weight.sum(),1)
print({'AUC_as_pairwise_win_rate':round(float(auc),3),'attention_sum':round(float(weight.sum()),3)})""",
        },
        {
            "slug": "7_1_multitask_foundations",
            "title": "7.1 导读与数学基础：多目标学习",
            "goal": "在进入 MMoE 与 PLE 前，理解多任务损失、共享与专属参数、Softmax gate、梯度冲突和负迁移。",
            "source": "[MMoE](https://dl.acm.org/doi/10.1145/3219819.3220007) · [PLE](https://dl.acm.org/doi/10.1145/3383313.3412236)",
            "layout": """## 本章布局与选型地图

| 子章节 | 共享方式 | 任务专属结构 | 主要风险 |
|---|---|---|---|
| 7.2 MMoE | 所有任务共享专家 | 每任务 gate 与 tower | 专家塌缩、梯度冲突 |
| 7.3 PLE | 逐层共享专家 | 每层保留任务专属专家 | 参数与调参成本 |
| 7.4 总结 | 相同 KuaiRand 曝光切片 | 逐任务指标 | 平均值掩盖跷跷板 |

MMoE 适合相关任务的灵活共享；PLE 适合确有负迁移且数据足够支撑更复杂结构的场景。上线仍需明确最终业务效用函数。""",
            "papers": """## 来源论文解读

- **MMoE (2018)** 用任务独立 gate 混合同一组专家，在任务相关性变化时比 hard sharing 更灵活。
- **PLE (2020)** 在每层区分共享和任务专属专家，逐步提取可共享信息，针对负迁移和跷跷板问题。

二者都不能自动解决标签空间不一致。例如 CVR 只在点击样本上观测，必须配合样本掩码、ESMM 类概率分解或因果校正。""",
            "math": r"""## 共同数学：加权损失、Gate 与梯度夹角

多任务总损失常写成 $L=\sum_t\lambda_tL_t$。权重 $\lambda_t$ 不只是数学常数，它决定优化器更重视哪个任务。

MMoE 的任务表示为 $z_t=\sum_e g_{t,e}(x)f_e(x)$，Softmax gate 满足 $\sum_e g_{t,e}=1$。

两个任务梯度的余弦 $\cos(g_1,g_2)=g_1^\top g_2/(\|g_1\|\|g_2\|)$：大于 0 表示大致同向，小于 0 表示一步更新可能帮助一个任务却伤害另一个任务。""",
            "demo": """import numpy as np, matplotlib.pyplot as plt
gate_logits=np.array([[2.,1.,0.],[.2,.8,1.8]])
gate=np.exp(gate_logits); gate/=gate.sum(1,keepdims=True)
fig,ax=plt.subplots(figsize=(6,3.2)); left=np.zeros(2)
for expert in range(3):
    ax.barh(['click','long view'],gate[:,expert],left=left,label=f'expert {expert+1}'); left+=gate[:,expert]
ax.set(xlim=(0,1),title='Task-specific Softmax gates'); ax.legend(ncol=3,loc='lower center'); plt.show()
grad_click=np.array([1.,.2]); grad_convert=np.array([-.5,1.])
cosine=grad_click@grad_convert/(np.linalg.norm(grad_click)*np.linalg.norm(grad_convert))
print({'gate_sums':gate.sum(1).round(3).tolist(),'gradient_cosine':round(float(cosine),3)})""",
            "checks": """assert np.allclose(gate.sum(1),1)
assert -1<=cosine<=1
print('PASS：任务 gate 是合法权重，梯度夹角位于合法范围。')""",
        },
        {
            "slug": "8_1_generative_foundations",
            "title": "8.1 导读与数学基础：生成式推荐",
            "goal": "在进入 OpenOneRec 与 HSTU 前，理解 token、Semantic ID、自回归概率、交叉熵、因果 mask、约束解码与列表指标。",
            "source": "[TIGER](https://arxiv.org/abs/2305.05065) · [OpenOneRec](https://github.com/Kuaishou-OneRec/OpenOneRec) · [HSTU](https://arxiv.org/abs/2402.17152)",
            "layout": """## 本章布局与选型地图

| 子章节 | 输出对象 | 核心数学 | 系统难点 |
|---|---|---|---|
| 8.2 OpenOneRec | Semantic ID / 推荐列表 | 自回归、trie、偏好优化 | 合法性、重复、解码 P99 |
| 8.3 DLRM HSTU | 下一 item / 行为序列 | 因果序列建模、交叉熵 | 长序列吞吐与状态缓存 |
| 8.4 总结 | 两个实验结果 | 指标方向与 ROI | 不同任务不能只看单一分数 |

生成式召回生成 item 标识；生成式排序输出分数、标签或排列；召排融合直接生成列表/session。三者的输入、输出和成本并不相同。""",
            "papers": """## 来源论文与工业路径

- **TIGER (2023)** 用 RQ-VAE 将 item 表示成多级 Semantic ID，再把全库召回转为受约束生成。
- **HSTU (2024)** 针对高基数、非平稳行为流设计序列模型与系统协同。
- **OpenOneRec** 公开列表生成、奖励模型、偏好优化和 RecIF-Bench 流程。

生成式方案不是把 LLM API 接到推荐器上。目录合法性、增量更新、重复控制、缓存、吞吐和 GPU ROI 都是模型定义的一部分。""",
            "math": r"""## 共同数学：自回归概率、因果性与 NDCG

序列概率分解为

$$P(y_1,\ldots,y_T|x)=\prod_{t=1}^TP(y_t|y_{<t},x)$$

训练用 teacher forcing：第 $t$ 步输入真实前缀，最小化下一 token 的交叉熵。推理时输入自己已生成的前缀，因此错误可能累积。

因果 mask 让当前位置只能看过去；目录 trie 只保留合法后继。列表质量常用
$\mathrm{DCG@K}=\sum_{r=1}^K rel_r/\log_2(r+1)$，再除以理想 DCG 得 NDCG。越相关的 item 越靠前，分数越高。""",
            "demo": """import numpy as np, matplotlib.pyplot as plt
length=6; causal=np.tril(np.ones((length,length)))
fig,axes=plt.subplots(1,2,figsize=(9,3.3))
axes[0].imshow(causal,cmap='YlGn',vmin=0,vmax=1); axes[0].set(title='causal mask',xlabel='visible key',ylabel='prediction step')
ranks=np.arange(1,7); discount=1/np.log2(ranks+1); axes[1].bar(ranks,discount,color='#7ca832'); axes[1].set(title='NDCG position discount',xlabel='rank',ylabel='weight')
plt.tight_layout(); plt.show()
catalog={(1,2,3),(1,2,4),(2,1,5)}; prefix=(1,2)
allowed=sorted({code[len(prefix)] for code in catalog if code[:len(prefix)]==prefix})
print({'prefix':prefix,'allowed_next_tokens':allowed})""",
            "checks": """relevant={3,4}; ranked=[3,8,4]
dcg=sum((item in relevant)/np.log2(rank+2) for rank,item in enumerate(ranked))
ideal=sum(1/np.log2(rank+2) for rank in range(len(relevant)))
ndcg=dcg/ideal
assert allowed==[3,4] and 0<=ndcg<=1
print({'NDCG@3':round(float(ndcg),3),'trie_constraint':'PASS'})""",
        },
    ]

    specs = {}
    for chapter in chapters:
        cells = [
            md("## 如何使用本导读\n\n先阅读布局和论文问题，再运行共同数学演示。完成 Checks 后进入独立算法 Notebook；各算法会重新给出本模型的公式和更小的 Python 演示，不要求记住本页所有公式。"),
            md(chapter["layout"]),
            md(chapter["papers"]),
            md(chapter["math"]),
            code(chapter["demo"]),
            md("## 学习顺序\n\n1. 说清业务阶段和输入输出；2. 手算共享数学；3. 进入每个独立算法；4. 执行训练与推理；5. 最后打开章节总结读取实际结果。"),
            md("## Checks"),
            code(chapter["checks"]),
            md("## Next Steps\n\n从左侧 Notebook 导航进入本章第一个算法。遇到公式时依次检查：符号代表什么、数组形状是什么、用小数字怎么算、代码输出是否符合直觉。"),
        ]
        specs[chapter["slug"]] = notebook(chapter["title"], chapter["goal"], chapter["source"], cells)
    return specs
