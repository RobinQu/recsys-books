from __future__ import annotations

import json
from pathlib import Path


PROTOCOLS = json.loads((Path(__file__).resolve().parents[1] / "config" / "reproduction_protocols.json").read_text(encoding="utf-8"))

ROOT = Path(__file__).resolve().parents[1]
FIGURES_CFG = json.loads((ROOT / "config" / "paper_figures.json").read_text(encoding="utf-8"))
MODEL_LAYERS_CFG = json.loads((ROOT / "config" / "model_layers.json").read_text(encoding="utf-8"))


# Generic mathematics belongs to the progressive 3.0 curriculum.  Algorithm notebooks
# link back to those lessons and only derive the mathematics introduced by the
# paper itself.  Keep these URLs in sync with the anchors documented in
# .slim/deepwork/math-curriculum.md.
DEEP_MATH_PREREQUISITES = {
    "3_2_1_dssm": {
        "generic": [
            ("隐式反馈、正例与未知项", "/notebooks/3_0_1_data_ml_basics#implicit-feedback"),
            ("向量、点积与余弦", "/notebooks/3_0_2_linear_algebra#elementwise-dot"),
            ("Softmax、温度与归一化", "/notebooks/3_0_5_information_theory#softmax-temperature"),
            ("多类交叉熵", "/notebooks/3_0_5_information_theory#cross-entropy-kl"),
        ],
        "paper": "两侧独立编码、点击对中的负文档采样，以及把余弦分数变成条件概率；ANN 是推荐迁移后的工程接口，不是 DSSM 原论文实验。",
    },
    "3_2_2_mind": {
        "generic": [
            ("隐式反馈与未观察物品", "/notebooks/3_0_1_data_ml_basics#implicit-feedback"),
            ("张量形状与轴", "/notebooks/3_0_2_linear_algebra#tensors-shapes"),
            ("向量模长、点积与加权和", "/notebooks/3_0_2_linear_algebra#elementwise-dot"),
            ("Softmax 与温度", "/notebooks/3_0_5_information_theory#softmax-temperature"),
        ],
        "paper": "胶囊动态路由（squash 与投票更新）、归一化 Label-aware Attention，以及候选采样下的 sampled-softmax。",
    },
    "3_2_3_sasrec": {
        "generic": [
            ("序列、会话与时间切分", "/notebooks/3_0_1_data_ml_basics#split-leakage"),
            ("矩阵乘法、embedding 与加权和", "/notebooks/3_0_2_linear_algebra#matmul-embedding"),
            ("Q/K/V 形状", "/notebooks/3_0_2_linear_algebra#low-rank-attention"),
            ("Softmax 与温度", "/notebooks/3_0_5_information_theory#softmax-temperature"),
        ],
        "paper": "可学习位置编码、只许看过去的因果遮罩，以及逐位置正负样本的 softplus pairwise 损失。",
    },
    "3_3_1_deepfm": {
        "generic": [
            ("样本、特征与点击标签", "/notebooks/3_0_1_data_ml_basics#observation-label"),
            ("向量点积", "/notebooks/3_0_2_linear_algebra#elementwise-dot"),
            ("函数复合与激活", "/notebooks/3_0_3_calculus#functions"),
            ("二元交叉熵", "/notebooks/3_0_5_information_theory#cross-entropy-kl"),
        ],
        "paper": "FM 二阶项从枚举特征对化简到 O(nd)，以及 FM 与 DNN 共享同一组 embedding。",
    },
    "3_3_2_din": {
        "generic": [
            ("曝光、点击与未知项", "/notebooks/3_0_1_data_ml_basics#implicit-feedback"),
            ("张量形状与轴", "/notebooks/3_0_2_linear_algebra#tensors-shapes"),
            ("逐元素乘积、点积与加权和", "/notebooks/3_0_2_linear_algebra#elementwise-dot"),
            ("正则化与过拟合", "/notebooks/3_0_6_optimization#regularization"),
        ],
        "paper": "候选感知的 Local Activation Unit、原论文不归一化的兴趣汇总、Dice 激活与 mini-batch aware 正则化。",
    },
    "3_3_3_dien": {
        "generic": [
            ("序列与时间泄漏", "/notebooks/3_0_1_data_ml_basics#split-leakage"),
            ("逐元素运算", "/notebooks/3_0_2_linear_algebra#elementwise-dot"),
            ("函数复合与激活", "/notebooks/3_0_3_calculus#functions"),
            ("链式求导与反向传播", "/notebooks/3_0_3_calculus#chain-rule"),
        ],
        "paper": "GRU 兴趣抽取、逐步辅助损失，以及用候选注意力缩放更新门的 AUGRU。",
    },
    "3_4_1_mmoe": {
        "generic": [
            ("多任务标签", "/notebooks/3_0_1_data_ml_basics#observation-label"),
            ("张量形状与矩阵乘法", "/notebooks/3_0_2_linear_algebra#matmul-embedding"),
            ("Softmax 与加权和", "/notebooks/3_0_5_information_theory#softmax-temperature"),
            ("多任务梯度冲突", "/notebooks/3_0_6_optimization#gradient-conflict"),
        ],
        "paper": "让每个任务用自己的 gate 混合同一组专家，并辨认任务梯度在共享专家中的汇合路径。",
    },
    "3_4_2_ple": {
        "generic": [
            ("多任务标签", "/notebooks/3_0_1_data_ml_basics#observation-label"),
            ("Softmax 与加权和", "/notebooks/3_0_5_information_theory#softmax-temperature"),
            ("多任务梯度冲突", "/notebooks/3_0_6_optimization#gradient-conflict"),
            ("正则化与泛化", "/notebooks/3_0_6_optimization#regularization"),
        ],
        "paper": "CGC 的可见专家约束、多层渐进分离，以及样本空间掩码和动态任务权重。",
    },
    "4_2_openonerec_practice": {
        "generic": [
            ("序列、标签与反馈", "/notebooks/3_0_1_data_ml_basics#observation-label"),
            ("embedding 与向量量化接口", "/notebooks/3_0_2_linear_algebra#matmul-embedding"),
            ("条件概率链式法则", "/notebooks/3_0_4_probability_statistics#conditional-chain"),
            ("序列 NLL 与 DPO", "/notebooks/3_0_5_information_theory#sequence-nll-dpo"),
        ],
        "paper": "残差量化到均衡 Semantic ID 的接口、trie 约束列表解码，以及 reward 模型边界内的迭代偏好对齐。",
    },
    "4_3_dlrm_hstu_practice": {
        "generic": [
            ("序列与严格时间切分", "/notebooks/3_0_1_data_ml_basics#split-leakage"),
            ("Q/K/V 张量形状", "/notebooks/3_0_2_linear_algebra#low-rank-attention"),
            ("逐元素运算与 SiLU", "/notebooks/3_0_3_calculus#functions"),
            ("序列负对数似然", "/notebooks/3_0_5_information_theory#sequence-nll-dpo"),
        ],
        "paper": "不经 softmax 的 Pointwise Aggregated Attention、相对位置/时间偏置，以及统一序列转导块。",
    },
}


def prerequisite_markdown(slug: str) -> str:
    spec = DEEP_MATH_PREREQUISITES[slug]
    links = "\n".join(f"- [{label}]({url})" for label, url in spec["generic"])
    return f"""### 通用先修（先回看 3.0 基础课程）

{links}

### 本论文新增数学（本节详细推导）

{spec['paper']}"""


def figure_markdown(paper_id: str) -> str:
    """Embed the cropped paper structure figure and the key-module list into a notebook cell.

    The image is referenced by absolute /static/ path so it renders in the Web preview
    (the primary reading surface). Key modules come from config/model_layers.json, shared
    with the on-page paper-guide cards.
    """
    spec = FIGURES_CFG.get(paper_id)
    if not spec:
        return ""
    label = spec["label"]
    page = spec["page"]
    bullets = "\n".join(f"- **{name}**：{desc}" for name, desc in MODEL_LAYERS_CFG.get(paper_id, []))
    return (
        f"![{label}](/static/paper-figures/{paper_id}.webp)\n\n"
        f"> **论文原图节选** · {label} · PDF p.{page}。"
        "下图直接截取自原文，用于对照下方公式与代码。\n\n"
        "### 关键模块\n\n"
        f"{bullets}"
    )


def build_deep_specs(md, code, notebook):
    """Build one-algorithm-per-notebook tutorials for chapters 3.2–4."""

    algorithms = [
        {
            "slug": "3_2_1_dssm",
            "title": "3.2.1 DSSM 双塔召回",
            "chapter": "chapter_3_2",
            "function": "run_dssm",
            "source": "[Huang et al., 2013, DSSM](https://www.microsoft.com/en-us/research/publication/learning-deep-structured-semantic-models-for-web-search-using-clickthrough-data/)",
            "problem": "怎样把用户与物品分别编码，使物品向量可以离线建索引，同时仍能从全库找回相关候选？",
            "paper": "DSSM 最初解决的是 Web 搜索：query 很短，document title 较长，两边即使没有相同单词也可能表达同一概念。训练正例是搜索日志里的 query-clicked-title 对，负文档由训练候选采样；最终 NDCG 则来自人工相关性标签，不是点击率。论文真正验证的是两侧独立映射、余弦相关分、点击数据判别式训练与 letter-trigram word hashing，没有直接提出推荐系统 ANN 双塔。教程把 query/document 迁移为 user/item，把评分转成“观察正例/未观察候选”，并接 ANN；这是结构迁移，不是原论文数据或实验结论。",
            "math": r"""先把 $u$、$v$ 想成两支从原点出发的箭头。点积 $u^\top v$ 同时受方向和长度影响；除以两者长度后，$\cos(u,v)=u^\top v/(\|u\|_2\|v\|_2)$ 只比较方向。若归一化后的用户为 $(0.8,0.6)$，物品 A 为 $(0.9,0.4)$、物品 B 为 $(-0.2,0.9)$，就分别代入“横坐标乘横坐标 + 纵坐标乘纵坐标”。批量计算时，$[B,d]$ 乘 $[N,d]^\top$ 只是一次同时完成 $B\times N$ 组点积，输出 $[B,N]$。""",
            "hand": """import numpy as np, matplotlib.pyplot as plt
user=np.array([.8,.6]); positive=np.array([.9,.4]); negative=np.array([-.2,.9])
norm=lambda x:x/np.linalg.norm(x)
scores=[norm(user)@norm(v) for v in [positive,negative]]
fig,ax=plt.subplots(figsize=(5.5,4.4))
for vector,label,color in [(user,'user','#4f8f00'),(positive,'positive item','#e36b2c'),(negative,'negative item','#5b7c99')]:
    ax.quiver(0,0,*vector,angles='xy',scale_units='xy',scale=1,label=label,color=color)
ax.set(xlim=(-.4,1.1),ylim=(-.1,1.1),title=f'cosine: positive={scores[0]:.2f}, negative={scores[1]:.2f}')
ax.grid(alpha=.25); ax.legend(); plt.show()
print({'user@positive':round(scores[0],3),'user@negative':round(scores[1],3)})""",
            "smoke_dataset": "仓库随附的真实 Amazon 评分教学子集；`rating>=4` 仅在这个迁移实验中定义为正反馈，按时间切出训练/测试并按用户留出最后正例。它没有曝光日志，未评分物品只能称为“未观察”，不能称为论文里的未点击文档。",
            "framework": "实际使用 torch_rechub.models.matching.DSSM。full profile 映射 TorchEasyRec 双塔配置，并把 item tower 导出到 Faiss、HNSW 或 Milvus。",
            "primary": "recall@10", "secondary": "test_auc", "baseline": None,
            "inference": "用户塔在线编码 → L2 归一化 → ANN Top-K；物品塔离线批量更新。监控索引新鲜度、负采样分布与向量范数。",
            "caveat": "Amazon review 仍是主动评分而非完整曝光日志；它比电影评分更接近电商目录与冷启动，但 Recall 仍不能直接代表线上曝光收益。",
        },
        {
            "slug": "3_2_2_mind",
            "title": "3.2.2 MIND 多兴趣召回",
            "chapter": "chapter_3_2",
            "function": "run_mind",
            "source": "[Li et al., 2019, MIND](https://arxiv.org/abs/1904.08030)",
            "problem": "当用户同时喜欢科幻、跑步和烹饪时，为什么一个平均向量会丢失兴趣？如何用多个向量分别检索？",
            "paper": "MIND 从 Tmall 的匹配阶段出发：一个用户可能同时关心服饰、运动和食品，而单个平均向量会把不同方向混在一起。论文把行为转为多个兴趣胶囊；训练时目标物品通过 label-aware attention 选择应监督的兴趣，服务时则移除这层，让多个兴趣各自 ANN。原文还给出随机 19:1 离线切分和一周线上 A/B；本教程的时间切分是更严格的迁移协议，不能把两套数值直接相减。",
            "math": r"""设四条历史在二维图上是 $(1,0.1),(0.9,0.2),(0.1,1),(0.2,0.9)$。全部平均得到 $(0.55,0.55)$，它既不像横向兴趣也不像纵向兴趣；分成两组后，两个中心约为 $(0.95,0.15)$ 与 $(0.15,0.95)$。MIND 用可学习的软分组代替手工分组：每条行为对 K 个兴趣都有权重，权重经 softmax 后和为 1，再对行为向量做加权和。候选 $e_i$ 的服务分数取 $\max_k v_k^\top e_i$。""",
            "hand": """import numpy as np, matplotlib.pyplot as plt
history=np.array([[1,.1],[.9,.2],[.1,1],[.2,.9]])
a,b,single=history[:2].mean(0),history[2:].mean(0),history.mean(0)
fig,ax=plt.subplots(figsize=(5.5,4.4)); ax.scatter(history[:,0],history[:,1],s=90,label='history')
ax.scatter(*a,s=180,marker='*',label='interest 1'); ax.scatter(*b,s=180,marker='*',label='interest 2')
ax.scatter(*single,s=120,marker='x',label='single average'); ax.set(title='Two interests avoid mixed intent',xlabel='topic A',ylabel='topic B')
ax.legend(); ax.grid(alpha=.2); plt.show()
print({'interest_1':a.tolist(),'interest_2':b.tolist(),'single':single.tolist()})""",
            "smoke_dataset": "仓库随附的真实 Amazon 评分教学子集；从每位用户的已观察序列构造 next-item 正例，并从其未观察目录中确定性采负例。它不满足论文的 10-core 后统计，只验证 `[B,L] -> [B,K,d]` 与多兴趣检索链路。",
            "framework": "实际使用 torch_rechub.models.matching.MIND，执行 CapsuleNetwork、目标兴趣选择和多兴趣推理；full profile 映射 TorchEasyRec MIND。",
            "primary": "recall@10", "secondary": "positive_top1", "baseline": None,
            "inference": "user tower 输出 [B,K,d]；每个兴趣独立检索，再按最高分合并、去重和配额控制。",
            "caveat": "只有 full 档同时满足论文数据版本、10-core 后统计、K=3、d=36 和候选协议时才允许对照 Table 2；smoke 指标只证明代码可运行。",
        },
        {
            "slug": "3_3_1_deepfm",
            "title": "3.3.1 DeepFM 排序",
            "chapter": "chapter_3_3",
            "function": "run_deepfm",
            "source": "[Guo et al., 2017, DeepFM](https://arxiv.org/abs/1703.04247)",
            "problem": "如何在不手工枚举所有组合的情况下，同时学习一阶、二阶和高阶稀疏特征交互？",
            "paper": "DeepFM（IJCAI 2017，华为诺亚方舟实验室）针对 CTR 预估中低阶与高阶交互难以兼得的问题：FNN 要先拿 FM 预训练 embedding 才能用，PNN 只学高阶，Wide & Deep 的 wide 侧仍要人工叉乘特征。论文让 FM 与 DNN 共享同一份输入与 embedding，端到端联合训练，离线在 Criteo 与公司数据上同时报告最优 AUC 与 LogLoss。它没有做的：没有自己的在线 A/B（只引用 Wide & Deep 的线上经验），不建模行为序列，不处理曝光偏差与概率校准；Criteo 用随机 9:1 切分，比时间切分宽松。",
            "math": r"""FM 二阶项是 $\sum_{i<j}\langle v_i,v_j\rangle x_ix_j$：每对非零特征贡献一次隐向量内积；Deep 分支把同一组 embedding 展平送入 MLP；最终 logit 为 linear + FM + DNN 三路相加，再经 Sigmoid 得点击概率。下面的代码用三个二维 embedding（user、item、hour）手算：user·item $=1\times0.8+0\times0.2=0.8$，user·hour $=0.2$，item·hour $=0.8\times0.2+0.2\times0.9=0.34$，三项相加约 $1.34$，就是这条样本的全部二阶交互分。""",
            "hand": """import numpy as np, matplotlib.pyplot as plt
embedding=np.array([[1.,0.],[.8,.2],[.2,.9]]); pair=embedding@embedding.T
fig,ax=plt.subplots(figsize=(4.8,4)); image=ax.imshow(pair,cmap='YlGn',vmin=0,vmax=1)
ax.set_xticks(range(3),['user','item','hour']); ax.set_yticks(range(3),['user','item','hour'])
for i in range(3):
    for j in range(3): ax.text(j,i,f'{pair[i,j]:.2f}',ha='center',va='center')
ax.set_title('FM interaction = embedding dot product'); plt.colorbar(image,ax=ax); plt.show()
print('sum of three pair interactions =',round(pair[0,1]+pair[0,2]+pair[1,2],3))""",
            "smoke_dataset": "KuaiRand-Pure 的真实短视频曝光教学子集：user、video、场景 tab、hour、视频时长桶为特征，标签直接读取 `is_click`，按日志顺序切分。它不是 full 协议指定的 Criteo。",
            "framework": "实际使用 torch_rechub.models.ranking.DeepFM。full profile 在 TorchEasyRec 配置 sparse feature、embedding group、MLP、分布式 embedding 与模型导出。",
            "primary": "auc", "secondary": "logloss", "baseline": "lr_auc",
            "inference": "读取请求和候选特征 → 查 embedding → FM 与 DNN 并行 → Sigmoid；线上校验词表、缺失值和 embedding 版本。",
            "caveat": "AUC 衡量排序而非概率准确度，必须同时看 LogLoss 与校准；KuaiRand 的随机干预比例和不同 tab 策略也会造成分布差异。",
        },
        {
            "slug": "3_3_2_din",
            "title": "3.3.2 DIN 候选感知排序",
            "chapter": "chapter_3_3",
            "function": "run_din",
            "source": "[Zhou et al., 2018, DIN](https://arxiv.org/abs/1706.06978)",
            "problem": "为什么同一位用户面对相机和跑鞋时应该激活不同历史？如何让用户表示依赖当前候选？",
            "paper": "DIN（KDD 2018，阿里妈妈）从展示广告的真实瓶颈出发：Embedding&MLP 把用户历史压成与候选无关的固定长度向量。论文的贡献不止结构——mini-batch aware 正则化让上亿参数的稀疏网络用得起 ℓ2，Dice 让激活整流点随输入分布自适应；阿里线上 A/B 报告 CTR 最高 +10.0%。代价是每个候选都要重算历史注意力；论文自己尝试 LSTM 建模行为序列没有收益，把序列建模留给后续工作（即 DIEN）。",
            "math": r"""注意力分数 $a_j=g(e_j,e_t,e_j-e_t,e_j\odot e_t)$。原论文直接算 $v=\sum_j a_je_j$，不要求权重和为 1；下面代码为了对应 Torch-RecHub 的 `use_softmax=True` 实现，额外把原始分数归一化。候选『相机』为 $(1,0)$ 时，前两条历史获得较高比例；换成候选『跑鞋』$(0,1)$，比例立刻翻到后两条历史——这张图只验证“候选改变历史选择”，不声称两种汇总公式等价。""",
            "hand": """import numpy as np, matplotlib.pyplot as plt
history=np.array([[1.,0.],[.8,.2],[0.,1.],[.1,.9]]); targets=[np.array([1.,0.]),np.array([0.,1.])]
fig,axes=plt.subplots(1,2,figsize=(9,3.4))
for ax,target,name in zip(axes,targets,['camera candidate','running candidate']):
    raw=history@target; weight=np.exp(raw)/np.exp(raw).sum(); ax.bar(range(4),weight,color='#7ca832')
    ax.set(ylim=(0,.4),title=name,xlabel='history position',ylabel='attention weight')
plt.tight_layout(); plt.show(); print('same history, different candidate -> different weights')""",
            "smoke_dataset": "KuaiRand-Pure 的真实 feed 曝光教学子集：候选是当前视频，历史只含该曝光之前的真实点击，标签直接读取 `is_click`。对 DIN，未点击曝光是当前样本的负标签，不会伪装成用户“主动不喜欢”。",
            "framework": "实际使用 torch_rechub.models.ranking.DIN，调用 SequenceFeature、ActivationUnit 与 Dice MLP；full profile 配置序列截断、缓存和候选批处理。",
            "primary": "auc", "secondary": "logloss", "baseline": "static_overlap_auc",
            "inference": "对每个候选计算目标感知 attention，再拼接上下文打分；候选多时需缩短历史、缓存 embedding 或先预排。",
            "caveat": "真实小数据上静态基线可能更高；这正说明必须设置诚实 baseline，模型复杂度本身不是收益。",
        },
        {
            "slug": "3_3_3_dien",
            "title": "3.3.3 DIEN 兴趣演化排序",
            "chapter": "chapter_3_3",
            "function": "run_dien",
            "source": "[Zhou et al., 2019, DIEN](https://arxiv.org/abs/1809.03672)",
            "problem": "当兴趣从旧主题转向新主题时，怎样区分“出现过什么”和“现在更想要什么”？",
            "paper": "DIEN（AAAI 2019，同一团队）认为 DIN 把行为直接当作兴趣、又忽略次序：真实兴趣藏在行为背后且随时间演化。它用 GRU 抽取逐时刻兴趣状态，以辅助下一行为损失给每步监督，再用目标感知的 AUGRU 控制状态演化。它比 DIN 重得多：收益依赖严格时间顺序和高质量负序列；线上要靠 kernel 融合、batching 与 Rocket Launching 压缩（GRU 隐状态 108→32 维）才把延迟从 38.2 ms 压到 6.6 ms。",
            "math": r"""GRU 是带记忆的递推函数 $h_t=\mathrm{GRU}(e_t,h_{t-1})$：同样的集合换一种顺序，末状态就不同。辅助损失要求 $h_t$ 更像下一次真实行为、远离负样本；AUGRU 用候选相关权重控制每步写入多少。下面的代码用 $\alpha=0.65$ 的指数递推模拟这种记忆：四个事件『旧、旧、新、新』演完后新兴趣分量约 $0.58$，顺序反过来只剩约 $0.24$——次序本身就是信息。""",
            "hand": """import numpy as np, matplotlib.pyplot as plt
alpha=.65
def evolve(sequence):
    state=np.zeros(2); path=[]
    for event in sequence:
        state=alpha*state+(1-alpha)*event; path.append(state.copy())
    return np.array(path)
old=np.array([1.,0.]); new=np.array([0.,1.]); paths=[evolve([old,old,new,new]),evolve([new,new,old,old])]
fig,ax=plt.subplots(figsize=(6,3.5)); ax.plot(paths[0][:,1],marker='o',label='old to new'); ax.plot(paths[1][:,1],marker='o',label='new to old')
ax.set(title='Order changes current interest',xlabel='time',ylabel='new-topic state'); ax.legend(); ax.grid(alpha=.2); plt.show()""",
            "smoke_dataset": "KuaiRand-Pure 的真实时间序列教学子集：点击视频进入正序列，已曝光未点击视频进入 `negative_history`；padding=0，未来曝光不进入历史。这里的负序列是“已曝光未点击”，不是从未曝光物品。",
            "framework": "实际使用 torch_rechub.models.ranking.DIEN，执行 GRU、auxiliary loss 与 AUGRU；full profile 需核对序列打包、负采样和线上顺序。",
            "primary": "auc", "secondary": "logloss", "baseline": "static_overlap_auc",
            "inference": "严格按时间编码历史，候选控制 AUGRU 更新，最终状态进入 MLP；同时测 P99、吞吐和校准。",
            "caveat": "小数据下 DIEN 可能 AUC 尚可但 LogLoss 很差，应使用早停、校准和用户分桶，不能只看训练损失。",
        },
        {
            "slug": "3_4_1_mmoe",
            "title": "3.4.1 MMoE 多目标学习",
            "chapter": "chapter_3_4",
            "function": "run_mmoe",
            "source": "[Ma et al., 2018, MMoE](https://dl.acm.org/doi/10.1145/3219819.3220007)",
            "problem": "点击与转化既共享信号又不完全相同，怎样让每个任务选择不同的共享专家？",
            "paper": "MMoE（KDD 2018，Google）针对 Shared-Bottom 对任务相关性敏感的问题：先用合成数据证明相关性越低共享底层越差，再让一组专家被所有任务共享、每个任务配独立 gate。它比 hard sharing 灵活，参数量只多几个轻量 gate；但所有专家仍全量共享，低相关或复杂相关任务下仍可能互相干扰（PLE 论文后来在腾讯系统上测得它对 VCR 的 MTL gain 仅 +0.0001）。",
            "math": r"""任务 $k$ 的表示是专家输出的加权和 $z_k=\sum_e g_{k,e}(x)f_e(x)$，其中 $g_k(x)=\mathrm{softmax}(W_{g,k}x)$：softmax 把任意实数变成和为 1 的权重。手算一组：2 个专家、某条样本的 gate logits 为 $(2,0)$，则 $e^2\approx7.389$、$e^0=1$，权重为 $(7.389/8.389,\ 1/8.389)\approx(0.881,0.119)$，表示近九成来自专家 1；另一条样本 logits 为 $(0,2)$ 时配方整个反过来。专家像擅长不同题型的老师：各任务听同一组老师，但自行决定每位老师占多少。总损失 $L=\sum_k\lambda_kL_k$，$\lambda_k$ 表达业务权衡。""",
            "hand": """import numpy as np, matplotlib.pyplot as plt
weights=np.array([[.65,.25,.10],[.15,.30,.55]]); fig,ax=plt.subplots(figsize=(6,3.3)); left=np.zeros(2)
for expert in range(3):
    ax.barh(['click','conversion'],weights[:,expert],left=left,label=f'expert {expert+1}'); left+=weights[:,expert]
ax.set(xlim=(0,1),title='Task gates mix shared experts'); ax.legend(ncol=3,loc='lower center'); plt.show()
print('gate sums =',weights.sum(1))""",
            "smoke_dataset": "KuaiRand-Pure 的真实曝光教学子集，以 `is_click` 和 `long_view` 为两个标签，并共享同一组上下文特征。它只检查多任务梯度与 gate 链路，不冒充 Census-Income 的 income/marital-status 任务。",
            "framework": "实际使用 torch_rechub.models.multi_task.MMOE；full profile 在 TorchEasyRec 配置专家、任务塔、loss 权重、样本掩码和分布式训练。",
            "primary": "click_auc", "secondary": "long_view_auc", "baseline": None,
            "inference": "共享特征编码一次，任务 gate 混合专家，各塔输出概率；监控逐任务 AUC、校准、专家利用率与业务价值。",
            "caveat": "MMoE 不保证胜过独立模型；任务相关性、标签噪声、loss 尺度和采样空间决定是否正迁移。",
        },
        {
            "slug": "3_4_2_ple",
            "title": "3.4.2 PLE 渐进式专家抽取",
            "chapter": "chapter_3_4",
            "function": "run_ple",
            "source": "[Tang et al., 2020, PLE](https://dl.acm.org/doi/10.1145/3383313.3412236)",
            "problem": "如果共享专家仍让任务互相干扰，怎样显式保留任务专属知识，并逐层控制共享？",
            "paper": "PLE（RecSys 2020，腾讯）先命名并量化跷跷板现象：在腾讯视频推荐的 VTR/VCR 复杂相关任务组上，当时 SOTA 多任务模型提升一个任务常牺牲另一个（Figure 3 中无基线落在双优象限）。它给每个任务独立专属专家、CGC 门控只读本任务与共享专家，再堆叠成多层渐进分离路由；代价是专家数、层数与动态损失权重等更多超参。",
            "math": r"""CGC 的关键是「谁能被选」：任务 $k$ 的 gate 只在自己的 $m_k$ 个专属专家和 $m_s$ 个共享专家之间分配 softmax 权重，其它任务的专属专家被结构性排除，权重恒为 0。手算一组：$m_k=1, m_s=1$，gate logits $(1,0)$ 得权重 $(e/(e+1), 1/(e+1))\approx(0.731,0.269)$；若专属专家输出 $(2,0)$、共享专家输出 $(0,2)$，融合结果为 $0.731\times(2,0)+0.269\times(0,2)=(1.462,0.538)$。直觉像公共课与专业课：都听公共课，但不被迫共享全部专业细节。堆叠多层即 progressive extraction：低层充分混合，高层逐步分离。""",
            "hand": """import numpy as np, matplotlib.pyplot as plt
matrix=np.array([[1,1,0],[1,0,1]],dtype=float); fig,ax=plt.subplots(figsize=(5.5,2.8)); image=ax.imshow(matrix,cmap='YlGn')
ax.set_xticks(range(3),['shared','click-only','convert-only']); ax.set_yticks(range(2),['click gate','convert gate'])
for i in range(2):
    for j in range(3): ax.text(j,i,int(matrix[i,j]),ha='center',va='center')
ax.set_title('PLE: shared plus task-specific experts'); plt.colorbar(image,ax=ax,ticks=[0,1]); plt.show()""",
            "smoke_dataset": "与 MMoE 使用同一 KuaiRand-Pure 真实曝光教学子集和 `is_click`/`long_view` 标签，使 smoke 横向差异主要来自路由结构。它不冒充 full 的 Census-Income 官方切分。",
            "framework": "实际使用 torch_rechub.models.multi_task.PLE，执行两层 CGC、共享/专属专家与任务塔；full profile 映射 TorchEasyRec PLE。",
            "primary": "click_auc", "secondary": "long_view_auc", "baseline": None,
            "inference": "共享/专属专家逐层计算，各任务塔输出；服务关注参数量、专家并行和输出版本兼容。",
            "caveat": "简单或小数据上 PLE 可能不如 MMoE；复杂结构只有在负迁移真实存在且数据充足时才可能回本。",
        },
        {
            "slug": "3_2_3_sasrec",
            "title": "3.2.3 SASRec 序列召回",
            "chapter": "chapter_3_2",
            "function": "run_sasrec",
            "source": "[Kang & McAuley, 2018, SASRec](https://arxiv.org/abs/1808.09781) · [作者官方实现](https://github.com/kang205/SASRec)",
            "problem": "如何用因果自注意力同时建模近期转移与较长兴趣，并预测用户下一部可能喜欢的电影？",
            "paper": "SASRec 把每位用户的交互按时间排成序列，在每个位置预测下一物品。它不是简单把 Transformer 原样搬过来：论文明确规定左侧 padding、最近 n 项截断、可学习位置向量、因果遮罩、残差/LayerNorm 和逐位置负采样。注意力能在稀疏数据上偏向最近行为，在稠密数据上使用更远历史；代价是长度 n 增大时注意力矩阵按 n² 增长。",
            "math": r"""把 Q、K、V 读成“问题、标签、内容”：位置 t 的 Q 去和每个过去位置的 K 做点积，得到该读多少 V 的权重。若一行原始分数是 $[2,1,0]$，softmax 后最相关位置权重大，但其余位置仍可贡献；因果 mask 先把未来格子改成 $-\infty$，指数后它们严格变成 0。除以 $\sqrt d$ 是为了防止维度多时点积过大，让 softmax 过早接近全 0/全 1。""",
            "hand": "import numpy as np, matplotlib.pyplot as plt\nlength=6\nlogits=np.fromfunction(lambda row,col: 2.0-np.abs(row-col)*.55,(length,length))\nlogits[np.triu(np.ones((length,length),dtype=bool),1)]=-np.inf\nweights=np.exp(logits-np.max(logits,axis=1,keepdims=True)); weights/=weights.sum(1,keepdims=True)\nfig,ax=plt.subplots(figsize=(5.4,4.2)); image=ax.imshow(weights,cmap='YlGn'); ax.set(title='SASRec causal attention',xlabel='history position',ylabel='prediction position'); plt.colorbar(image,ax=ax); plt.show()\nprint('row sums =',weights.sum(1).round(3))",
            "smoke_dataset": "仓库随附的真实 Amazon 评分序列教学子集；把已观察评分按时间组成隐式序列，从未观察目录采训练负例。它缩短序列和模型，只验证因果注意力与 next-item 链路，不使用 MovieLens-1M 论文候选协议。",
            "framework": "实际使用 torch_rechub.models.matching.SASRec，包含 item/position embedding、causal multi-head attention 与 pairwise loss；线上可分离用户表示和 item 向量。",
            "primary": "hr@10", "secondary": "popularity_hr@10", "baseline": None,
            "inference": "截取最近 L 个真实行为 → 因果 Transformer → 最后位置用户向量 → 全库点积/ANN Top-K；屏蔽已见物品并监控序列截断与 P99。",
            "caveat": "full 档参照论文 ML-1M 设置使用 2 blocks、d=50、长度 200、dropout 0.2 与 lr=0.001；只有该档可与 Table III 的 ML-1M HR/NDCG 对照。",
        },
        {
            "slug": "4_2_openonerec_practice",
            "title": "4.2 OpenOneRec：受约束列表生成实战",
            "chapter": "chapter_4",
            "function": "run_openonerec",
            "source": "[Kuaishou OpenOneRec](https://github.com/Kuaishou-OneRec/OpenOneRec)",
            "problem": "如何把推荐列表变成 token 序列，同时保证解码结果属于真实目录、没有重复并可接受奖励对齐？",
            "paper": "OneRec（快手，2025）把召回、粗排、精排的级联流水线替换成单一 encoder-decoder 生成模型：内容 embedding 经均衡 K-Means 残差量化成 Semantic ID，decoder 自回归生成整屏 session，再用 IPA（自采样 + reward 模型 + 迭代 DPO）对齐偏好。OpenOneRec 展示这一流程的开放实现。smoke 档不冒充官方大模型训练，而是复现最关键的数据契约：item 到 Semantic ID、teacher forcing、trie 约束和合法性压力测试。",
            "math": r"""自回归分解 $P(y_1,\ldots,y_T|x)=\prod_tP(y_t|y_{<t},x)$，表示每一步根据上下文和已生成 token 预测下一 token。trie 像目录树：前缀 $(1,2)$ 下只允许真实后继 $\{3,4\}$。DPO 只需要 $\sigma$ 与 $\log$ 的直觉：$-\log\sigma(z)$ 在 $z$ 增大时减小，因此损失鼓励 chosen 列表相对参考模型的对数概率增量高于 rejected。""",
            "hand": """import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(7,3)); ax.axis('off'); ax.text(.05,.5,'root',bbox=dict(boxstyle='round',fc='#e8f2d6'))
for y,node in zip([.25,.5,.75],['1','2','3']):
    ax.text(.32,y,node,bbox=dict(boxstyle='round',fc='white')); ax.plot([.13,.30],[.5,y],color='#809070')
ax.text(.58,.35,'prefix (1,2)',bbox=dict(boxstyle='round',fc='#ffe8d7')); ax.text(.88,.25,'3'); ax.text(.88,.48,'4')
ax.plot([.72,.86],[.38,.27]); ax.plot([.72,.86],[.38,.49]); ax.set_title('Catalog trie allows valid next tokens'); plt.show()""",
            "smoke_dataset": "KuaiRand-Pure 的真实视频 tag、music type 与 item 分区形成教学 Semantic ID；真实反馈只用于小型 token/约束链路。CUDA 与 CPU 都不自动获得 gated 的 RecIF-Bench，也不把本地 chosen/rejected 当成官方 reward 数据。",
            "framework": "OpenOneRec 官方训练配置负责 full profile；本 Notebook 的 PyTorch 小生成器验证相同 token/trie 契约、训练损失和合法性。",
            "primary": "invalid_constrained", "secondary": "invalid_unconstrained", "baseline": None,
            "inference": "自回归每一步应用目录 trie、去重和长度约束；随后可用奖励模型或 DPO 调整整列价值。",
            "caveat": "合法率只是底线，不等于相关性；还需看 Recall/NDCG、重复率、多样性、P99、目录更新和奖励偏差。",
        },
        {
            "slug": "4_3_dlrm_hstu_practice",
            "title": "4.3 DLRM HSTU：长行为序列实战",
            "chapter": "chapter_4",
            "function": "run_hstu",
            "source": "[Zhai et al., 2024, HSTU](https://arxiv.org/abs/2402.17152) · [Meta generative-recommenders](https://github.com/meta-recsys/generative-recommenders)",
            "problem": "如何把长期用户行为作为统一自回归序列训练，并让模型结构与大规模推荐系统共同设计？",
            "paper": "HSTU（Meta，2024）把推荐重新定义为序列转导任务：所有特征序列化成统一流，用 SiLU 逐点聚合注意力（不做整序列 softmax 归一化）加相对位置/时间偏置的单一块，替代 DLRM 的特征抽取、特征交互与表示变换三类拼装模块。本节默认在 CUDA 上用 Torch-RecHub HSTUModel 运行 next-item 训练；完整工业配置对齐 Meta DLRM-v3、TorchRec/FBGEMM 和 M-FALCON。",
            "math": r"""给定 $[i_1,\ldots,i_t]$，模型在每个位置预测下一 item。因果 mask 保证位置 $t$ 只能看不晚于 $t$ 的历史。训练把所有位置交叉熵相加；推理只取最后位置 logits 做 Top-K。HSTU 把每行 softmax 归一化换成逐点 SiLU$(z)=z\sigma(z)$：各行不再共享总和为 1 的预算，多个强相关历史可以同时保持高贡献。""",
            "hand": """import numpy as np, matplotlib.pyplot as plt
length=7; mask=np.tril(np.ones((length,length))); fig,ax=plt.subplots(figsize=(5,4)); image=ax.imshow(mask,cmap='YlGn',vmin=0,vmax=1)
ax.set(title='Causal mask: only past is visible',xlabel='history position',ylabel='prediction position')
ax.set_xticks(range(length)); ax.set_yticks(range(length)); plt.colorbar(image,ax=ax,ticks=[0,1]); plt.show(); print(mask.astype(int))""",
            "smoke_dataset": "KuaiRand-Pure 的真实短视频点击流教学适配器：按毫秒时间排序，最后行为只作测试，训练只看更早的 next-video 转移。CUDA 只扩大该适配器规模；它不会自动切换到 Meta 的 MovieLens-20M + Amazon Books full 协议。",
            "framework": "实际使用 torch_rechub.models.generative.HSTUModel 完成 next-item 训练；工业档切换 Meta DLRM-v3、TorchRec/FBGEMM 和多 GPU。",
            "primary": "hr@5", "secondary": "popularity_hr@5", "baseline": None,
            "inference": "最后位置 logits → Top-K 或 M-FALCON；服务需缓存状态、过滤非法/已见 item，并监控吞吐、显存和目录新鲜度。",
            "caveat": "真实小数据上的 HR 可能低于热门基线；这比可学习的人工序列更诚实，需通过严格时间切分、强 baseline 和成本收益共同判断。",
        },
    ]

    # These summaries are taken from the locally initialized PDFs.  They keep
    # paper claims, tutorial reruns, and production extrapolation visibly apart.
    paper_evidence = {
        "3_2_1_dssm": ("dssm", "论文用约 1 亿 query-title 点击对训练，在 16,510 个查询、每个约 15 个带人工相关性标签的文档上报告 NDCG。Table 2 的完整 L-WH DNN 为 0.362/0.425/0.498（NDCG@1/3/10），而 TF-IDF 为 0.319/0.382/0.462。教程改用 Amazon 行为与 Recall@10，只验证从两侧编码到 Top-K 的迁移链路，不声称复现搜索 NDCG，也不把 ANN 说成原论文实验。"),
        "3_2_2_mind": ("mind", "原文对 Amazon Books 做 user/item 10-core 后得到 351,356 用户、393,801 商品、6,271,511 行，并随机按 19:1 划分，再用 HitRate@10/50/100 比较方法。Amazon 上 K=3、d=36 的 MIND-K HR@10 为 0.0309，YouTube DNN 为 0.0231。full 档恢复数据版本与统计，但教程使用更严格的时间协议；只有明确对齐的字段才允许对照，smoke 不参与。"),
        "3_2_3_sasrec": ("sasrec", "原文对 Amazon Beauty/Games、Steam、MovieLens-1M 做 5-core 和 leave-two-out，并在 1 个真值加 100 个未观察负例上计算 Hit@10/NDCG@10。ML-1M 的 SASRec 为 0.8245/0.5905。full 档使用论文公开的 ML-1M、相同候选协议和主要超参数；smoke 不进入论文数值比较。"),
        "3_3_1_deepfm": ("deepfm", "原文 Table 2：Criteo（约 4,500 万条点击记录，13 连续 + 26 离散特征，随机 9:1 切分）上 DeepFM AUC 0.8007、LogLoss 0.45083，优于 FM（0.7892/0.46077）与最佳 PNN 变体（0.7987/0.45214）；Company*（约 10 亿条记录，7 天训练 + 次日测试）上 AUC 0.8715、LogLoss 0.02618。相对 LR，AUC 提升 0.86%/4.18%（Company*/Criteo），LogLoss 降低 1.15%/5.60%；相对用独立 embedding 的 LR&DNN/FM&DNN 仍高 0.48%/0.33%。深度模型统一 400-400-400、dropout 0.5，FM 隐向量维度 10。教程使用 KuaiRand 完整曝光重新测量并保留 LR 对照，两套数值不能相减。"),
        "3_3_2_din": ("din", "原文采用用户加权 AUC（GAUC）。Table 3（公开集各重复 5 次，初始化波动 <0.0002）：Amazon Electronics（192,403 用户、1,689,188 样本）DIN AUC 0.8818、Dice 版 0.8871，BaseModel 0.8624，RelaImpr 6.82%；MovieLens DIN+Dice 0.7348 对 BaseModel 0.7300。Table 5（阿里工业数据，约 20 亿训练样本）：DIN 0.6029，叠加 MBA 正则与 Dice 后 0.6083，RelaImpr 11.65%；论文注明该规模下 0.001 绝对 AUC 即值得上线。线上 A/B（2017-05~06）相对线上 BaseModel CTR 最高 +10.0%、RPM +3.8%。这些是阿里广告系统口径，教程在 KuaiRand 的数值另行报告，不能相减。"),
        "3_3_3_dien": ("dien", "Table 2（公开集各重复 5 次）：Amazon Electronics DIEN AUC 0.7792±0.00243（DIN 0.7603、BaseModel 0.7435），Books 0.8453±0.00476（DIN 0.7880）。Table 3（工业数据，70 亿样本、历史截断 50）：DIEN 0.6541（DIN 0.6428、BaseModel 0.6350）。Table 4 消融：AUGRU 优于 AIGRU/AGRU；辅助损失在公开集贡献最大、工业集收益变小。Table 5 在线 A/B（2018-06-07~07-12，淘宝展示广告）：相对 BaseModel，DIEN CTR +20.7%、eCPM +17.1%、PPC −3.0%（DIN 为 +8.9%/+6.7%/−2.0%）。服务侧 GRU 隐状态 108→32 维，延迟 38.2→6.6 ms。以上均为论文系统口径，教程结果另行报告。"),
        "3_4_1_mmoe": ("mmoe", "原文先用合成数据控制任务相关性：输入 100 维、8 个 16 单元专家、tower 8 单元，每个设置独立重复 200 次——相关性降到 0.5 时 Shared-Bottom 与单门 OMoE 明显退化而 MMoE 几乎不变，且 MMoE 的最终损失方差更小（trainability）。UCI Census-income：299,285 条、40 特征、199,523 训练 + 99,762 测试，两组任务相关性 0.1768/0.2373，每种方法调参后独立训练 400 次；相关性更低的第二组 MMoE 主任务 AUC 最佳 0.8860、均值 0.8826，所有均值优于其他多任务模型。Google 排序：300 亿隐式反馈训练、100 万留出，Table 3 中 AUC@6M 0.6908（Shared-Bottom 0.6900）、R²@6M 0.09362（0.09287）；Table 4 线上 MMoE 相对 Shared-Bottom engagement +0.25%、satisfaction +2.65%。以上是论文系统口径，教程在 Census-income 官方切分与 KuaiRand 上重测的数值不可相减。"),
        "3_4_2_ple": ("ple", "原文工业数据：腾讯视频推荐 8 天日志采样，46.926M 用户、2.682M 视频、0.995B 样本，前 7 天训练。Table 1（复杂相关 VTR/VCR）：PLE VCR MSE 0.1307 最优、VTR AUC 0.6831，双任务 MTL gain 同时为正；MMoE 的 VCR gain 仅 +0.0001，ML-MMOE 升 VTR 损 VCR，即跷跷板。Table 3（4 周在线 A/B）：PLE 相对单任务总播放量 +4.17%、总观看时长 +3.57%（CGC +3.92%/+2.75%，MMoE +1.94%/+1.73%）；摘要另给相对 SOTA MTL +2.23%/+1.84% 的口径。Table 5：Census-income 两任务 0.9522/0.9945、Ali-CCP 0.6112/0.6097，均超单任务与 MMoE（MMoE 在 Ali-CCP CVR 上 gain −0.0302）；合成数据 PLE 平均 MTL gain 高 MMoE 87.2%。教程数值另行报告，不可相减。"),
        "4_2_openonerec_practice": ("onerec", "原文 Table 1（离线，指标由 reward 模型估计而非真实行为）：OneRec-1B+IPA 的 swt mean/max 为 0.1025/0.1933，OneRec-1B 为 0.0991/0.1529，TIGER-1B 为 0.0873/0.1368，SASRec 仅 0.0375/0.0803；IPA 优于 DPO/IPO/cDPO/rDPO/CPO/simPO/S-DPO 全部对比变体。Table 2（快手主页面 1% 流量在线 A/B）：OneRec-1B+IPA 总观看时长 +1.68%、平均观看时长 +6.56%（摘要约写作 1.6%）。实现细节：Semantic ID 3 层、每层 8192 簇，历史 n=256、session m=5，MoE 24 专家每 token 激活 2 个，beam size 128，推理仅 13% 参数激活；0.05B→1B 规模消融持续提升。以上是快手业务口径，教程在 KuaiRand 上复现 Semantic ID、trie 约束与列表指标契约，数值不可相减。"),
        "4_3_dlrm_hstu_practice": ("hstu", "原文 Table 4（公开数据，multi-pass full-shuffle 设定）：相对论文自跑的 SASRec，ML-1M HR@10 0.2853→0.3097（+8.6%，HSTU-large +15.5%），ML-20M +11.9%（large +22.8%），Books 0.0292→0.0404（+38.4%，large +60.6%），Books NDCG@10 最高 +65.8%。Table 5（工业单遍流式）消融显示换成 softmax 注意力或去掉相对偏置均退化；编码器与 FlashAttention-2 同配置对比，训练最高快 15.2 倍、推理快 5.6 倍（H100、bfloat16、序列 1024–8192）。Table 7（工业排序在线 A/B）：GR 相对 DLRM E-Task +12.4%、C-Task +4.4%；模型质量随训练算力幂律扩展，最大 1.5 万亿参数。以上是 Meta 系统口径，轻量实验只验证接口，数值不可相减。"),
    }

    # Keep the narrative close to the mathematics, while leaving the runnable
    # implementation in recsys_lab reusable and testable.  Each entry follows
    # the same chain: input tensors -> representation -> score -> loss.
    derivations = {
        "3_2_1_dssm": (r"""
### 结构：两条独立编码路径

先按原论文读图：底部是高维稀疏输入，中间是逐层变窄的非线性投影，顶部是 128 维语义向量；query 与多个 document 直到余弦相似度处才相遇。推荐迁移后，用户特征 $x_u$ 进入用户塔 $f_\theta$，物品特征 $x_i$ 进入物品塔 $g_\phi$。两座塔可以有不同输入，但最后都输出同样的 $d$ 维向量。

| 符号 | 含义 | 形状 |
|---|---|---|
| $B$ | 一次训练的用户数 | 标量 |
| $N$ | 参与比较的物品数 | 标量 |
| $d$ | 公共向量空间维度 | 标量 |
| $U$ | 一批用户向量 | $[B,d]$ |
| $V$ | 物品向量库 | $[N,d]$ |

因此 $UV^\top$ 的第 $(b,n)$ 个格子正好是第 b 个用户和第 n 个物品的点积，一次矩阵乘法得到 $[B,N]$ 个分数。

```text
x_u -> Embedding/MLP -> u --\
                              dot product -> score -> sampled loss
x_i -> Embedding/MLP -> v --/
```

### 从相似度到训练目标

向量长度 $\|u\|_2=\sqrt{u_1^2+\cdots+u_d^2}$。先做归一化 $\bar u=u/\|u\|_2,\ \bar v=v/\|v\|_2$，两支箭头长度都变成 1，于是 $s(u,i)=\bar u^\top\bar v$ 就等于余弦并落在 $[-1,1]$。对一个正物品 $i^+$ 和若干负物品 $i^-$，温度（temperature）$\tau$ 控制分布尖锐程度：

$$P(i^+|u)=\frac{\exp(s(u,i^+)/\tau)}{\sum_{j\in\{i^+,i^-\}}\exp(s(u,j)/\tau)},\qquad L=-\log P(i^+|u).$$

把 $-\log$ 想成“答错惩罚”：若正确物品概率为 0.9，损失约 0.105；若只有 0.1，损失约 2.303。正物品分数上升会增大分子占比并降低损失。独立物品塔使 $v$ 能离线计算并建立 ANN 索引；这一步是推荐工程迁移，不是 DSSM 论文的 ANN 实验。""", "`run_dssm` 负责构造正负对、调用 Torch-RecHub DSSM、反向传播并做全库 Top-K；数据加载与时间切分在源码讲解页逐函数展开。"),
        "3_2_2_mind": (r"""
### 结构：一段历史变成多个兴趣向量

先说人话：每条历史行为都可以给 K 个“兴趣小组”投票，而且不是非黑即白地只进一组。历史商品 embedding 组成 $H\in\mathbb R^{B\times L\times d}$：$B$ 是用户数，$L$ 是保留的历史长度，$d$ 是每件商品的坐标维度。输出是 $[B,K,d]$，表示每位用户有 K 支兴趣箭头。

| 符号 | 含义 | 形状 |
|---|---|---|
| $h_j$ | 第 $j$ 条历史行为向量 | $[d]$ |
| $b_{jk}$ | 行为 $j$ 投给兴趣 $k$ 的未归一化票数 | 标量 |
| $c_{jk}$ | 对固定 $j$ 经 softmax 后的路由权重 | 标量，$\sum_k c_{jk}=1$ |
| $s_k$ | 兴趣 $k$ 收到的行为加权和 | $[d]$ |
| $v_k$ | squash 后的第 $k$ 个兴趣向量 | $[d]$ |
| $e_i$ | 训练目标或候选物品向量 | $[d]$ |
| $a_k$ | 目标物品对兴趣 $k$ 的 Label-aware 权重 | 标量，$\sum_k a_k=1$ |

动态路由维护行为 $j$ 到兴趣胶囊 $k$ 的权重 $c_{jk}=\mathrm{softmax}_k(b_{jk})$。对固定行为 j，所有 $c_{jk}$ 的和为 1；先加权汇总 $s_k=\sum_jc_{jk}h_j$，再用 squash 保留方向并把长度限制在 0～1：

$$v_k=\frac{\|s_k\|^2}{1+\|s_k\|^2}\frac{s_k}{\|s_k\|},\qquad b_{jk}\leftarrow b_{jk}+h_j^\top v_k.$$

当 $\|s_k\|$ 很小时，前面的比例接近 0；有很多一致行为时，长度逐渐接近 1。更新式表示：行为 j 与兴趣 k 越同向，下一轮越愿意把票投给它。

训练时才使用目标商品做 **Label-aware Attention**。先算匹配分 $r_k=v_k^\top e_i$，再归一化：

$$a_k=\frac{\exp(r_k^p)}{\sum_{q=1}^{K}\exp(r_q^p)},\qquad v_u=\sum_{k=1}^{K}a_kv_k.$$

$p$ 控制选择有多“硬”：$p$ 增大时，最高分兴趣更接近独占监督；权重始终和为 1。若两个兴趣分数是 $(2,1)$、取 $p=1$，权重约为 $(0.731,0.269)$。选出训练用户向量后，对一个正物品 $i^+$ 和采到的负物品集合 $\mathcal N$ 优化 sampled-softmax：

$$L=-\log\frac{\exp(v_u^\top e_{i^+})}{\exp(v_u^\top e_{i^+})+\sum_{j\in\mathcal N}\exp(v_u^\top e_j)}.$$

正物品分数上升会增大分子占比，负物品分数上升会增大分母并受罚。这里的“label-aware”也划出边界：线上没有目标标签，不能先偷看候选来生成用户兴趣；服务时必须让 K 个兴趣分别 ANN，再合并、去重与限额。

```text
history [B,L,d] -> routing -> interests [B,K,d] -> K-way ANN -> merge
```""", "`run_mind` 把真实时间序列整理为定长张量；Torch-RecHub 的 CapsuleNetwork 完成路由，训练标签只用于选择兴趣而不会进入线上用户特征。"),
        "3_3_1_deepfm": (r"""
### 结构：Linear、FM 与 DNN 三路相加

一条样本是 m 个 field 的稀疏向量 $x$。field 就是特征分组：性别是一个 field，视频类别是一个 field，小时是一个 field。每个特征 $i$ 同时拥有一阶权重 $w_i$（标量）和隐向量 $V_i\in\mathbb R^k$。

最终 logit 分三路：一阶线性项记住单特征分量，FM 二阶项显式算特征对交互，DNN 从同一组 embedding 学高阶组合：

$$z=w_0+\sum_iw_ix_i+\sum_{i<j}\langle V_i,V_j\rangle x_ix_j+\mathrm{MLP}([V_ix_i]_i),\quad p=\sigma(z).$$

| 符号 | 含义 |
|---|---|
| $x_i$ | 特征 $i$ 的取值（one-hot 中为 0/1） |
| $w_i$ | 特征 $i$ 的一阶权重 |
| $V_i$ | 特征 $i$ 的 $k$ 维隐向量 |
| $\langle V_i,V_j\rangle$ | 内积 $\sum_f V_{i,f}V_{j,f}$，即二阶交互强度 |
| $\sigma$ | Sigmoid，把 logit 压成 0～1 的点击概率 |

### 为什么 FM 不用真的枚举特征对

直接算二阶项要枚举所有特征对：$n$ 个非零特征有 $n(n-1)/2$ 对，每对一次 $k$ 维内积，共 $O(n^2k)$。把内积按维度展开并交换求和顺序，可得恒等式

$$\sum_{i<j}\langle V_i,V_j\rangle x_ix_j=\frac12\sum_{f=1}^{k}\left[\left(\sum_i V_{i,f}x_i\right)^2-\sum_i (V_{i,f}x_i)^2\right],$$

右边每个维度 $f$ 只做 $n$ 次乘加，总共 $O(nk)$（实现按非零特征计，常写作 $O(nd)$，$d$ 即隐向量维度 $k$）。手算验证：取 $k=2$、两个特征 $V_1=(1,2)$、$V_2=(3,4)$、$x_1=x_2=1$。左边 $\langle V_1,V_2\rangle=1\times3+2\times4=11$；右边 $f=1$ 时 $(1+3)^2-(1^2+3^2)=16-10=6$，$f=2$ 时 $(2+4)^2-(2^2+4^2)=36-20=16$，$\tfrac12(6+16)=11$，两边相等。直觉：$(\sum_i a_i)^2$ 展开后包含所有两两乘积 $2a_ia_j$ 和平方项 $a_i^2$，减掉平方项再除以 2 正好留下 $i<j$ 的配对。

### 共享 embedding 与损失

共享 embedding 指同一组 $V_i$ 同时喂给 FM 分支和 DNN 分支：反向传播时，低阶交互和高阶组合的梯度落在同一组表示上，而不是各练一套互不相干的特征——论文表 2 里它因此优于使用独立 embedding 的 LR&DNN 与 FM&DNN。输出 $p=\sigma(z)$ 后用二元交叉熵 $L=-y\log p-(1-y)\log(1-p)$：$y=1$ 时只留 $-\log p$，预测越接近 1 损失越小。例：$y=1$、$p=0.9$ 时 $L\approx0.105$；$p=0.1$ 时 $L\approx2.303$。""", "`run_deepfm` 将 KuaiRand 的 user、video、场景和时段编码为 field，Torch-RecHub DeepFM 内部实现三路 logit，并在真实点击标签上优化 BCE。"),
        "3_3_2_din": (r"""
### 结构：候选先提问，历史再回答

记号：用户有 $L$ 条历史行为，第 $j$ 条的 embedding 是 $e_j\in\mathbb R^d$；当前候选（目标物品）的 embedding 是 $e_t\in\mathbb R^d$。一个 batch 的历史张量形状是 $[B,L,d]$：$B$ 个用户、每人 $L$ 条历史、每条 $d$ 维。

激活单元 $a(\cdot)$ 是一个小型全连接网络，输入是把四个向量拼成的 $4d$ 维向量 $[e_j,\ e_t,\ e_j-e_t,\ e_j\odot e_t]$，逐段含义：

- $e_j$：这条历史本身是什么；
- $e_t$：候选是什么；
- $e_j-e_t$：两者逐维差异，差的绝对值越小说明该维度上越接近；
- $e_j\odot e_t$：逐维乘积，两者同向的维度给出大的正值，是『共同激活』的显式信号。

手算例子：$d=2$，候选 $e_t=(1,0)$（想象成相机），历史 $e_1=(0.9,0.1)$（相关）与 $e_2=(0.1,0.9)$（不相关）。差异向量 $e_1-e_t=(-0.1,0.1)$、$e_2-e_t=(-0.9,0.9)$；逐维积 $e_1\odot e_t=(0.9,0)$、$e_2\odot e_t=(0.1,0)$。网络很容易从『差异小、乘积大』学出 $e_1$ 应得高权重。

输出相关分 $a_j=a(e_j,e_t)$ 后，必须区分论文与本教程实现的两种汇总：

$$\text{DIN 原论文：}\quad v_U(e_t)=\sum_{j=1}^{L}a_j e_j\quad\text{（不做 softmax）},$$
$$\text{Torch-RecHub 教学实现：}\quad \alpha_j=\frac{\exp(a_j)}{\sum_{j'}\exp(a_{j'})},\qquad v_U(e_t)=\sum_j\alpha_j e_j.$$

两者都让“哪个历史重要”依赖候选，但语义不同。原论文保留权重总量：面对 T 恤时，十条相关历史可以比一条相关历史产生更强激活；实现变体令权重和恒为 1，数值更稳定，却丢掉这部分强度。Notebook 的 NumPy 图演示的是 **softmax 实现变体**，不能倒写成原论文公式。张量链路都是 $[B,L,d]\to[B,L]\to[B,d]$。

### Dice：整流门槛随数据移动

ReLU 固定在 0 处截断，但 CTR 网络某一层的输入均值可能不是 0。Dice 先用当前 mini-batch 的均值 $\mu$ 和标准差 $\sigma$ 估计“高于常态”的概率，再在原斜率 1 与可学习斜率 $\alpha$ 之间切换：

$$p(s)=\operatorname{sigmoid}\!\left(\frac{s-\mu}{\sqrt{\sigma^2+\epsilon}}\right),\qquad \operatorname{Dice}(s)=p(s)s+[1-p(s)]\alpha s.$$

若 $s$ 明显高于本批均值，$p(s)\approx1$，输出接近 $s$；明显低于均值时输出接近 $\alpha s$。所以门槛跟随输入分布，而不是永远卡在 0。

### Mini-batch aware 正则：只惩罚本批真正查到的 embedding

完整 embedding 表可能有上亿行，每步遍历全表计算 $\ell_2$ 不现实。设第 $j$ 个特征域在当前 batch 激活的 ID 集合为 $S_j$，ID $s$ 的向量是 $w_j^{(s)}$，在全部训练样本中出现 $n_j^{(s)}$ 次；论文使用近似项

$$\Omega(W)=\lambda\sum_j\sum_{s\in S_j}\frac{\|w_j^{(s)}\|_2^2}{n_j^{(s)}}.$$

只读取本批出现的行，把计算量从“整张词表”降到“本批唯一 ID”；除以出现次数是为了避免高频 ID 因为被重复抽到而受到过强惩罚。最终 BCE 与正则相加。$v_U$ 再与候选、用户、上下文特征拼接进 MLP；每个候选都要重算权重，这既是 DIN 的表达力来源，也是在线成本。""", "`run_din` 严格在当前曝光之前构造历史；Torch-RecHub DIN 的 ActivationUnit 使用 `use_softmax=True`，因此源码映射明确标注它是归一化实现变体；Dice 与论文 MBA 正则是论文知识，是否启用必须以实际模型配置为准。"),
        "3_3_3_dien": (r"""
### 结构：GRU 抽取兴趣，AUGRU 按候选演化

GRU 是带两个『门』的递推网络，每个门都是 0～1 之间的数（Sigmoid 输出）。重置门 $r_t$ 决定计算候选新状态时保留多少旧记忆，更新门 $z_t$ 决定最终把多少新状态写进记忆：

$$r_t=\sigma(W_re_t+U_rh_{t-1}),\quad z_t=\sigma(W_ze_t+U_zh_{t-1}),$$
$$\tilde h_t=\tanh(W_he_t+U_h(r_t\odot h_{t-1})),\quad h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t.$$

其中 $e_t$ 是第 $t$ 步行为的 embedding，$h_{t-1}$ 是上一步记忆，$\odot$ 是逐维相乘，$W,U$ 是可学习权重。手算一步：设 $h_{t-1}=(0.8,0.2)$，当前行为算出 $z_t=0.25$、$\tilde h_t=(0.1,0.9)$，则

$$h_t=0.75\times(0.8,0.2)+0.25\times(0.1,0.9)=(0.625,0.325),$$

记忆向新兴趣方向挪了 25%。若某步 $z_t\approx0$，状态几乎不变——『门』的意义就是让网络自己学会哪一步该忘、哪一步该写。

辅助损失解决『中间状态没人教』的问题：主损失只看最后一次点击，长序列中第 $t$ 步的 $h_t$ 得不到监督。论文让每一步用 $h_t$ 区分下一次真实行为 $e_{t+1}^+$ 与负采样行为 $e_{t+1}^-$，即 $L_{aux}=-\sum_t[\log\sigma(h_t^\top e_{t+1}^+)+\log(1-\sigma(h_t^\top e_{t+1}^-))]$，总损失 $L=L_{target}+\alpha\cdot L_{aux}$（公开集实验 $\alpha=1$）。

AUGRU 再把 DIN 的候选注意力嵌进更新门：先对每步算与候选的相关分 $a_t$（softmax，和为 1），再缩放更新门 $\tilde z_t=a_t\,z_t$。于是与候选无关的历史 $a_t\approx0\Rightarrow\tilde z_t\approx0$，状态几乎不被改写，兴趣漂移被抑制；相关历史则正常参与演化。同样的历史集合换一种排列会得到不同的 $h_t$ 序列——这正是 DIEN 比 DIN 多出的『次序』能力。""", "`run_dien` 显式构造正历史、负历史和 mask；Torch-RecHub DIEN 同时返回主任务预测与辅助损失，训练代码展示两者如何相加。"),
        "3_4_1_mmoe": (r"""
### 结构：共享专家，任务各自选课

先定记号：输入 $x\in\mathbb R^m$ 是一条样本的 $m$ 维特征；$E$ 个专家网络 $f_1,\dots,f_E$ 结构相同、参数独立，各自输出 $d$ 维向量 $f_e(x)\in\mathbb R^d$；共 $K$ 个任务，任务 $k$ 有自己的门控 $g_k$ 与塔 $t_k$。下标 $e$ 始终表示专家编号，下标 $k$ 始终表示任务编号。

任务 $k$ 的门控是一个线性变换加 softmax：

$$g_k(x)=\mathrm{softmax}(W_{g,k}x),\qquad W_{g,k}\in\mathbb R^{E\times m},$$

即先算 $E$ 个 logit $(W_{g,k}x)_e$，再做 $g_{k,e}(x)=\dfrac{\exp((W_{g,k}x)_e)}{\sum_{e'}\exp((W_{g,k}x)_{e'})}$，得到的 $E$ 个权重和为 1。

手算一组：$E=2$，某条样本的 logits 为 $(2,0)$。$\exp(2)\approx7.389$，$\exp(0)=1$，权重为 $(7.389/8.389,\ 1/8.389)\approx(0.881,0.119)$。若两个专家输出 $f_1(x)=(1,0)$、$f_2(x)=(0,1)$，则任务表示

$$z_k=\sum_{e=1}^{E}g_{k,e}(x)f_e(x)=0.881\times(1,0)+0.119\times(0,1)=(0.881,\ 0.119),$$

近九成取自专家 1；另一条样本若 logits 为 $(0,2)$，配方整个反过来——同一组专家，不同任务、不同样本得到不同混合比例。这正是论文 Figure 6 在真实系统里观察到的门控分布差异。

最终预测再过任务自己的塔：$\hat y_k=\sigma(t_k(z_k))$，$\sigma$ 是 Sigmoid。形状链路：一个 batch 的专家输出堆成 $[B,E,d]$（$B$ 为批大小），task gate 输出 $[B,E]$，逐任务加权求和压成 $[B,d]$，塔输出 $[B,1]$。

总损失是逐任务损失的加权和 $L=\sum_k\lambda_kL_k$；$\lambda_k$ 不只是数学常数，也表达业务权衡（转化样本稀疏时调大 $\lambda_{\text{转化}}$）。注意梯度路径：任务 $k$ 的损失只直接训练自己的 gate 与塔，但所有任务的梯度都会流过共享专家——专家学到的是「按各任务 gate 加权的共同知识」。低相关任务在专家内部打架，正是这一结构的残留风险，也是下一节 PLE 的动机。""", "`run_mmoe` 用相同曝光行产生 click/long-view 两个真实标签；Torch-RecHub MMOE 的专家、gate、任务塔和逐任务 BCE 都能在源码映射中定位。"),
        "3_4_2_ple": (r"""
### 结构：共享专家之外，再给每个任务专属空间

MMoE 的每个 gate 面对所有专家；CGC（Customized Gate Control）先改「谁能被选」。第 $l$ 层有三组专家：任务 A 的专属专家 $m_A$ 个、任务 B 的专属专家 $m_B$ 个、所有任务共用的共享专家 $m_s$ 个。$E_k^{(l)}$ 表示任务 $k$ 在第 $l$ 层的专属专家集合，$E_s^{(l)}$ 表示共享专家集合。

任务 $k$ 的 gate 只在自己的专属专家和共享专家之间分配权重：

$$w_k^{(l)}(x)=\mathrm{softmax}(W_{g,k}^{(l)}x)\in\mathbb R^{m_k+m_s},\qquad z_k^{(l)}=\sum_{e\in E_k^{(l)}\cup E_s^{(l)}}w_{k,e}^{(l)}(x)\,f_e^{(l)}(x).$$

连通性一句话讲清：任务 A 的 gate 读「A 专属专家 + 共享专家」，任务 B 的 gate 读「B 专属专家 + 共享专家」；其它任务的专属专家被结构性排除，权重恒为 0，梯度不再互相污染。论文把允许被选的专家输出拼成选择矩阵 $S_k(x)$，门控输出写作 $g_k(x)=w_k(x)\,S_k(x)$（公式 2-4），再过任务塔 $y_k=t_k(g_k(x))$（公式 5）。

手算一组：$m_k=1, m_s=1$，任务 A 的 gate logits 为 $(1,0)$，softmax 权重为 $(e/(e+1),\ 1/(e+1))\approx(0.731,0.269)$。若 A 专属专家输出 $(2,0)$、共享专家输出 $(0,2)$，融合结果为 $0.731\times(2,0)+0.269\times(0,2)=(1.462,\ 0.538)$。若这是 MMoE，B 的专属专家输出（比如 $(0,-2)$）也会进入 A 的候选集，权重一旦非零就成为干扰；CGC 从结构上让它恒为 0。

PLE 把 CGC 堆成多层，称为 progressive extraction（渐进抽取）：第 $j$ 层任务 $k$ 的门控以第 $j-1$ 层的融合结果为输入，即 $g_{k,j}(x)=w_{k,j}(g_{k,j-1}(x))\,S_{k,j}(x)$；共享门控的选择矩阵包含本层全部专家，产出公共表示送给下一层。越往高层，共享知识与任务专属知识的边界越清晰——论文用化学提纯类比：低层先充分混合萃取，高层再逐步分离。最终仍是每任务 tower、Sigmoid 与 $L=\sum_k\lambda_kL_k$，PLE 论文另给 $\lambda_k$ 配动态更新 $\omega_k^{(t)}=\omega_{k,0}\,\gamma_k^t$，并用样本掩码处理「先点击才可能有评论」造成的样本空间错位。与 MMoE 的关键差异不在损失，而在「哪些专家允许被哪个 gate 读取」。""", "`run_ple` 保持与 MMoE 相同数据和目标，仅替换为 Torch-RecHub PLE；这样结果差异才主要来自 CGC 结构而非样本口径。"),
        "3_2_3_sasrec": (r"""
### 结构：只看过去的 Transformer

输入序列先按时间排序；过长保留最近 n 项，过短在左侧补 0。第 $t$ 个有效输入是商品与位置 embedding 之和 $x_t=e_{i_t}+p_t$：商品向量回答“是什么”，位置向量回答“在第几步”。把所有位置堆成 $X\in\mathbb R^{L\times d}$，线性投影得到同形状的 $Q=XW_Q,K=XW_K,V=XW_V$。

Q 是当前位置提出的问题，K 是历史位置的标签，V 是实际要汇总的内容。$QK^\top$ 的形状为 $[L,L]$，第 t 行列出位置 t 对所有历史位置的相关分：

$$A=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt d}+M\right),\qquad H=AV.$$

因果 mask $M_{t,j}$ 在 $j>t$ 时为 $-\infty$，因为 $\exp(-\infty)=0$，softmax 后未来权重严格变成 0。除以 $\sqrt d$ 是为了避免 d 个乘积累加后数值过大、softmax 过早只剩一个接近 1 的格子。$H=AV$ 是对过去内容的加权和。注意力后还有残差连接（保留原信息）、LayerNorm（稳定尺度）和逐位置前馈网络（加入非线性）。最后用位置 $t$ 的表示区分下一件正商品与负商品：

$$L_t=\mathrm{softplus}(-h_t^\top e^+)+\mathrm{softplus}(h_t^\top e^-).$$

softplus$(z)=\log(1+e^z)$ 始终为正。正物品点积越大，第一项越接近 0；负物品点积越小，第二项越接近 0。训练时所有位置可并行，推理时必须取最后有效位置而不是最后一个 padding，再做全库点积或 ANN。""", "`run_sasrec` 从真实时间戳生成序列、正目标和负目标；Torch-RecHub SASRec 实现 item/position embedding、causal attention 与 pairwise loss。"),
        "4_2_openonerec_practice": (r"""
### 结构：把物品变成可以“写”出来的 Semantic ID

先定记号：物品 $i$ 的内容 embedding（标题、标签等文本经预训练编码器得到）记为 $x_i$；量化器把它映射为 $m$ 级 token 元组 $s(i)=(c_1,\ldots,c_m)$，每个 $c_j$ 是第 $j$ 层码本里的一个编号。

RQ-VAE 的“残差”就是上一层还没解释掉的部分。令编码器输出 $z_i=E(x_i)$、初始残差 $r_i^{(0)}=z_i$；第 $\ell$ 层从码本 $C_\ell$ 选最近码字，再把它扣掉：

$$c_\ell=\arg\min_c\|r_i^{(\ell-1)}-C_\ell[c]\|_2^2,\qquad r_i^{(\ell)}=r_i^{(\ell-1)}-C_\ell[c_\ell],$$
$$\hat z_i=\sum_{\ell=1}^{m}C_\ell[c_\ell],\qquad s(i)=(c_1,\ldots,c_m).$$

例如 $z=(0.9,0.7)$，第一层选 $(0.8,0.4)$ 后残差是 $(0.1,0.3)$；第二层再选 $(0.1,0.25)$，重建为 $(0.9,0.65)$，只剩 $(0,0.05)$ 未解释。RQ-VAE 用重建误差让 $\hat z$ 接近 $z$，并用 codebook/commitment 项让编码器输出和码字彼此靠近；这些损失训练的是“如何编码物品”，不是推荐点击损失。

OneRec 指出普通残差量化可能形成 hourglass：少数码点挤满物品，多数闲置，热门前缀导致 trie 分支失衡。它改用逐层均衡 K-Means，并约束每个簇的分配数近似 $|V|/K$（论文 Algorithm 1，每层 8192 簇、共 3 层）。两种方法的**接口相同**：都把物品交给生成器一个固定长度 token 元组；内部目标不同：RQ-VAE 优先重建，均衡量化还显式控制 token 使用率。于是不能把“使用 Semantic ID”自动写成“运行了 RQ-VAE”。

| 符号 | 含义 |
|---|---|
| $H^u$ | 用户 $u$ 的历史行为序列（OneRec 取 $n=256$ 条） |
| $S$ | 一个 session：一次请求返回的列表（训练取 $m=5$ 个物品） |
| $s_i^j$ | 第 $i$ 个物品第 $j$ 层 Semantic ID token |
| $y_t$ | 生成序列第 $t$ 步的 token |
| $x$ | 编码器对用户历史的表示（条件上下文） |

### 链式分解：一个列表的概率是一串条件概率的乘积

自回归生成把“生成整个 session”分解成一步步“接龙”：

$$P(y|x)=\prod_{t=1}^{T}P(y_t\mid y_{<t},x),\qquad L_{CE}=-\sum_t\log P(y_t^*|y_{<t}^*,x).$$

手算一步：词表只有 $\{a,b,c\}$，某步模型给出 $P(a|x)=0.5,\ P(b|x)=0.3,\ P(c|x)=0.2$。若正确 token 是 $a$，该步损失 $-\log 0.5\approx0.693$；若模型把概率押错，比如 $P(a|x)=0.1$，损失变成 $2.303$，约 3.3 倍。整个列表的损失是各步相加，所以模型必须步步都把概率质量放在真实 token 上。再算一条两步链：$P(y_1=a|x)=0.5$、$P(y_2=b|y_1=a,x)=0.4$，则 $P(a,b|x)=0.5\times0.4=0.2$，链越长概率越小，$-\log$ 把它们变成可相加的损失。

teacher forcing 是训练时的“提示许可”：第 $t$ 步的输入不是模型自己上一步的输出，而是真实前缀 $y_{<t}^*$。好处是误差不会逐步放大；代价是训练与推理不一致——推理时模型只能看到自己刚生成的（可能出错的）前缀。

### trie：把“不许编造物品”变成查树

自回归模型可能生成目录里不存在的 token 组合。trie（前缀树）把全部合法 Semantic ID 存成一棵树：从根到叶子的一条路径就是一个真实物品。解码第 $t$ 步时，已生成前缀 $y_{<t}$ 定位到树上某个节点，logits 里只保留该节点的子节点，其余置 $-\infty$ 再 softmax。

走一遍小例子：目录里有 4 个物品，Semantic ID 为 $(1,3)$、$(1,4)$、$(2,3)$、$(2,5)$。根节点的合法后继是 $\{1,2\}$；若第一步生成 $1$，第二步只允许 $\{3,4\}$——即使模型给 $5$ 打了 0.9 的概率也会被屏蔽。约束生成因此保证非法率为 0，但合法不等于相关：模型仍要把最高的合法概率给对物品。

### DPO：只拉开“好列表”和“差列表”的差距

偏好对齐阶段不教模型“什么是标准答案”，只教它“哪个列表更好”。记号：$y^+$ 是 chosen（reward 模型打分最高的采样列表），$y^-$ 是 rejected（打分最低），$\pi$ 是当前模型，$\pi_{ref}$ 是冻结的参考模型（防止跑偏），$\beta$ 控制偏离参考模型的惩罚强度：

$$L_{DPO}=-\log\sigma\{\beta[(\log\pi(y^+|x)-\log\pi_{ref}(y^+|x))-(\log\pi(y^-|x)-\log\pi_{ref}(y^-|x))]\}.$$

直觉来自大括号的符号：它是“chosen 相对参考模型的对数概率增量”减去“rejected 的”。若 chosen 被抬高 $+0.3$、rejected 变化 $-0.1$，差值为 $0.4$，$\sigma(0.4)\approx0.60$，损失 $-\log0.60\approx0.51$；若方向反了，差值 $-0.4$，损失 $-\log0.40\approx0.92$。梯度只做一件事——把 chosen 与 rejected 的差距拉大。OneRec 的 IPA 把差距来源换成“自己 beam search 采样 128 个列表、reward 模型打分、最好当 chosen 最差当 rejected”，每轮用新模型重新采样，且只用 1% 训练数据做对齐（论文 Figure 4 显示比例提到 5% 收益有限，GPU 成本却线性上升）。

### Reward 的证据边界

reward 模型 $R(x,y)$ 只是用历史反馈训练出的代理评分器，不是真实用户满意度，也不是线上因果收益。IPA 在模型自己采样的 128 个列表中按 $R$ 选 best/worst；DPO 随后只学习这对相对顺序，既不会直接优化真实观看时长，也无法修复 reward 模型没见过的偏差。trie 又只保证“目录合法”，不保证相关、多样或公平。因此三层验收必须分开：token 交叉熵检验模仿数据，reward/DPO 检验代理偏好，真正业务收益只能由同口径离线标签和线上 A/B 验证。smoke 实验只覆盖第一层与约束接口。""", "`run_openonerec` 展开教学 Semantic ID、teacher forcing 与 trie 解码；它没有训练官方 RQ-VAE/均衡量化器或 reward 模型。完整 OpenOneRec 配置留在官方框架，教程明确区分接口、代理 reward 和线上效果。"),
        "4_3_dlrm_hstu_practice": (r"""
### 结构：一个模块替代 DLRM 的三类拼装件

DLRM 通常把特征抽取（embedding 查表）、特征交互（FM/DCN）、表示变换（MLP/MoE）拼成三段。HSTU 的主张是：把所有特征序列化成统一流 $X\in\mathbb R^{B\times L\times d}$：$B$ 个用户、每人最多 $L$ 个事件、每个事件 $d$ 维，然后反复堆叠同一种模块。论文 Eq.1/2/3 给出每个 HSTU 层的三个子层：

$$U,Q,K,V=\mathrm{Split}(\phi_1(f_1(X))),$$
$$A(X)V(X)=\phi_2\!\left(QK^\top+rab^{p,t}\right)V,$$
$$Y=f_2\!\left(\mathrm{Norm}(A(X)V(X))\odot U\right).$$

| 符号 | 含义 | 多头形状 |
|---|---|---|
| $f_1,f_2$ | 线性投影（论文为省算力各用一层） | 输入输出见上下文 |
| $\phi_1,\phi_2$ | 非线性激活，均取 SiLU | 不改变形状 |
| $Q,K$ | 每个位置提出的问题、供匹配的键；最后一维做点积 | $[B,L,H,d_q]$ |
| $V$ | 匹配后真正汇总的内容 | $[B,L,H,d_v]$ |
| $U$ | 与汇总结果逐元素相乘的门（类似 SwiGLU） | $[B,L,H,d_v]$ |
| $rab^{p,t}$ | 相对位置与时间偏置 | $[B,H,L,L]$（可广播） |
| $\mathrm{Norm}$ | 逐点聚合后稳定数值的 LayerNorm | 不改变形状 |
| $\odot$ | 逐元素相乘 | 两侧同形状 |

对每个 head，把 $q_{t,h}$ 与 $k_{j,h}$ 的 $d_q$ 个对应坐标相乘再求和，得到 $QK^\top\in\mathbb R^{B\times H\times L\times L}$；第 $(t,j)$ 格是位置 $t$ 对历史位置 $j$ 的内容相关分。它乘 $V$ 后回到 $[B,L,H,d_v]$，再与同形状的 $U$ 逐元素相乘、拼回 head 并经 $f_2$ 回到 $[B,L,d]$。

相对偏置可以读成“内容分之外的时间提示”。例如两次都看同一类别，内容点积一样；昨天发生的行为可获得比一年前行为更高的 $r_{\Delta p,\Delta t}$。它只依赖两个事件相隔多少位置、多少时间，不记绝对日期，所以同一条“刚看完就继续看”的规律能平移到任何一天。

### pointwise aggregated attention：不除总和的注意力

标准 softmax 注意力先把每行分数归一化成和为 1 再加权 $V$。HSTU 的 pointwise aggregated attention 直接对 $QK^\top+rab^{p,t}$ 施加 SiLU 后乘 $V$，不做整行归一化。

手算对比：设某位置对三条历史的相关分为 $(2,1,0.1)$。softmax 得 $(e^2,e^1,e^{0.1})/(e^2+e^1+e^{0.1})\approx(7.39,2.72,1.11)/11.22\approx(0.66,0.24,0.10)$——三条历史被迫瓜分固定为 1 的权重。SiLU$(z)=z\sigma(z)$ 则给出 $(2\times0.881,\ 1\times0.731,\ 0.1\times0.525)\approx(1.76,0.73,0.05)$：强相关历史的贡献不被“预算”压扁，“有几条相关历史”这个强度信号得以保留——论文指出这对预测观看时长这类强度目标至关重要。代价是数值不再自动有界，所以后面必须接 LayerNorm。SiLU 还有个顺手性质：负分数被压到接近 0（SiLU$(-4)\approx-0.07$），相当于软阈值。

### 因果约束、next-item 分数与交叉熵

$j\le t$ 的因果 mask 保证位置 $t$ 只能聚合不晚于 $t$ 的历史（$j>t$ 的格子被掩掉）。设位置 $t$ 的输出为 $h_t\in\mathbb R^d$，物品 $i$ 的输出 embedding 为 $e_i\in\mathbb R^d$，则下一物品分数和概率是

$$z_{t,i}=h_t^\top e_i+b_i,\qquad P(i\mid i_{\le t})=\frac{\exp z_{t,i}}{\sum_{j\in\mathcal V}\exp z_{t,j}},$$
$$L_{\text{next}}=-\sum_{t\ \text{有效}}\log P(i_{t+1}\mid i_{\le t}).$$

若某步正确物品概率从 0.2 提到 0.8，该步惩罚就从 $-\log0.2\approx1.609$ 降到 $-\log0.8\approx0.223$。一次前向会产生 logits `[B,L,|V|]`；padding 位置不计损失，推理只取最后有效位置的 `[B,|V|]` 做 Top-K。工业词表过大时可以使用 sampled-softmax，但候选采样必须写进协议；本教程小模型直接对教学词表做完整交叉熵。

HSTU 的论文级差异不在 next-item 目标，而在统一序列化特征、相对位置/时间偏置与不归一化的逐点聚合注意力。论文另配生成式训练采样（按 $1/n_i$ 采样用户把成本降一个 $O(N)$ 因子）与 M-FALCON 微批推理，属于工业配置，教学 Notebook 不展开。教程默认在 CUDA 上运行 Torch-RecHub 模型并启用混合精度；CPU basic smoke 只保留核心张量契约。""", "`run_hstu` 从 KuaiRand 真实 feed 流生成 next-item 样本并直接计算教学词表交叉熵；它不是 Meta full 数据/采样协议，实现映射会明确两者边界。"),
    }

    specs = {}
    for meta in algorithms:
        cuda_first = meta["slug"].startswith("4_")
        run_expression = (
            f"{meta['function']}(cpu_smoke=not torch.cuda.is_available())"
            if cuda_first else f"{meta['function']}()"
        )
        keys = [meta["primary"], meta["secondary"]] + ([meta["baseline"]] if meta["baseline"] else [])
        metrics_literal = "{" + ", ".join(f"{key!r}: result[{key!r}]" for key in keys) + "}"
        baseline_line = (
            f"- 对照指标 {meta['baseline']} = **{{result[{meta['baseline']!r}]:.4f}}**。"
            if meta["baseline"] else
            "- 本节没有把不同任务的数值伪装成 baseline；章节总结只做同口径比较。"
        )
        derivation, implementation_map = derivations[meta["slug"]]
        paper_id, findings = paper_evidence[meta["slug"]]
        protocol = PROTOCOLS.get(meta["slug"])
        protocol_literal = json.dumps(protocol or {}, ensure_ascii=False)
        protocol_text = (
            f"**正式数据：** {protocol['dataset']}  \n**资源 ID：** `{protocol['resource']}`  \n"
            f"**切分：** {protocol['split']}  \n**指标：** {', '.join(protocol['metrics'])}  \n"
            f"**与论文比较边界：** {protocol.get('paper_comparison', '按配置中的 paper_targets 逐指标对照')}"
            if protocol else "本节暂无可公开取得且与论文完全等价的数据；报告只验证代码契约，不生成伪复现结论。"
        )
        cells = [
            md(f"""## 学习地图

1. 从原始论文理解系统约束；
2. 用可手算数字读懂公式和形状；
3. 检查数据、切分与标签；
4. 使用工业框架模型类训练；
5. 分开验证训练、推理和测试；
6. 用实际输出讨论失败边界。

**本节问题：** {meta['problem']}

**阅读约定：** 通用数学通过 3.0 基础课程链接回看；本页只详细推导论文引入或改造的数学。第一次阅读先追踪输入、输出和形状，再看梯度。"""),
            md(f"""## Paper & Context

{meta['paper']}

**来源：** {meta['source']}

### 原文实验设计与关键结论

{findings}

请区分三层证据：论文中的离线实验、本 Notebook 验证的代码链路、生产系统尚需验证的在线收益。三者不能互相替代。"""),
            md(f"""## Reproduction Contract

{protocol_text}

`full` 只有在运行输出证明数据、切分、候选集、模型配置与指标均对齐时，才可能进入论文数值比较；它不是把教学适配器自动变成论文复现的开关。`smoke` 只做张量、损失和推理链路回归。"""),
            md(f"""## Model Structure & Formula Walkthrough

{figure_markdown(paper_id)}
{derivation}

### 公式到代码

{implementation_map}

阅读源码时按“张量形状 → 前向计算 → score → loss → metric”五步追踪，不需要一次读完整个工程文件。"""),
            md(f"""## Math by Hand

{prerequisite_markdown(meta['slug'])}

{meta['math']}

下面用 NumPy/Matplotlib 验证直觉。二维图只是教学投影，工业 embedding 虽有更多维，计算规则相同。"""),
            code(meta["hand"]),
            md(f"""## Data

### 权威 full 协议（效果验收目标）

{protocol_text}

### smoke 教学适配器（默认 runner 实际读取）

{meta['smoke_dataset']}

下方运行结果打印的 provenance 才是本次执行事实；若资源、统计或切分与 full 协议不一致，就必须标记为不可比较。

**防泄漏清单：**按时间切分；item 映射只表达已知目录，不读取测试标签；低评分或未点击负反馈均来自数据中的已观察行；序列只看预测时刻以前；测试集只在最后评价。{'无 CUDA 时的 CPU basic smoke 只验证基本功能，不产出精度结论。' if cuda_first else 'CPU 档使用真实数据的确定性子集，**不是统一 benchmark 成绩**。'}"""),
            md(f"""## Model & Framework

{meta['framework']}

smoke 档强调模型类、张量契约和指标链路真实可运行；full 档应替换原始数据、分布式配置、索引/服务和资源预算，而不是只增加 epoch。{' 本节默认使用 CUDA、混合精度与 TF32；没有 CUDA 时只运行缩小后的 CPU basic smoke，结果不进入完整精度结论。' if cuda_first else ''}"""),
            code(f"""import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
from importlib import import_module
from recsys_lab.runtime import save_records

# 算法实现就在当前章节目录，不再通过公共模块隐藏。
chapter_train = import_module("chapter_code.{meta['slug']}.train")
{meta['function']} = chapter_train.{meta['function']}

print("实际执行函数源码（包含数据、训练、推理和测试）：")
print(inspect.getsource({meta['function']}))"""),
            md("## Train & Inference\n\n下一格固定 seed、构造数据、实例化模型、训练并进入推理路径。生成式章节在 CUDA 上执行完整评测；CPU 环境只验证缩小后的基本张量与约束链路。"),
            code(f"""result = {run_expression}
print({{'framework': result['framework'], 'dataset': result.get('dataset', {{}}),
       'device': result.get('device'), 'validation_mode': result.get('validation_mode')}})
print('inference contract:', {meta['inference']!r})
assert np.isfinite(result['loss_curve']).all()
print('loss:', round(result['loss_curve'][0],4), '→', round(result['loss_curve'][-1],4))"""),
            code(f"""fig,axes=plt.subplots(1,2,figsize=(10.5,3.5))
axes[0].plot(result['loss_curve'],color='#4f8f00',lw=2); axes[0].set(title='Training loss',xlabel='epoch',ylabel='loss'); axes[0].grid(alpha=.2)
metrics={metrics_literal}
axes[1].bar(range(len(metrics)),list(metrics.values()),color=['#7ca832','#e1874b','#6d88a4'][:len(metrics)])
axes[1].set_xticks(range(len(metrics)),list(metrics),rotation=18); axes[1].set(title='Executed test metrics',ylim=(0,max(1.0,max(metrics.values())*1.15)))
for index,value in enumerate(metrics.values()): axes[1].text(index,value,f'{{value:.3f}}',ha='center',va='bottom')
plt.tight_layout(); plt.show(); display(pd.Series(metrics,name='value').to_frame())"""),
            code(f"""# 论文数字只能在数据、切分、候选和指标全部同口径时相减。
paper_protocol = json.loads({protocol_literal!r})
paper_targets = paper_protocol.get('paper_targets', {{}})
metric_key = {{'HitRate@10':'paper_protocol_hr@10', 'NDCG@10':'paper_protocol_ndcg@10',
              'AUC':'auc', 'LogLoss':'logloss'}}
dataset_name = result.get('dataset', {{}}).get('dataset', '')
dataset_aligned = paper_protocol.get('dataset', '').split(',')[0].casefold() in dataset_name.casefold()
comparison_eligible = PROFILE == 'full' and dataset_aligned
rows=[]
for paper_metric,target in paper_targets.items():
    result_key=metric_key.get(paper_metric)
    value=result.get(result_key) if result_key else None
    rows.append({{'metric':paper_metric,'tutorial':value,'paper':target,
                 'absolute_gap':None if value is None or not comparison_eligible else float(value)-float(target),
                 'comparable':comparison_eligible and value is not None}})
if rows:
    display(pd.DataFrame(rows))
    if not comparison_eligible:
        print('NOT COMPARABLE：当前运行的数据/协议与论文不完全一致，不计算复现差值。')
else:
    print('论文没有可公开、可同口径复现的绝对目标；本节只报告结构与公开协议验证。')"""),
            md("## Test & Results Discussion"),
            code(f"""display(Markdown(f'''### 本次已执行结果

- 主指标 {meta['primary']} = **{{result[{meta['primary']!r}]:.4f}}**。
- 辅助指标 {meta['secondary']} = **{{result[{meta['secondary']!r}]:.4f}}**。
{baseline_line}
- 训练损失从 **{{result['loss_curve'][0]:.4f}}** 降到 **{{result['loss_curve'][-1]:.4f}}**。损失下降只说明优化工作，不等于泛化或业务收益。
- **结果解释：** {meta['caveat']}

### 工业边界

{meta['inference']}

上线前还需验证延迟、吞吐、内存/显存、数据新鲜度、校准、回滚和线上 A/B。
'''))"""),
            code(f"""record={{
    'algorithm': {meta['title'].split(' ',1)[1]!r},
    'primary_metric': {meta['primary']!r}, 'primary_value': float(result[{meta['primary']!r}]),
    'secondary_metric': {meta['secondary']!r}, 'secondary_value': float(result[{meta['secondary']!r}]),
    'baseline_metric': {meta['baseline']!r},
    'baseline_value': float(result[{meta['baseline']!r}]) if {bool(meta['baseline'])} else None,
    'framework': result['framework'], 'source_notebook': {meta['slug']!r},
    'validation_mode': result.get('validation_mode', 'standard'),
    'dataset': result['dataset']['dataset'],
    'randomly_fabricated_rows': int(result['dataset']['randomly_fabricated_rows'])
}}
path=save_records({meta['chapter']!r},{meta['slug']!r},[record]); print('saved:',path.relative_to(ARTIFACT_ROOT))"""),
            md("## Checks\n\n自动断言用于防止数据、训练和指标链路静默失效，不是效果证明。"),
            code(f"""assert result['loss_curve'][-1] < result['loss_curve'][0]
assert 0 <= float(result[{meta['primary']!r}]) <= 1
assert np.isfinite(float(result[{meta['secondary']!r}]))
print('PASS：数据、训练、推理、测试和结果产物均已验证。')"""),
            md("## Next Steps\n\n1. 换成对应公开数据的完整时间切分；2. 增加强 baseline 与消融；3. 记录效果、延迟和成本；4. 映射到 TorchEasyRec/官方 full profile；5. 只在相同候选与数据口径下比较。"),
        ]
        specs[meta["slug"]] = notebook(meta["title"], meta["problem"], meta["source"], cells)

    summaries = [
        ("3_2_summary", "3.2", "DSSM、MIND 与 SASRec 召回", "chapter_3_2", 3,
         "DSSM 用单向量换取简单 ANN；MIND 用多向量覆盖并行意图；SASRec 用因果注意力刻画意图随顺序的变化。三者都可输出用户向量并接全库检索，Recall 必须与 Coverage、索引成本和序列延迟一起看。", "DSSM · MIND · SASRec 原始论文"),
        ("3_3_summary", "3.3", "DeepFM、DIN 与 DIEN 排序", "chapter_3_3", 3,
         "DeepFM 处理静态 field 交互；DIN 让历史随候选变化；DIEN 再利用次序。复杂模型不保证胜出，AUC、LogLoss、GAUC 与 P99 必须同时评估。", "DeepFM · DIN · DIEN 原始论文"),
        ("3_4_summary", "3.4", "MMoE 与 PLE 多目标", "chapter_3_4", 2,
         "MMoE 为任务选择共享专家；PLE 再隔离任务专属专家。必须逐任务报告，避免平均指标掩盖跷跷板，并检查专家利用率与负迁移。", "MMoE · PLE 原始论文"),
        ("4_1_generative_overview", "4.1", "生成式召回、排序与召排融合", "chapter_4", 2,
         "OpenOneRec 关注 Semantic ID、列表解码与奖励对齐；HSTU 关注统一长序列。共同验收维度是相关性、合法性、重复率、目录更新、解码延迟与 GPU ROI。", "TIGER · OpenOneRec · HSTU 论文及官方仓库"),
    ]
    paper_audits = {
        "3_2_summary": [
            {
                "algorithm": "DSSM",
                "source_notebook": "3_2_1_dssm",
                "paper_result": "NDCG@10=0.498",
                "paper_protocol": "商业搜索日志训练约 1 亿 query-title 对；16,510 个查询、每个约 15 个带人工相关性标签的文档",
                "verdict": "不可直接比较：任务、候选集和指标均不同；教程仅验证双塔检索链路。",
            },
            {
                "algorithm": "MIND",
                "source_notebook": "3_2_2_mind",
                "paper_result": "Amazon Books HR@10=0.0309",
                "paper_protocol": "351,356 用户、393,801 商品、6,271,511 样本；随机 19:1；K=3、d=36",
                "verdict": "只有 full 档数据统计可对齐；教程时间切分比论文随机目标更严格，smoke 不参与复现。",
            },
            {
                "algorithm": "SASRec",
                "source_notebook": "3_2_3_sasrec",
                "paper_result": "ML-1M HR@10=0.8245；NDCG@10=0.5905",
                "paper_protocol": "完整读取后做 user/item 5-core；6,040 用户、3,416 物品；leave-two-out；1 真值 + 100 未观察负例；2 个自注意力块、ML-1M 序列长度 200（其余数据集 50）、d 从 {10,…,50} 调参（消融用 d=50）、默认单头（双头略差）、验证集 20 轮无提升早停",
                "verdict": "协议最接近论文；只有 full profile 的最终产物可比较，smoke 不覆盖效果结论。",
            },
        ],
        "3_3_summary": [
            {
                "algorithm": "DeepFM",
                "source_notebook": "3_3_1_deepfm",
                "paper_result": "Criteo AUC=0.8007；LogLoss=0.45083",
                "paper_protocol": "Criteo 约 4,500 万条点击记录、13 连续 + 26 离散特征，随机 9:1；400-400-400 MLP、dropout 0.5、FM 维度 10；Company* 约 10 亿条记录、7 天训练 + 次日测试",
                "verdict": "不可直接比较：数据集、特征口径与切分完全不同；教程数值来自 results JSON，只验证共享 embedding 的三路结构可训练。",
            },
            {
                "algorithm": "DIN",
                "source_notebook": "3_3_2_din",
                "paper_result": "Amazon Electronics AUC=0.8818（Dice 版 0.8871）；Alibaba AUC=0.6083（MBA+Dice）",
                "paper_protocol": "Amazon 192,403 用户、1,689,188 样本；阿里约 20 亿训练样本；用户加权 AUC（GAUC）；线上 A/B CTR +10.0%、RPM +3.8%",
                "verdict": "论文用 GAUC 且候选为广告；教程在 KuaiRand 曝光上测量，数值不可相减，只验证 target-aware 路径可运行。",
            },
            {
                "algorithm": "DIEN",
                "source_notebook": "3_3_3_dien",
                "paper_result": "Electronics AUC=0.7792；Books AUC=0.8453；工业 AUC=0.6541",
                "paper_protocol": "公开集 192,403/603,668 用户、各重复 5 次；工业 70 亿样本、历史截断 50；在线 A/B CTR +20.7%、eCPM +17.1%",
                "verdict": "不可跨数据集比较绝对值；教程只验证 GRU+辅助损失+AUGRU 链路可运行。",
            },
        ],
        "3_4_summary": [
            {
                "algorithm": "MMoE",
                "source_notebook": "3_4_1_mmoe",
                "paper_result": "Census-income 低相关组主任务 AUC best=0.8860、mean=0.8826；Google engagement AUC@6M=0.6908、R²=0.09362",
                "paper_protocol": "合成数据 200 次重复控制任务相关性；Census-income 299,285 条、199,523 训练 + 99,762 测试、每法 400 次重复；Google 300 亿隐式反馈训练、100 万留出；线上相对 Shared-Bottom engagement +0.25%、satisfaction +2.65%",
                "verdict": "full 档使用同一 Census-income 官方切分，但目标构造与特征处理为教学复刻，只有完全对齐的字段才可对照；Google 与线上数字为论文系统口径，smoke/KuaiRand 数值不相减。",
            },
            {
                "algorithm": "PLE",
                "source_notebook": "3_4_2_ple",
                "paper_result": "VTR AUC=0.6831、VCR MSE=0.1307（双任务 MTL gain 同时为正）；在线总播放量 +4.17%、总观看时长 +3.57%",
                "paper_protocol": "腾讯 8 天日志采样：46.926M 用户、2.682M 视频、0.995B 样本，前 7 天训练；4 周在线 A/B 相对单任务模型；公开集 Census-income 0.9522/0.9945、Ali-CCP 0.6112/0.6097",
                "verdict": "工业数据与在线数字为腾讯单一系统口径；教程只验证 CGC 结构与逐任务 MTL gain 评估链路，数值不能与论文相减。",
            },
        ],
    }
    paper_relationships = {
        "4_1_generative_overview": [
            {
                "paper": "TIGER (2023)",
                "starts_from": "embedding+ANN 检索无法利用物品的层次语义，随机 ID 对新物品无泛化",
                "user_representation": "用户历史 = Semantic ID token 序列；物品 = RQ-VAE 多级码，相似物品共享前缀",
                "training_signal": "seq2seq 对下一物品 Semantic ID 的逐 token 交叉熵（teacher forcing）",
                "serving_shape": "自回归解码 + 目录约束，替代 ANN 召回",
                "hands_to_next": "只替代召回一级，排序仍靠级联；逐点生成后要靠人工规则拼列表",
            },
            {
                "paper": "OneRec / OpenOneRec (2025)",
                "starts_from": "级联召排各级独立优化、目标不一致，前级上限即后级天花板",
                "user_representation": "256 条历史行为编码 → decoder 自回归生成 5 物品 session",
                "training_signal": "session 级 NTP 交叉熵 + IPA：自采 128 个列表、RM 打分、best/worst 做 DPO（仅 1% 数据）",
                "serving_shape": "单模型端到端生成整屏列表；KV cache + beam 128 + 13% MoE 激活",
                "hands_to_next": "离线指标由 reward 模型估计而非真实行为；互动指标仍弱；1B 规模与 beam 解码成本高",
            },
            {
                "paper": "HSTU (2024)",
                "starts_from": "DLRM 依赖手工特征拼装且不随算力扩展；softmax 归一化抹掉相关历史强度",
                "user_representation": "全部特征序列化为统一流，HSTU 层逐位置输出表示",
                "training_signal": "生成式 next-item 交叉熵（按 1/n_i 采样用户，成本降一个 O(N) 因子）",
                "serving_shape": "长序列统一转导 + M-FALCON 微批推理；285x FLOPs 换 1.50x/2.99x QPS",
                "hands_to_next": "依赖 Meta 级流式基础设施与算力；公开集 multi-pass 评测与线上单遍流式口径不同",
            },
        ],
        "3_2_summary": [
            {
                "paper": "DSSM (2013)",
                "starts_from": "query 与 document 的词汇不一致",
                "user_representation": "原文是 query/document 各 1 个向量；推荐迁移为 user/item 双塔",
                "training_signal": "点击文档 vs 4 个随机未点击文档的条件概率",
                "serving_shape": "两侧独立编码；ANN 是后续推荐迁移",
                "hands_to_next": "单向量可检索，但会平均多个兴趣",
            },
            {
                "paper": "MIND (2019)",
                "starts_from": "单向量混合服饰、运动、食品等并行兴趣",
                "user_representation": "K 个动态路由兴趣胶囊",
                "training_signal": "目标物品选择兴趣 + sampled softmax",
                "serving_shape": "K 路 ANN 后合并去重",
                "hands_to_next": "表达多意图，但不显式利用行为先后顺序",
            },
            {
                "paper": "SASRec (2018)",
                "starts_from": "最近转移与长期偏好需要随数据密度自适应",
                "user_representation": "因果自注意力产生的最后有效位置向量",
                "training_signal": "每个位置的真实下一项 vs 未观察负项",
                "serving_shape": "1 路全库点积/ANN",
                "hands_to_next": "利用顺序，但长序列注意力成本为 O(n²d)",
            },
        ],
        "3_3_summary": [
            {
                "paper": "DeepFM (2017)",
                "starts_from": "Wide & Deep 的 wide 侧依赖人工交叉特征",
                "user_representation": "静态 field embedding：一阶线性 + FM 二阶 + DNN 高阶三路相加",
                "training_signal": "曝光样本的点击/未点击 BCE",
                "serving_shape": "查 embedding 后 FM 与 DNN 并行打分",
                "hands_to_next": "只有静态交互；用户表示不含行为历史、与候选无关",
            },
            {
                "paper": "DIN (2018)",
                "starts_from": "固定长度用户向量表达不下多样兴趣",
                "user_representation": "候选感知的历史加权和（一个候选一个向量）",
                "training_signal": "BCE + mini-batch aware 正则 + Dice 激活",
                "serving_shape": "每个候选重算历史注意力再进 MLP",
                "hands_to_next": "利用相关性但忽略行为先后次序；行为被直接当作兴趣",
            },
            {
                "paper": "DIEN (2019)",
                "starts_from": "行为不等于兴趣，且兴趣随时间演化、会漂移",
                "user_representation": "GRU 兴趣状态序列 + 候选感知 AUGRU 演化末状态",
                "training_signal": "BCE + 逐步辅助损失（下一真实行为 vs 负采样）",
                "serving_shape": "串行 GRU/AUGRU，需 kernel 融合与压缩降延迟",
                "hands_to_next": "串行结构限制长序列；长期兴趣留存与多峰演化仍开放",
            },
        ],
        "3_4_summary": [
            {
                "paper": "MMoE (2018)",
                "starts_from": "Shared-Bottom 对任务相关性敏感：合成数据上相关性越低，共享底层损失越差（负迁移）",
                "user_representation": "任务表示 = 任务专属 gate 对全部共享专家输出的 softmax 加权和",
                "training_signal": "逐任务损失的静态加权和 Σλ_k·L_k",
                "serving_shape": "一次前向算出所有专家，每任务 gate + tower 输出",
                "hands_to_next": "所有专家被所有任务无差别共享；复杂相关任务上仍有跷跷板（PLE 论文测得其 VCR gain 仅 +0.0001）",
            },
            {
                "paper": "PLE (2020)",
                "starts_from": "MMoE 共享专家不区分共享/专属知识，工业系统上跷跷板现象仍在",
                "user_representation": "CGC：任务 gate 只读本任务专属专家 + 共享专家；多层渐进分离路由",
                "training_signal": "带样本空间掩码的逐任务损失 + 动态损失权重 ω_k,0·γ_k^t",
                "serving_shape": "多层专家逐层前向，各任务 tower 输出",
                "hands_to_next": "专家数/层数/动态权重超参增多；分层任务组相关性建模仍开放",
            },
        ],
    }
    future_maps = {
        "4_1_generative_overview": [
            ["物品表示", "RQ-VAE 层次 Semantic ID", "均衡 K-Means 残差量化（每层 8192 簇）", "统一特征序列化，无显式 ID", "多模态内容与行为联合、可增量更新的标识"],
            ["生成单位", "下一物品的 ID token", "整屏 session 列表", "任意未来事件序列", "列表级多样性/新颖性的显式建模"],
            ["训练信号", "逐 token 交叉熵", "NTP + 自采样迭代 DPO", "生成式 next-item + 流式采样", "在线反馈闭环、多目标对齐与去偏"],
            ["服务形态", "受约束自回归解码替代 ANN", "单模型端到端 + beam 128", "统一转导 + M-FALCON 微批", "模型与索引/推理内核联合设计、目录增量更新"],
            ["必须补测", "非法 ID 率 + 冷启动泛化", "合法率/重复率 + RM 估计偏差", "长序列成本 + 流式口径一致性", "P99 延迟、GPU ROI、线上因果收益与公平性"],
        ],
        "3_2_summary": [
            ["表示数量", "单向量", "多兴趣向量", "按上下文动态表示", "可变数量兴趣 + 时序融合"],
            ["序列范围", "无显式次序", "行为集合软聚类", "最近 n 项因果注意力", "长序列稀疏/分层注意力与状态缓存"],
            ["训练信号", "随机未点击文档", "sampled softmax", "逐位置正负 next-item", "去偏负采样、自监督预训练、多行为目标"],
            ["检索系统", "后续迁移到单路 ANN", "K 路 ANN 合并", "序列用户向量 + ANN", "模型与索引联合设计、增量更新"],
            ["必须补测", "Recall + 索引延迟", "覆盖/去重/每兴趣配额", "全库指标 + 长度/P99", "新鲜度、公平性、多样性、成本与线上因果收益"],
        ],
        "3_3_summary": [
            ["交互对象", "静态 field 低阶/高阶交互", "历史行为 × 候选交互", "兴趣状态 × 候选逐步交互", "多模态内容、上下文与图的统一交互"],
            ["用户表示", "无显式用户向量（field 拼接）", "候选感知单向量", "候选感知演化状态", "长期兴趣记忆 + 多兴趣演化"],
            ["训练信号", "点击 BCE", "BCE + MBA 正则 + Dice", "BCE + 序列辅助损失", "去偏、多行为目标、自监督预训练"],
            ["服务成本", "一次查表 + 两路前向", "每候选重算注意力", "串行 GRU，108→32 维压缩", "长序列稀疏化、状态缓存、蒸馏"],
            ["必须补测", "AUC + LogLoss + 校准", "GAUC + 候选批量延迟", "在线 A/B + P99 延迟", "公平性、生态长期收益与成本 ROI"],
        ],
        "3_4_summary": [
            ["共享粒度", "全量共享专家 + 任务 gate", "共享/专属专家显式分离 + 渐进路由", "按样本与任务阶段动态决定共享多少"],
            ["任务冲突处理", "gate 权重隐式调节任务差异", "结构性切断其它任务专属专家 + MTL gain 逐任务审计", "任务自动分组、冲突检测与结构搜索"],
            ["训练信号", "逐任务损失静态加权", "样本空间掩码 + 动态损失权重", "不确定性加权、梯度手术与公平性约束"],
            ["服务成本", "专家共享一次前向，gate 轻量", "多层专家堆叠，参数与延迟上升", "top-k 稀疏 gate、专家并行与蒸馏"],
            ["必须补测", "逐任务指标 + 专家利用率", "跷跷板审计 + 在线 A/B", "长期生态指标、新任务冷启动迁移与 ROI"],
        ],
    }
    relationship_intros = {
        "4_1_generative_overview": "三篇论文不是同一赛道上的升级排名，而是把“生成”逐层推深：TIGER 先回答“物品怎么变成可生成的标识”，OneRec 回答“生成能不能统一召回与排序并对齐偏好”，HSTU 回答“整个推荐能不能变成随算力扩展的序列转导”。下表中的“交接问题”比发表年份更重要。",
        "3_2_summary": "三篇论文不是同一赛道上的简单升级排名。DSSM 提供“可独立编码并比较”的结构起点；MIND 沿着**表示数量**扩展，把一个用户拆成多个并行兴趣；SASRec 沿着**时间顺序**扩展，让当前表示随历史位置变化。下表中的“交接问题”比发表年份更重要。",
        "3_3_summary": "三篇论文沿同一条 CTR 排序线各自补一块短板：DeepFM 解决“静态特征交互要不要手工”，DIN 解决“用户表示与候选无关”，DIEN 解决“历史没有次序、行为不等于兴趣”。下表中的“交接问题”比发表年份更重要。",
        "3_4_summary": "两篇论文沿同一条多任务结构线：MMoE 解决“共享底层对任务关系太敏感”，PLE 解决“共享专家仍被无差别共享、跷跷板仍在”。下表中的“交接问题”比发表年份更重要。",
    }
    future_columns = {
        "4_1_generative_overview": ["dimension", "TIGER line", "OneRec line", "HSTU line", "next questions"],
        "3_2_summary": ["dimension", "DSSM line", "MIND line", "SASRec line", "next questions"],
        "3_3_summary": ["dimension", "DeepFM line", "DIN line", "DIEN line", "next questions"],
        "3_4_summary": ["dimension", "MMoE line", "PLE line", "next questions"],
    }
    for slug, number, title, directory, expected, interpretation, sources in summaries:
        if slug in paper_relationships:
            relationship_cells = [
                md(f"""## 论文关联关系：后一篇在修补前一篇的哪块短板？

{relationship_intros[slug]}"""),
                code(f"""import pandas as pd
paper_relationships = pd.DataFrame({paper_relationships[slug]!r})
display(paper_relationships)"""),
            ]
        else:
            relationship_cells = [
                md("## Paper Evidence Map\n\n本页直接列出本章论文证据与对应 Notebook，便于在总结中核对来源；各算法页负责逐页 PDF 导读。"),
                code(f"""from app.evidence import load_evidence, _chapter_items
evidence = load_evidence()
rows = []
for item in _chapter_items(evidence.get({slug!r})):
    rows.append({{'paper': item['paper_id'], 'page': item['page'], 'keyword': item['keyword'], 'quote': item['quote'][:120] + '...'}})
display(pd.DataFrame(rows))"""),
            ]
        audit_cells = []
        if slug in paper_audits:
            audit_code = f"""paper_audit = pd.DataFrame({paper_audits[slug]!r})
display(paper_audit)
print('结论：当前结果是可执行的 smoke/教学实验，不是 paper reproduction。')"""
            if slug in ("3_2_summary", "3_3_summary", "3_4_summary"):
                audit_code = f"""paper_audit = pd.DataFrame({paper_audits[slug]!r})
live = comparison.copy()
live['tutorial_result'] = live.apply(
    lambda row: f"{{row.primary_metric}}={{row.primary_value:.4f}}；{{row.secondary_metric}}={{row.secondary_value:.4f}}", axis=1
)
paper_audit = paper_audit.merge(live[['source_notebook', 'tutorial_result']], on='source_notebook', how='left')
display(paper_audit[['algorithm', 'tutorial_result', 'paper_result', 'paper_protocol', 'verdict']])
print('结论：教程数值来自本次 results JSON；smoke/迁移实验不是 paper reproduction。')"""
            audit_cells = [
                md("""## 与原论文的可比性审计

下表逐项核对本教程实际产物与论文表格。**“不可直接比较”不是回避差距**：候选集合、样本规模、切分和指标任一不同，数值相减就没有统计含义。这里同时指出当前实验能证明什么、不能证明什么。"""),
                code(audit_code),
            ]
        future_cells = []
        if slug in future_maps:
            future_cells = [
                md("""## 未来发展：沿着什么约束继续前进？

未来路线不是一味堆更深网络，而是逐项解除当前系统约束。表格从左到右给出本章已经走过的变化和仍待验证的方向；最后一列是研究/工程问题，不是预告一定会获得线上提升。"""),
                code(f"""future = pd.DataFrame({future_maps[slug]!r}, columns={future_columns[slug]!r})
display(future)"""),
            ]
        specs[slug] = notebook(
            f"{number} 总结：{title}",
            "读取本章节每个独立算法 Notebook 的实际结果产物，在统一口径下比较和选型。",
            sources,
            [
                md(f"""## 开篇回顾

本章节固定为 **开篇导读 → 独立算法教程 → 结果总结**。每篇算法 Notebook 都包含论文、数学、数据、训练、推理、测试与讨论；本页不重新训练，也不手填数字。

{interpretation}"""),
                *relationship_cells,
                md("## Results\n\n读取 results 目录。若缺文件，请先按章节顺序执行算法 Notebook。"),
                code(f"""import json
import pandas as pd
import matplotlib.pyplot as plt
result_dir=ARTIFACT_ROOT/'results'/{directory!r}; files=sorted(result_dir.glob('*.json'))
assert len(files)=={expected},f'期望 {expected} 个结果，实际 {{[p.name for p in files]}}'
records=[]
for path in files: records.extend(json.loads(path.read_text(encoding='utf-8'))['records'])
comparison=pd.DataFrame(records); display(comparison.round(4)); print('数据来源:',[p.name for p in files])"""),
                code("""from matplotlib import font_manager

# Matplotlib 默认的 DejaVu Sans 不包含中文字形。优先选择容器中安装的
# Noto CJK；在精简宿主机上找不到中文字体时，退回纯 ASCII 的 Notebook 编号，
# 从根源避免 missing glyph 警告，而不是用 warnings.filterwarnings 隐藏它。
cjk_candidates = ('Noto Sans CJK SC', 'Noto Sans CJK JP', 'Microsoft YaHei', 'SimHei', 'PingFang SC')
installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
cjk_font = next((name for name in cjk_candidates if name in installed_fonts), None)
if cjk_font:
    plt.rcParams['font.sans-serif'] = [cjk_font, 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
chart_labels = comparison.algorithm if cjk_font else comparison.source_notebook
print('图表字体:', cjk_font or 'ASCII fallback（宿主机未安装 CJK 字体）')

fig,ax=plt.subplots(figsize=(max(7,len(comparison)*1.5),3.8))
bars=ax.bar(chart_labels,comparison.primary_value,color='#7ca832')
ax.set(title='Primary metrics from executed notebooks',ylabel='metric value',ylim=(0,max(1.0,comparison.primary_value.max()*1.18)))
ax.tick_params(axis='x',rotation=12)
for bar,value in zip(bars,comparison.primary_value): ax.text(bar.get_x()+bar.get_width()/2,value,f'{value:.3f}',ha='center',va='bottom')
plt.tight_layout(); plt.show()"""),
                *audit_cells,
                *future_cells,
                md(f"""## Takeaways

{interpretation}

先固定业务阶段和候选口径，再比较主指标、辅助指标、baseline 与系统成本。smoke 数值用于代码回归和学习，不能跨数据或跨公司宣称优劣。"""),
                md("## Checks"),
                code(f"""assert len(comparison)=={expected}
assert comparison.source_notebook.nunique()=={expected}
assert comparison.primary_value.between(0,1).all()
print('PASS：总结完全来自独立 Notebook 的执行产物。')"""),
                md("## Next Steps\n\n在相同完整数据、时间切分、负样本和候选集上重跑；加入效果—延迟—成本三维表，再决定是否进入线上 A/B。"),
            ],
        )
    return specs
