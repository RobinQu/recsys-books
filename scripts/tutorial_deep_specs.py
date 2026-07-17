from __future__ import annotations


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
            "paper": "DSSM 最初面向搜索点击，把 query 与 document 映射到同一语义空间。推荐系统继承的是“两侧独立编码 + 相似度学习”这一可检索结构。双塔牺牲候选前的深交互，换取 item embedding 可预计算和 ANN 检索。",
            "math": r"""用户塔输出 $u=f_\theta(x_u)\in\mathbb R^d$，物品塔输出 $v=g_\phi(x_i)\in\mathbb R^d$。归一化后，$s(u,i)=u^\top v$ 就是余弦相似度。形状是用户批次 $[B,d]$ 乘物品库 $[N,d]^\top$，得到 $[B,N]$ 分数。线上只算一次用户向量，再从索引找 Top-K。""",
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
            "dataset": "Amazon Reviews 2023 的 Video Games 5-core 真实电商行为。用户、商品、评分与毫秒时间戳来自 McAuley Lab 官方文件；`rating>=4` 为正反馈，并按时间留出目标。",
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
            "paper": "MIND 用动态路由把行为序列聚成多个兴趣胶囊，训练时用 label-aware attention 选择与目标物品最相关的兴趣。关键变化是单用户单向量变成单用户多向量；代价是多路 ANN、去重和兴趣数选择。",
            "math": r"""设用户有 $K$ 个兴趣向量 $V_u=[v_1,\ldots,v_K]$。候选物品 $e_i$ 的分数取 $\max_k v_k^\top e_i$。可先把历史点分成两团，各自平均得到两个质心；若强行平均成一个点，它可能落在两团之间，反而不像任何真实兴趣。""",
            "hand": """import numpy as np, matplotlib.pyplot as plt
history=np.array([[1,.1],[.9,.2],[.1,1],[.2,.9]])
a,b,single=history[:2].mean(0),history[2:].mean(0),history.mean(0)
fig,ax=plt.subplots(figsize=(5.5,4.4)); ax.scatter(history[:,0],history[:,1],s=90,label='history')
ax.scatter(*a,s=180,marker='*',label='interest 1'); ax.scatter(*b,s=180,marker='*',label='interest 2')
ax.scatter(*single,s=120,marker='x',label='single average'); ax.set(title='Two interests avoid mixed intent',xlabel='topic A',ylabel='topic B')
ax.legend(); ax.grid(alpha=.2); plt.show()
print({'interest_1':a.tolist(),'interest_2':b.tolist(),'single':single.tolist()})""",
            "dataset": "Amazon Reviews 2023 Video Games 真实序列。选择该数据是因为 MIND 原论文使用 Amazon Books 与天猫日志；新版 Amazon 数据保留多商品兴趣和长序列，并把时间覆盖更新到 2023 年。",
            "framework": "实际使用 torch_rechub.models.matching.MIND，执行 CapsuleNetwork、目标兴趣选择和多兴趣推理；full profile 映射 TorchEasyRec MIND。",
            "primary": "recall@10", "secondary": "positive_top1", "baseline": None,
            "inference": "user tower 输出 [B,K,d]；每个兴趣独立检索，再按最高分合并、去重和配额控制。",
            "caveat": "商品兴趣数不固定，胶囊仍可能重复或塌缩；确定性 CPU 切片用于比较兴趣多样性，不等同原论文的 35 万 Amazon Books 用户规模。",
        },
        {
            "slug": "3_3_1_deepfm",
            "title": "3.3.1 DeepFM 排序",
            "chapter": "chapter_3_3",
            "function": "run_deepfm",
            "source": "[Guo et al., 2017, DeepFM](https://arxiv.org/abs/1703.04247)",
            "problem": "如何在不手工枚举所有组合的情况下，同时学习一阶、二阶和高阶稀疏特征交互？",
            "paper": "DeepFM 让 FM 与 DNN 共享 embedding：FM 显式承担低阶交互，DNN 隐式组合高阶模式。相对 Wide&Deep，它减少宽侧人工交叉，但没有自动解决序列、曝光偏差或概率校准。",
            "math": r"""FM 二阶项是 $\sum_{i<j}\langle v_i,v_j\rangle x_ix_j$；Deep 分支把同一组 embedding 展平送入 MLP。最终 logit 为 linear + FM + DNN，再经 Sigmoid 得点击概率。当前样本只有非零 field 参与交互。""",
            "hand": """import numpy as np, matplotlib.pyplot as plt
embedding=np.array([[1.,0.],[.8,.2],[.2,.9]]); pair=embedding@embedding.T
fig,ax=plt.subplots(figsize=(4.8,4)); image=ax.imshow(pair,cmap='YlGn',vmin=0,vmax=1)
ax.set_xticks(range(3),['user','item','hour']); ax.set_yticks(range(3),['user','item','hour'])
for i in range(3):
    for j in range(3): ax.text(j,i,f'{pair[i,j]:.2f}',ha='center',va='center')
ax.set_title('FM interaction = embedding dot product'); plt.colorbar(image,ax=ax); plt.show()
print('sum of three pair interactions =',round(pair[0,1]+pair[0,2]+pair[1,2],3))""",
            "dataset": "KuaiRand-Pure 真实短视频曝光：user、video、场景 tab、hour、视频时长桶为特征，标签直接使用日志中的 `is_click`，按时间切分。",
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
            "paper": "DIN 不再先把历史压成与候选无关的固定向量，而用 activation unit 比较每条历史与目标商品。贡献是 target-aware representation；代价是每个候选都要重新读取历史。",
            "math": r"""注意力分数 $a_j=g(e_j,e_t,e_j-e_t,e_j\odot e_t)$，用户兴趣为 $v=\sum_j a_je_j$。候选像问题，历史像资料：先给相关资料更高权重，再做加权平均。权重是针对当前候选临时计算的。""",
            "hand": """import numpy as np, matplotlib.pyplot as plt
history=np.array([[1.,0.],[.8,.2],[0.,1.],[.1,.9]]); targets=[np.array([1.,0.]),np.array([0.,1.])]
fig,axes=plt.subplots(1,2,figsize=(9,3.4))
for ax,target,name in zip(axes,targets,['camera candidate','running candidate']):
    raw=history@target; weight=np.exp(raw)/np.exp(raw).sum(); ax.bar(range(4),weight,color='#7ca832')
    ax.set(ylim=(0,.4),title=name,xlabel='history position',ylabel='attention weight')
plt.tight_layout(); plt.show(); print('same history, different candidate -> different weights')""",
            "dataset": "KuaiRand-Pure 的真实 feed impression：候选是当前视频，历史只含此前真实点击，标签直接读取 `is_click`；未点击曝光进入 observed negative history。",
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
            "paper": "DIEN 用 GRU 抽取逐时刻兴趣状态，以辅助下一行为损失提供额外监督，再用目标感知 AUGRU 控制状态演化。它比 DIN 更重，收益依赖严格时间顺序和高质量负序列。",
            "math": r"""GRU 是带记忆的递推函数 $h_t=\mathrm{GRU}(e_t,h_{t-1})$。辅助损失要求 $h_t$ 更像下一次真实行为、远离负样本。AUGRU 用候选相关权重控制每步写入多少。因此两个相同集合的不同排列会得到不同末状态。""",
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
            "dataset": "KuaiRand-Pure 真实时间序列：点击视频进入正序列，已曝光未点击视频进入 negative_history；padding=0，未来曝光不进入历史。",
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
            "paper": "MMoE 让所有任务共享一组专家，但为每个任务学习独立 gate。它比 hard sharing 灵活，却仍可能发生专家塌缩、任务梯度冲突和样本空间错位。",
            "math": r"""第 $k$ 个任务表示为 $z_k=\sum_e g_{k,e}(x)f_e(x)$。gate 权重经 softmax 后和为 1。专家像擅长不同题型的老师：各任务听同一组老师，但自行决定每位老师占多少。""",
            "hand": """import numpy as np, matplotlib.pyplot as plt
weights=np.array([[.65,.25,.10],[.15,.30,.55]]); fig,ax=plt.subplots(figsize=(6,3.3)); left=np.zeros(2)
for expert in range(3):
    ax.barh(['click','conversion'],weights[:,expert],left=left,label=f'expert {expert+1}'); left+=weights[:,expert]
ax.set(xlim=(0,1),title='Task gates mix shared experts'); ax.legend(ncol=3,loc='lower center'); plt.show()
print('gate sums =',weights.sum(1))""",
            "dataset": "KuaiRand-Pure 真实短视频曝光及场景、时段、时长、热度和活跃度特征；两个目标直接使用日志中的 `is_click` 与 `long_view`。",
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
            "paper": "PLE 每层同时维护共享专家和任务专属专家，用 CGC 门控逐层抽取。它给“不该共享的知识”明确位置，目标是缓解负迁移与跷跷板；代价是结构与调参更复杂。",
            "math": r"""任务 gate 只从共享专家和自己的专属专家中选择；共享 gate 汇总所有专家供下一层使用。层层重复后，共享信息逐步提纯。直觉上像公共课和专业课：都听公共课，但不被迫共享全部专业细节。""",
            "hand": """import numpy as np, matplotlib.pyplot as plt
matrix=np.array([[1,1,0],[1,0,1]],dtype=float); fig,ax=plt.subplots(figsize=(5.5,2.8)); image=ax.imshow(matrix,cmap='YlGn')
ax.set_xticks(range(3),['shared','click-only','convert-only']); ax.set_yticks(range(2),['click gate','convert gate'])
for i in range(2):
    for j in range(3): ax.text(j,i,int(matrix[i,j]),ha='center',va='center')
ax.set_title('PLE: shared plus task-specific experts'); plt.colorbar(image,ax=ax,ticks=[0,1]); plt.show()""",
            "dataset": "与 MMoE 完全相同的 KuaiRand-Pure 真实曝光行、时间切分和 `is_click`/`long_view` 目标，保证横向口径一致。",
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
            "paper": "SASRec 用单向 Transformer 编码按时间排列的行为，并在每个位置预测下一物品。它在稀疏数据上可以聚焦少数近期行为，在较密数据上也能利用更长依赖；相对 RNN，训练可并行，但注意力成本随序列长度平方增长。",
            "math": r"""把 item embedding 与位置 embedding 相加得到 $X$。单头注意力为 $\mathrm{softmax}(QK^\top/\sqrt d)V$，其中因果 mask 把未来位置设为 $-\infty$。位置 $t$ 的表示只能读取 $i_1,\ldots,i_t$，再与正/负物品向量做点积并优化 pairwise logistic loss。""",
            "hand": "import numpy as np, matplotlib.pyplot as plt\nlength=6\nlogits=np.fromfunction(lambda row,col: 2.0-np.abs(row-col)*.55,(length,length))\nlogits[np.triu(np.ones((length,length),dtype=bool),1)]=-np.inf\nweights=np.exp(logits-np.max(logits,axis=1,keepdims=True)); weights/=weights.sum(1,keepdims=True)\nfig,ax=plt.subplots(figsize=(5.4,4.2)); image=ax.imshow(weights,cmap='YlGn'); ax.set(title='SASRec causal attention',xlabel='history position',ylabel='prediction position'); plt.colorbar(image,ax=ax); plt.show()\nprint('row sums =',weights.sum(1).round(3))",
            "dataset": "Amazon Reviews 2023 Video Games 5-core 的真实商品序列。每位用户按毫秒 timestamp 排序，倒数一项测试；负序列来自真实低评分商品。",
            "framework": "实际使用 torch_rechub.models.matching.SASRec，包含 item/position embedding、causal multi-head attention 与 pairwise loss；线上可分离用户表示和 item 向量。",
            "primary": "hr@10", "secondary": "popularity_hr@10", "baseline": None,
            "inference": "截取最近 L 个真实行为 → 因果 Transformer → 最后位置用户向量 → 全库点积/ANN Top-K；屏蔽已见物品并监控序列截断与 P99。",
            "caveat": "CPU 切片只保留最活跃的 500 位用户，目录仍稀疏；SASRec 未必超过热门基线，结果用于验证 2023 电商序列与严格时间协议。",
        },
        {
            "slug": "4_2_openonerec_practice",
            "title": "4.2 OpenOneRec：受约束列表生成实战",
            "chapter": "chapter_4",
            "function": "run_openonerec",
            "source": "[Kuaishou OpenOneRec](https://github.com/Kuaishou-OneRec/OpenOneRec)",
            "problem": "如何把推荐列表变成 token 序列，同时保证解码结果属于真实目录、没有重复并可接受奖励对齐？",
            "paper": "OpenOneRec 展示列表生成、奖励模型和偏好优化的开放流程。smoke 档不冒充官方大模型训练，而是复现最关键的数据契约：item 到 Semantic ID、teacher forcing、trie 约束和合法性压力测试。",
            "math": r"""自回归分解 $P(y_1,\ldots,y_T|x)=\prod_tP(y_t|y_{<t},x)$，表示每一步根据上下文和已生成 token 预测下一 token。trie 像目录树：前缀 $(1,2)$ 下只允许真实后继 $\{3,4\}$。""",
            "hand": """import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(7,3)); ax.axis('off'); ax.text(.05,.5,'root',bbox=dict(boxstyle='round',fc='#e8f2d6'))
for y,node in zip([.25,.5,.75],['1','2','3']):
    ax.text(.32,y,node,bbox=dict(boxstyle='round',fc='white')); ax.plot([.13,.30],[.5,y],color='#809070')
ax.text(.58,.35,'prefix (1,2)',bbox=dict(boxstyle='round',fc='#ffe8d7')); ax.text(.88,.25,'3'); ax.text(.88,.48,'4')
ax.plot([.72,.86],[.38,.27]); ax.plot([.72,.86],[.38,.49]); ax.set_title('Catalog trie allows valid next tokens'); plt.show()""",
            "dataset": "KuaiRand-Pure 的真实视频 tag、music type 与 item 分区形成可约束代码；真实点击/长播和未点击曝光构成训练及 chosen/rejected。官方 RecIF-Bench 当前为 gated 数据，full profile 需授权访问。",
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
            "paper": "HSTU 针对高基数、非平稳推荐序列重新设计 attention 与系统实现。本节用 Torch-RecHub HSTUModel 跑 next-item smoke；full profile 对齐 Meta DLRM-v3、TorchRec/FBGEMM 和 M-FALCON。",
            "math": r"""给定 $[i_1,\ldots,i_t]$，模型在每个位置预测下一 item。因果 mask 保证位置 $t$ 只能看不晚于 $t$ 的历史。训练把所有位置交叉熵相加；推理只取最后位置 logits 做 Top-K。""",
            "hand": """import numpy as np, matplotlib.pyplot as plt
length=7; mask=np.tril(np.ones((length,length))); fig,ax=plt.subplots(figsize=(5,4)); image=ax.imshow(mask,cmap='YlGn',vmin=0,vmax=1)
ax.set(title='Causal mask: only past is visible',xlabel='history position',ylabel='prediction position')
ax.set_xticks(range(length)); ax.set_yticks(range(length)); plt.colorbar(image,ax=ax,ticks=[0,1]); plt.show(); print(mask.astype(int))""",
            "dataset": "KuaiRand-Pure 的真实短视频点击流按毫秒时间排序；倒数行为仅用于测试，训练只学习更早的 next-video 转移，更贴近 HSTU 面向非平稳行为流的设计目标。",
            "framework": "实际使用 torch_rechub.models.generative.HSTUModel 完成 next-item 训练；工业档切换 Meta DLRM-v3、TorchRec/FBGEMM 和多 GPU。",
            "primary": "hr@5", "secondary": "popularity_hr@5", "baseline": None,
            "inference": "最后位置 logits → Top-K 或 M-FALCON；服务需缓存状态、过滤非法/已见 item，并监控吞吐、显存和目录新鲜度。",
            "caveat": "真实小数据上的 HR 可能低于热门基线；这比可学习的人工序列更诚实，需通过严格时间切分、强 baseline 和成本收益共同判断。",
        },
    ]

    # Keep the narrative close to the mathematics, while leaving the runnable
    # implementation in recsys_lab reusable and testable.  Each entry follows
    # the same chain: input tensors -> representation -> score -> loss.
    derivations = {
        "3_2_1_dssm": (r"""
### 结构：两条独立编码路径

用户特征 $x_u$ 进入用户塔 $f_\theta$，物品特征 $x_i$ 进入物品塔 $g_\phi$。两座塔可以有不同输入，但最后都输出 $d$ 维向量。一个批次中，用户张量形状是 $[B,d]$，物品库是 $[N,d]$；矩阵乘法一次得到 $[B,N]$ 个候选分数。

```text
x_u -> Embedding/MLP -> u --\
                              dot product -> score -> sampled loss
x_i -> Embedding/MLP -> v --/
```

### 从相似度到训练目标

先做归一化 $\bar u=u/\|u\|_2,\ \bar v=v/\|v\|_2$，于是 $s(u,i)=\bar u^\top\bar v$ 落在 $[-1,1]$。对一个正物品 $i^+$ 和若干负物品 $i^-$，温度（temperature）$\tau$ 控制分布尖锐程度：

$$P(i^+|u)=\frac{\exp(s(u,i^+)/\tau)}{\sum_{j\in\{i^+,i^-\}}\exp(s(u,j)/\tau)},\qquad L=-\log P(i^+|u).$$

当正物品分数上升，分子占比变大，损失就下降。独立物品塔使 $v$ 能离线计算并建立 ANN 索引，这正是模型结构与召回效率之间的连接。""", "`run_dssm` 负责构造正负对、调用 Torch-RecHub DSSM、反向传播并做全库 Top-K；数据加载与时间切分在源码讲解页逐函数展开。"),
        "3_2_2_mind": (r"""
### 结构：一段历史变成多个兴趣向量

历史商品 embedding 组成 $H\in\mathbb R^{B\times L\times d}$。动态路由维护行为 $j$ 到兴趣胶囊 $k$ 的权重 $c_{jk}=\mathrm{softmax}_k(b_{jk})$，先加权汇总 $s_k=\sum_jc_{jk}h_j$，再用 squash 保留方向并限制长度：

$$v_k=\frac{\|s_k\|^2}{1+\|s_k\|^2}\frac{s_k}{\|s_k\|},\qquad b_{jk}\leftarrow b_{jk}+h_j^\top v_k.$$

相似的行为会逐轮增加对同一胶囊的赞成票。训练时目标商品 $e_i$ 选择最相关兴趣，常写成 $a_k\propto\exp((v_k^\top e_i)^p)$，再令 $v_u=\sum_ka_kv_k$ 并使用 sampled-softmax。推理时不知道目标，因此让 $K$ 个兴趣分别 ANN 检索后合并去重。

```text
history [B,L,d] -> routing -> interests [B,K,d] -> K-way ANN -> merge
```""", "`run_mind` 把真实时间序列整理为定长张量；Torch-RecHub 的 CapsuleNetwork 完成路由，训练标签只用于选择兴趣而不会进入线上用户特征。"),
        "3_3_1_deepfm": (r"""
### 结构：Linear、FM 与 DNN 三路相加

对稀疏输入 $x$，一阶分支记忆单特征贡献；FM 分支显式计算二阶交互；DNN 分支从共享 embedding 学高阶组合：

$$z=w_0+\sum_iw_ix_i+\sum_{i<j}\langle v_i,v_j\rangle x_ix_j+\mathrm{MLP}([v_ix_i]_i),\quad p=\sigma(z).$$

直接枚举二阶组合要 $O(n^2d)$。展开平方后可化成

$$\frac12\sum_f\left[\left(\sum_i v_{i,f}x_i\right)^2-\sum_i(v_{i,f}x_i)^2\right],$$

只需 $O(nd)$。最后用二元交叉熵 $L=-y\log p-(1-y)\log(1-p)$。共享 embedding 意味着低阶和高阶分支共同修改同一组表示，而不是训练两套互不相干的特征。""", "`run_deepfm` 将 KuaiRand 的 user、video、场景和时段编码为 field，Torch-RecHub DeepFM 内部实现三路 logit，并在真实点击标签上优化 BCE。"),
        "3_3_2_din": (r"""
### 结构：候选先提问，历史再回答

每条历史 $e_j$ 与目标 $e_t$ 组成 $[e_j,e_t,e_j-e_t,e_j\odot e_t]$。小型网络输出相关分数

$$a_j=g(e_j,e_t,e_j-e_t,e_j\odot e_t),\qquad v=\sum_j\alpha_je_j.$$

这里 $e_j-e_t$ 表示两者差异，$e_j\odot e_t$ 表示逐维共同激活。实现可对 $a_j$ 做 softmax 得到和为 1 的 $\alpha_j$；原论文也强调权重不必被解释为概率。兴趣向量 $v$ 与候选、用户和上下文拼接，经 MLP 得到 logit，再用 Sigmoid 与 BCE 训练。张量从历史 $[B,L,d]$ 经过逐位置打分变成 $[B,L]$ 权重，最终压缩为 $[B,d]$。""", "`run_din` 严格在当前曝光之前构造历史；Torch-RecHub DIN 的 ActivationUnit 对每个候选重新计算权重，所以代码同时展示 mask 与序列长度。"),
        "3_3_3_dien": (r"""
### 结构：GRU 抽取兴趣，AUGRU 按候选演化

GRU 用两个 0～1 的门控制记忆。重置门决定计算新内容时保留多少过去，更新门决定最终写入多少：

$$r_t=\sigma(W_re_t+U_rh_{t-1}),\quad z_t=\sigma(W_ze_t+U_zh_{t-1}),$$
$$\tilde h_t=\tanh(W_he_t+U_h(r_t\odot h_{t-1})),\quad h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t.$$

辅助损失让 $h_t$ 区分下一次真实行为 $e_{t+1}^+$ 与负行为 $e_{t+1}^-$，为每个时间步提供监督。AUGRU 再用候选注意力 $a_t$ 缩放更新门 $\tilde z_t=a_tz_t$：与候选无关的历史几乎不改状态。最后状态进入 MLP 和 BCE。顺序不同会产生不同 $h_t$，这是它相对 DIN 加入的新能力。""", "`run_dien` 显式构造正历史、负历史和 mask；Torch-RecHub DIEN 同时返回主任务预测与辅助损失，训练代码展示两者如何相加。"),
        "3_4_1_mmoe": (r"""
### 结构：共享专家，任务各自选课

输入 $x\in\mathbb R^m$ 同时进入 $E$ 个专家 $f_e(x)\in\mathbb R^d$。任务 $k$ 的 gate 产生和为 1 的权重：

$$g_k(x)=\mathrm{softmax}(W_{g,k}x),\quad z_k=\sum_{e=1}^E g_{k,e}(x)f_e(x),\quad \hat y_k=\sigma(t_k(z_k)).$$

因此同一条样本上，点击任务可以偏向专家 1，长播任务可以偏向专家 3。总损失是 $L=\sum_k\lambda_kL_k$；$\lambda_k$ 不只是数学常数，也表达业务权衡。形状从专家输出 $[B,E,d]$ 经 task gate $[B,E]$ 变成每任务的 $[B,d]$。""", "`run_mmoe` 用相同曝光行产生 click/long-view 两个真实标签；Torch-RecHub MMOE 的专家、gate、任务塔和逐任务 BCE 都能在源码映射中定位。"),
        "3_4_2_ple": (r"""
### 结构：共享专家之外，再给每个任务专属空间

第 $l$ 层中，任务 $k$ 只混合自己的专家 $E_k^{(l)}$ 与共享专家 $E_s^{(l)}$：

$$z_k^{(l)}=\sum_{e\in E_k^{(l)}\cup E_s^{(l)}}g_{k,e}^{(l)}f_e^{(l)}(z^{(l-1)}).$$

共享 gate 则读取所有任务与共享专家，为下一层产生公共表示。重复多层称为 progressive extraction：越往后，公共信息与任务特有信息的边界越清楚。最终仍是每任务 tower、Sigmoid 与 $L=\sum_k\lambda_kL_k$。与 MMoE 的关键差异不是损失，而是限制“哪些专家允许被哪个 gate 读取”。""", "`run_ple` 保持与 MMoE 相同数据和目标，仅替换为 Torch-RecHub PLE；这样结果差异才主要来自 CGC 结构而非样本口径。"),
        "3_2_3_sasrec": (r"""
### 结构：只看过去的 Transformer

第 $t$ 个输入是商品与位置 embedding 之和 $x_t=e_{i_t}+p_t$。线性投影得到 $Q=XW_Q,K=XW_K,V=XW_V$，再计算

$$A=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt d}+M\right),\qquad H=AV.$$

因果 mask $M_{t,j}$ 在 $j>t$ 时为 $-\infty$，softmax 后未来权重变成 0。除以 $\sqrt d$ 是为了避免维度变大时点积过大、softmax 过早饱和。注意力后还有残差连接、LayerNorm 和逐位置前馈网络。最后用位置 $t$ 的表示区分下一件正商品与负商品：

$$L_t=\mathrm{softplus}(-h_t^\top e^+)+\mathrm{softplus}(h_t^\top e^-).$$

训练时所有位置可并行，推理时取最后有效位置做全库点积或 ANN。""", "`run_sasrec` 从真实时间戳生成序列、正目标和负目标；Torch-RecHub SASRec 实现 item/position embedding、causal attention 与 pairwise loss。"),
        "4_2_openonerec_practice": (r"""
### 结构：把 item 变成可生成的 Semantic ID

量化器把 item embedding 映射为多级 token $s(i)=(c_1,\ldots,c_m)$。生成器对列表做自回归分解：

$$P(y|x)=\prod_{t=1}^{T}P(y_t\mid y_{<t},x),\qquad L_{CE}=-\sum_t\log P(y_t^*|y_{<t}^*,x).$$

teacher forcing 训练时给真实前缀，推理时给模型自己刚生成的前缀，因此必须用 trie 把下一 token 限制在真实目录分支中。偏好阶段可使用 DPO：

$$L_{DPO}=-\log\sigma\{\beta[(\log\pi(y^+|x)-\log\pi_{ref}(y^+|x))-(\log\pi(y^-|x)-\log\pi_{ref}(y^-|x))]\}.$$

它鼓励偏好列表 $y^+$ 相对 $y^-$ 的优势超过参考模型。smoke 实验验证 token、约束与指标契约，不把小模型冒充官方大规模复现。""", "`run_openonerec` 展开 Semantic ID、teacher forcing 与 trie 解码；完整 OpenOneRec 训练配置保留在官方框架，教程明确区分接口复现和规模复现。"),
        "4_3_dlrm_hstu_practice": (r"""
### 结构：把长期行为流当作同一种序列任务

每个事件由 item、动作与位置/时间信息组成 $x_t=e_{item}+e_{action}+e_{position}$。HSTU 不把注意力权重强制归一化为总和 1，而以相对关系 $r_{t,j}$ 调制 pointwise aggregated attention：

$$a_{t,j}=\mathrm{SiLU}(q_t\odot k_j+r_{t,j}),\qquad h_t=\sum_{j\le t}a_{t,j}\odot v_j.$$

$j\le t$ 是因果约束；SiLU$(z)=z\sigma(z)$ 允许多个历史事件同时贡献，不必彼此争夺固定概率质量。堆叠 HSTU block 后，用 $h_t$ 预测下一 item，并以 sampled softmax/交叉熵训练。教程中的 Torch-RecHub CPU 模型保留核心张量契约；Meta DLRM-v3 的长序列、时间特征和分布式检索属于 full profile。""", "`run_hstu` 从 KuaiRand 真实 feed 流生成 next-item 样本；实现映射会指出教学后端与 Meta generative-recommenders 完整 HSTU 的边界。"),
    }

    specs = {}
    for meta in algorithms:
        keys = [meta["primary"], meta["secondary"]] + ([meta["baseline"]] if meta["baseline"] else [])
        metrics_literal = "{" + ", ".join(f"{key!r}: result[{key!r}]" for key in keys) + "}"
        baseline_line = (
            f"- 对照指标 {meta['baseline']} = **{{result[{meta['baseline']!r}]:.4f}}**。"
            if meta["baseline"] else
            "- 本节没有把不同任务的数值伪装成 baseline；章节总结只做同口径比较。"
        )
        derivation, implementation_map = derivations[meta["slug"]]
        cells = [
            md(f"""## 学习地图

1. 从原始论文理解系统约束；
2. 用可手算数字读懂公式和形状；
3. 检查数据、切分与标签；
4. 使用工业框架模型类训练；
5. 分开验证训练、推理和测试；
6. 用实际输出讨论失败边界。

**本节问题：** {meta['problem']}

**先修知识：** 3.0 的向量、概率与损失函数。第一次阅读无需推导梯度，只要能解释输入、输出和形状。"""),
            md(f"""## Paper & Context

{meta['paper']}

**来源：** {meta['source']}

请区分三层证据：论文中的离线实验、本 Notebook 验证的代码链路、生产系统尚需验证的在线收益。三者不能互相替代。"""),
            md(f"""## Model Structure & Formula Walkthrough

{derivation}

### 公式到代码

{implementation_map}

阅读源码时按“张量形状 → 前向计算 → score → loss → metric”五步追踪，不需要一次读完整个工程文件。"""),
            md(f"""## Math by Hand

{meta['math']}

下面用 NumPy/Matplotlib 验证直觉。二维图只是教学投影，工业 embedding 虽有更多维，计算规则相同。"""),
            code(meta["hand"]),
            md(f"""## Data

{meta['dataset']}

**防泄漏清单：**按时间切分；item 映射只表达已知目录，不读取测试标签；低评分或未点击负反馈均来自数据中的已观察行；序列只看预测时刻以前；测试集只在最后评价。CPU 档使用真实数据的确定性子集，**不是统一 benchmark 成绩**。"""),
            md(f"""## Model & Framework

{meta['framework']}

smoke 档强调模型类、张量契约和指标链路真实可运行；full 档应替换原始数据、分布式配置、索引/服务和资源预算，而不是只增加 epoch。"""),
            code(f"""import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
from recsys_lab.industrial_experiments import {meta['function']}, save_records

print("实际执行函数源码（包含数据、训练、推理和测试）：")
print(inspect.getsource({meta['function']}))"""),
            md("## Train & Inference\n\n下一格实际执行完整 smoke：固定 seed、构造数据、实例化模型、训练、切换到推理路径并计算测试指标。"),
            code(f"""result = {meta['function']}()
print({{'framework': result['framework'], 'dataset': result.get('dataset', {{}})}})
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
    'dataset': result['dataset']['dataset'],
    'randomly_fabricated_rows': int(result['dataset']['randomly_fabricated_rows'])
}}
path=save_records({meta['chapter']!r},{meta['slug']!r},[record]); print('saved:',path.relative_to(ROOT))"""),
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
    for slug, number, title, directory, expected, interpretation, sources in summaries:
        specs[slug] = notebook(
            f"{number} 总结：{title}",
            "读取本章节每个独立算法 Notebook 的实际结果产物，在统一口径下比较和选型。",
            sources,
            [
                md(f"""## 开篇回顾

本章节固定为 **开篇导读 → 独立算法教程 → 结果总结**。每篇算法 Notebook 都包含论文、数学、数据、训练、推理、测试与讨论；本页不重新训练，也不手填数字。

{interpretation}"""),
                md("## Results\n\n读取 results 目录。若缺文件，请先按章节顺序执行算法 Notebook。"),
                code(f"""import json
import pandas as pd
import matplotlib.pyplot as plt
result_dir=ROOT/'results'/{directory!r}; files=sorted(result_dir.glob('*.json'))
assert len(files)=={expected},f'期望 {expected} 个结果，实际 {{[p.name for p in files]}}'
records=[]
for path in files: records.extend(json.loads(path.read_text(encoding='utf-8'))['records'])
comparison=pd.DataFrame(records); display(comparison.round(4)); print('数据来源:',[p.name for p in files])"""),
                code("""fig,ax=plt.subplots(figsize=(max(7,len(comparison)*1.5),3.8))
bars=ax.bar(comparison.algorithm,comparison.primary_value,color='#7ca832')
ax.set(title='Primary metrics from executed notebooks',ylabel='metric value',ylim=(0,max(1.0,comparison.primary_value.max()*1.18)))
ax.tick_params(axis='x',rotation=12)
for bar,value in zip(bars,comparison.primary_value): ax.text(bar.get_x()+bar.get_width()/2,value,f'{value:.3f}',ha='center',va='bottom')
plt.tight_layout(); plt.show()"""),
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
