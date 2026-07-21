"""Curated report content. Claims link to papers or official engineering sources."""

EVOLUTION = [
    {"period": "1990s–2008", "recall": "UserCF / ItemCF、共现倒排", "rank": "规则、流行度、LR", "shift": "从编辑规则转向群体行为", "limit": "稀疏、冷启动、覆盖率"},
    {"period": "2006–2013", "recall": "MF / 隐语义向量", "rank": "LR、GBDT、FM/FFM", "shift": "低维表示与自动特征交叉", "limit": "高阶非线性与时序不足"},
    {"period": "2013–2017", "recall": "DSSM 双塔 + ANN", "rank": "GBDT+LR、Wide&Deep、DeepFM", "shift": "召回—排序级联定型", "limit": "双塔交互弱、排序偏静态"},
    {"period": "2017–2020", "recall": "YouTube DNN、MIND、多兴趣", "rank": "DIN/DIEN、MMoE/PLE", "shift": "目标感知序列与多任务", "limit": "长序列成本、目标冲突"},
    {"period": "2018–2023", "recall": "SASRec/BERT4Rec、图与多模态", "rank": "Transformer、蒸馏、重排", "shift": "推荐成为行为序列建模", "limit": "训练服务一致性与算力"},
    {"period": "2023–2026", "recall": "TIGER / Semantic ID / HSTU-Match", "rank": "RankGPT、生成式列表打分", "shift": "OneRec/HSTU 尝试统一召排", "limit": "解码、合法性、更新与 ROI"},
]

MODELS = [
    {"id":"cf","chapter":"classic","stage":"经典","name":"协同过滤","idea":"从 user–item 共现估计相似度，以邻域投票召回或打分。","pros":"无需内容特征；解释直观；强基线。","cons":"稀疏和冷启动；热门偏置；全量相似度成本。","metric":"Recall@K / Coverage","notebook":"3_1_1_collaborative_filtering"},
    {"id":"mf","chapter":"classic","stage":"经典","name":"MF","idea":"把交互矩阵分解为用户与物品低维向量，内积重构偏好。","pros":"压缩稀疏矩阵；检索友好；易加入偏置。","cons":"依赖 ID；难用上下文和内容；内积表达受限。","metric":"RMSE / Recall@K","notebook":"3_1_2_matrix_factorization"},
    {"id":"fm","chapter":"classic","stage":"经典/排序","name":"FM","idea":"用特征隐向量内积参数化二阶交叉，线性时间计算全部 pair。","pros":"稀疏特征共享统计；少做手工交叉。","cons":"主要是二阶；不建模行为次序。","metric":"AUC / LogLoss","notebook":"3_1_3_factorization_machine"},
    {"id":"gbdtlr","chapter":"classic","stage":"排序","name":"GBDT+LR","idea":"树学习非线性分桶与组合，叶节点 one-hot 交给 LR 校准。","pros":"表格特征强；可解释；服务稳定。","cons":"两阶段；叶子空间膨胀且易陈旧。","metric":"AUC / LogLoss","notebook":"3_1_4_gbdt_lr"},
    {"id":"word2vec","chapter":"classic","stage":"经典/召回","name":"word2vec / Item2Vec","idea":"把行为序列当句子，用 Skip-gram 学物品向量，向量近邻做全库召回。","pros":"向量可预计算；ANN 召回；保留局部次序。","cons":"冷启动物品无向量；短序列不稳；共现继承热门偏置。","metric":"Recall@K / Coverage","notebook":"3_1_5_word2vec"},
    {"id":"dssm","chapter":"retrieval","stage":"召回","name":"DSSM 双塔","idea":"用户塔、物品塔独立编码，同空间内积训练并用 ANN 全库检索。","pros":"item 可预计算；延迟低；可扩到亿级目录。","cons":"打分前无深交互；负采样偏差；单向量混兴趣。","metric":"Recall@K / MRR","notebook":"3_2_1_dssm"},
    {"id":"mind","chapter":"retrieval","stage":"召回","name":"MIND","idea":"动态路由把历史聚成多个兴趣胶囊，每个兴趣分别检索。","pros":"缓解兴趣混叠；覆盖多意图与长尾。","cons":"兴趣数敏感；多路 ANN 合并成本。","metric":"Recall@K / Coverage","notebook":"3_2_2_mind"},
    {"id":"deepfm","chapter":"ranking","stage":"排序","name":"DeepFM","idea":"FM 一二阶分支与 DNN 高阶分支共享 embedding 并联合训练。","pros":"端到端；兼顾记忆与泛化；成熟基线。","cons":"高阶交互隐式；无序列和去偏专门机制。","metric":"AUC / LogLoss","notebook":"3_3_1_deepfm"},
    {"id":"din","chapter":"ranking","stage":"排序","name":"DIN","idea":"用候选感知 attention 激活与当前物品相关的历史。","pros":"同一用户面对不同候选产生不同表示。","cons":"逐候选读取历史，延迟随候选数增长。","metric":"AUC / GAUC / P99","notebook":"3_3_2_din"},
    {"id":"dien","chapter":"ranking","stage":"排序","name":"DIEN","idea":"用辅助损失、GRU 与 AUGRU 显式学习兴趣状态及其演化。","pros":"利用顺序与兴趣变化；监督更密集。","cons":"训练与服务更复杂，校准和时延压力更高。","metric":"AUC / LogLoss / P99","notebook":"3_3_3_dien"},
    {"id":"mmoe","chapter":"multitask","stage":"多目标","name":"MMoE","idea":"每个任务用独立 gate 混合同一组共享专家。","pros":"共享灵活；适合相关任务。","cons":"专家塌缩、梯度冲突与样本空间错位。","metric":"逐任务 AUC + business value","notebook":"3_4_1_mmoe"},
    {"id":"ple","chapter":"multitask","stage":"多目标","name":"PLE","idea":"逐层分离共享与任务专属专家，控制知识共享边界。","pros":"缓解负迁移与跷跷板。","cons":"结构、参数和调参成本更高。","metric":"逐任务 AUC + transfer gap","notebook":"3_4_2_ple"},
    {"id":"sasrec","chapter":"retrieval","stage":"召回/Transformer","name":"SASRec","idea":"以因果自注意力编码真实行为序列，输出当前用户向量并预测下一物品。","pros":"并行训练；兼顾近期转移与较长依赖；可接全库检索。","cons":"注意力随序列长度平方增长；严格依赖时间协议。","metric":"HR@K / NDCG@K / P99","notebook":"3_2_3_sasrec"},
    {"id":"openonerec","chapter":"generative","stage":"生成式","name":"OpenOneRec","idea":"将 item/list token 化，自回归生成并用目录约束和奖励对齐。","pros":"统一列表目标；支持偏好优化。","cons":"无效 ID、重复、解码延迟与目录更新。","metric":"NDCG / invalid rate / P99","notebook":"4_2_openonerec_practice"},
    {"id":"hstu","chapter":"generative","stage":"生成式","name":"DLRM HSTU","idea":"把长期行为流作为统一序列进行 next-item 预训练与下游迁移。","pros":"长序列、统一表征和规模效应。","cons":"训练服务协同复杂，GPU ROI 要求高。","metric":"HR/NDCG + throughput / P99","notebook":"4_3_dlrm_hstu_practice"},
]

DATASETS = [
    {"name":"MovieLens latest-small","year":"2018 固定版本","scale":"100,836 ratings · 610 users · 9,742 movies","fit":"3.0—3.1 的 CF、MF、FM、GBDT+LR 教学","note":"结构简单，适合手算和经典基线；不再承担深度 CTR/序列实验","url":"https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html"},
    {"name":"Amazon Reviews 2014 / Books & Electronics","year":"2014","scale":"Books / Electronics 5-core 与逐条时间戳","fit":"DSSM 完整迁移、MIND 10-core、DIN/DIEN 公开复现","note":"各算法使用的品类、过滤与切分协议不同；不能把同属 Amazon 的结果直接横比","url":"https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html"},
    {"name":"MovieLens 1M","year":"2003 固定版本","scale":"1,000,209 ratings · 6,040 users · 3,952 movies","fit":"SASRec 论文协议；HSTU 官方 debug 路径之一","note":"SASRec full 使用 leave-two-out 与 100 个未观察负例；与 latest-small 教学集不是同一口径","url":"https://grouplens.org/datasets/movielens/1m/"},
    {"name":"Criteo Display Advertising Challenge","year":"2014","scale":"约 4,500 万条点击记录 · 13 连续 + 26 离散特征","fit":"DeepFM full 论文复现","note":"按论文使用随机 9:1 切分；不把 KuaiRand smoke 指标与论文表格相减","url":"https://www.kaggle.com/c/criteo-display-ad-challenge"},
    {"name":"Census-Income KDD","year":"1996","scale":"199,523 train + 99,762 test · 40 features","fit":"MMoE / PLE full 多任务协议","note":"使用官方 train/test 与论文任务定义；KuaiRand 仅承担 smoke 路由验证","url":"https://archive.ics.uci.edu/dataset/117/census+income+kdd"},
    {"name":"Amazon Reviews 2023 / Video Games","year":"2023","scale":"5-core rating-only: 284,120 reviews · 13,926 users · 24,276 items","fit":"3.2 的确定性 smoke 适配器与结果汇总","note":"只验证真实行为上的数据、形状和训练链路；不冒充 Books 2014、Electronics 或 MovieLens-1M 的 full 协议","url":"https://amazon-reviews-2023.github.io/data_processing/5core.html"},
    {"name":"KuaiRand-Pure","year":"2022","scale":"1.44M standard + 1.19M random-policy interactions","fit":"排序、多任务、OpenOneRec 与 HSTU 的 smoke 适配器","note":"真实短视频曝光与多反馈用于可复现链路验证；full 档仍按各论文指定的 Criteo、Amazon、Census、RecIF 或 Meta 协议运行","url":"https://github.com/chongminggao/KuaiRand"},
    {"name":"MerRec","year":"2024","scale":"6 months, millions of C2C users/items","fit":"多动作、会话、电商多目标","note":"含 taxonomy、文本和买卖双重角色","url":"https://github.com/mercari/mercari-ml-merrec-pub-us"},
    {"name":"EB-NeRD","year":"2024","scale":"2.3M+ news users","fit":"新闻曝光、排序、多样性","note":"提供 demo/small/full，适合曝光偏差研究","url":"https://recsys.eb.dk/"},
    {"name":"Yambda-5B","year":"2025","scale":"4.79B events, 1M users, 9.39M tracks","fit":"工业级召回/排序/序列基础模型","note":"50M/500M/5B 三档；含 organic 标记和音频 embedding","url":"https://huggingface.co/datasets/yandex/yambda"},
    {"name":"Synerise UBM","year":"2025","scale":"multi-event retail logs, 1M target clients","fit":"通用行为表征、多任务","note":"购买/加购/移除/访问/搜索，Parquet","url":"https://recsys.synerise.com/summary"},
    {"name":"RecIF-Bench","year":"2025","scale":"100M interactions, 200K users","fit":"4.2 OpenOneRec 生成式召排","note":"短视频/广告/商品统一评测；遵循官方许可","url":"https://github.com/Kuaishou-OneRec/OpenOneRec"},
]

PRACTICES = [
    {"company":"YouTube / Google","system":"双塔候选生成 + 多任务排序；采样偏差校正","evidence":"A","lesson":"级联仍是可靠基线，训练负样本分布必须贴近全库。","url":"https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/"},
    {"company":"Meta","system":"HSTU 序列生成；SilverTorch 索引即模型；基础模型迁移","evidence":"A","lesson":"模型—系统共设计，先以教师/表征迁移获取 ROI。","url":"https://github.com/meta-recsys/generative-recommenders"},
    {"company":"Netflix","system":"自回归用户交互基础模型连接首页、搜索等场景","evidence":"B","lesson":"共享用户表示能降低专用模型维护成本。","url":"https://netflixtechblog.com/foundation-model-for-personalized-recommendation-1a0bd8e02d39"},
    {"company":"快手","system":"OneRec 端到端列表生成；OpenOneRec + RecIF-Bench","evidence":"A","lesson":"奖励模型、DPO 与合法候选约束和模型本体同等重要。","url":"https://github.com/Kuaishou-OneRec/OpenOneRec"},
    {"company":"阿里巴巴","system":"MIND/DIN/DIEN/ESMM/PLE；TorchEasyRec 工程底座","evidence":"A","lesson":"多兴趣、多任务与流式特征在电商漏斗中长期共存。","url":"https://github.com/alibaba/TorchEasyRec"},
    {"company":"字节跳动","system":"Monolith 在线训练、无冲突哈希表与实时特征","evidence":"A","lesson":"推荐质量不仅取决于模型，在线新鲜度和一致性常是第一约束。","url":"https://arxiv.org/abs/2209.07663"},
    {"company":"腾讯","system":"PLE 多任务渐进式抽取","evidence":"A","lesson":"共享与专属专家分离可缓解任务负迁移。","url":"https://dl.acm.org/doi/10.1145/3383313.3412236"},
]

NOTEBOOK_KIND_LABELS = {
    "foundation": "基础导读",
    "algorithm": "算法实验",
    "summary": "结果总结",
    "curriculum": "数学课程",
}

# Only these notebook roles own a chapter-local implementation package. The 3.0
# curriculum notebooks remain executable in Jupyter, but teach reusable concepts
# rather than pretending to have an algorithm-specific model/train/IDE surface.
CHAPTER_CODE_NOTEBOOK_KINDS = frozenset({"foundation", "algorithm", "summary"})

NOTEBOOKS = [
    {"slug":"3_0_math_foundations","title":"3.0 基础课程总览：从数据语义到模型训练","kind":"foundation","framework":"NumPy / Matplotlib","dataset":"MovieLens latest-small + 手算投影","runtime":"约 15 秒 CPU"},
    {"slug":"3_0_1_data_ml_basics","title":"3.0.1 数据与机器学习基础：一行数据到底代表什么","kind":"curriculum","framework":"pandas / NumPy","dataset":"MovieLens latest-small 真实行为 + 教学拆分","runtime":"约 15 秒 CPU"},
    {"slug":"3_0_2_linear_algebra","title":"3.0.2 线性代数：把推荐数据放进有形状的盒子","kind":"curriculum","framework":"NumPy / Matplotlib","dataset":"明确标注的数学教学数字","runtime":"约 15 秒 CPU"},
    {"slug":"3_0_3_calculus","title":"3.0.3 微积分：参数动一点，损失会怎样","kind":"curriculum","framework":"NumPy / Matplotlib","dataset":"明确标注的数学教学数字","runtime":"约 15 秒 CPU"},
    {"slug":"3_0_4_probability_statistics","title":"3.0.4 概率与统计：把不确定性说清楚","kind":"curriculum","framework":"NumPy / Matplotlib","dataset":"明确标注的数学教学数字","runtime":"约 15 秒 CPU"},
    {"slug":"3_0_5_information_theory","title":"3.0.5 信息论：一次意外有多少信息","kind":"curriculum","framework":"NumPy / Matplotlib","dataset":"明确标注的数学教学数字","runtime":"约 15 秒 CPU"},
    {"slug":"3_0_6_optimization","title":"3.0.6 优化：怎样让模型稳定地下山","kind":"curriculum","framework":"NumPy / Matplotlib","dataset":"明确标注的数学教学数字","runtime":"约 15 秒 CPU"},
    {"slug":"3_0_7_data_pipeline","title":"3.0.7 数据与实验基础：从 import 到训练循环","kind":"foundation","framework":"pandas / PyTorch / inspect","dataset":"MovieLens latest-small 真实行为","runtime":"约 20 秒 CPU"},
    {"slug":"3_1_0_classic_foundations","title":"3.1 导读与数学基础：经典算法","kind":"foundation","framework":"NumPy / Matplotlib","dataset":"MovieLens latest-small","runtime":"约 10 秒 CPU"},
    {"slug":"3_1_1_collaborative_filtering","title":"3.1.1 协同过滤：UserCF / ItemCF","kind":"algorithm","framework":"NumPy","dataset":"MovieLens latest-small","runtime":"约 20 秒 CPU"},
    {"slug":"3_1_2_matrix_factorization","title":"3.1.2 矩阵分解：BiasMF","kind":"algorithm","framework":"PyTorch","dataset":"MovieLens latest-small","runtime":"约 30 秒 CPU"},
    {"slug":"3_1_3_factorization_machine","title":"3.1.3 因子分解机：FM","kind":"algorithm","framework":"PyTorch / scikit-learn","dataset":"MovieLens latest-small 评分偏好任务","runtime":"约 30 秒 CPU"},
    {"slug":"3_1_4_gbdt_lr","title":"3.1.4 GBDT+LR","kind":"algorithm","framework":"XGBoost / scikit-learn","dataset":"MovieLens latest-small 评分偏好任务","runtime":"约 30 秒 CPU"},
    {"slug":"3_1_5_word2vec","title":"3.1.5 word2vec / Item2Vec","kind":"algorithm","framework":"PyTorch","dataset":"MovieLens latest-small 行为序列","runtime":"约 25 秒 CPU"},
    {"slug":"3_1_summary","title":"3.1 总结：经典算法横向对比","kind":"summary","framework":"结果聚合与选型","dataset":"MovieLens 真实实验 JSON","runtime":"约 10 秒 CPU"},
    {"slug":"3_2_0_retrieval_foundations","title":"3.2 导读与数学基础：向量召回","kind":"foundation","framework":"NumPy / Matplotlib","dataset":"full：各论文公开协议；smoke：Amazon Reviews 2023","runtime":"约 10 秒 CPU"},
    {"slug":"3_2_1_dssm","title":"3.2.1 DSSM 双塔召回","kind":"algorithm","framework":"Torch-RecHub DSSM","dataset":"full：Amazon Books 5-core 迁移；smoke：Amazon 真实切片","runtime":"约 20 秒 CPU"},
    {"slug":"3_2_2_mind","title":"3.2.2 MIND 多兴趣召回","kind":"algorithm","framework":"Torch-RecHub MIND","dataset":"full：Amazon Books 2014 10-core；smoke：Amazon 真实切片","runtime":"约 20 秒 CPU"},
    {"slug":"3_2_3_sasrec","title":"3.2.3 SASRec 序列召回","kind":"algorithm","framework":"Torch-RecHub SASRec","dataset":"full：MovieLens-1M 论文协议；smoke：Amazon 真实切片","runtime":"约 35 秒 CPU"},
    {"slug":"3_2_summary","title":"3.2 总结：DSSM、MIND 与 SASRec","kind":"summary","framework":"结果聚合与选型","dataset":"full：按算法分别审计；smoke：Amazon 2023 实验 JSON","runtime":"约 5 秒 CPU"},
    {"slug":"3_3_0_ranking_foundations","title":"3.3 导读与数学基础：CTR 排序","kind":"foundation","framework":"NumPy / Matplotlib","dataset":"full：Criteo / Amazon Electronics；smoke：KuaiRand","runtime":"约 10 秒 CPU"},
    {"slug":"3_3_1_deepfm","title":"3.3.1 DeepFM 排序","kind":"algorithm","framework":"Torch-RecHub DeepFM","dataset":"full：Criteo；smoke：KuaiRand-Pure is_click","runtime":"约 20 秒 CPU"},
    {"slug":"3_3_2_din","title":"3.3.2 DIN 候选感知排序","kind":"algorithm","framework":"Torch-RecHub DIN","dataset":"full：Amazon Electronics；smoke：KuaiRand feed 序列","runtime":"约 25 秒 CPU"},
    {"slug":"3_3_3_dien","title":"3.3.3 DIEN 兴趣演化排序","kind":"algorithm","framework":"Torch-RecHub DIEN","dataset":"full：Amazon Electronics；smoke：KuaiRand feed 序列","runtime":"约 40 秒 CPU"},
    {"slug":"3_3_summary","title":"3.3 总结：DeepFM、DIN 与 DIEN","kind":"summary","framework":"结果聚合与选型","dataset":"full：按算法分别审计；smoke：KuaiRand 实验 JSON","runtime":"约 5 秒 CPU"},
    {"slug":"3_4_0_multitask_foundations","title":"3.4 导读与数学基础：多目标学习","kind":"foundation","framework":"NumPy / Matplotlib","dataset":"full：Census-Income；smoke：KuaiRand 多反馈","runtime":"约 10 秒 CPU"},
    {"slug":"3_4_1_mmoe","title":"3.4.1 MMoE 多目标学习","kind":"algorithm","framework":"Torch-RecHub MMOE","dataset":"full：Census-Income；smoke：KuaiRand 双目标","runtime":"约 20 秒 CPU"},
    {"slug":"3_4_2_ple","title":"3.4.2 PLE 渐进式专家抽取","kind":"algorithm","framework":"Torch-RecHub PLE","dataset":"full：Census-Income；smoke：KuaiRand 双目标","runtime":"约 25 秒 CPU"},
    {"slug":"3_4_summary","title":"3.4 总结：MMoE 与 PLE","kind":"summary","framework":"结果聚合与选型","dataset":"full：Census-Income；smoke：KuaiRand 实验 JSON","runtime":"约 5 秒 CPU"},
    {"slug":"4_0_generative_foundations","title":"4.0 导读与数学基础：生成式推荐","kind":"foundation","framework":"NumPy / Matplotlib","dataset":"full：按算法官方协议；smoke：KuaiRand 适配器","runtime":"默认 CUDA；CPU 仅基础检查"},
    {"slug":"4_1_generative_overview","title":"4.1 总结：生成式召回、排序与融合","kind":"summary","framework":"结果聚合与工业选型","dataset":"full：按算法分别审计；smoke：KuaiRand 实验 JSON","runtime":"默认 CUDA；CPU 仅阅读"},
    {"slug":"4_2_openonerec_practice","title":"4.2 OpenOneRec 实战","kind":"algorithm","framework":"OpenOneRec 契约 + PyTorch 约束生成器","dataset":"KuaiRand taxonomy adapter；RecIF full 需授权","runtime":"CUDA 混合精度；CPU basic smoke"},
    {"slug":"4_3_dlrm_hstu_practice","title":"4.3 DLRM HSTU 实战","kind":"algorithm","framework":"Torch-RecHub HSTU + Meta DLRM-v3 配置","dataset":"full：Meta MovieLens-20M + Amazon Books；smoke：KuaiRand feed 序列","runtime":"CUDA 混合精度；官方配置需 4×24GB GPU"},
]

# The six-course curriculum lives contiguously inside 3.0. Appendix A.4 is only
# a bounded knowledge-graph projection of these concepts, not another hierarchy.
# Each concept points at an exact course heading and carries explicit model and
# algorithm-notebook joins for coverage tests and graph navigation.
_MODEL_NOTEBOOK_BY_ID = {model["id"]: model["notebook"] for model in MODELS}


def _math_topic(
    topic_id: str,
    area: str,
    topic: str,
    intuition: str,
    used_by: str,
    notebook: str,
    anchor: str,
    prerequisites: tuple[str, ...],
    model_ids: tuple[str, ...],
) -> dict:
    return {
        "id": topic_id,
        "area": area,
        "topic": topic,
        "intuition": intuition,
        "used_by": used_by,
        "notebook": notebook,
        "anchor": anchor,
        "prerequisites": list(prerequisites),
        "model_ids": list(model_ids),
        "notebook_slugs": [_MODEL_NOTEBOOK_BY_ID[model_id] for model_id in model_ids],
    }


MATH_PREREQUISITES = [
    # 3.0.1 数据与机器学习基础
    _math_topic(
        "data-observation-label", "数据与机器学习", "样本、实体、特征与标签",
        "一行数据不是现实本身，而是对一次用户—物品事件的记录；特征是答题线索，标签是希望模型学会预测的答案。",
        "全部算法的数据入口与训练目标", "3_0_1_data_ml_basics", "observation-label", (),
        tuple(_MODEL_NOTEBOOK_BY_ID),
    ),
    _math_topic(
        "data-implicit-feedback", "数据与机器学习", "显式反馈、隐式反馈与未知样本",
        "评分直接说出偏好；点击和停留只留下行为痕迹。没有点击通常表示未知，不能不加条件地当成负例。",
        "CF/MF、序列召回、CTR、多任务与生成式推荐", "3_0_1_data_ml_basics", "implicit-feedback",
        ("data-observation-label",),
        ("cf", "mf", "word2vec", "dssm", "mind", "sasrec", "deepfm", "din", "dien", "mmoe", "ple", "openonerec", "hstu"),
    ),
    _math_topic(
        "data-split-leakage", "数据与机器学习", "数据切分、泄漏、基线与泛化",
        "训练集像练习题，测试集像没见过的考试题；若把未来行为或答案偷偷放进训练，分数再高也不能说明模型会泛化。",
        "全部算法的实验协议与结果比较", "3_0_1_data_ml_basics", "split-leakage",
        ("data-observation-label", "data-implicit-feedback"), tuple(_MODEL_NOTEBOOK_BY_ID),
    ),
    # 3.0.2 线性代数
    _math_topic(
        "linear-tensors-shapes", "线性代数", "标量、向量、矩阵、张量与形状",
        "它们只是把数字按零维、一维、二维或更多维摆放；先写清每条轴表示什么，就能避免多数维度错误。",
        "embedding、行为序列、多兴趣与多任务批次", "3_0_2_linear_algebra", "tensors-shapes", (),
        ("mf", "fm", "word2vec", "dssm", "mind", "sasrec", "deepfm", "din", "dien", "mmoe", "ple", "openonerec", "hstu"),
    ),
    _math_topic(
        "linear-elementwise-dot", "线性代数", "逐元素运算、点积、范数与余弦",
        "逐元素运算是同位置配对，点积则把配对乘积再相加；除以两边长度后，余弦只比较方向。",
        "CF 相似度、MF/FM 交互、双塔与候选匹配", "3_0_2_linear_algebra", "elementwise-dot",
        ("linear-tensors-shapes",),
        ("cf", "mf", "fm", "dssm", "mind", "sasrec", "deepfm", "din", "dien", "mmoe", "ple", "hstu"),
    ),
    _math_topic(
        "linear-matmul-embedding", "线性代数", "矩阵乘法、转置与 embedding 查表",
        "矩阵乘法是批量完成许多次点积；one-hot 乘 embedding 矩阵等价于按编号取出其中一行。",
        "MF/FM、word2vec、双塔、排序网络与生成模型", "3_0_2_linear_algebra", "matmul-embedding",
        ("linear-elementwise-dot",),
        ("mf", "fm", "word2vec", "dssm", "mind", "sasrec", "deepfm", "din", "dien", "mmoe", "ple", "openonerec", "hstu"),
    ),
    _math_topic(
        "linear-low-rank-attention", "线性代数", "低秩表示、加权和与 Q/K/V 注意力",
        "低秩表示用少量坐标概括大表格；注意力让一个查询给历史内容分配票数，再把值向量加权汇总。",
        "MF、多兴趣路由、SASRec、DIN/DIEN 与 HSTU", "3_0_2_linear_algebra", "low-rank-attention",
        ("linear-matmul-embedding",), ("mf", "dssm", "mind", "sasrec", "din", "dien", "openonerec", "hstu"),
    ),
    # 3.0.3 微积分
    _math_topic(
        "calculus-functions", "微积分", "函数、复合函数与激活函数",
        "模型是函数套函数：前一层输出成为后一层输入；激活函数负责把简单直线组合变成能表达弯曲规律的映射。",
        "LR、FM、深度排序、序列与生成模型", "3_0_3_calculus", "functions", (),
        ("fm", "gbdtlr", "word2vec", "dssm", "mind", "sasrec", "deepfm", "din", "dien", "mmoe", "ple", "openonerec", "hstu"),
    ),
    _math_topic(
        "calculus-derivative-gradient", "微积分", "导数、偏导、梯度与方向",
        "导数测量一个量轻微改变时结果怎样变；多个参数各有一个偏导，把它们排成向量就是梯度。",
        "除邻域 CF 外所有可训练参数的更新", "3_0_3_calculus", "derivative-gradient",
        ("calculus-functions",),
        ("mf", "fm", "gbdtlr", "word2vec", "dssm", "mind", "sasrec", "deepfm", "din", "dien", "mmoe", "ple", "openonerec", "hstu"),
    ),
    _math_topic(
        "calculus-chain-rule", "微积分", "链式法则、计算图与反向传播",
        "复合函数的变化要沿计算路径逐段相乘；反向传播只是从损失出发，把这条规则有次序地重复应用。",
        "FM 与全部神经网络的自动求导", "3_0_3_calculus", "chain-rule",
        ("calculus-derivative-gradient",),
        ("mf", "fm", "word2vec", "dssm", "mind", "sasrec", "deepfm", "din", "dien", "mmoe", "ple", "openonerec", "hstu"),
    ),
    # 3.0.4 概率与统计
    _math_topic(
        "probability-random-variable", "概率与统计", "随机事件、随机变量与分布",
        "随机变量是给不确定结果贴上的数字标签，分布则说明每种结果有多大可能；推荐分数由此才能解释成概率。",
        "反馈建模、负采样、CTR 与序列预测", "3_0_4_probability_statistics", "random-variable", (),
        tuple(_MODEL_NOTEBOOK_BY_ID),
    ),
    _math_topic(
        "probability-conditional-chain", "概率与统计", "联合、边缘、条件概率与链式分解",
        "条件概率是在已知用户或历史之后重新计算可能性；一段序列的联合概率可拆成从左到右的一串条件概率。",
        "召回条件概率、多任务漏斗与自回归生成", "3_0_4_probability_statistics", "conditional-chain",
        ("probability-random-variable",),
        ("cf", "word2vec", "dssm", "mind", "sasrec", "din", "dien", "mmoe", "ple", "openonerec", "hstu"),
    ),
    _math_topic(
        "probability-expectation-variance", "概率与统计", "期望、方差与抽样估计",
        "期望是按概率加权的长期平均，方差表示结果围绕平均值的波动；小批次梯度就是用样本平均估计总体方向。",
        "指标汇总、mini-batch 训练与稳定性判断", "3_0_4_probability_statistics", "expectation-variance",
        ("probability-random-variable",), tuple(_MODEL_NOTEBOOK_BY_ID),
    ),
    _math_topic(
        "probability-likelihood-calibration", "概率与统计", "odds、logit、似然、采样偏差与校准",
        "似然问参数对已观察数据解释得多好；校准则检查预测 0.7 的样本是否真的约有七成发生。",
        "GBDT+LR、CTR 模型、负采样召回与多任务概率", "3_0_4_probability_statistics", "likelihood-calibration",
        ("probability-conditional-chain", "probability-expectation-variance"),
        ("gbdtlr", "word2vec", "dssm", "mind", "sasrec", "deepfm", "din", "dien", "mmoe", "ple", "openonerec", "hstu"),
    ),
    # 3.0.5 信息论与损失
    _math_topic(
        "information-entropy", "信息论", "信息量与熵",
        "越意外的事件带来的信息越多；熵是把所有结果的信息量按概率加权，衡量整个分布有多不确定。",
        "分类不确定性、推荐分布与归一化指标", "3_0_5_information_theory", "information-entropy",
        ("probability-random-variable",),
        ("word2vec", "dssm", "mind", "sasrec", "deepfm", "din", "dien", "mmoe", "ple", "openonerec", "hstu"),
    ),
    _math_topic(
        "information-cross-entropy-kl", "信息论", "二元/多类交叉熵与 KL 散度",
        "交叉熵衡量模型分布给正确答案的代价；KL 衡量用一个分布近似另一个分布时多付出的信息成本，而且方向不能互换。",
        "CTR LogLoss、sampled softmax、分类与分布对齐", "3_0_5_information_theory", "cross-entropy-kl",
        ("information-entropy", "probability-conditional-chain"),
        ("gbdtlr", "word2vec", "dssm", "mind", "sasrec", "deepfm", "din", "dien", "mmoe", "ple", "openonerec", "hstu"),
    ),
    _math_topic(
        "information-softmax-temperature", "信息论", "Softmax、温度与 Normalized Entropy",
        "Softmax 把一组分数变成总和为 1 的比例；温度调节分布尖锐程度，归一化熵让不同候选规模更容易比较。",
        "召回分类、多兴趣路由、注意力与受控采样", "3_0_5_information_theory", "softmax-temperature",
        ("information-cross-entropy-kl", "calculus-functions"),
        ("word2vec", "dssm", "mind", "sasrec", "din", "dien", "mmoe", "ple", "openonerec", "hstu"),
    ),
    _math_topic(
        "information-sequence-nll-dpo", "信息论", "序列 NLL、困惑度与 DPO log-ratio",
        "序列 NLL 把每一步正确 token 的负对数相加；DPO 比较偏好答案与非偏好答案相对参考模型的对数概率差。",
        "SASRec/HSTU 的逐位置目标与 OpenOneRec 偏好优化", "3_0_5_information_theory", "sequence-nll-dpo",
        ("information-softmax-temperature", "probability-conditional-chain"),
        ("sasrec", "openonerec", "hstu"),
    ),
    # 3.0.6 优化
    _math_topic(
        "optimization-sgd", "优化", "目标函数、GD、SGD 与 mini-batch",
        "优化就是寻找让训练损失更小的参数；SGD 用一小批样本估计下坡方向，以较低成本反复修正。",
        "MF/FM、embedding、深度与生成模型训练", "3_0_6_optimization", "sgd",
        ("calculus-derivative-gradient", "probability-expectation-variance"),
        ("mf", "fm", "gbdtlr", "word2vec", "dssm", "mind", "sasrec", "deepfm", "din", "dien", "mmoe", "ple", "openonerec", "hstu"),
    ),
    _math_topic(
        "optimization-learning-rate", "优化", "学习率、Momentum、Adam 与初始化",
        "学习率是每步迈多远；Momentum 记住近期方向，Adam 还会按参数缩放步长，而初始化决定从哪处开始下山。",
        "全部梯度训练算法的收敛速度与稳定性", "3_0_6_optimization", "learning-rate",
        ("optimization-sgd",),
        ("mf", "fm", "word2vec", "dssm", "mind", "sasrec", "deepfm", "din", "dien", "mmoe", "ple", "openonerec", "hstu"),
    ),
    _math_topic(
        "optimization-regularization", "优化", "L1/L2、过拟合、早停与梯度裁剪",
        "模型若只背训练题就会过拟合；正则化限制参数，早停限制训练轮数，裁剪限制一次更新的极端幅度。",
        "MF/FM、深度排序、序列与生成模型", "3_0_6_optimization", "regularization",
        ("optimization-learning-rate", "data-split-leakage"),
        ("mf", "fm", "word2vec", "dssm", "mind", "sasrec", "deepfm", "din", "dien", "mmoe", "ple", "openonerec", "hstu"),
    ),
    _math_topic(
        "optimization-gradient-conflict", "优化", "多任务梯度冲突与负迁移",
        "若两个任务希望共享参数朝相反方向移动，一个任务进步时另一个可能退步；专家分工用结构限制这种拉扯。",
        "MMoE 与 PLE 的共享边界和跷跷板现象", "3_0_6_optimization", "gradient-conflict",
        ("calculus-chain-rule", "optimization-sgd"), ("mmoe", "ple"),
    ),
]

CHAPTERS = {
    "foundations": {
        "number": "3.0", "title": "共性基础课程：从数据语义到实验管线",
        "intro": "面向高中毕业生，从一行数据的含义开始，循序学习线性代数、微积分、概率统计、信息论损失与优化，再把这些概念接入可执行训练管线。",
        "layout": "总览负责路线与最小直觉；3.0.1—3.0.6 各自完整讲解一组共性知识；3.0.7 打开项目代码，串起时间切分、张量化、训练、推理、测试与结果落盘。",
        "use_cases": "这是 3.1—4.3 的共同先修课程。算法页只展开论文新增数学，其余概念精确回链到 3.0 的稳定锚点。",
        "model_ids": [],
        "opening": "3_0_math_foundations",
        "notebooks": [
            "3_0_math_foundations", "3_0_1_data_ml_basics", "3_0_2_linear_algebra",
            "3_0_3_calculus", "3_0_4_probability_statistics", "3_0_5_information_theory",
            "3_0_6_optimization", "3_0_7_data_pipeline",
        ],
        "summary": "3_0_7_data_pipeline",
        "sources": [
            ("MIT OpenCourseWare：Linear Algebra", "https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/", "用于进一步理解向量、矩阵乘法与低秩表示；本章只提取推荐系统真正需要的部分。"),
            ("Google Machine Learning Crash Course：Logistic Regression", "https://developers.google.com/machine-learning/crash-course/logistic-regression", "用概率、Sigmoid 和 LogLoss 解释二分类预测，是 CTR 排序的数学起点。"),
            ("scikit-learn：Model evaluation", "https://scikit-learn.org/stable/modules/model_evaluation.html", "提供分类、回归和排序指标的正式定义；本章重点解释指标在推荐问题中的直觉。"),
        ],
    },
    "classic": {
        "number": "3.1", "title": "经典算法：协同过滤、MF、FM、GBDT+LR、word2vec",
        "intro": "从显式邻域、低秩表示、稀疏特征交叉到行为序列嵌入，建立召回与排序的五个最小原型。每个算法独立阅读与执行，最后只在相同任务口径内比较指标。",
        "layout": "先用小矩阵理解相似度与低秩，再进入真实 MovieLens Top-K/评分任务；随后把真实评分按阈值派生为二分类任务，比较 FM 与 GBDT+LR；最后把正反馈序列当句子，用 word2vec 学物品向量做召回。",
        "use_cases": "ItemCF 适合相关推荐与兜底召回；MF 适合 ID 向量召回/评分；FM 与 GBDT+LR 适合稀疏表格 CTR 基线；word2vec / Item2Vec 适合序列向量召回。",
        "model_ids": ["cf", "mf", "fm", "gbdtlr", "word2vec"],
        "opening": "3_1_0_classic_foundations",
        "notebooks": ["3_1_1_collaborative_filtering", "3_1_2_matrix_factorization", "3_1_3_factorization_machine", "3_1_4_gbdt_lr", "3_1_5_word2vec"],
        "summary": "3_1_summary",
        "sources": [
            ("Resnick et al., 1994：GroupLens 协同过滤", "https://dl.acm.org/doi/10.1145/192844.192905", "把群体评分压缩为邻域加权预测，是 UserCF 的早期系统证据。"),
            ("Sarwar et al., 2001：Item-based CF", "https://dl.acm.org/doi/10.1145/371920.372071", "把相似度转移到较稳定的 item 侧，奠定相关推荐与离线近邻表。"),
            ("Koren et al., 2009：Matrix Factorization", "https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf", "以偏置加低秩向量解释评分结构，也是后续双塔内积的最小原型。"),
            ("Rendle, 2010：Factorization Machines", "https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf", "用共享隐向量估计稀疏二阶交叉，并将计算降到线性复杂度。"),
            ("He et al., 2014：GBDT+LR", "https://research.facebook.com/publications/practical-lessons-from-predicting-clicks-on-ads-at-facebook/", "树负责学习分桶与组合，LR 负责稀疏线性校准，形成经典两阶段 CTR 管线。"),
            ("Mikolov et al., 2013：word2vec", "https://arxiv.org/abs/1301.3781", "用中心词预测上下文高效学词向量；推荐系统据此把行为序列当句子学物品向量（Item2Vec）做召回。"),
        ],
    },
    "retrieval": {
        "number": "3.2", "title": "召回模型：双塔、多兴趣与序列召回",
        "intro": "召回的核心约束是全库可检索。DSSM 以独立编码换取 ANN；MIND 用多个向量表达并行兴趣；SASRec 用因果自注意力表达兴趣随顺序的变化。",
        "layout": "先理解双塔、负采样与向量索引，再观察多兴趣抽取，最后学习 SASRec 的位置编码、因果注意力和 next-item 召回，统一比较 Recall、Coverage 与服务成本。",
        "use_cases": "DSSM 适合亿级目录通用召回；MIND 适合多意图用户；SASRec 适合时间顺序强、近期兴趣变化明显的内容与电商场景。",
        "model_ids": ["dssm", "mind", "sasrec"], "opening": "3_2_0_retrieval_foundations", "notebooks": ["3_2_1_dssm", "3_2_2_mind", "3_2_3_sasrec"], "summary": "3_2_summary",
        "sources": [("Huang et al., 2013：DSSM", "https://www.microsoft.com/en-us/research/publication/learning-deep-structured-semantic-models-for-web-search-using-clickthrough-data/", "以点击数据联合学习两侧表示，双塔的工程价值来自 item 向量可离线预计算。"), ("Li et al., 2019：MIND", "https://arxiv.org/abs/1904.08030", "动态路由从行为中抽取多个兴趣胶囊，用 label-aware attention 选择训练兴趣。"), ("Kang & McAuley, 2018：SASRec", "https://arxiv.org/abs/1808.09781", "用因果自注意力从有序历史产生 next-item 表示，可作为全库序列召回用户塔。")],
    },
    "ranking": {
        "number": "3.3", "title": "排序模型：DeepFM、DIN 与 DIEN",
        "intro": "排序在有限候选集上恢复深交互：DeepFM 处理静态交叉，DIN 让历史表示依赖候选，DIEN 再显式学习兴趣演化。",
        "layout": "从静态特征交叉基线进入候选感知注意力，再加入时序状态和辅助损失，统一比较 AUC、GAUC、LogLoss 与 P99。",
        "use_cases": "DeepFM 适合通用 CTR；DIN 适合强行为历史排序；DIEN 适合兴趣变化明显且收益覆盖时序成本的场景。",
        "model_ids": ["deepfm", "din", "dien"], "opening": "3_3_0_ranking_foundations", "notebooks": ["3_3_1_deepfm", "3_3_2_din", "3_3_3_dien"], "summary": "3_3_summary",
        "sources": [("Guo et al., 2017：DeepFM", "https://arxiv.org/abs/1703.04247", "共享 embedding 联合训练 FM 与 DNN，减少宽深模型的人工交叉。"), ("Zhou et al., 2018：DIN", "https://arxiv.org/abs/1706.06978", "候选感知激活历史，使同一用户对不同候选得到不同兴趣表示。"), ("Zhou et al., 2019：DIEN", "https://arxiv.org/abs/1809.03672", "用辅助损失监督兴趣抽取，再以目标感知 GRU 建模演化。")],
    },
    "multitask": {
        "number": "3.4", "title": "多目标模型：MMoE 与 PLE",
        "intro": "多目标学习的难点不是多加几个 loss，而是控制共享、专属知识与梯度冲突。MMoE 以任务门控选专家，PLE 进一步分层隔离。",
        "layout": "先建立 hard-sharing 对照，再观察 MMoE 的专家利用率和每任务 AUC，最后用 PLE 分析负迁移与跷跷板。",
        "use_cases": "适合联合 CTR、CVR、时长、互动和负反馈；任务相关性低或样本空间不同必须单独治理。",
        "model_ids": ["mmoe", "ple"], "opening": "3_4_0_multitask_foundations", "notebooks": ["3_4_1_mmoe", "3_4_2_ple"], "summary": "3_4_summary",
        "sources": [("Ma et al., 2018：MMoE", "https://dl.acm.org/doi/10.1145/3219819.3220007", "为每个任务学习门控，在共享专家上形成不同混合。"), ("Tang et al., 2020：PLE", "https://dl.acm.org/doi/10.1145/3383313.3412236", "逐层分离共享与任务专属专家，针对负迁移和跷跷板优化。")],
    },
    "generative": {
        "number": "4", "title": "生成式推荐：生成召回、排序与召排融合",
        "intro": "生成式推荐把 item、Semantic ID 或列表当作输出序列，探索从生成候选、生成排列到端到端 session 生成的统一路径。",
        "layout": "先比较 TIGER、生成式排序与 OneRec/HSTU 的任务边界，再分别通过 OpenOneRec 和 DLRM HSTU 实战验证合法解码、列表指标与序列契约。",
        "use_cases": "适合跨域统一建模、长序列和语义冷启；必须同时评估无效 ID、重复率、目录更新、P99 与 GPU ROI。",
        "model_ids": ["openonerec", "hstu"], "opening": "4_0_generative_foundations", "notebooks": ["4_2_openonerec_practice", "4_3_dlrm_hstu_practice"], "summary": "4_1_generative_overview",
        "sources": [("Rajput et al., 2023：TIGER", "https://arxiv.org/abs/2305.05065", "以 RQ-VAE Semantic ID 将全库召回转成受约束自回归生成。"), ("Kuaishou, 2025：OneRec / OpenOneRec", "https://github.com/Kuaishou-OneRec/OpenOneRec", "公开列表生成、奖励建模与 RecIF-Bench 流程，是召排融合的工业化样本。"), ("Zhai et al., 2024：HSTU", "https://arxiv.org/abs/2402.17152", "针对高基数、非平稳行为流重新设计序列转换与系统协同。")],
    },
}


def notebook_has_paper_guide(slug: str) -> bool:
    """Whether the detail page should render the 论文导读 tab.

    The foundations chapter (3.0) and every chapter opening / 导读 page focus on
    math and intuition, so they skip the paper-evidence tab. Summary pages do
    their cross-paper comparison inside the notebook and also skip this tab.
    Algorithm detail pages keep it.
    """
    notebook = next((item for item in NOTEBOOKS if item["slug"] == slug), None)
    if notebook and notebook["kind"] == "curriculum":
        return False
    foundations = CHAPTERS["foundations"]
    if slug in {foundations["opening"], *foundations["notebooks"], foundations["summary"]}:
        return False
    for key in ("classic", "retrieval", "ranking", "multitask", "generative"):
        if slug in {CHAPTERS[key]["opening"], CHAPTERS[key]["summary"]}:
            return False
    return True


SOURCES = [
    ("Collaborative Filtering","https://dl.acm.org/doi/10.1145/192844.192905","论文"),
    ("Matrix Factorization Techniques","https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf","论文"),
    ("Factorization Machines","https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf","论文"),
    ("DSSM","https://www.microsoft.com/en-us/research/publication/learning-deep-structured-semantic-models-for-web-search-using-clickthrough-data/","论文/官方"),
    ("MIND","https://arxiv.org/abs/1904.08030","论文"),
    ("GBDT+LR at Facebook","https://research.facebook.com/publications/practical-lessons-from-predicting-clicks-on-ads-at-facebook/","论文/官方"),
    ("word2vec","https://arxiv.org/abs/1301.3781","论文"),
    ("DeepFM","https://arxiv.org/abs/1703.04247","论文"),
    ("DIN","https://arxiv.org/abs/1706.06978","论文"),
    ("DIEN","https://arxiv.org/abs/1809.03672","论文"),
    ("MMoE","https://dl.acm.org/doi/10.1145/3219819.3220007","论文"),
    ("PLE","https://dl.acm.org/doi/10.1145/3383313.3412236","论文"),
    ("SASRec","https://arxiv.org/abs/1808.09781","论文"),
    ("SASRec official implementation","https://github.com/kang205/SASRec","官方代码"),
    ("BERT4Rec","https://arxiv.org/abs/1904.06690","论文"),
    ("TIGER","https://arxiv.org/abs/2305.05065","论文"),
    ("OneRec","https://arxiv.org/abs/2502.18965","论文"),
    ("OpenOneRec","https://github.com/Kuaishou-OneRec/OpenOneRec","官方代码"),
    ("HSTU","https://arxiv.org/abs/2402.17152","论文"),
    ("Meta Generative Recommenders","https://github.com/meta-recsys/generative-recommenders","官方代码"),
    ("TorchEasyRec","https://github.com/alibaba/TorchEasyRec","官方代码"),
    ("Torch-RecHub","https://github.com/datawhalechina/torch-rechub","官方代码"),
    ("Amazon Reviews 2023","https://amazon-reviews-2023.github.io/","官方数据"),
    ("KuaiRand","https://github.com/chongminggao/KuaiRand","官方数据/论文"),
    ("OpenOneRec RecIF-Bench","https://huggingface.co/datasets/OpenOneRec/OpenOneRec-RecIF","官方数据（需授权）"),
]
