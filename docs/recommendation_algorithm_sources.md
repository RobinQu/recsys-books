# 互联网推荐算法技术发展：来源与研究说明

更新时间：2026-07-17（Asia/Shanghai）

## 研究边界

- 主题覆盖经典协同过滤与矩阵分解、FM/FFM、深度召回、深度排序、序列与多任务建模、生成式召回、生成式排序和召排融合。
- “工业实践”按证据强度区分：A=官方明确部署/上线且披露线上结果；B=公司论文说明生产部署但细节有限；C=工业数据或离线验证，不能据此断言全面上线。
- 不横向比较不同公司披露的 CTR、观看时长、转化率或满意度增益，因为产品、流量、基线、实验周期和指标定义均不同。
- 生成式推荐不等同于“用通用聊天 LLM 写推荐理由”。本文关注模型是否生成 item/item token、列表或排序决策，以及是否统一推荐任务。

## 经典与深度学习基础论文

1. Koren, Bell, Volinsky. [Matrix Factorization Techniques for Recommender Systems](https://doi.org/10.1109/MC.2009.263), IEEE Computer, 2009.
2. Rendle. [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf), ICDM, 2010.
3. Huang et al. [Learning Deep Structured Semantic Models for Web Search using Clickthrough Data](https://www.microsoft.com/en-us/research/publication/learning-deep-structured-semantic-models-for-web-search-using-clickthrough-data/), CIKM, 2013.
4. He et al. [Practical Lessons from Predicting Clicks on Ads at Facebook](https://ai.meta.com/research/publications/practical-lessons-from-predicting-clicks-on-ads-at-facebook/), ADKDD, 2014（GBDT+LR）.
5. Cheng et al. [Wide & Deep Learning for Recommender Systems](https://research.google/pubs/wide-deep-learning-for-recommender-systems/), 2016.
6. Covington, Adams, Sargin. [Deep Neural Networks for YouTube Recommendations](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/), RecSys, 2016.
7. Guo et al. [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/abs/1703.04247), IJCAI, 2017.
8. Wang et al. [Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123), 2017.
9. Zhou et al. [Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1706.06978), 2017/2018.
10. Ma et al. [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/), KDD, 2018.
11. Zhou et al. [Deep Interest Evolution Network for Click-Through Rate Prediction](https://arxiv.org/abs/1809.03672), AAAI, 2019.
12. Kang, McAuley. [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781), ICDM, 2018.
13. Sun et al. [BERT4Rec](https://arxiv.org/abs/1904.06690), CIKM, 2019.
14. Li et al. [MIND: Multi-Interest Network with Dynamic Routing for Recommendation at Tmall](https://arxiv.org/abs/1904.08030), CIKM, 2019.
15. Tang et al. [Progressive Layered Extraction (PLE)](https://doi.org/10.1145/3383313.3412236), RecSys, 2020.

## 数据集选择证据

1. GroupLens. [MovieLens latest-small README](https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html)：用于经典矩阵、评分和特征交互教学，不外推为 feed CTR。
2. Hou et al. [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/)：571.54M reviews、细粒度时间戳、交互延伸到 2023 年；本教程采用官方 Video Games 5-core。
3. Li et al. [MIND](https://arxiv.org/abs/1904.08030)：原论文离线数据为 Amazon Books 与 TmallData，支持用 Amazon 电商长序列展示多兴趣召回，而非继续使用电影评分。
4. Gao et al. [KuaiRand](https://github.com/chongminggao/KuaiRand)：真实短视频 feed、标准/随机干预策略、12 类反馈和时间戳；用于 DeepFM、DIN/DIEN、MMoE/PLE 与 HSTU。
5. Wu et al. [Microsoft News Dataset](https://learn.microsoft.com/en-us/azure/open-datasets/dataset-microsoft-news)：真实 news impression、点击与候选未点击，可作为 DSSM/新闻排序的 full-profile 替代数据。
6. OpenOneRec. [RecIF-Bench dataset card](https://huggingface.co/datasets/OpenOneRec/OpenOneRec-RecIF)：多域生成式推荐基准；当前仓库 gated，教程明确区分 KuaiRand smoke adapter 与授权后的 RecIF full profile。

选择原则不是“越新越好”：经典方法优先可解释、可手算；CTR 模型必须优先真实曝光；多目标模型需要原生多反馈；Transformer/HSTU 需要严格时间序列、较长历史和更贴近线上流的数据。

## 生成式推荐与统一建模

15. Rajput et al. [Recommender Systems with Generative Retrieval (TIGER)](https://arxiv.org/abs/2305.05065), NeurIPS, 2023.
16. Zhai et al. [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations (HSTU)](https://arxiv.org/abs/2402.17152), ICML, 2024.
17. Deng et al. [OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment](https://arxiv.org/abs/2502.18965), 2025.
18. Huang et al. [Towards Large-scale Generative Ranking (RankGPT)](https://arxiv.org/abs/2505.04180), 2025.
19. Subbiah et al. [Efficient Single-Step Item ID Generation for Large-Scale LLM-based Recommendation](https://research.google/pubs/efficient-single-step-item-id-generation-for-large-scalellm-based-recommendation/), 2025.
20. Hou et al. [Generating Long Semantic IDs in Parallel for Recommendation (RPG)](https://arxiv.org/abs/2506.05781), KDD, 2025.
21. Netflix. [Foundation Model for Personalized Recommendation](https://netflixtechblog.com/foundation-model-for-personalized-recommendation-1a0bd8e02d39), 2025.
22. Netflix Research. [Integrating Netflix's Foundation Model into Personalization applications](https://research.netflix.com/publication/integrating-netflixs-foundation-model-into-personalization-applications), 2026.

## 工业系统与公开部署证据

23. Yi et al. [Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations](https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/), YouTube NDR, RecSys 2019.
24. Kumthekar et al. [Recommending What Video to Watch Next: A Multitask Ranking System](https://research.google/pubs/recommending-what-video-to-watch-next-a-multitask-ranking-system/), YouTube, RecSys 2019.
25. Meta Engineering. [Sequence learning: A paradigm shift for personalized ads recommendations](https://engineering.fb.com/2024/11/19/data-infrastructure/sequence-learning-personalized-ads-recommendations/), 2024.
26. Meta Engineering. [Meta's Generative Ads Model (GEM)](https://engineering.fb.com/2025/11/10/ml-applications/metas-generative-ads-model-gem-the-central-brain-accelerating-ads-recommendation-ai-innovation/), 2025.
27. Meta Engineering. [Meta Adaptive Ranking Model](https://engineering.fb.com/2026/03/31/ml-applications/meta-adaptive-ranking-model-bending-the-inference-scaling-curve-to-serve-llm-scale-models-for-ads/), 2026.
28. Meta Engineering. [SilverTorch: Index as Model](https://engineering.fb.com/2026/05/26/ml-applications/silvertorch-index-as-model-new-retrieval-paradigm-recommendation-systems/), 2026.
29. Hou et al. [Kunlun: Establishing Scaling Laws for Massive-Scale Recommendation Systems](https://arxiv.org/abs/2602.10016), 2026.
30. Liu et al. [Monolith: Real Time Recommendation System With Collisionless Embedding Table](https://arxiv.org/abs/2209.07663), ByteDance/BytePlus, 2022.
31. Zhou et al. [DIN](https://arxiv.org/abs/1706.06978) and Zhou et al. [DIEN](https://arxiv.org/abs/1809.03672), Alibaba/Taobao.
32. Li et al. [MIND](https://arxiv.org/abs/1904.08030), Alibaba/Tmall.
33. Deng et al. [OneRec](https://arxiv.org/abs/2502.18965), Kuaishou, 2025.
34. Huang et al. [RankGPT](https://arxiv.org/abs/2505.04180), Xiaohongshu, 2025.
35. Tang et al. [PLE](https://recsys.acm.org/recsys20/session-4/), Tencent PCG, 2020.

## 解释与限制

- 论文时间以首次公开或会议年份为主，个别模型存在 arXiv 与正式会议年份不同的情况。
- “DSSM 双塔”在推荐领域常泛指独立编码 user/query 与 item/document、以点积或余弦相似度检索的架构；DSSM 原始论文面向 Web 搜索，但它奠定了双塔语义匹配范式。
- HSTU 的“生成式”更接近对用户动作序列做生成式序列转换，不等同于 TIGER/OneRec 那种必须自回归解码 Semantic ID。
- Meta 2025-2026 的 GEM、Adaptive Ranking、SilverTorch 分别代表基础模型知识迁移、超大规模实时排序、以及检索系统模型化；它们不是同一个模型，也不能合并解读为单一端到端系统。
- 由于大厂只公开局部栈，报告中的“参考架构”是对公开证据的工程归纳，不代表任何公司的完整内部实现。
