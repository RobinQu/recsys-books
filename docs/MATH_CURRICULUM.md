# 数学课程审计与内容契约

本文件记录课程边界与审计结论，不是概念清单。概念、锚点和算法覆盖的唯一结构化真源是 `app/content.py:MATH_PREREQUISITES`。

## 三类页面的边界

- **3.0 总览与实验管线**：`3_0_math_foundations` 给出共同路线和最小手算直觉，`3_0_7_data_pipeline` 串起数据切分、张量化、训练、推理、测试与结果落盘。
- **3.0.1—3.0.6 共性课程**：6 个可独立学习的 Notebook，依次负责数据/机器学习基础、线性代数、微积分、概率统计、信息论与损失、优化。稳定锚点供算法页精确回链。
- **算法 Notebook**：只详细推导论文新引入或特殊组合的数学；通用知识通过 `/notebooks/3_0_*#anchor` 引用。论文报告、仓库复现和生产推断必须分开陈述。
- **Appendix A.4 数学知识图**：只投影 `MATH_PREREQUISITES` 中已登记的课程、模型与依赖关系。默认仅显示 6 个章节点，聚焦视图最多 16 个节点、最多展开 2 层；它不是第二套课程目录，也不增加页面层级。

## 15 个算法页审计

等级含义：P1 表示公式或数据口径边界必须重点补强；P2 表示手算/链接需补齐；A 表示原内容可用，仅用统一契约防回退。

| slug | 初审 | 已处理的缺口 |
|---|---:|---|
| `3_1_1_collaborative_filtering` | P2 | 补相似度/邻域手算与通用向量链接 |
| `3_1_2_matrix_factorization` | P2 | 补低秩单格重构、损失到更新与优化链接 |
| `3_1_3_factorization_machine` | A | 固化二阶恒等式、符号和数据任务边界 |
| `3_1_4_gbdt_lr` | P1 | 补叶编码→LR 公式，并隔离 Facebook NE 与 MovieLens 教学结果 |
| `3_1_5_word2vec` | P2 | 补负采样逐项代入、未知项语义与信息论链接 |
| `3_2_1_dssm` | A | 固化双塔/余弦/条件概率与搜索论文到推荐迁移边界 |
| `3_2_2_mind` | P1 | 补动态路由、squash、label-aware attention 及 19:1/时间切分边界 |
| `3_2_3_sasrec` | A | 固化 Q/K/V、因果遮罩、逐位置损失与候选协议 |
| `3_3_1_deepfm` | A | 固化 FM 化简、共享 embedding 和 Criteo/KuaiRand 边界 |
| `3_3_2_din` | P1 | 补未归一化激活、Dice/MBA 正则，并隔离阿里广告与 KuaiRand 指标 |
| `3_3_3_dien` | A | 固化辅助损失、GRU/AUGRU 与严格时序语义 |
| `3_4_1_mmoe` | A | 固化任务 gate、张量形状和共享梯度路径 |
| `3_4_2_ple` | A | 固化 CGC、渐进抽取、样本空间与任务权重 |
| `4_1_openonerec_practice` | P1 | 补自回归/DPO/trie 推导，并隔离官方 RecIF/reward 与本地 smoke |
| `4_2_dlrm_hstu_practice` | P1 | 补非 Softmax 的 SiLU 聚合，并隔离 Meta full 协议与 KuaiRand 适配器 |

## 面向高中毕业生的教学微契约

每节遵循：日常问题与直觉 → 先定义符号和形状 → 至少一次 2–4 个数字的逐步代入 → NumPy/图形验证 → 常见误区 → 精确算法回链 → Checks / Next Steps。算法页还必须按“通用先修 / 本论文新增数学”分层，并明确 loss 如何驱动训练。

## 真源与生成关系

- `app/content.py`：Notebook 角色、3.0 课程 manifest、模型到课程覆盖关系。
- `scripts/tutorial_math_specs.py`：6 门课程正文与稳定锚点。
- `scripts/generate_notebooks.py`：课程 helper、3.0 与经典算法生成源。
- `scripts/tutorial_deep_specs.py`：深度/生成算法的论文数学、先修链接和复现实验口径。
- `scripts/tutorial_opening_specs.py`：各大章开篇的共性路线图。
- `notebooks/*.ipynb` 是生成物，不应成为长期手工真源。

## 验证契约

`tests/test_notebook_content.py` 以 `NOTEBOOKS.kind` 区分课程与实验：非课程页继续验证真实数据及零伪造行；课程页验证 6 组锚点、学习路径、符号表、数字代入、误区、算法回链、至少两个代码演示、Checks/Next Steps、无错误输出与 `recsys.kind=curriculum`。同一测试解析全部 15 个算法页的 3.0 href，确认目标 slug 和 anchor 均存在，并对五个 P1 页面锁定论文公式、论文数据协议和本仓库教学协议三组证据。`tests/test_knowledge_graph.py` 另行约束 A.4 投影的节点上限、展开深度、稳定顺序和已解析边。
