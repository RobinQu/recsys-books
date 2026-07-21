from __future__ import annotations


def build_math_specs(md, code, curriculum_notebook):
    """Six progressive 3.0 foundation lessons written for a high-school graduate."""

    specs = {}

    specs["3_0_1_data_ml_basics"] = curriculum_notebook(
        "3.0.1 数据与机器学习基础：一行数据到底代表什么",
        "从真实 MovieLens 行出发，分清观察、实体、特征、标签、隐式反馈、切分与泄漏，并建立监督、自监督、多任务、基线与泛化的共同语言。",
        [
            md(r"""
## 学习路径

1. 先问“一行是谁在什么时刻对什么做了什么”，再决定预测目标。
2. 分清观察到的反馈、没有观察到的未知项，以及真实负反馈。
3. 把类别、数值、序列整理成模型能读取的形状。
4. 沿时间切分训练/验证/测试，最后才谈指标、基线和泛化。

## 符号表

| 符号 | 含义 | 例子 |
|---|---|---|
| $u,i,t$ | 用户、物品、时间 | 用户 7、电影 12、时间戳 |
| $x$ | 输入特征 | 用户/电影 ID、类型、历史 |
| $y$ | 标签，即希望模型回答的答案 | 是否点击、评分、下一物品 |
| $D_{train},D_{val},D_{test}$ | 训练、调参、最终检查数据 | 边界确定后不能互相偷看 |
| $\hat y$ | 模型预测 | 点击概率 0.73 |

<a id="observation-label"></a>
## 1. 观察、实体、特征与标签

一行记录首先是一次**观察**，不天然等于一个独立的人。MovieLens 的一行表示某用户在某时刻给某电影评分；同一个用户会出现许多行。实体是用户和电影，特征是预测前可知道的信息，标签是这道题要求模型预测的答案。把评分既放进特征又当标签，就像把答案印在试卷正面。

类别 ID 只是名字，不表示距离：电影 20 不比电影 10“大一倍”。one-hot 用一个位置表示一个类别，embedding 则把类别查成一行可学习数字。稀疏向量大多为 0；embedding 是短而稠密的表示。序列和会话还保留顺序：`[电影A, 电影B]` 与反序不同。

### 数字代入 1：样本数不等于用户数

若用户 A 有 3 行、B 有 2 行、C 有 1 行，则观察数 $N=3+2+1=6$，用户实体数却只有 $3$。若按行随机切分，同一用户的未来记录很可能泄漏进训练集。

### 数字代入 2：类别编码

三种类型的 one-hot 可写成喜剧 $[1,0,0]$、剧情 $[0,1,0]$、动画 $[0,0,1]$。它们两两距离相同，不会误导模型认为“动画编号 3 比喜剧编号 1 大两倍”。
"""),
            code(r"""
# Demo 1：这些是真实 MovieLens 行，不是人工交互。
columns = ["userId", "movieId", "rating", "timestamp"]
display(real_ratings[columns].head())
entity_counts = np.array([len(real_ratings), real_ratings.userId.nunique(), real_ratings.movieId.nunique()])
print({"observations": int(entity_counts[0]), "user_entities": int(entity_counts[1]),
       "item_entities": int(entity_counts[2])})
plt.bar(["observations", "users", "items"], entity_counts, color=["#446f82", "#79a53b", "#d09b3d"])
plt.title("真实行数与实体数不是同一个概念"); plt.tight_layout(); plt.show()
assert len(real_ratings) > real_ratings.userId.nunique()
"""),
            md(r"""
<a id="implicit-feedback"></a>
## 2. 显式、隐式反馈与“未知不等于负例”

评分是显式反馈；点击、观看、收藏是隐式反馈。隐式反馈只说明行为发生了，不保证用户真正喜欢。更关键的是：目录中没出现的物品可能没被曝光，所以“没有点击”可能是未知，而不是真负例。只有日志明确记录“曝光后未点击”，才有资格直接成为未点击标签；训练时人工抽出的负样本必须标明采样规则。

监督学习用已有标签预测答案；自监督学习从数据自身构造答案，例如遮住序列中的电影再预测；多任务学习同时预测点击、长播、收藏。三者改变的是学习题目，不会改变原始事实。

### 数字代入 3：分母决定点击率含义

用户得到 100 次曝光，点击 8 次，曝光点击率为 $8/100=8\%$。若日志只保存 8 次点击，不能写成 $8/8=100\%$；后者的分母已经丢失。
"""),
            code(r"""
# Demo 2：用真实评分派生一个“高评分”教学标签，同时保留来源行。
sample = real_ratings[["userId", "movieId", "rating", "timestamp"]].head(200).copy()
sample["high_rating_label"] = (sample.rating >= 4.0).astype(int)
rates = sample.groupby("userId").high_rating_label.mean().head(12)
rates.plot(kind="bar", color="#79a53b", title="真实 MovieLens 用户的高评分比例（前 12 位）")
plt.ylabel("fraction"); plt.tight_layout(); plt.show()
print(sample.head())
assert set(sample.high_rating_label.unique()).issubset({0, 1})
"""),
            md(r"""
<a id="split-leakage"></a>
## 3. 切分、泄漏、基线与泛化

训练集用于拟合参数，验证集用于选模型，测试集只在最终报告时使用。推荐系统通常按用户内时间切分：用过去预测未来。泄漏包括把未来行为放进历史、用全数据计算归一化统计、反复查看测试结果调参。

基线是最简单的合理方案，例如热门推荐或逻辑回归；新模型必须在同一切分、候选集和指标上比较。泛化指模型能否在未见过的未来样本上继续工作。离线指标只能回答既定日志和协议中的问题，还要警惕曝光偏差、热门偏差、假负例和概率校准。

### 数字代入 4：时间边界

某用户事件时间为 $[2,5,9,12]$。leave-last-out 得训练 $[2,5,9]$、测试 $[12]$。若把时间 12 的电影加入训练历史，再预测时间 12，答案已经泄漏。

## 常见误区

- “没点就是不喜欢”：未曝光与曝光未点必须分开。
- “随机切分更平均”：它可能把未来信息放到过去。
- “ID 是普通数值”：ID 只用于定位类别，不能直接解释大小。
- “AUC 高就一定上线好”：概率校准、候选分布、时延与业务目标仍需检查。

## 算法回链

- [GBDT+LR：曝光标签与校准](/notebooks/3_1_4_gbdt_lr)
- [DSSM：负采样与训练/服务分布](/notebooks/3_2_1_dssm)
- [SASRec：严格的时间序列切分](/notebooks/3_2_3_sasrec)
- [DIN：候选与行为历史的语义](/notebooks/3_3_2_din)
- [MMoE：点击和长播多任务](/notebooks/3_4_1_mmoe)

## Checks

1. 为什么“目录中没出现”不能直接写成负反馈？
2. 预测点击时，哪些列在预测发生前可知？
3. 为什么测试集不能参与选择学习率？
"""),
            code(r"""
ordered = real_ratings.sort_values(["userId", "timestamp", "movieId"])
last = ordered.groupby("userId").tail(1)
train = ordered.drop(last.index)
assert set(train.index).isdisjoint(last.index)
assert (train.groupby("userId").timestamp.max() <= last.set_index("userId").timestamp).all()
print("PASS：真实行可追溯；标签、实体计数和按用户时间切分检查通过。")
"""),
            md("## Next Steps\n\n下一课把 one-hot、embedding、行为矩阵与序列统一成向量、矩阵和张量；回到算法页时，先写清楚一行数据的语义，再看模型公式。"),
        ],
        uses_movielens=True,
    )

    specs["3_0_2_linear_algebra"] = curriculum_notebook(
        "3.0.2 线性代数：把推荐数据放进有形状的盒子",
        "理解标量、向量、矩阵、张量及其轴和形状，亲手计算逐元素乘法、点积、矩阵乘法、范数、余弦、embedding、低秩分解与注意力形状。",
        [
            md(r"""
## 学习路径

先辨认形状，再辨认运算；先手算一个格子，再让 NumPy 批量算；最后把矩阵乘法连接到 embedding、低秩模型和 Q/K/V 注意力。

## 符号表

| 符号 | 含义 | 形状示例 |
|---|---|---|
| $a$ | 标量，一个数 | `()` |
| $x$ | 向量，一排数 | $(d,)$ |
| $X$ | 矩阵，行×列 | $(B,d)$ |
| $T$ | 张量，三轴及以上 | $(B,L,d)$ |
| $X^\top$ | 转置，交换行列 | $(d,B)$ |
| $\|x\|_2$ | 向量长度 | 标量 |

> 本课所有数组都是**数学教学对象**，不是用户、曝光或交互记录。

<a id="tensors-shapes"></a>
## 1. 标量、向量、矩阵、张量、轴与形状

可把形状想成盒子的尺寸。$X\in\mathbb R^{3\times2}$ 有 3 行 2 列；序列张量 $(B,L,d)$ 依次表示批大小、序列长度、每个位置的表示维数。轴不是抽象术语：沿轴 1 求和，就是把每行的列加起来。

### 数字代入 1：沿轴求和

$$X=\begin{bmatrix}1&2\\3&4\\5&6\end{bmatrix}.$$

沿列方向合并每行得到 $[1+2,3+4,5+6]=[3,7,11]$；沿行方向合并每列得到 $[1+3+5,2+4+6]=[9,12]$。

<a id="elementwise-dot"></a>
## 2. 逐元素运算、点积、范数与余弦

逐元素乘法保留形状：$[1,2]\odot[3,4]=[3,8]$。点积再求和：$[1,2]\cdot[3,4]=1\times3+2\times4=11$。长度 $\|x\|_2=\sqrt{\sum x_j^2}$；余弦除去长度，只比较方向。

### 数字代入 2：长度与余弦

$x=[3,4]$ 的长度为 $5$；$y=[6,8]$ 长度为 $10$，点积 $50$，所以余弦 $50/(5\times10)=1$：长度不同但方向完全相同。
"""),
            code(r"""
# Demo 1：形状、轴、逐元素、点积与余弦。
X = np.array([[1., 2.], [3., 4.], [5., 6.]])
x, y = np.array([3., 4.]), np.array([6., 8.])
cosine = (x @ y) / (np.linalg.norm(x) * np.linalg.norm(y))
print({"shape": X.shape, "sum_axis_1": X.sum(axis=1).tolist(),
       "elementwise": (x*y).tolist(), "dot": float(x@y), "cosine": cosine})
fig, ax = plt.subplots(figsize=(5, 4))
ax.quiver([0,0], [0,0], [x[0],y[0]], [x[1],y[1]], angles="xy", scale_units="xy", scale=1)
ax.set(xlim=(0,9), ylim=(0,9), title="同方向、不同长度的教学向量"); ax.grid(); plt.show()
"""),
            md(r"""
<a id="matmul-embedding"></a>
## 3. 矩阵乘法与 embedding 查表

若 $A$ 形状 $(m,n)$、$B$ 形状 $(n,p)$，中间的 $n$ 必须相同，结果是 $(m,p)$。结果第 $(r,c)$ 格是 $A$ 第 $r$ 行与 $B$ 第 $c$ 列的点积。one-hot 乘 embedding 表等价于查一行。

### 数字代入 3：手算矩阵的一个格子

$$A=\begin{bmatrix}1&2\\0&1\end{bmatrix},\quad B=\begin{bmatrix}3&4\\5&6\end{bmatrix}.$$

左上角为 $1\times3+2\times5=13$，右上角为 $1\times4+2\times6=16$，所以 $AB=\begin{bmatrix}13&16\\5&6\end{bmatrix}$。one-hot $[0,1,0]$ 乘三行 embedding 表，会精确取出第二行。

加权和也是矩阵乘法：权重 $[0.2,0.8]$ 加权两个向量 $[1,0]$、$[0,1]$ 得 $[0.2,0.8]$。归一化常把权重总和变为 1，方便解释“分票”。

<a id="low-rank-attention"></a>
## 4. 低秩、转置与 Q/K/V 形状

低秩近似 $R\approx PQ^\top$ 用两个小矩阵描述大矩阵：$P$ 每行是一个用户坐标，$Q$ 每行是一个物品坐标。注意力中 $QK^\top$ 把每个 query 与每个 key 做点积；若 $Q,K,V$ 均为 $(B,L,d)$，则分数为 $(B,L,L)$，乘 $V$ 后回到 $(B,L,d)$。

### 数字代入 4：低秩重构一个格子

用户坐标 $p=[1,2]$，物品坐标 $q=[3,-1]$，重构分数为 $p^\top q=1\times3+2\times(-1)=1$。这只是数学对象，不是声称某用户真的给了 1 分。
"""),
            code(r"""
# Demo 2：embedding 查表、低秩重构和注意力形状。
embedding_table = np.array([[1.,0.], [0.,2.], [-1.,1.]])
one_hot = np.array([0.,1.,0.])
print("one-hot @ table =", one_hot @ embedding_table)
R_math = np.array([[5.,4.,1.], [4.,3.,1.], [1.,1.,5.]])  # 数学教学矩阵，不是交互行
U, s, Vt = np.linalg.svd(R_math, full_matrices=False)
rank2 = (U[:, :2] * s[:2]) @ Vt[:2]
Q = np.arange(2*3*4, dtype=float).reshape(2,3,4) / 10
K, V = Q + .1, Q - .1
scores = Q @ K.transpose(0,2,1)
output = scores @ V
print({"rank2_shape": rank2.shape, "attention_scores": scores.shape, "output": output.shape})
fig, axes = plt.subplots(1,2,figsize=(8,3))
axes[0].imshow(R_math); axes[0].set_title("教学矩阵")
axes[1].imshow(rank2); axes[1].set_title("rank-2 近似")
plt.tight_layout(); plt.show()
"""),
            md(r"""
## 常见误区

- `*` 与 `@` 相同：前者通常逐元素，后者是矩阵乘法。
- 形状能广播就一定语义正确：程序能跑不代表轴对齐正确。
- embedding 维度有固定含义：各坐标通常联合学习，不能单独命名。
- 低秩重构是恢复事实：它是模型估计，会有误差和偏差。

## 算法回链

- [ItemCF：矩阵共现与余弦](/notebooks/3_1_1_collaborative_filtering)
- [BiasMF：低秩分解](/notebooks/3_1_2_matrix_factorization)
- [DSSM：批量 user-item 点积](/notebooks/3_2_1_dssm)
- [MIND：多个兴趣的加权聚合](/notebooks/3_2_2_mind)
- [SASRec：Q/K/V 与因果注意力](/notebooks/3_2_3_sasrec)

## Checks

1. $(8,20,16)$ 的序列张量中三个轴各可能代表什么？
2. 为什么 $(3,4)@(4,5)$ 得 $(3,5)$？
3. 余弦为 1 是否表示两个向量完全相等？
"""),
            code(r"""
assert X.shape == (3,2)
assert np.isclose(cosine, 1.0)
assert (np.array([[1,2],[0,1]]) @ np.array([[3,4],[5,6]])).tolist() == [[13,16],[5,6]]
assert scores.shape == (2,3,3) and output.shape == (2,3,4)
print("PASS：轴、点积、矩阵乘法、低秩和注意力形状检查通过。")
"""),
            md("## Next Steps\n\n下一课把模型看作函数的复合：线性代数负责算出预测，微积分负责回答参数该往哪边改。"),
        ],
    )

    specs["3_0_3_calculus"] = curriculum_notebook(
        "3.0.3 微积分：参数动一点，损失会怎样",
        "从斜率和有限差分进入导数、偏导、梯度、方向导数、链式法则、计算图、反向传播与激活饱和。",
        [
            md(r"""
## 学习路径

先把模型看成输入到输出的函数；用相邻两点估计坡度；再把一个变量推广到许多参数；最后沿计算图反向应用链式法则。

## 符号表

| 符号 | 含义 |
|---|---|
| $f(x)$ | 输入 $x$ 经过函数后的输出 |
| $f'(x)$ 或 $df/dx$ | $x$ 动一点时输出的瞬时变化率 |
| $\partial L/\partial w$ | 其他变量暂时固定时，损失对 $w$ 的偏导 |
| $\nabla L$ | 把所有参数偏导排成的梯度向量 |
| $\eta$ | 学习率，即每步走多远 |

> 下文数值全部是数学教学对象，不代表真实交互。

<a id="functions"></a>
## 1. 函数与复合

函数像一台规则明确的机器。模型常是复合函数：$x\xrightarrow{g}z\xrightarrow{h}\hat y\xrightarrow{\ell}L$。例如 $z=wx+b$，$hat y=\sigma(z)$，再由标签和预测计算损失。

### 数字代入 1：复合三步

令 $x=2,w=3,b=-1$，则 $z=3\times2-1=5$；若 $h(z)=z^2$，则 $h(5)=25$；若目标为 20，平方误差 $(25-20)^2=25$。每个中间量都能单独核对。

<a id="derivative-gradient"></a>
## 2. 导数、偏导、梯度与方向

有限差分用 $[f(x+h)-f(x)]/h$ 估计斜率。对 $f(x)=x^2$、$x=3$、$h=0.01$，得到 $(3.01^2-3^2)/0.01=6.01$，接近导数 $2x=6$。

多个参数时分别求偏导，再组成梯度。方向导数是梯度与单位方向 $v$ 的点积 $\nabla L\cdot v$；负梯度是局部下降最快方向。

### 数字代入 2：二维梯度

$L(w,b)=(w-2)^2+(b+1)^2$，在 $(w,b)=(4,2)$ 处，梯度为 $[2(4-2),2(2+1)]=[4,6]$。沿 $-[4,6]$ 走一小步会降低损失。
"""),
            code(r"""
# Demo 1：有限差分靠近解析导数，并画出切线。
f = lambda x: x**2
x0 = 3.0
for h in [1., .1, .01, .001]:
    print(h, (f(x0+h)-f(x0))/h)
xs = np.linspace(0,6,200)
plt.plot(xs, f(xs), label="$x^2$")
plt.plot(xs, f(x0)+2*x0*(xs-x0), "--", label="x=3 的切线")
plt.scatter([x0],[f(x0)]); plt.legend(); plt.grid(); plt.show()
"""),
            md(r"""
<a id="chain-rule"></a>
## 3. 链式法则、计算图与反向传播

若 $y=h(z)$、$z=g(x)$，那么 $dy/dx=(dy/dz)(dz/dx)$。反向传播不是另一套数学，而是把这条规则从损失端沿计算图逐边相乘，并把分叉路径的贡献相加。

### 数字代入 3：一条链反向算

$z=2x+1$，$y=z^2$。在 $x=3$ 时 $z=7$。$dy/dz=2z=14$，$dz/dx=2$，所以 $dy/dx=14\times2=28$。直接展开 $y=(2x+1)^2$ 求导，也得 $2(2x+1)\times2=28$。

激活函数会影响梯度。Sigmoid 在 $z=0$ 的导数为 $0.25$，到 $z=8$ 时接近 0，叫“饱和”；许多层连续乘很小的导数，会造成梯度消失。ReLU 正区间导数为 1，但负区间为 0，也要留意神经元长期不激活。

### 数字代入 4：饱和

$\sigma(0)=0.5$，导数 $0.5(1-0.5)=0.25$；$\sigma(8)\approx0.9997$，导数约 $0.00034$，一层就把传回的梯度缩小约 735 倍。
"""),
            code(r"""
# Demo 2：计算图的数值梯度与解析梯度；同时观察 Sigmoid 饱和。
def composed(x):
    z = 2*x + 1
    return z**2
x = 3.; eps = 1e-5
numeric = (composed(x+eps)-composed(x-eps))/(2*eps)
analytic = 2*(2*x+1)*2
z = np.linspace(-10,10,300); sig = 1/(1+np.exp(-z)); derivative = sig*(1-sig)
fig, ax = plt.subplots(1,2,figsize=(9,3))
ax[0].plot(z,sig); ax[0].set_title("Sigmoid")
ax[1].plot(z,derivative); ax[1].set_title("Sigmoid derivative")
for a in ax: a.grid()
plt.tight_layout(); plt.show()
print({"numeric": numeric, "analytic": analytic})
"""),
            md(r"""
## 常见误区

- 导数是“函数值”：导数是局部变化率，可能为负。
- 偏导时所有变量一起变：求某一偏导时暂时固定其他变量。
- 梯度就是更新方向：梯度指向上升最快，梯度下降走负梯度。
- 反向传播会自动保证模型正确：它只忠实求当前计算图的导数。

## 算法回链

- [BiasMF：评分误差对隐向量的梯度](/notebooks/3_1_2_matrix_factorization)
- [DeepFM：共享 embedding 的两条梯度路径](/notebooks/3_3_1_deepfm)
- [DIEN：序列网络与辅助损失反传](/notebooks/3_3_3_dien)
- [HSTU：深层序列模型的梯度稳定性](/notebooks/4_3_dlrm_hstu_practice)

## Checks

1. $f(x)=3x+2$ 的导数为何处处为 3？
2. 梯度 $[4,6]$ 时，下降应取哪个方向？
3. 链式法则为什么是相乘而不是相加？
"""),
            code(r"""
assert abs(numeric-analytic) < 1e-4
gradient = np.array([4.,6.]); before = (4-2)**2 + (2+1)**2
after_point = np.array([4.,2.]) - .1*gradient
after = (after_point[0]-2)**2 + (after_point[1]+1)**2
assert after < before
print("PASS：有限差分、链式法则与负梯度下降检查通过。")
"""),
            md("## Next Steps\n\n下一课解释标签为何能写成概率和似然；之后再把梯度与学习率、正则化组合成完整优化过程。"),
        ],
    )

    specs["3_0_4_probability_statistics"] = curriculum_notebook(
        "3.0.4 概率与统计：把不确定性说清楚",
        "理解事件、随机变量、Bernoulli/类别分布、联合/边缘/条件概率、独立性、链式法则、期望、方差、odds、likelihood、采样偏差与校准。",
        [
            md(r"""
## 学习路径

从一次不确定事件开始；用随机变量给结果编码；用条件概率描述“已知什么”；再用期望和方差概括分布，最后区分概率、似然、采样偏差与校准。

## 符号表

| 符号 | 含义 |
|---|---|
| $P(A)$ | 事件 A 发生的概率 |
| $X$ | 取值不确定的随机变量 |
| $P(A\mid B)$ | 已知 B 后 A 的概率 |
| $E[X]$ | 长期平均值，即期望 |
| $Var(X)$ | 围绕期望的波动大小 |
| $p/(1-p)$ | odds，发生与不发生机会之比 |

> 所有抛硬币、类别和分数数组都是数学教学对象，不是交互数据。

<a id="random-variable"></a>
## 1. 事件、随机变量与分布

事件是可判断真假的集合，例如“点击”；随机变量把结果映射成数。Bernoulli 变量只取 0/1，参数 $p$；类别变量在多个互斥类别中取一个，所有类别概率和为 1。分布不是一串已经发生的数据，而是对可能结果及其概率的说明。

### 数字代入 1：Bernoulli

若 $P(X=1)=0.3$，则 $P(X=0)=0.7$。重复 10 次的期望成功数是 $10\times0.3=3$，但某次实际出现 2 或 5 次都不矛盾。

<a id="conditional-chain"></a>
## 2. 联合、边缘、条件、独立与概率链式法则

联合概率 $P(A,B)$ 描述两件事同时发生；边缘概率把另一变量的所有可能求和；条件概率 $P(A\mid B)=P(A,B)/P(B)$。独立要求 $P(A,B)=P(A)P(B)$，不是简单的“不相等”。链式法则为 $P(A,B)=P(A\mid B)P(B)$，序列可继续展开。

### 数字代入 2：条件与链式法则

100 个教学对象中 40 个满足 B，其中 10 个同时满足 A，则 $P(B)=0.4$、$P(A,B)=0.1$、$P(A\mid B)=10/40=0.25$；反乘 $0.25\times0.4=0.1$ 回到联合概率。
"""),
            code(r"""
# Demo 1：从一个明确给定的数学联合分布求边缘与条件概率。
joint = np.array([[.42,.18], [.28,.12]])  # rows A=0/1, cols B=0/1；教学分布
p_a = joint.sum(axis=1); p_b = joint.sum(axis=0)
p_a1_given_b1 = joint[1,1] / p_b[1]
print({"P(A)": p_a.tolist(), "P(B)": p_b.tolist(), "P(A=1|B=1)": p_a1_given_b1})
fig, ax = plt.subplots(figsize=(4,3)); im=ax.imshow(joint,cmap="YlGn")
for r in range(2):
    for c in range(2): ax.text(c,r,f"{joint[r,c]:.2f}",ha="center",va="center")
ax.set(xlabel="B", ylabel="A", title="数学联合分布"); plt.colorbar(im); plt.show()
"""),
            md(r"""
<a id="expectation-variance"></a>
## 3. 期望与方差

$E[X]=\sum_x xP(X=x)$ 是按概率加权的平均；$Var(X)=E[(X-E[X])^2]$ 先看每个值离均值多远，再平方加权。两组预测平均相同，方差可能完全不同。

### 数字代入 3：同均值不同波动

教学变量等概率取 $[1,3]$，期望为 2，方差为 $[(1-2)^2+(3-2)^2]/2=1$。若等概率取 $[-1,5]$，期望仍为 2，方差变为 9。

<a id="likelihood-calibration"></a>
## 4. odds、logit、似然、采样偏差与校准

概率 $p=0.8$ 的 odds 是 $0.8/0.2=4$，logit 为 $\log4\approx1.386$；意思是发生机会约为不发生的 4 倍。概率固定参数后问“数据会怎样”；似然固定观察数据后问“哪个参数更支持这些数据”。

采样偏差指样本来源改变了总体比例，例如只记录热门物品曝光。校准要求预测为 0.7 的一组样本中，长期约 70% 为正；排序很好不代表校准就好。

### 数字代入 4：似然与校准

观察教学标签 $[1,1,0]$。若 Bernoulli 参数 $p=0.8$，似然为 $0.8\times0.8\times0.2=0.128$；若 $p=0.5$，为 $0.125$。在这个很小样本里两者相近，不能过度下结论。若 10 个预测都为 0.7 而只有 4 个为正，经验正率 0.4 与 0.7 相差 0.3，提示未校准。
"""),
            code(r"""
# Demo 2：期望/方差和一个简单可靠性图。
values1, values2 = np.array([1.,3.]), np.array([-1.,5.])
print({"mean1": values1.mean(), "var1": values1.var(),
       "mean2": values2.mean(), "var2": values2.var()})
pred = np.array([.1,.2,.2,.4,.5,.6,.7,.8,.8,.9])       # 教学预测
label = np.array([0,0,1,0,1,1,1,1,0,1])               # 教学标签
bins = np.array([0,.33,.66,1.01]); ids=np.digitize(pred,bins)-1
mean_pred=[pred[ids==i].mean() for i in range(3)]
rate=[label[ids==i].mean() for i in range(3)]
plt.plot([0,1],[0,1],"--",label="perfect"); plt.plot(mean_pred,rate,"o-",label="teaching model")
plt.xlabel("mean prediction"); plt.ylabel("positive rate"); plt.legend(); plt.grid(); plt.show()
"""),
            md(r"""
## 常见误区

- 概率 0.7 意味着本次一定发生：它描述重复条件下的不确定性。
- $P(A\mid B)=P(B\mid A)$：分母不同，通常不相等。
- 不相关就等于独立：独立条件更强。
- 排序指标高就概率准确：AUC 与校准回答不同问题。

## 算法回链

- [GBDT+LR：logit 与点击概率](/notebooks/3_1_4_gbdt_lr)
- [DSSM：条件概率与负采样](/notebooks/3_2_1_dssm)
- [DIN：候选条件下的点击概率](/notebooks/3_3_2_din)
- [MMoE：多个 Bernoulli 任务](/notebooks/3_4_1_mmoe)
- [OpenOneRec：序列条件概率](/notebooks/4_2_openonerec_practice)

## Checks

1. 为什么 $P(A\mid B)$ 与 $P(B\mid A)$ 通常不同？
2. 期望相同是否代表风险相同？
3. 如何用一组 0.8 预测检查校准？
"""),
            code(r"""
assert np.isclose(joint.sum(),1)
assert np.isclose(p_a1_given_b1*p_b[1], joint[1,1])
assert values1.mean()==values2.mean() and values1.var()<values2.var()
assert np.isclose((.8**2)*.2, .128)
print("PASS：联合/条件、期望/方差、似然计算检查通过。")
"""),
            md("## Next Steps\n\n下一课从信息量与熵出发，解释交叉熵、Softmax、序列 NLL 等概率损失为何以对数为共同语言。"),
        ],
    )

    specs["3_0_6_optimization"] = curriculum_notebook(
        "3.0.6 优化：怎样让模型稳定地下山",
        "区分训练目标与评测指标，理解 GD/SGD/mini-batch、学习率、Momentum/Adam、凸与非凸、初始化、L1/L2、早停、梯度裁剪和多任务梯度冲突。",
        [
            md(r"""
## 学习路径

先画出目标函数地形；用梯度决定方向、学习率决定步长；再理解随机小批次的噪声，最后加入正则化、早停和梯度稳定措施。

## 符号表

| 符号 | 含义 |
|---|---|
| $\theta$ | 模型全部可学习参数 |
| $L(\theta)$ | 训练目标/损失 |
| $\nabla L$ | 当前点的梯度 |
| $\eta$ | 学习率 |
| $\lambda$ | 正则强度 |
| $B$ | mini-batch 样本数 |

> 本课曲线、参数和梯度均为数学教学对象，不是实验效果声明。

<a id="sgd"></a>
## 1. 目标、指标、GD、SGD 与 mini-batch

目标是训练时直接最小化的数，如交叉熵；指标是最终关心的评价，如 Recall@K，它可能不可导。全批 GD 用全部数据求梯度，SGD 用一个样本，mini-batch 用一小批：速度、内存与噪声之间折中。

### 数字代入 1：一次梯度下降

$L(w)=(w-3)^2$，在 $w=0$ 的梯度 $2(w-3)=-6$。学习率 $\eta=0.1$ 时，$w\leftarrow0-0.1(-6)=0.6$；损失从 9 降到 $(0.6-3)^2=5.76$。

<a id="learning-rate"></a>
## 2. 学习率、Momentum、Adam、凸与非凸

学习率太小走得慢，太大会跨过谷底甚至发散。Momentum 像带惯性的球，累积一致方向；Adam 再按每个参数近期梯度平方调整步长。凸函数像单碗，任意局部最低也是全局最低；深度网络通常非凸，更像有坡、鞍点和平坦区的山地，因此初始化和随机性重要。

### 数字代入 2：大步跨谷

仍取 $w=0$、梯度 $-6$。若 $\eta=1.1$，一步到 $6.6$，损失变 $12.96$，反而比 9 大。更新公式没错，错的是步长与曲率不匹配。
"""),
            code(r"""
# Demo 1：同一教学目标，不同学习率的轨迹。
def path(lr, steps=12):
    w=0.; out=[w]
    for _ in range(steps):
        w -= lr*2*(w-3); out.append(w)
    return np.array(out)
fig, ax=plt.subplots(figsize=(7,3.5))
for lr in [.05,.3,1.1]:
    p=path(lr); ax.plot((p-3)**2,"o-",label=f"lr={lr}")
ax.set_yscale("log"); ax.set(xlabel="step",ylabel="loss",title="学习率改变下山轨迹"); ax.legend(); ax.grid(); plt.show()
"""),
            md(r"""
<a id="regularization"></a>
## 3. L1/L2、过拟合与早停

训练误差低而验证误差升高叫过拟合。L2 在目标中加入 $\lambda\sum_jw_j^2$，平滑地压小权重；L1 加 $\lambda\sum_j|w_j|$，更容易把部分权重推到 0。早停在验证指标不再改善时保存最佳点，不能根据测试集早停。

### 数字代入 3：L2 代价

$w=[3,4]$，原损失为 2，$\lambda=0.1$。L2 项为 $0.1(3^2+4^2)=2.5$，总目标 $4.5$。若权重减半，L2 项降为 $0.625$，但数据误差可能上升；训练在两者间权衡。

梯度爆炸指梯度过大，裁剪可令 $g\leftarrow g\min(1,c/\|g\|)$；梯度消失则需检查激活、初始化、归一化和网络路径，单靠裁剪无效。

<a id="gradient-conflict"></a>
## 4. 多任务梯度冲突

多任务共享参数时，各任务梯度可能同向也可能冲突。若点击梯度 $g_1=[1,0]$、长播梯度 $g_2=[-0.8,0.2]$，点积为 $-0.8<0$，说明对一个任务下降的方向可能伤害另一个。任务加权、专家分离或梯度处理都在管理这类跷跷板。

### 数字代入 4：梯度裁剪

$g=[6,8]$ 的长度是 10；阈值 $c=5$ 时乘 $5/10$，得到 $[3,4]$，方向不变、长度变 5。
"""),
            code(r"""
# Demo 2：L2 改变最优点；并画出两个冲突梯度。
xs=np.linspace(-1,5,300); data_loss=(xs-3)**2
fig,ax=plt.subplots(1,2,figsize=(9,3.3))
for lam in [0,.1,1.]: ax[0].plot(xs,data_loss+lam*xs**2,label=f"lambda={lam}")
ax[0].set(title="L2 改变教学目标",xlabel="w",ylabel="objective"); ax[0].legend(); ax[0].grid()
g1=np.array([1.,0.]); g2=np.array([-.8,.2])
ax[1].quiver([0,0],[0,0],[*g1],[*g2],angles="xy",scale_units="xy",scale=1)
ax[1].set(xlim=(-1.1,1.1),ylim=(-.3,.5),title=f"gradient dot={g1@g2:.1f}"); ax[1].grid()
plt.tight_layout(); plt.show()
g=np.array([6.,8.]); clipped=g*min(1,5/np.linalg.norm(g)); print("clipped",clipped)
"""),
            md(r"""
## 常见误区

- 损失越低，业务指标一定越好：训练代理目标与最终指标不同。
- Adam 不需要学习率：它自适应缩放，但全局步长仍关键。
- 正则越强越好：过强会欠拟合。
- 多任务总损失下降就都变好：某个任务可能被牺牲。

## 算法回链

- [BiasMF：SGD 与 L2](/notebooks/3_1_2_matrix_factorization)
- [DeepFM：mini-batch 与共享参数](/notebooks/3_3_1_deepfm)
- [MMoE：gate 与任务冲突](/notebooks/3_4_1_mmoe)
- [PLE：共享/专属专家缓解负迁移](/notebooks/3_4_2_ple)
- [HSTU：长序列训练稳定性](/notebooks/4_3_dlrm_hstu_practice)

## Checks

1. 为什么测试指标不能用来早停？
2. 学习率太大时损失为何可能来回变大？
3. 两任务梯度点积为负说明什么？
"""),
            code(r"""
assert path(.1)[-1] > 0 and (path(.1)[-1]-3)**2 < 9
assert np.isclose(np.linalg.norm(clipped),5)
assert g1@g2 < 0
assert np.isclose(2 + .1*(3**2+4**2),4.5)
print("PASS：学习率、L2、裁剪与梯度冲突检查通过。")
"""),
            md("## Next Steps\n\n继续进入 3.0.7 数据与实验基础，把数据切分、张量化、训练、推理和测试连接成完整工程管线。"),
        ],
    )

    specs["3_0_5_information_theory"] = curriculum_notebook(
        "3.0.5 信息论：一次意外有多少信息",
        "从信息量和熵进入二元/多类交叉熵、CE=H+KL、KL 非对称、Softmax/温度、Normalized Entropy、序列 NLL、perplexity 与 DPO log-ratio。",
        [
            md(r"""
## 学习路径

先比较常见与意外消息的信息量；把平均意外程度写成熵；再用交叉熵评估模型概率，最后连接分类、序列生成和偏好优化。

## 符号表

| 符号 | 含义 |
|---|---|
| $p(x)$ | 真实/目标分布 |
| $q(x)$ | 模型给出的分布 |
| $I(x)=-\log_2p(x)$ | 事件的信息量（bit） |
| $H(p)$ | 分布自身的不确定性 |
| $H(p,q)$ | 用模型 q 编码来自 p 的结果的交叉熵 |
| $D_{KL}(p\|q)$ | q 相对 p 多付出的平均编码代价 |

> 本课概率、token 与偏好分数均为数学教学对象，不是假造行为序列。

<a id="information-entropy"></a>
## 1. 信息量与熵

越意外的结果信息量越大：$I(x)=-\log_2p(x)$。概率 1 的必然事件信息量为 0；概率减半，信息量增加 1 bit。熵 $H(p)=-\sum_xp(x)\log_2p(x)$ 是信息量的平均。

### 数字代入 1：公平硬币与偏硬币

公平硬币每面概率 0.5，单次信息量都是 1 bit，熵也是 1 bit。若概率为 $[0.9,0.1]$，熵约 $-(0.9\log_20.9+0.1\log_20.1)=0.469$ bit：结果更可预测。

<a id="cross-entropy-kl"></a>
## 2. 交叉熵、KL 与 `CE = H + KL`

交叉熵 $H(p,q)=-\sum p(x)\log q(x)$ 用模型 q 为真实分布 p 编码。它可拆成 $H(p)+D_{KL}(p\|q)$：真实世界固有的不确定性加上模型不准确的额外代价。KL 非对称，因为“用 q 覆盖 p”的漏项与反方向不同。

二分类标签 $y\in\{0,1\}$ 的交叉熵为 $-[y\log q+(1-y)\log(1-q)]$；one-hot 多分类只保留正确类别的 $-\log q_y$。

### 数字代入 2：自信答错更痛

正确类别是第 1 类。模型 A 给它概率 0.8，损失 $-\ln0.8\approx0.223$；模型 B 只给 0.1，损失 $-\ln0.1\approx2.303$，约大 10 倍。
"""),
            code(r"""
# Demo 1：熵、交叉熵和 KL 的数值分解。
def entropy(p):
    p=np.asarray(p,float); return -np.sum(p*np.log2(p))
p=np.array([.7,.2,.1]); q=np.array([.5,.3,.2])
H=entropy(p); CE=-np.sum(p*np.log2(q)); KL=np.sum(p*np.log2(p/q))
print({"H(p)":H,"CE(p,q)":CE,"KL(p||q)":KL,"H+KL":H+KL,
       "KL(q||p)":np.sum(q*np.log2(q/p))})
probs=np.linspace(.01,.99,200)
plt.plot(probs,-np.log(probs)); plt.xlabel("正确类别概率"); plt.ylabel("NLL"); plt.title("答得越自信且越错，惩罚越大"); plt.grid(); plt.show()
"""),
            md(r"""
<a id="softmax-temperature"></a>
## 3. Softmax、温度与 Normalized Entropy

Softmax 把任意分数变为和为 1 的概率：$q_j=\exp(z_j/\tau)/\sum_k\exp(z_k/\tau)$。小温度放大差异，大温度让分布平坦。实现时先减最大分数，避免指数溢出。

若有 $K$ 类，最大熵为 $\log K$。Normalized Entropy 常写成 $H(q)/\log K$，落在 0 到 1：接近 0 很集中，接近 1 很均匀。具体论文可能用另一归一化口径，比较前必须读定义。

### 数字代入 3：温度

分数 $[2,1]$：$\tau=1$ 时概率约 $[0.731,0.269]$；$\tau=0.5$ 等价于分数差翻倍，约 $[0.881,0.119]$，分布更尖。

<a id="sequence-nll-dpo"></a>
## 4. 序列 NLL、perplexity 与 DPO log-ratio

自回归序列概率是条件概率连乘：$P(y_{1:T})=\prod_tP(y_t\mid y_{<t})$。取负对数后变成可相加的序列 NLL：$-\sum_t\log P(y_t\mid y_{<t})$。平均每 token NLL 为 $\bar L$ 时，perplexity $=e^{\bar L}$，可直觉理解为模型平均在多少个等可能选项间犹豫。

DPO 比较偏好答案 $y^+$ 与不偏好答案 $y^-$ 的 log-probability 差，并减去参考模型的同一差值。log-ratio 抵消共同尺度；正值表示当前策略相对参考更偏向 $y^+$。

### 数字代入 4：序列与偏好

三个正确 token 概率为 $[0.5,0.25,0.8]$，序列概率 $0.1$，NLL $=-\ln0.1=2.303$；平均 NLL $0.768$，perplexity $e^{0.768}\approx2.15$。若策略的 $\log P(y^+)-\log P(y^-)=0.9$，参考模型为 0.2，DPO 的相对偏好 margin 为 $0.7$。
"""),
            code(r"""
# Demo 2：温度、Normalized Entropy 与序列 NLL。
scores=np.array([2.,1.,0.])
fig,ax=plt.subplots(figsize=(6,3.3))
for tau in [.3,1.,3.]:
    stable=(scores-scores.max())/tau; soft=np.exp(stable); soft/=soft.sum()
    ne=-np.sum(soft*np.log(soft))/np.log(len(soft))
    ax.plot(range(3),soft,"o-",label=f"tau={tau}, NE={ne:.2f}")
ax.set(xticks=range(3),ylabel="probability",title="Softmax 温度教学对象"); ax.legend(); ax.grid(); plt.show()
token_p=np.array([.5,.25,.8]); nll=-np.log(token_p).sum(); ppl=np.exp(nll/len(token_p))
print({"sequence_probability":token_p.prod(),"NLL":nll,"perplexity":ppl,"DPO_margin":.9-.2})
"""),
            md(r"""
## 常见误区

- 熵越低模型越好：过度自信且错误的分布也可能低熵。
- KL 是普通距离：它不对称，也不满足距离的全部性质。
- perplexity 可跨词表直接比较：token 划分和数据协议不同会改变数值。
- DPO margin 就是线上提升：它只是训练中的相对偏好信号。

## 算法回链

- [word2vec：负采样二元交叉熵](/notebooks/3_1_5_word2vec)
- [DSSM：sampled softmax](/notebooks/3_2_1_dssm)
- [SASRec：next-item NLL](/notebooks/3_2_3_sasrec)
- [OpenOneRec：序列生成与 DPO](/notebooks/4_2_openonerec_practice)
- [HSTU：序列预测与 Normalized Entropy](/notebooks/4_3_dlrm_hstu_practice)

## Checks

1. 为什么概率减半会多 1 bit 信息？
2. `CE=H+KL` 中哪一项能通过改模型降低？
3. 温度变小为什么让 Softmax 更尖？
"""),
            code(r"""
assert np.isclose(CE,H+KL)
assert not np.isclose(KL,np.sum(q*np.log2(q/p)))
assert np.isclose(token_p.prod(),.1)
assert np.isclose(nll,-np.log(.1)) and ppl > 1
print("PASS：熵、CE/KL、Softmax、序列 NLL 与 DPO margin 检查通过。")
"""),
            md("## Next Steps\n\n下一课进入 3.0.6 优化，学习怎样用 mini-batch、学习率、正则化和早停稳定地降低这些损失。"),
        ],
    )

    canonical_order = (
        "3_0_1_data_ml_basics",
        "3_0_2_linear_algebra",
        "3_0_3_calculus",
        "3_0_4_probability_statistics",
        "3_0_5_information_theory",
        "3_0_6_optimization",
    )
    return {slug: specs[slug] for slug in canonical_order}
