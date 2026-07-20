"""Train item2vec (Skip-gram on item sequences) and evaluate next-item Recall@5."""
import numpy as np
import torch

from .model import SkipGram, negative_sampling_loss
from recsys_lab.data import leave_last_out, load_movielens


def _skip_gram_pairs(sequences, window: int = 3):
    """把每个序列展开为 (中心, 上下文) 物品对，窗口内邻居都算正样本。"""
    pairs: list[tuple[int, int]] = []
    for sequence in sequences:
        for i, center in enumerate(sequence):
            lo = max(0, i - window)
            hi = min(len(sequence), i + window + 1)
            for j in range(lo, hi):
                if j != i:
                    pairs.append((center, sequence[j]))
    return pairs


def train_and_evaluate(epochs: int = 8) -> dict:
    torch.manual_seed(2026)
    # 时间留一法：每个用户最后一个正反馈物品作为测试目标，其余构成训练序列。
    ratings, _ = load_movielens(max_users=80, max_items=360, min_user_events=12)
    train, test = leave_last_out(ratings)

    # 物品 ID 从 1 开始（0 留作 padding）；按时间序构造每个用户的训练序列。
    positive = train[train.rating >= 4.0].sort_values(["user_id", "timestamp", "item_id"])
    sequences = positive.groupby("user_id").item_id.apply(lambda v: [int(x) + 1 for x in v]).to_dict()
    sequences = {int(user): seq for user, seq in sequences.items() if len(seq) >= 4}
    pairs = _skip_gram_pairs(sequences.values(), window=3)
    if not pairs:
        return {"dataset": "MovieLens latest-small", "randomly_fabricated_rows": 0,
                "recall@5": 0.0, "loss_curve": []}

    num_items = int(ratings.item_id.max()) + 2  # +1 偏移、+1 容纳 1-indexing
    center = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    context = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    num_neg = 5
    rng = np.random.default_rng(2026)

    model = SkipGram(num_items, dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    losses: list[float] = []
    for _ in range(epochs):
        # 对每个正样本采样若干负物品，构成正负对比；负采样避免遍历全物品表。
        neg = torch.tensor(rng.integers(1, num_items, size=(len(pairs), num_neg)), dtype=torch.long)
        score_pos = model(center, context)
        score_neg = model(center[:, None].expand(-1, num_neg).reshape(-1), neg.reshape(-1))
        loss = negative_sampling_loss(score_pos, score_neg)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        losses.append(float(loss.detach()))

    # 用中心嵌入作为物品向量；归一为单位向量后用余弦相似度召回，量纲稳定且与 item2vec 检索惯例一致。
    with torch.no_grad():
        embeddings = model.center.weight.detach().numpy()
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.maximum(norms, 1e-12)
    test_frame = test.sort_values("user_id")
    targets = (test_frame.item_id.to_numpy() + 1)
    users = test_frame.user_id.to_numpy()
    seen = {user: set(seq) for user, seq in sequences.items()}
    hits = 0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for user, target in zip(users, targets):
            history = sequences.get(int(user), [])
            if not history:
                continue
            user_vec = normed[history].mean(axis=0)
            user_vec = user_vec / max(float(np.linalg.norm(user_vec)), 1e-12)
            scores = normed @ user_vec
            scores[list(seen.get(int(user), []))] = -1.0  # 余弦范围 [-1,1]，屏蔽已见物品
            k = min(5, len(scores) - 1)
            top5 = np.argpartition(-scores, kth=k)[:5]
            hits += int(target in top5)
    recall = hits / max(len(targets), 1)
    return {"dataset": "MovieLens latest-small", "randomly_fabricated_rows": 0,
            "recall@5": float(recall), "loss_curve": losses}
