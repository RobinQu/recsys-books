"""Train item2vec (Skip-gram on item sequences) and evaluate next-item Recall@5."""
import numpy as np
import torch

from .model import SkipGram, negative_sampling_loss
from recsys_lab.data import leave_last_out, load_movielens
from recsys_lab.runtime import ProgressCallback, emit_progress, full_profile, progress_due


def _skip_gram_pair_count(sequences, window: int = 3) -> int:
    """Count all positive pairs without materializing millions of Python tuples."""
    return sum(2 * sum(max(0, len(sequence) - offset) for offset in range(1, window + 1))
               for sequence in sequences)


def _skip_gram_pair_batches(sequences, batch_size: int, window: int = 3):
    """Stream complete skip-gram pairs as compact int64 arrays.

    Per-offset NumPy slices replace one Python tuple per pair. This preserves
    every window neighbor and the negative-sampling objective while bounding
    peak memory independently of the full dataset size.
    """
    pairs = np.empty((batch_size, 2), dtype=np.int64)
    used = 0
    for sequence in sequences:
        values = np.asarray(sequence, dtype=np.int64)
        for offset in range(1, min(window, len(values) - 1) + 1):
            for centers, contexts in ((values[:-offset], values[offset:]), (values[offset:], values[:-offset])):
                cursor = 0
                while cursor < len(centers):
                    take = min(batch_size - used, len(centers) - cursor)
                    pairs[used:used + take, 0] = centers[cursor:cursor + take]
                    pairs[used:used + take, 1] = contexts[cursor:cursor + take]
                    used += take
                    cursor += take
                    if used == batch_size:
                        yield pairs
                        pairs = np.empty((batch_size, 2), dtype=np.int64)
                        used = 0
    if used:
        yield pairs[:used]


def train_and_evaluate(epochs: int = 8, *, progress: ProgressCallback | None = None) -> dict:
    torch.manual_seed(2026)
    # 时间留一法：每个用户最后一个正反馈物品作为测试目标，其余构成训练序列。
    emit_progress(progress, stage="data_prepare", current=0, total=1, message="加载 MovieLens 并构造正反馈序列")
    ratings, _ = load_movielens(max_users=80, max_items=360, min_user_events=12)
    train, test = leave_last_out(ratings)

    # 物品 ID 从 1 开始（0 留作 padding）；按时间序构造每个用户的训练序列。
    positive = train[train.rating >= 4.0].sort_values(["user_id", "timestamp", "item_id"])
    sequences = positive.groupby("user_id").item_id.apply(lambda v: [int(x) + 1 for x in v]).to_dict()
    sequences = {int(user): seq for user, seq in sequences.items() if len(seq) >= 4}
    pair_count = _skip_gram_pair_count(sequences.values(), window=3)
    emit_progress(progress, stage="data_prepare", current=1, total=1, metrics={"pairs": pair_count})
    if not pair_count:
        return {"dataset": "MovieLens latest-small", "randomly_fabricated_rows": 0,
                "recall@5": 0.0, "loss_curve": []}

    num_items = int(ratings.item_id.max()) + 2  # +1 偏移、+1 容纳 1-indexing
    num_neg = 5
    rng = np.random.default_rng(2026)

    model = SkipGram(num_items, dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    losses: list[float] = []
    pair_batch = min(pair_count, 1_000_000)
    batches_per_epoch = (pair_count + pair_batch - 1) // pair_batch
    total_batches = epochs * batches_per_epoch
    completed_batches = 0
    emit_progress(progress, stage="train", current=0, total=total_batches, message="流式训练 Item2Vec")
    for _ in range(epochs):
        weighted = 0.0
        for chunk in _skip_gram_pair_batches(sequences.values(), pair_batch, window=3):
            center = torch.from_numpy(chunk[:, 0])
            context = torch.from_numpy(chunk[:, 1])
            # 对每个正样本采样若干负物品，构成正负对比；负采样避免遍历全物品表。
            neg = torch.tensor(rng.integers(1, num_items, size=(len(chunk), num_neg)), dtype=torch.long)
            score_pos = model(center, context)
            score_neg = model(center[:, None].expand(-1, num_neg).reshape(-1), neg.reshape(-1))
            loss = negative_sampling_loss(score_pos, score_neg)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            weighted += float(loss.detach()) * len(chunk)
            completed_batches += 1
            if progress_due(completed_batches, total_batches):
                emit_progress(progress, stage="train", current=completed_batches, total=total_batches)
        losses.append(weighted / pair_count)

    # 用中心嵌入作为物品向量；归一为单位向量后用余弦相似度召回，量纲稳定且与 item2vec 检索惯例一致。
    emit_progress(progress, stage="inference", current=0, total=1, message="提取归一化物品向量")
    with torch.inference_mode():
        embeddings = model.center.weight.detach().numpy()
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.maximum(norms, 1e-12)
    emit_progress(progress, stage="inference", current=1, total=1)
    test_frame = test.sort_values("user_id")
    if full_profile() and len(test_frame) > 5000:
        # 33M 全量逐用户评估是平方级成本：用固定种子抽 5000 个用户评估，口径写入 provenance。
        test_frame = test_frame.iloc[rng.choice(len(test_frame), size=5000, replace=False)]
    targets = (test_frame.item_id.to_numpy() + 1)
    users = test_frame.user_id.to_numpy()
    seen = {user: set(seq) for user, seq in sequences.items()}
    hits = 0
    recommended_items: set[int] = set()
    emit_progress(progress, stage="evaluate", current=0, total=len(targets), message="计算下一物品 Recall@5")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for row_index, (user, target) in enumerate(zip(users, targets), start=1):
            history = sequences.get(int(user), [])
            if history:
                user_vec = normed[history].mean(axis=0)
                user_vec = user_vec / max(float(np.linalg.norm(user_vec)), 1e-12)
                scores = normed @ user_vec
                scores[list(seen.get(int(user), []))] = -1.0  # 余弦范围 [-1,1]，屏蔽已见物品
                k = min(5, len(scores) - 1)
                top5 = np.argpartition(-scores, kth=k)[:5]
                hits += int(target in top5)
                recommended_items.update(map(int, top5))
            if progress_due(row_index, len(targets)):
                emit_progress(progress, stage="evaluate", current=row_index, total=len(targets))
    emit_progress(
        progress,
        stage="evaluate",
        current=len(targets),
        total=len(targets),
        metrics={"recall@5": hits / max(len(targets), 1)},
    )
    recall = hits / max(len(targets), 1)
    coverage = len(recommended_items) / max(num_items - 1, 1)
    dataset = "MovieLens latest (33M, 5,000 用户种子抽样评估)" if full_profile() else "MovieLens latest-small"
    return {"dataset": dataset, "randomly_fabricated_rows": 0,
            "recall@5": float(recall), "coverage": coverage, "loss_curve": losses}
