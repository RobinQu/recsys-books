"""Train and evaluate ItemCF on the real MovieLens time split."""
import numpy as np

from .model import ItemCF
from recsys_lab.data import leave_last_out, load_movielens
from recsys_lab.runtime import ProgressCallback, emit_progress, full_profile, progress_due


def _sparse_itemcf_recall(
    ratings, train, test, neighbors: int = 50, *, progress: ProgressCallback | None = None,
) -> tuple[float, float]:
    """33M 评分上的稀疏 ItemCF：稀疏共现 + 余弦 top-k，分块评估避免稠密矩阵。"""
    from scipy import sparse

    implicit = train.assign(weight=(train.rating >= 4.0).astype(np.float32))
    matrix = sparse.csr_matrix(
        (implicit.weight.to_numpy(), (implicit.user_id.to_numpy(), implicit.item_id.to_numpy())),
        shape=(ratings.user_id.nunique(), ratings.item_id.nunique()),
    )
    cooccurrence = (matrix.T @ matrix).tocsr()
    norms = np.sqrt(cooccurrence.diagonal()); norms[norms == 0] = 1.0
    cooccurrence = cooccurrence.multiply(1.0 / norms[:, None]).multiply(1.0 / norms[None, :]).tocsr()
    # 每个物品只保留 top-k 近邻，避免稠密化 83k × 83k 相似度。
    indptr, indices, values = [0], [], []
    emit_progress(progress, stage="train", current=0, total=cooccurrence.shape[0], message="筛选物品 top-k 近邻")
    for row in range(cooccurrence.shape[0]):
        start, end = cooccurrence.indptr[row], cooccurrence.indptr[row + 1]
        cols, vals = cooccurrence.indices[start:end], cooccurrence.data[start:end]
        if len(cols) > neighbors:
            keep = np.argpartition(-vals, neighbors)[:neighbors]
            cols, vals = cols[keep], vals[keep]
        indices.extend(cols); values.extend(vals); indptr.append(len(indices))
        if progress_due(row + 1, cooccurrence.shape[0]):
            emit_progress(progress, stage="train", current=row + 1, total=cooccurrence.shape[0])
    similarity = sparse.csr_matrix((values, indices, indptr), shape=cooccurrence.shape)
    ordered = test.sort_values("user_id")
    users, targets = ordered.user_id.to_numpy(), ordered.item_id.to_numpy()
    hits, chunk = 0, 2000
    recommended_items: set[int] = set()
    total_chunks = max(1, (len(users) + chunk - 1) // chunk)
    emit_progress(progress, stage="evaluate", current=0, total=total_chunks, message="分块计算 Recall@5")
    for chunk_index, start in enumerate(range(0, len(users), chunk), start=1):
        block = matrix[users[start:start + chunk]]
        block_scores = (block @ similarity).toarray()
        block_scores[block.toarray() > 0] = -np.inf
        top5 = np.argpartition(-block_scores, kth=4, axis=1)[:, :5]
        hits += sum(target in candidates for target, candidates in zip(targets[start:start + chunk], top5))
        recommended_items.update(map(int, top5.ravel()))
        if progress_due(chunk_index, total_chunks):
            emit_progress(progress, stage="evaluate", current=chunk_index, total=total_chunks)
    return hits / max(1, len(users)), len(recommended_items) / max(1, matrix.shape[1])


def train_and_evaluate(epochs: int = 8, *, progress: ProgressCallback | None = None) -> dict:
    # 时间留一法保证测试物品发生在用户训练历史之后。
    emit_progress(progress, stage="data_prepare", current=0, total=1, message="加载 MovieLens 并构造时间切分")
    ratings, _ = load_movielens(max_users=80, max_items=360, min_user_events=12)
    train, test = leave_last_out(ratings)
    emit_progress(progress, stage="data_prepare", current=1, total=1, metrics={"rows": len(ratings)})
    if full_profile():
        recall, coverage = _sparse_itemcf_recall(ratings, train, test, progress=progress)
        return {
            "dataset": "MovieLens latest (33M, 稀疏 top-k ItemCF)",
            "randomly_fabricated_rows": 0,
            "recall@5": recall,
            "itemcf_recall@5": recall,
            "usercf_recall@5": None,
            "coverage": coverage,
        }
    emit_progress(progress, stage="train", current=0, total=1, message="拟合 ItemCF 相似度矩阵")
    matrix = np.zeros((ratings.user_id.nunique(), ratings.item_id.nunique()), dtype=np.float32)
    for row in train.itertuples():
        matrix[row.user_id, row.item_id] = float(row.rating >= 4.0)
    scores = ItemCF().fit(matrix).score(matrix)
    emit_progress(progress, stage="train", current=1, total=1)
    emit_progress(progress, stage="evaluate", current=0, total=1, message="计算 Recall@5")
    scores[matrix > 0] = -np.inf
    targets = test.sort_values("user_id").item_id.to_numpy()
    top5 = np.argpartition(-scores, kth=4, axis=1)[:, :5]
    recall = float(np.mean([target in candidates for target, candidates in zip(targets, top5)]))
    coverage = len(set(map(int, top5.ravel()))) / max(1, matrix.shape[1])
    emit_progress(progress, stage="evaluate", current=1, total=1, metrics={"recall@5": recall, "coverage": coverage})
    return {
        "dataset": "MovieLens latest-small",
        "randomly_fabricated_rows": 0,
        "recall@5": recall,
        "itemcf_recall@5": recall,
        "usercf_recall@5": None,
        "coverage": coverage,
    }
