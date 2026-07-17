"""CPU-sized industrial-model experiments trained on task-appropriate real data.

Amazon Reviews 2023 supplies e-commerce retrieval/sequences; KuaiRand supplies
real short-video impressions and multi-feedback targets. Deterministic
truncation keeps tests fast; seeds initialize parameters only.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from torch_rechub.basic.features import DenseFeature, SequenceFeature, SparseFeature
from torch_rechub.models.generative import HSTUModel
from torch_rechub.models.matching import DSSM, MIND, SASRec
from torch_rechub.models.multi_task import MMOE, PLE
from torch_rechub.models.ranking import DIEN, DIN, DeepFM

from .data import (
    amazon_provenance,
    clicked_sequences,
    kuairand_provenance,
    kuairand_sequence_classification_rows,
    load_amazon_2023,
    load_kuairand,
    positive_sequences,
)


ROOT = Path(__file__).resolve().parents[1]


def seed_everything(seed: int = 2026) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def save_records(chapter: str, name: str, records: list[dict]) -> Path:
    path = ROOT / "results" / chapter / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else 0.5


def _recall_single_target(scores: np.ndarray, targets: np.ndarray, k: int, seen: list[set[int]] | None = None) -> float:
    scores = scores.copy()
    if seen:
        for row, items in enumerate(seen):
            if items:
                scores[row, list(items)] = -np.inf
    topk = np.argpartition(-scores, kth=min(k, scores.shape[1]) - 1, axis=1)[:, :k]
    return float(np.mean([target in candidates for target, candidates in zip(targets, topk)]))


def _train_binary(model, data: dict[str, torch.Tensor], labels: torch.Tensor, epochs: int, lr: float, dien: bool = False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        output = model(data)
        if dien:
            probability, auxiliary_loss = output
            loss = torch.nn.functional.binary_cross_entropy(probability, labels) + auxiliary_loss
        else:
            probability = output
            loss = torch.nn.functional.binary_cross_entropy(probability, labels)
        optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
    return losses


def _real_amazon(max_users: int = 128):
    ratings = load_amazon_2023(max_users=max_users, min_user_events=12)
    return ratings, amazon_provenance(ratings)


def _real_kuairand(max_users: int = 96, max_items: int = 2200):
    interactions, videos = load_kuairand(max_users=max_users, max_items=max_items)
    return interactions, videos, kuairand_provenance(interactions)


def run_dssm(epochs: int = 24) -> dict:
    seed_everything(); ratings, provenance = _real_amazon()
    events = ratings.sort_values("timestamp").tail(5200).reset_index(drop=True)
    split = int(len(events) * .8)
    train, test = events.iloc[:split], events.iloc[split:]
    n_users, n_items = ratings.user_id.nunique(), ratings.item_id.nunique()
    user_features = [SparseFeature("user_id", n_users + 1, 16, padding_idx=0)]
    item_features = [SparseFeature("item_id", n_items + 1, 16, padding_idx=0)]
    tower = {"dims": [32, 16], "activation": "relu", "dropout": 0.0}
    model = DSSM(user_features, item_features, tower, tower, temperature=.12)

    def fields(frame):
        return {
            "user_id": torch.tensor(frame.user_id.to_numpy() + 1, dtype=torch.long),
            "item_id": torch.tensor(frame.item_id.to_numpy() + 1, dtype=torch.long),
        }

    losses = _train_binary(model, fields(train), torch.tensor(train.like.to_numpy()), epochs, .004)
    with torch.no_grad():
        probability = model(fields(test)).numpy()
        model.mode = "user"
        user_embedding = model({"user_id": torch.arange(1, n_users + 1)}).numpy()
        model.mode = "item"
        item_embedding = model({"item_id": torch.arange(1, n_items + 1)}).numpy()
        model.mode = None
    user_embedding = np.nan_to_num(user_embedding); item_embedding = np.nan_to_num(item_embedding)
    user_embedding /= np.linalg.norm(user_embedding, axis=1, keepdims=True) + 1e-8
    item_embedding /= np.linalg.norm(item_embedding, axis=1, keepdims=True) + 1e-8
    scores = user_embedding @ item_embedding.T
    targets, seen, valid_users = [], [], []
    for user, frame in ratings.sort_values("timestamp").groupby("user_id"):
        positives = frame[frame.rating >= 4.0].item_id.tolist()
        if len(positives) >= 2:
            valid_users.append(int(user)); targets.append(int(positives[-1])); seen.append(set(map(int, positives[:-1])))
    recall = _recall_single_target(scores[valid_users], np.asarray(targets), 10, seen)
    return {
        "framework": "torch_rechub.models.matching.DSSM",
        "dataset": provenance | {"train_rows": len(train), "test_rows": len(test), "label": "observed Amazon rating >= 4.0"},
        "loss_curve": losses,
        "test_auc": _safe_auc(test.like.to_numpy(), probability),
        "recall@10": recall,
        "embedding_shape": [list(user_embedding.shape), list(item_embedding.shape)],
        "score_sample": scores[:3, :8].round(3).tolist(),
    }


def _mind_rows(ratings, history_length=10, negatives=5):
    global_disliked = [int(v) + 1 for v in ratings[ratings.rating <= 2.5].item_id.tolist()]
    rows = []
    for user, frame in ratings.sort_values("timestamp").groupby("user_id"):
        positives = [int(v) + 1 for v in frame[frame.rating >= 4.0].item_id]
        disliked = [int(v) + 1 for v in frame[frame.rating <= 3.0].item_id]
        if len(positives) < 4:
            continue
        pool = disliked or global_disliked
        negative_items = [pool[(int(user) + offset) % len(pool)] for offset in range(negatives)]
        history = positives[:-2][-history_length:]
        history = [0] * (history_length - len(history)) + history
        inference_history = positives[:-1][-history_length:]
        inference_history = [0] * (history_length - len(inference_history)) + inference_history
        rows.append((int(user) + 1, history, positives[-2], negative_items, inference_history, positives[-1]))
    return rows


def run_mind(epochs: int = 26) -> dict:
    seed_everything(); ratings, provenance = _real_amazon()
    rows = _mind_rows(ratings); n_users, n_items = ratings.user_id.nunique(), ratings.item_id.nunique(); history_length = 10
    user_feature = [SparseFeature("user_id", n_users + 1, 12)]
    item_feature = [SparseFeature("item_id", n_items + 1, 12, padding_idx=0)]
    history_feature = [SequenceFeature("history", n_items + 1, 12, pooling="concat", shared_with="item_id", padding_idx=0)]
    negative_feature = [SequenceFeature("negative_items", n_items + 1, 12, pooling="concat", shared_with="item_id", padding_idx=0)]
    model = MIND(user_feature, history_feature, item_feature, negative_feature, history_length, interest_num=2)
    train_data = {
        "user_id": torch.tensor([r[0] for r in rows]),
        "history": torch.tensor([r[1] for r in rows]),
        "item_id": torch.tensor([r[2] for r in rows]),
        "negative_items": torch.tensor([r[3] for r in rows]),
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=.005); losses = []
    for _ in range(epochs):
        logits = model(train_data); loss = torch.nn.functional.cross_entropy(logits, torch.zeros(len(rows), dtype=torch.long))
        optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
    inference = {"user_id": train_data["user_id"], "history": torch.tensor([r[4] for r in rows])}
    with torch.no_grad():
        train_top1 = float((model(train_data).argmax(1) == 0).float().mean())
        model.mode = "user"; user_interests = model(inference).numpy()
        model.mode = "item"; item_embeddings = model({"item_id": torch.arange(1, n_items + 1)}).numpy(); model.mode = None
    scores = np.einsum("ukd,id->uki", user_interests, item_embeddings).max(1)
    seen = [set(np.asarray(r[4])[np.asarray(r[4]) > 0] - 1) for r in rows]
    targets = np.asarray([r[5] - 1 for r in rows])
    return {
        "framework": "torch_rechub.models.matching.MIND",
        "dataset": provenance | {"sequence_users": len(rows), "history_length": history_length, "negative_source": "observed Amazon rating <= 3.0"},
        "loss_curve": losses, "positive_top1": train_top1,
        "recall@10": _recall_single_target(scores, targets, 10, seen),
        "interest_shape": list(user_interests.shape),
        "interest_cosine": float(np.mean(np.sum(user_interests[:, 0] * user_interests[:, 1], axis=1))),
    }


def _ranking_fields(interactions):
    frame = interactions.sort_values("timestamp").tail(7000).copy()
    return {
        "user_id": frame.user_id.to_numpy() + 1,
        "item_id": frame.item_id.to_numpy() + 1,
        "tab": frame.tab.to_numpy() + 1,
        "hour": frame.hour.to_numpy() + 1,
        "duration_bucket": frame.duration_bucket.to_numpy() + 1,
    }, frame.is_click.to_numpy(dtype=np.float32), frame


def run_deepfm(epochs: int = 28) -> dict:
    seed_everything(); interactions, _, provenance = _real_kuairand(); fields, labels, frame = _ranking_fields(interactions); split = int(len(labels) * .8)
    features = [
        SparseFeature("user_id", int(max(fields["user_id"])) + 1, 12),
        SparseFeature("item_id", int(max(fields["item_id"])) + 1, 12),
        SparseFeature("tab", int(max(fields["tab"])) + 1, 12),
        SparseFeature("hour", 25, 12),
        SparseFeature("duration_bucket", int(max(fields["duration_bucket"])) + 1, 12),
    ]
    model = DeepFM(features, features, {"dims": [48, 24], "activation": "relu", "dropout": 0.0})
    tensors = {name: torch.tensor(value, dtype=torch.long) for name, value in fields.items()}
    losses = _train_binary(model, {k: v[:split] for k, v in tensors.items()}, torch.tensor(labels[:split]), epochs, .015)
    with torch.no_grad(): probability = model({k: v[split:] for k, v in tensors.items()}).numpy()
    baseline_x = np.c_[frame.tab.to_numpy(), frame.hour.to_numpy(), frame.duration_bucket.to_numpy()]
    baseline = LogisticRegression(max_iter=300).fit(baseline_x[:split], labels[:split])
    baseline_probability = baseline.predict_proba(baseline_x[split:])[:, 1]
    return {
        "framework": "torch_rechub.models.ranking.DeepFM",
        "dataset": provenance | {"rows": len(labels), "train_rows": split, "label": "observed KuaiRand is_click"},
        "loss_curve": losses, "auc": _safe_auc(labels[split:], probability),
        "logloss": float(log_loss(labels[split:], np.clip(probability, 1e-6, 1 - 1e-6))),
        "lr_auc": _safe_auc(labels[split:], baseline_probability), "probability_sample": probability[:8].round(3).tolist(),
    }


def _run_sequence_ranker(kind: str, epochs: int) -> dict:
    seed_everything(); interactions, _, provenance = _real_kuairand(); rows = kuairand_sequence_classification_rows(interactions, max_len=20, limit=2600)
    labels = rows.pop("label"); timestamps = rows.pop("timestamp"); split = int(len(labels) * .8)
    n_users, n_items = interactions.user_id.nunique(), interactions.item_id.nunique()
    user = [SparseFeature("user_id", n_users + 1, 12)]
    item = [SparseFeature("item_id", n_items + 1, 12, padding_idx=0)]
    history = [SequenceFeature("history", n_items + 1, 12, pooling="concat", shared_with="item_id", padding_idx=0)]
    tensors = {name: torch.tensor(value, dtype=torch.long) for name, value in rows.items()}
    if kind == "din":
        model = DIN(user, history, item, {"dims": [40, 20], "dropout": 0.0}, {"dims": [24, 12], "activation": "prelu", "use_softmax": True}); used = ["user_id", "history", "item_id"]
    else:
        negative = [SequenceFeature("negative_history", n_items + 1, 12, pooling="concat", shared_with="item_id", padding_idx=0)]
        model = DIEN(user, history, negative, item, {"dims": [40, 20], "dropout": 0.0}, alpha=.1); used = list(tensors)
    train = {name: tensors[name][:split] for name in used}; y_train = torch.tensor(labels[:split])
    losses = _train_binary(model, train, y_train, epochs, .012, dien=kind == "dien")
    with torch.no_grad():
        output = model({name: tensors[name][split:] for name in used}); probability = (output[0] if kind == "dien" else output).numpy()
    history_values = rows["history"]
    overlap = (history_values == rows["item_id"][:, None]).mean(1)
    return {
        "framework": f"torch_rechub.models.ranking.{kind.upper()}",
        "dataset": provenance | {"rows": len(labels), "sequence_length": 20, "label": "observed KuaiRand is_click", "negative_history": "observed skipped impressions", "time_ordered": True},
        "loss_curve": losses, "auc": _safe_auc(labels[split:], probability),
        "logloss": float(log_loss(labels[split:], np.clip(probability, 1e-6, 1 - 1e-6))),
        "static_overlap_auc": _safe_auc(labels[split:], overlap[split:]), "probability_sample": probability[:8].round(3).tolist(),
    }


def run_din(epochs: int = 26) -> dict: return _run_sequence_ranker("din", epochs)
def run_dien(epochs: int = 30) -> dict: return _run_sequence_ranker("dien", epochs)


def _multitask_view(interactions):
    frame = interactions.sort_values("timestamp").tail(8000).copy()
    tabs = np.eye(int(interactions.tab.max()) + 1, dtype=np.float32)[frame.tab.to_numpy()]
    duration = np.eye(int(interactions.duration_bucket.max()) + 1, dtype=np.float32)[frame.duration_bucket.to_numpy()]
    continuous = np.c_[
        np.sin(frame.hour.to_numpy() / 24 * 2 * np.pi), np.cos(frame.hour.to_numpy() / 24 * 2 * np.pi),
        np.log1p(frame.item_popularity.to_numpy()) / np.log1p(interactions.item_popularity.max()),
        np.log1p(frame.user_activity.to_numpy()) / np.log1p(interactions.user_activity.max()),
    ].astype(np.float32)
    return np.c_[tabs, duration, continuous].astype(np.float32), np.c_[frame.is_click, frame.long_view].astype(np.float32)


def _run_multitask(kind: str, epochs: int) -> dict:
    seed_everything(); interactions, _, provenance = _real_kuairand(); x, labels = _multitask_view(interactions); split = int(len(x) * .8)
    features = [DenseFeature(f"x{i}") for i in range(x.shape[1])]
    expert = {"dims": [24, 12], "activation": "relu", "dropout": 0.0}; towers = [{"dims": [12], "activation": "relu", "dropout": 0.0}] * 2
    model = MMOE(features, ["classification", "classification"], 4, expert, towers) if kind == "mmoe" else PLE(features, ["classification", "classification"], 2, 2, 2, expert, towers)
    data = {f"x{i}": torch.tensor(x[:, i]) for i in range(x.shape[1])}; target = torch.tensor(labels)
    optimizer = torch.optim.Adam(model.parameters(), lr=.012); losses = []
    for _ in range(epochs):
        probability = model({name: value[:split] for name, value in data.items()})
        loss = sum(torch.nn.functional.binary_cross_entropy(probability[:, task], target[:split, task]) for task in range(2))
        optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
    with torch.no_grad(): probability = model({name: value[split:] for name, value in data.items()}).numpy()
    independent_auc = []
    for task in range(2):
        baseline = LogisticRegression(max_iter=300, solver="liblinear").fit(x[:split], labels[:split, task])
        independent_auc.append(_safe_auc(labels[split:, task], baseline.predict_proba(x[split:])[:, 1]))
    return {
        "framework": f"torch_rechub.models.multi_task.{kind.upper()}",
        "dataset": provenance | {"rows": len(x), "features": x.shape[1], "tasks": ["observed is_click", "observed long_view"]},
        "loss_curve": losses, "click_auc": _safe_auc(labels[split:, 0], probability[:, 0]),
        "long_view_auc": _safe_auc(labels[split:, 1], probability[:, 1]),
        "conversion_auc": _safe_auc(labels[split:, 1], probability[:, 1]), "independent_lr_auc": independent_auc,
    }


def run_mmoe(epochs: int = 28) -> dict: return _run_multitask("mmoe", epochs)
def run_ple(epochs: int = 28) -> dict: return _run_multitask("ple", epochs)


class TinyListGenerator(torch.nn.Module):
    def __init__(self, vocabulary_size: int, hidden: int = 32):
        super().__init__(); self.embedding = torch.nn.Embedding(vocabulary_size, hidden); self.gru = torch.nn.GRU(hidden, hidden, batch_first=True); self.head = torch.nn.Linear(hidden, vocabulary_size)
    def forward(self, tokens):
        states, _ = self.gru(self.embedding(tokens)); return self.head(states)


def _semantic_catalog(videos):
    def first_tag(value) -> int:
        raw = str(value).split(",")[0]
        try: return int(float(raw))
        except ValueError: return 0
    return {
        int(row.item_id): (
            first_tag(row.tag) % 128 + 1,
            int(0 if np.isnan(row.music_type) else row.music_type) % 32 + 1,
            int(row.item_id % 64) + 1,
        )
        for row in videos.itertuples()
    }


def run_openonerec(epochs: int = 32) -> dict:
    seed_everything(); interactions, videos, provenance = _real_kuairand(); item_to_code = _semantic_catalog(videos); catalog = set(item_to_code.values())
    positive = interactions[interactions.is_click == 1].sort_values("timestamp").tail(2000)
    sequences = np.asarray([item_to_code[int(item)] for item in positive.item_id], dtype=np.int64)
    vocabulary_size = int(sequences.max()) + 2
    model_input = torch.tensor(np.c_[np.zeros((len(sequences), 1), dtype=np.int64), sequences[:, :-1]])
    target = torch.tensor(sequences); model = TinyListGenerator(vocabulary_size); optimizer = torch.optim.Adam(model.parameters(), lr=.02); losses=[]
    for _ in range(epochs):
        logits = model(model_input); loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
        optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
    prefix_counts = {}
    for code in catalog: prefix_counts.setdefault(code[:2], set()).add(code[2])
    prefix = sorted(prefix_counts, key=lambda value: (-len(prefix_counts[value]), value))[0]
    allowed = sorted(prefix_counts[prefix])
    with torch.no_grad(): next_logits = model(torch.tensor([[0, *prefix]]))[0, -1].numpy()
    stress_logits = next_logits.copy(); invalid_token = vocabulary_size - 1
    while prefix + (invalid_token,) in catalog: invalid_token -= 1
    stress_logits[invalid_token] = float(stress_logits.max() + .5)
    unconstrained = int(stress_logits.argmax()); constrained = int(max(allowed, key=lambda token: stress_logits[token]))
    chosen_row = interactions[(interactions.is_click == 1) & (interactions.long_view == 1)].iloc[0]
    rejected_row = interactions[interactions.is_click == 0].iloc[0]
    return {
        "framework": "OpenOneRec contract + PyTorch executable proxy",
        "dataset": provenance | {
            "semantic_catalog_size": len(catalog), "training_codes": len(sequences),
            "code_source": "observed KuaiRand video tag + music type + item id partition",
            "recif_bench_access": "official OpenOneRec-RecIF repository is gated; full profile requires authenticated acceptance",
        },
        "loss_curve": losses, "catalog_size": len(catalog), "prefix": prefix, "allowed_tokens": allowed,
        "unconstrained_token": unconstrained, "constrained_token": constrained,
        "invalid_unconstrained": float(prefix + (unconstrained,) not in catalog), "invalid_constrained": float(prefix + (constrained,) not in catalog),
        "dpo_pair": {"chosen": item_to_code[int(chosen_row.item_id)], "rejected": item_to_code[int(rejected_row.item_id)]},
    }


def _sequence_windows_from_sequences(sequences, length=12):
    train_input, train_target, eval_input, eval_target = [], [], [], []
    for sequence in sequences.values():
        prior = sequence[:-1]
        train = prior[-(length + 1):]
        if len(train) < length + 1: continue
        train_input.append(train[:-1]); train_target.append(train[1:])
        eval_input.append(sequence[-(length + 1):-1]); eval_target.append(sequence[-1])
    return np.asarray(train_input), np.asarray(train_target), np.asarray(eval_input), np.asarray(eval_target)


def run_sasrec(epochs: int = 30) -> dict:
    seed_everything(); ratings, provenance = _real_amazon(max_users=160); length = 20
    sequences = positive_sequences(ratings, threshold=4.0, min_length=length + 3)
    train_input, train_target, eval_input, eval_target = _sequence_windows_from_sequences(sequences, length)
    low_items = [int(v) + 1 for v in ratings[ratings.rating <= 2.5].item_id]
    negative = np.asarray([[low_items[(row + col) % len(low_items)] for col in range(length)] for row in range(len(train_input))])
    vocab = ratings.item_id.nunique() + 1
    seq = SequenceFeature("seq", vocab, 24, pooling="concat", padding_idx=0)
    pos = SequenceFeature("pos", vocab, 24, pooling="concat", shared_with="seq", padding_idx=0)
    neg = SequenceFeature("neg", vocab, 24, pooling="concat", shared_with="seq", padding_idx=0)
    model = SASRec([seq, pos, neg], max_len=length, dropout_rate=.1, num_blocks=1, num_heads=2)
    data = {"seq": torch.tensor(train_input), "pos": torch.tensor(train_target), "neg": torch.tensor(negative)}
    optimizer = torch.optim.Adam(model.parameters(), lr=.008); losses=[]
    for _ in range(epochs):
        pos_logits, neg_logits = model(data)
        loss = torch.nn.functional.softplus(-pos_logits).mean() + torch.nn.functional.softplus(neg_logits).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
    with torch.no_grad():
        user_vectors = model.user_tower({"seq": torch.tensor(eval_input)}).squeeze(1)
        item_vectors = model.item_emb.embed_dict["seq"].weight[1:]
        scores = (user_vectors @ item_vectors.T).numpy()
    seen = [set(row[row > 0] - 1) for row in eval_input]
    popularity = np.bincount(train_input.ravel(), minlength=vocab)[1:]
    popularity_top10 = np.argsort(-popularity)[:10]
    popularity_hr = float(np.mean([target - 1 in popularity_top10 for target in eval_target]))
    return {
        "framework": "torch_rechub.models.matching.SASRec",
        "dataset": provenance | {"sequence_users": len(eval_target), "sequence_length": length, "negative_source": "observed Amazon rating <= 2.5"},
        "loss_curve": losses, "hr@10": _recall_single_target(scores, eval_target - 1, 10, seen),
        "popularity_hr@10": popularity_hr, "embedding_dim": 24,
    }


def run_hstu(epochs: int = 26) -> dict:
    seed_everything(); interactions, _, provenance = _real_kuairand(max_users=128, max_items=2500); length = 24
    sequences = clicked_sequences(interactions, min_length=length + 3)
    train_input, train_target, eval_input, eval_target = _sequence_windows_from_sequences(sequences, length)
    vocab = interactions.item_id.nunique() + 1
    inputs, targets = torch.tensor(train_input), torch.tensor(train_target)
    model = HSTUModel(vocab, d_model=32, n_heads=2, n_layers=1, dqk=8, dv=8, max_seq_len=length, dropout=0.0, use_time_embedding=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=.008); losses=[]
    for _ in range(epochs):
        logits = model(inputs); loss = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1))
        optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
    with torch.no_grad():
        final_logits = model(torch.tensor(eval_input))[:, -1]; top5 = final_logits.topk(5, dim=1).indices
        hit = (top5 == torch.tensor(eval_target)[:, None]).any(1).float().mean().item()
    popularity = np.bincount(train_input.ravel(), minlength=vocab); popular_top5 = np.argsort(-popularity[1:])[:5] + 1
    baseline = float(np.isin(eval_target, popular_top5).mean())
    return {
        "framework": "torch_rechub.models.generative.HSTUModel",
        "dataset": provenance | {"sequence_users": len(eval_target), "sequence_length": length, "vocab": vocab, "strict_time_split": True},
        "loss_curve": losses, "hr@5": hit, "popularity_hr@5": baseline, "logits_shape": list(final_logits.shape),
    }
