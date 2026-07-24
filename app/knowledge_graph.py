"""Deterministic, bounded knowledge-graph projection for the tutorial home page.

The graph is deliberately derived from existing content registries.  It does
not create a new content hierarchy and it never runs an unbounded force-layout:
the server selects a small, ordered neighbourhood before the browser renders it.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


DEFAULT_VISIBLE_LIMIT = 6
FOCUS_VISIBLE_LIMIT = 16
MAX_EXPANSION_DEPTH = 2
MAX_MODELS_PER_CHAPTER = 5
MAX_CONCEPTS_PER_CHAPTER = 5
MAX_FILTER_CONCEPTS = 3


# Stable editorial priorities.  A chapter may own more registry entries later,
# but its first graph view remains bounded and predictable until this list is
# consciously revised.
CHAPTER_MODEL_PRIORITIES: dict[str, tuple[str, ...]] = {
    "foundations": (),
    "classic": ("cf", "mf", "fm", "gbdtlr", "word2vec"),
    "retrieval": ("dssm", "mind", "sasrec"),
    "ranking": ("deepfm", "din", "dien"),
    "multitask": ("mmoe", "ple"),
    "generative": ("openonerec", "hstu"),
}


CHAPTER_CONCEPT_PRIORITIES: dict[str, tuple[str, ...]] = {
    "foundations": (
        "data-observation-label",
        "linear-tensors-shapes",
        "probability-random-variable",
        "calculus-derivative-gradient",
        "optimization-sgd",
    ),
    "classic": (
        "data-implicit-feedback",
        "linear-elementwise-dot",
        "linear-low-rank-attention",
        "information-cross-entropy-kl",
        "optimization-sgd",
    ),
    "retrieval": (
        "data-implicit-feedback",
        "linear-elementwise-dot",
        "linear-low-rank-attention",
        "information-softmax-temperature",
        "probability-likelihood-calibration",
    ),
    "ranking": (
        "data-observation-label",
        "linear-elementwise-dot",
        "calculus-functions",
        "information-cross-entropy-kl",
        "optimization-regularization",
    ),
    "multitask": (
        "data-observation-label",
        "probability-conditional-chain",
        "information-softmax-temperature",
        "optimization-gradient-conflict",
        "calculus-chain-rule",
    ),
    "generative": (
        "data-split-leakage",
        "linear-low-rank-attention",
        "probability-conditional-chain",
        "information-sequence-nll-dpo",
        "optimization-learning-rate",
    ),
}


# These edges describe the tutorial's conceptual hand-off, not a claim that the
# target paper is a strict superset or direct descendant of the source paper.
MODEL_EVOLUTION: dict[str, tuple[tuple[str, str, str], ...]] = {
    "classic": (
        ("cf", "mf", "邻域关系 → 低秩表示"),
        ("mf", "fm", "ID 向量 → 通用特征交互"),
        ("fm", "gbdtlr", "低秩交互 → 树叶组合与校准"),
        ("mf", "word2vec", "静态偏好 → 局部序列共现"),
    ),
    "retrieval": (
        ("dssm", "mind", "单一用户向量 → 多兴趣向量"),
        ("mind", "sasrec", "并行兴趣 → 有序兴趣状态"),
    ),
    "ranking": (
        ("deepfm", "din", "静态交叉 → 候选感知历史"),
        ("din", "dien", "候选激活 → 兴趣演化"),
    ),
    "multitask": (("mmoe", "ple", "共享专家 → 共享/专属渐进分离"),),
    "generative": (("hstu", "openonerec", "统一行为序列 → 受约束列表生成"),),
}


def _append_unique(target: list[str], values: Sequence[str]) -> None:
    for value in values:
        if value not in target:
            target.append(value)


def _edge(edge_type: str, source: str, target: str, label: str) -> dict[str, str]:
    return {
        "id": f"edge:{edge_type}:{source}:{target}",
        "type": edge_type,
        "source": source,
        "target": target,
        "label": label,
    }


def _normalise_focus(focus_id: str | None, nodes: Mapping[str, dict[str, Any]]) -> str | None:
    if focus_id is None:
        return None
    if focus_id in nodes:
        return focus_id
    for prefix in ("chapter", "model", "math"):
        candidate = f"{prefix}:{focus_id}"
        if candidate in nodes:
            return candidate
    raise ValueError(f"Unknown knowledge-graph focus: {focus_id}")


def _chapter_models(chapter_key: str, chapter: Mapping[str, Any]) -> tuple[str, ...]:
    registered = tuple(chapter.get("model_ids", ()))
    priority = CHAPTER_MODEL_PRIORITIES.get(chapter_key, registered)
    chosen = [model_id for model_id in priority if model_id in registered]
    for model_id in registered:
        if len(chosen) >= MAX_MODELS_PER_CHAPTER:
            break
        if model_id not in chosen:
            chosen.append(model_id)
    return tuple(chosen[:MAX_MODELS_PER_CHAPTER])


def _chapter_concepts(chapter_key: str, known_topics: set[str]) -> tuple[str, ...]:
    return tuple(
        topic_id
        for topic_id in CHAPTER_CONCEPT_PRIORITIES.get(chapter_key, ())
        if topic_id in known_topics
    )[:MAX_CONCEPTS_PER_CHAPTER]


def build_knowledge_graph(
    chapters: Mapping[str, Mapping[str, Any]],
    models: Sequence[Mapping[str, Any]],
    math_prerequisites: Sequence[Mapping[str, Any]],
    *,
    focus_id: str | None = None,
    depth: int = MAX_EXPANSION_DEPTH,
    limit: int | None = None,
) -> dict[str, Any]:
    """Return a small visible graph with deterministic order and resolved edges.

    With no focus, the six chapter nodes form the overview.  A focused node gets
    an editorially ordered neighbourhood; callers cannot request more than 16
    nodes or expand beyond two relation steps.  Every returned edge has both
    endpoints in the returned ``nodes`` list.
    """

    model_rows = [dict(model) for model in models]
    topic_rows = [dict(topic) for topic in math_prerequisites]
    model_by_id = {str(model["id"]): model for model in model_rows}
    topic_by_id = {str(topic["id"]): topic for topic in topic_rows}
    if len(model_by_id) != len(model_rows):
        raise ValueError("Duplicate model id in knowledge-graph input")
    if len(topic_by_id) != len(topic_rows):
        raise ValueError("Duplicate math topic id in knowledge-graph input")

    nodes: dict[str, dict[str, Any]] = {}
    chapter_order: list[str] = []
    model_order = {str(model["id"]): index for index, model in enumerate(model_rows)}
    topic_order = {str(topic["id"]): index for index, topic in enumerate(topic_rows)}
    model_chapter: dict[str, str] = {}

    for order, (chapter_key, chapter) in enumerate(chapters.items()):
        node_id = f"chapter:{chapter_key}"
        chapter_order.append(node_id)
        nodes[node_id] = {
            "id": node_id,
            "type": "chapter",
            "key": chapter_key,
            "label": str(chapter.get("title", chapter_key)),
            "number": str(chapter.get("number", "")),
            "summary": str(chapter.get("intro", "")),
            "url": f"/#{'foundations' if chapter_key == 'foundations' else chapter_key}",
            "order": order,
        }
        for model_id in chapter.get("model_ids", ()):
            if model_id not in model_by_id:
                raise ValueError(f"Chapter {chapter_key} refers to unknown model {model_id}")
            if model_id in model_chapter:
                raise ValueError(f"Model {model_id} belongs to more than one chapter")
            model_chapter[str(model_id)] = chapter_key

    for model_id, model in model_by_id.items():
        chapter_key = model_chapter.get(model_id, str(model.get("chapter", "")))
        if chapter_key not in chapters:
            raise ValueError(f"Model {model_id} has no resolved chapter")
        node_id = f"model:{model_id}"
        nodes[node_id] = {
            "id": node_id,
            "type": "model",
            "key": model_id,
            "label": str(model.get("name", model_id)),
            "chapter": chapter_key,
            "stage": str(model.get("stage", "")),
            "summary": str(model.get("idea", "")),
            "url": f"/notebooks/{model['notebook']}",
            "order": model_order[model_id],
        }

    for topic_id, topic in topic_by_id.items():
        node_id = f"math:{topic_id}"
        nodes[node_id] = {
            "id": node_id,
            "type": "math",
            "key": topic_id,
            "label": str(topic.get("topic", topic_id)),
            "area": str(topic.get("area", "")),
            "summary": str(topic.get("intuition", "")),
            "url": f"/notebooks/{topic['notebook']}#{topic['anchor']}",
            "order": topic_order[topic_id],
        }

    edges: list[dict[str, str]] = []
    for chapter_key, chapter in chapters.items():
        chapter_id = f"chapter:{chapter_key}"
        for model_id in chapter.get("model_ids", ()):
            edges.append(_edge("contains", chapter_id, f"model:{model_id}", "章节算法"))
        for topic_id in _chapter_concepts(chapter_key, set(topic_by_id)):
            edges.append(_edge("emphasizes", chapter_id, f"math:{topic_id}", "本章共性先修"))

    for topic_id, topic in topic_by_id.items():
        target = f"math:{topic_id}"
        for prerequisite in topic.get("prerequisites", ()):
            if prerequisite not in topic_by_id:
                raise ValueError(f"Math topic {topic_id} has unknown prerequisite {prerequisite}")
            edges.append(_edge("prerequisite", f"math:{prerequisite}", target, "先修于"))
        for model_id in topic.get("model_ids", ()):
            if model_id not in model_by_id:
                raise ValueError(f"Math topic {topic_id} refers to unknown model {model_id}")
            edges.append(_edge("requires", f"model:{model_id}", target, "依赖"))

    for chapter_key, relations in MODEL_EVOLUTION.items():
        chapter_models = set(chapters.get(chapter_key, {}).get("model_ids", ()))
        for source, target, label in relations:
            if source in chapter_models and target in chapter_models:
                edges.append(_edge("evolves", f"model:{source}", f"model:{target}", label))

    edge_ids = [edge["id"] for edge in edges]
    if len(edge_ids) != len(set(edge_ids)):
        raise ValueError("Duplicate edge id in knowledge graph")
    if any(edge["source"] not in nodes or edge["target"] not in nodes for edge in edges):
        raise ValueError("Knowledge graph contains an unresolved edge endpoint")

    focus = _normalise_focus(focus_id, nodes)
    effective_depth = min(MAX_EXPANSION_DEPTH, max(0, int(depth)))
    hard_limit = DEFAULT_VISIBLE_LIMIT if focus is None else FOCUS_VISIBLE_LIMIT
    minimum_limit = DEFAULT_VISIBLE_LIMIT
    if focus is not None and nodes[focus]["type"] != "chapter":
        minimum_limit += 1  # retain all chapters and the non-chapter focus itself
    effective_limit = hard_limit if limit is None else min(hard_limit, max(minimum_limit, int(limit)))

    visible: list[str] = []
    # Navigation context never disappears: focused views keep the same six
    # ordered chapters first, then add the bounded local neighbourhood.
    _append_unique(visible, chapter_order)
    if focus is not None:
        focus_node = nodes[focus]
        if focus_node["type"] != "chapter":
            _append_unique(visible, [focus])
        if effective_depth >= 1:
            if focus_node["type"] in {"chapter", "model"}:
                chapter_key = (
                    str(focus_node["key"])
                    if focus_node["type"] == "chapter"
                    else model_chapter[str(focus_node["key"])]
                )
                _append_unique(
                    visible,
                    [f"model:{model_id}" for model_id in _chapter_models(chapter_key, chapters[chapter_key])],
                )
            else:
                topic_id = str(focus_node["key"])
                related_models = sorted(
                    topic_by_id[topic_id].get("model_ids", ()),
                    key=lambda model_id: (model_order.get(str(model_id), 10**6), str(model_id)),
                )
                _append_unique(visible, [f"model:{item}" for item in related_models[:MAX_MODELS_PER_CHAPTER]])

        if effective_depth >= 2:
            if focus_node["type"] in {"chapter", "model"}:
                chapter_key = (
                    str(focus_node["key"])
                    if focus_node["type"] == "chapter"
                    else model_chapter[str(focus_node["key"])]
                )
                _append_unique(
                    visible,
                    [f"math:{topic_id}" for topic_id in _chapter_concepts(chapter_key, set(topic_by_id))],
                )
            else:
                topic_id = str(focus_node["key"])
                related_math = list(topic_by_id[topic_id].get("prerequisites", ()))
                related_math.extend(
                    candidate_id for candidate_id, topic in topic_by_id.items()
                    if topic_id in topic.get("prerequisites", ())
                )
                _append_unique(
                    visible,
                    [f"math:{item}" for item in related_math[: MAX_CONCEPTS_PER_CHAPTER - 1]],
                )

    visible = visible[:effective_limit]
    visible_set = set(visible)
    visible_edges = [
        edge for edge in edges
        if edge["source"] in visible_set and edge["target"] in visible_set
    ]

    return {
        "nodes": [nodes[node_id] for node_id in visible],
        "edges": visible_edges,
        "state": {
            "focus": focus,
            "depth": effective_depth,
            "visible_limit": effective_limit,
            "truncated": len(visible) == effective_limit and len(nodes) > effective_limit,
        },
        "limits": {
            "default": DEFAULT_VISIBLE_LIMIT,
            "focus": FOCUS_VISIBLE_LIMIT,
            "max_depth": MAX_EXPANSION_DEPTH,
        },
    }


def build_model_filter_graph(
    chapters: Mapping[str, Mapping[str, Any]],
    models: Sequence[Mapping[str, Any]],
    math_prerequisites: Sequence[Mapping[str, Any]],
    stage_filter: str,
    *,
    limit: int = FOCUS_VISIBLE_LIMIT,
) -> dict[str, Any]:
    """Project the model index into the graph for one visible stage filter.

    Unlike ``build_knowledge_graph``'s single-focus neighbourhood, this view
    keeps every matching model (within the same 16-node visual bound), the
    chapters that own those models, and the most widely shared prerequisite
    concepts.  It is used by chapter 2 so one filter controls both the model
    catalogue and its knowledge graph.
    """

    model_rows = [dict(model) for model in models]
    topic_rows = [dict(topic) for topic in math_prerequisites]
    model_by_id = {str(model["id"]): model for model in model_rows}
    if len(model_by_id) != len(model_rows):
        raise ValueError("Duplicate model id in knowledge-graph input")

    matching_models = [
        model for model in model_rows
        if stage_filter.casefold() in str(model.get("stage", "")).casefold()
    ]
    matching_ids = [str(model["id"]) for model in matching_models]
    matching_id_set = set(matching_ids)
    chapter_keys = [
        chapter_key for chapter_key, chapter in chapters.items()
        if matching_id_set.intersection(str(model_id) for model_id in chapter.get("model_ids", ()))
    ]

    nodes: list[dict[str, Any]] = []
    for order, chapter_key in enumerate(chapter_keys):
        chapter = chapters[chapter_key]
        nodes.append({
            "id": f"chapter:{chapter_key}",
            "type": "chapter",
            "key": chapter_key,
            "label": str(chapter.get("title", chapter_key)),
            "number": str(chapter.get("number", "")),
            "summary": str(chapter.get("intro", "")),
            "url": "/catalog#model-explorer",
            "order": order,
        })
    for order, model in enumerate(matching_models):
        nodes.append({
            "id": f"model:{model['id']}",
            "type": "model",
            "key": str(model["id"]),
            "label": str(model.get("name", model["id"])),
            "chapter": str(model.get("chapter", "")),
            "stage": str(model.get("stage", "")),
            "summary": str(model.get("idea", "")),
            "url": f"/notebooks/{model['notebook']}",
            "order": order,
        })

    remaining = max(0, min(FOCUS_VISIBLE_LIMIT, max(DEFAULT_VISIBLE_LIMIT, int(limit))) - len(nodes))
    related_topics = [
        topic for topic in topic_rows
        if matching_id_set.intersection(str(model_id) for model_id in topic.get("model_ids", ()))
    ]
    related_topics.sort(key=lambda topic: (
        -len(matching_id_set.intersection(str(model_id) for model_id in topic.get("model_ids", ()))),
        topic_rows.index(topic),
    ))
    # A stage filter can match models from multiple chapters.  Showing every
    # shared prerequisite produces a dense cross-column mesh, so keep the three
    # most widely shared concepts and expose the rest through each node's
    # linked curriculum page.
    selected_topics = related_topics[: min(MAX_FILTER_CONCEPTS, remaining)]
    for order, topic in enumerate(selected_topics):
        nodes.append({
            "id": f"math:{topic['id']}",
            "type": "math",
            "key": str(topic["id"]),
            "label": str(topic.get("topic", topic["id"])),
            "area": str(topic.get("area", "")),
            "summary": str(topic.get("intuition", "")),
            "url": f"/notebooks/{topic['notebook']}#{topic['anchor']}",
            "order": order,
        })

    visible_ids = {node["id"] for node in nodes}
    edges: list[dict[str, str]] = []
    for chapter_key in chapter_keys:
        for model_id in chapters[chapter_key].get("model_ids", ()):
            if str(model_id) in matching_id_set:
                edges.append(_edge("contains", f"chapter:{chapter_key}", f"model:{model_id}", "章节算法"))
    for topic in selected_topics:
        topic_id = str(topic["id"])
        for model_id in topic.get("model_ids", ()):
            if str(model_id) in matching_id_set:
                edges.append(_edge("requires", f"model:{model_id}", f"math:{topic_id}", "依赖"))
        for prerequisite in topic.get("prerequisites", ()):
            edge = _edge("prerequisite", f"math:{prerequisite}", f"math:{topic_id}", "先修于")
            if edge["source"] in visible_ids:
                edges.append(edge)
    for chapter_key, relations in MODEL_EVOLUTION.items():
        for source, target, label in relations:
            if source in matching_id_set and target in matching_id_set:
                edges.append(_edge("evolves", f"model:{source}", f"model:{target}", label))

    edges = [
        edge for edge in edges
        if edge["source"] in visible_ids and edge["target"] in visible_ids
    ]
    return {
        "nodes": nodes,
        "edges": edges,
        "state": {
            "focus": f"filter:{stage_filter}",
            "depth": MAX_EXPANSION_DEPTH,
            "visible_limit": FOCUS_VISIBLE_LIMIT,
            "truncated": len(related_topics) > len(selected_topics),
            "filter": stage_filter,
            "model_count": len(matching_models),
        },
        "limits": {
            "default": DEFAULT_VISIBLE_LIMIT,
            "focus": FOCUS_VISIBLE_LIMIT,
            "max_depth": MAX_EXPANSION_DEPTH,
        },
    }
