from __future__ import annotations

from pathlib import Path

import pytest

from app.content import CHAPTERS, MATH_PREREQUISITES, MODELS
from app.knowledge_graph import (
    CHAPTER_CONCEPT_PRIORITIES,
    CHAPTER_MODEL_PRIORITIES,
    DEFAULT_VISIBLE_LIMIT,
    FOCUS_VISIBLE_LIMIT,
    MAX_CONCEPTS_PER_CHAPTER,
    MAX_EXPANSION_DEPTH,
    MAX_MODELS_PER_CHAPTER,
    MODEL_EVOLUTION,
    build_knowledge_graph,
)


def _ids(graph: dict) -> list[str]:
    return [node["id"] for node in graph["nodes"]]


def _assert_resolved_visible_edges(graph: dict) -> None:
    visible = set(_ids(graph))
    assert len(visible) == len(graph["nodes"])
    assert len({edge["id"] for edge in graph["edges"]}) == len(graph["edges"])
    assert all(edge["source"] in visible and edge["target"] in visible for edge in graph["edges"])


ORDERED_CHAPTER_IDS = [f"chapter:{chapter}" for chapter in CHAPTERS]


def test_default_graph_is_six_ordered_namespaced_chapters() -> None:
    graph = build_knowledge_graph(CHAPTERS, MODELS, MATH_PREREQUISITES)

    assert _ids(graph) == ORDERED_CHAPTER_IDS
    assert len(graph["nodes"]) == DEFAULT_VISIBLE_LIMIT == 6
    assert graph["edges"] == []
    assert graph["state"] == {
        "focus": None,
        "depth": MAX_EXPANSION_DEPTH,
        "visible_limit": DEFAULT_VISIBLE_LIMIT,
        "truncated": True,
    }
    _assert_resolved_visible_edges(graph)


def test_editorial_priorities_are_explicit_stable_and_bounded() -> None:
    assert tuple(CHAPTER_MODEL_PRIORITIES) == tuple(CHAPTERS)
    assert tuple(CHAPTER_CONCEPT_PRIORITIES) == tuple(CHAPTERS)
    assert all(len(models) <= MAX_MODELS_PER_CHAPTER == 5 for models in CHAPTER_MODEL_PRIORITIES.values())
    assert all(len(topics) <= MAX_CONCEPTS_PER_CHAPTER == 5 for topics in CHAPTER_CONCEPT_PRIORITIES.values())

    known_models = {model["id"] for model in MODELS}
    known_topics = {topic["id"] for topic in MATH_PREREQUISITES}
    assert all(set(models) <= known_models for models in CHAPTER_MODEL_PRIORITIES.values())
    assert all(set(topics) <= known_topics for topics in CHAPTER_CONCEPT_PRIORITIES.values())

    retrieval = build_knowledge_graph(
        CHAPTERS, MODELS, MATH_PREREQUISITES, focus_id="chapter:retrieval", depth=2
    )
    assert _ids(retrieval) == [
        *ORDERED_CHAPTER_IDS,
        "model:dssm", "model:mind", "model:sasrec",
        "math:data-implicit-feedback",
        "math:linear-elementwise-dot",
        "math:linear-low-rank-attention",
        "math:information-softmax-temperature",
        "math:probability-likelihood-calibration",
    ]


def test_focus_is_bounded_and_depth_is_clamped_to_two() -> None:
    depth_zero = build_knowledge_graph(
        CHAPTERS, MODELS, MATH_PREREQUISITES,
        focus_id="retrieval", depth=0, limit=999,
    )
    assert _ids(depth_zero) == ORDERED_CHAPTER_IDS
    assert depth_zero["edges"] == []

    focused = build_knowledge_graph(
        CHAPTERS, MODELS, MATH_PREREQUISITES,
        focus_id="chapter:retrieval", depth=99, limit=999,
    )
    assert focused["state"]["depth"] == MAX_EXPANSION_DEPTH == 2
    assert focused["state"]["visible_limit"] == FOCUS_VISIBLE_LIMIT == 16
    assert len(focused["nodes"]) <= FOCUS_VISIBLE_LIMIT
    assert set(_ids(depth_zero)) <= set(_ids(focused))
    _assert_resolved_visible_edges(focused)


@pytest.mark.parametrize(
    "focus_id",
    ["chapter:classic", "chapter:ranking", "model:dssm", "math:optimization-sgd"],
)
def test_every_visible_edge_has_no_hidden_endpoint(focus_id: str) -> None:
    graph = build_knowledge_graph(
        CHAPTERS, MODELS, MATH_PREREQUISITES, focus_id=focus_id
    )
    assert _ids(graph)[:DEFAULT_VISIBLE_LIMIT] == ORDERED_CHAPTER_IDS
    assert len(graph["nodes"]) <= FOCUS_VISIBLE_LIMIT
    assert sum(node["type"] == "model" for node in graph["nodes"]) <= MAX_MODELS_PER_CHAPTER
    assert sum(node["type"] == "math" for node in graph["nodes"]) <= MAX_CONCEPTS_PER_CHAPTER
    _assert_resolved_visible_edges(graph)


def test_curated_evolution_edges_stay_inside_their_chapter_and_keep_order() -> None:
    for chapter_key, relations in MODEL_EVOLUTION.items():
        chapter_models = set(CHAPTERS[chapter_key]["model_ids"])
        assert all(source in chapter_models and target in chapter_models for source, target, _ in relations)

    ranking = build_knowledge_graph(
        CHAPTERS, MODELS, MATH_PREREQUISITES,
        focus_id="chapter:ranking", depth=1,
    )
    evolution = [
        (edge["source"], edge["target"])
        for edge in ranking["edges"]
        if edge["type"] == "evolves"
    ]
    assert evolution == [
        ("model:deepfm", "model:din"),
        ("model:din", "model:dien"),
    ]


def test_invalid_references_fail_before_rendering() -> None:
    duplicate_models = [*MODELS, dict(MODELS[0])]
    with pytest.raises(ValueError, match="Duplicate model id"):
        build_knowledge_graph(CHAPTERS, duplicate_models, MATH_PREREQUISITES)

    bad_math = [dict(topic) for topic in MATH_PREREQUISITES]
    bad_math[0] = {**bad_math[0], "prerequisites": ["missing-topic"]}
    with pytest.raises(ValueError, match="unknown prerequisite"):
        build_knowledge_graph(CHAPTERS, MODELS, bad_math)

    with pytest.raises(ValueError, match="Unknown knowledge-graph focus"):
        build_knowledge_graph(CHAPTERS, MODELS, MATH_PREREQUISITES, focus_id="paper:unknown")


def test_dormant_frontend_uses_requested_columns_and_bounded_mobile_grid() -> None:
    javascript = Path("app/static/app.js").read_text(encoding="utf-8")
    css = Path("app/static/app.css").read_text(encoding="utf-8")

    assert "if (node.type === 'math') return 1" in javascript
    assert "if (node.type === 'chapter') return 2" in javascript
    assert "const MAX_VISIBLE_NODES = 16" in javascript
    assert "const VISUAL_EDGE_TYPES = new Set(['contains', 'emphasizes', 'evolves', 'prerequisite'])" in javascript
    assert "edge.source !== currentView.state?.focus" in javascript
    assert "grid-template-columns:repeat(2,minmax(0,1fr))" in css
    assert "grid-column:auto!important" in css and "grid-row:auto!important" in css
    assert "min-width:0" in css and "min-height:48px" in css
