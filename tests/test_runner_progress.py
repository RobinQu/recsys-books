"""Progress and bounded-memory contracts for chapter-local experiment runners."""
from importlib import import_module
from inspect import Parameter, signature

import pandas as pd

from recsys_lab.industrial_experiments import CHAPTER_RUNNERS, load_runner


CLASSIC_SLUGS = [
    "4_2_collaborative_filtering",
    "4_3_matrix_factorization",
    "4_4_factorization_machine",
    "4_5_gbdt_lr",
    "4_6_word2vec",
]


def test_all_algorithm_runners_accept_keyword_only_progress():
    callables = [load_runner(name) for name in CHAPTER_RUNNERS]
    callables.extend(
        import_module(f"chapter_code.{slug}.train").train_and_evaluate
        for slug in CLASSIC_SLUGS
    )
    for runner in callables:
        assert signature(runner).parameters["progress"].kind is Parameter.KEYWORD_ONLY

    for name in ("run_openonerec", "run_hstu"):
        parameters = signature(load_runner(name)).parameters
        assert parameters["cpu_smoke"].kind is Parameter.POSITIONAL_OR_KEYWORD


def test_item2vec_stream_contains_every_window_pair_once():
    module = import_module("chapter_code.4_6_word2vec.train")
    sequences = [[1, 2, 3, 4], [7, 8]]
    batches = list(module._skip_gram_pair_batches(sequences, batch_size=3, window=2))
    actual = [tuple(pair) for batch in batches for pair in batch.tolist()]
    expected = [
        (center, sequence[context_index])
        for sequence in sequences
        for center_index, center in enumerate(sequence)
        for context_index in range(max(0, center_index - 2), min(len(sequence), center_index + 3))
        if context_index != center_index
    ]
    assert sorted(actual) == sorted(expected)
    assert len(actual) == module._skip_gram_pair_count(sequences, window=2)
    assert max(len(batch) for batch in batches) <= 3


def test_matrix_factorization_runner_reports_bounded_stage_progress(monkeypatch):
    module = import_module("chapter_code.4_3_matrix_factorization.train")
    ratings = pd.DataFrame(
        {
            "user_id": [0, 0, 0, 1, 1, 1],
            "item_id": [0, 1, 2, 0, 2, 1],
            "rating": [4.0, 3.0, 5.0, 2.0, 4.0, 5.0],
            "timestamp": [1, 2, 3, 1, 2, 3],
        }
    )
    monkeypatch.setattr(module, "load_movielens", lambda **_: (ratings, {}))
    events = []
    result = module.train_and_evaluate(epochs=2, progress=events.append)

    assert result["rmse"] >= 0
    assert {event["stage"] for event in events} == {"data_prepare", "train", "inference", "evaluate"}
    assert len(events) <= 12
    for stage in ("data_prepare", "train", "inference", "evaluate"):
        stage_events = [event for event in events if event["stage"] == stage]
        assert stage_events[0]["current"] == 0
        assert stage_events[-1]["current"] == stage_events[-1]["total"]
