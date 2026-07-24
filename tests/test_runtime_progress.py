from __future__ import annotations

import builtins

import torch

from recsys_lab import experiments, industrial_experiments, runtime


def test_emit_progress_is_silent_by_default_and_omits_empty_fields(capsys):
    runtime.emit_progress(None, stage="load", current=0, total=4)
    assert capsys.readouterr().out == ""

    events: list[dict[str, object]] = []
    runtime.emit_progress(events.append, stage="load", message="dataset ready")

    assert events == [{"stage": "load", "message": "dataset ready"}]


def test_print_progress_is_stable_single_line_and_flushes(monkeypatch):
    calls = []

    def fake_print(value, *, flush):
        calls.append((value, flush))

    monkeypatch.setattr(builtins, "print", fake_print)
    runtime.print_progress(
        {
            "stage": "train",
            "current": 3,
            "total": 8,
            "message": "epoch\n1/2",
            "metrics": {"loss": 0.123456789, "auc": 0.75},
        }
    )

    assert calls == [("[train] 3/8 epoch 1/2 auc=0.75 loss=0.123457", True)]


def test_progress_due_keeps_long_loops_bounded():
    checkpoints = [current for current in range(1001) if runtime.progress_due(current, 1000, max_updates=20)]

    assert checkpoints[0] == 0
    assert checkpoints[-1] == 1000
    assert len(checkpoints) <= 21


def test_train_binary_reports_bounded_train_steps():
    model = torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.Sigmoid())
    features = {"features": torch.arange(80, dtype=torch.float32).reshape(40, 2) / 80}
    labels = torch.tensor(([0.0, 1.0] * 20), dtype=torch.float32).reshape(-1, 1)
    events: list[dict[str, object]] = []

    class MappingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.network = model

        def forward(self, batch):
            return self.network(batch["features"])

    losses = runtime.train_binary(
        MappingModel(),
        features,
        labels,
        epochs=3,
        lr=0.01,
        batch_size=4,
        progress=events.append,
    )

    assert len(losses) == 3
    assert events[0] == {"stage": "train", "current": 0, "total": 30, "message": "starting"}
    assert events[-1]["stage"] == "train"
    assert events[-1]["current"] == events[-1]["total"] == 30
    assert events[-1]["metrics"] == {"loss": losses[-1]}
    assert len(events) <= 22


def test_train_binary_progress_does_not_change_seeded_results():
    torch.manual_seed(2026)
    initial = torch.nn.Linear(3, 1)
    initial_state = {name: value.detach().clone() for name, value in initial.state_dict().items()}
    features = {"features": torch.arange(72, dtype=torch.float32).reshape(24, 3) / 72}
    labels = torch.tensor(([0.0, 1.0] * 12), dtype=torch.float32).reshape(-1, 1)

    class MappingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.network = torch.nn.Sequential(torch.nn.Linear(3, 1), torch.nn.Sigmoid())
            self.network[0].load_state_dict(initial_state)

        def forward(self, batch):
            return self.network(batch["features"])

    silent_model = MappingModel()
    visible_model = MappingModel()
    silent_losses = runtime.train_binary(
        silent_model,
        features,
        labels,
        epochs=4,
        lr=0.01,
        batch_size=6,
    )
    events: list[dict[str, object]] = []
    visible_losses = runtime.train_binary(
        visible_model,
        features,
        labels,
        epochs=4,
        lr=0.01,
        batch_size=6,
        progress=events.append,
    )

    assert visible_losses == silent_losses
    for silent_parameter, visible_parameter in zip(silent_model.parameters(), visible_model.parameters()):
        torch.testing.assert_close(visible_parameter, silent_parameter, rtol=0, atol=0)
    assert events[0]["current"] == 0
    assert events[-1]["current"] == events[-1]["total"] == 16


def test_seed_everything_uses_bounded_smoke_threads_and_bounded_full_threads(monkeypatch):
    configured: list[int] = []
    monkeypatch.setattr(runtime.torch, "set_num_threads", configured.append)
    monkeypatch.setattr(runtime.os, "cpu_count", lambda: 24)

    monkeypatch.setenv("RECSYS_PROFILE", "smoke")
    monkeypatch.setenv("RECSYS_TORCH_THREADS", "6")
    runtime.seed_everything()

    monkeypatch.setenv("RECSYS_PROFILE", "full")
    monkeypatch.delenv("RECSYS_TORCH_THREADS")
    runtime.seed_everything()

    monkeypatch.setenv("RECSYS_TORCH_THREADS", "6")
    runtime.seed_everything()

    assert configured == [4, 8, 6]


def test_industrial_wrapper_forwards_keyword_progress(monkeypatch):
    callback = lambda event: None
    calls = []

    def runner(epochs, *, progress):
        calls.append((epochs, progress))
        return {"ok": True}

    monkeypatch.setattr(industrial_experiments, "load_runner", lambda name: runner)

    assert industrial_experiments.run_dssm(epochs=3, progress=callback) == {"ok": True}
    assert calls == [(3, callback)]


def test_aggregate_api_forwards_progress(monkeypatch):
    callback = lambda event: None
    calls = []

    def run_deepfm(epochs, *, progress):
        calls.append((epochs, progress))
        return {"auc": 0.7, "lr_auc": 0.6}

    monkeypatch.setattr(experiments, "run_deepfm", run_deepfm)

    result = experiments.run_ranking(epochs=5, progress=callback)

    assert result["deepfm_auc"] == 0.7
    assert calls == [(5, callback)]
