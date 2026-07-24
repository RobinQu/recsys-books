"""Backward-compatible entry points for chapter-local experiments.

The tutorial used to hide every algorithm in this module. Implementations now
live beside their chapters; this file only provides stable cross-chapter routing
for older notebooks, summary APIs and external callers.
"""
from __future__ import annotations

from importlib import import_module
from typing import Callable

from .runtime import ProgressCallback, save_records


CHAPTER_RUNNERS = {
    "run_dssm": "5_2_dssm",
    "run_mind": "5_3_mind",
    "run_sasrec": "5_4_sasrec",
    "run_deepfm": "6_2_deepfm",
    "run_din": "6_3_din",
    "run_dien": "6_4_dien",
    "run_mmoe": "7_2_mmoe",
    "run_ple": "7_3_ple",
    "run_openonerec": "8_2_openonerec_practice",
    "run_hstu": "8_3_dlrm_hstu_practice",
}


def load_runner(name: str) -> Callable[..., dict]:
    """Resolve a public runner to the chapter's visible ``train.py`` file."""
    slug = CHAPTER_RUNNERS[name]
    module = import_module(f"chapter_code.{slug}.train")
    return getattr(module, name)


def run_dssm(epochs: int = 24, *, progress: ProgressCallback | None = None) -> dict:
    return load_runner("run_dssm")(epochs, progress=progress)


def run_mind(epochs: int = 26, *, progress: ProgressCallback | None = None) -> dict:
    return load_runner("run_mind")(epochs, progress=progress)


def run_sasrec(epochs: int = 30, *, progress: ProgressCallback | None = None) -> dict:
    return load_runner("run_sasrec")(epochs, progress=progress)


def run_deepfm(epochs: int = 28, *, progress: ProgressCallback | None = None) -> dict:
    return load_runner("run_deepfm")(epochs, progress=progress)


def run_din(epochs: int = 26, *, progress: ProgressCallback | None = None) -> dict:
    return load_runner("run_din")(epochs, progress=progress)


def run_dien(epochs: int = 30, *, progress: ProgressCallback | None = None) -> dict:
    return load_runner("run_dien")(epochs, progress=progress)


def run_mmoe(epochs: int = 28, *, progress: ProgressCallback | None = None) -> dict:
    return load_runner("run_mmoe")(epochs, progress=progress)


def run_ple(epochs: int = 28, *, progress: ProgressCallback | None = None) -> dict:
    return load_runner("run_ple")(epochs, progress=progress)


def run_openonerec(
    epochs: int = 32,
    cpu_smoke: bool = False,
    *,
    progress: ProgressCallback | None = None,
) -> dict:
    return load_runner("run_openonerec")(epochs, cpu_smoke=cpu_smoke, progress=progress)


def run_hstu(
    epochs: int = 26,
    cpu_smoke: bool = False,
    *,
    progress: ProgressCallback | None = None,
) -> dict:
    return load_runner("run_hstu")(epochs, cpu_smoke=cpu_smoke, progress=progress)


__all__ = [*CHAPTER_RUNNERS, "load_runner", "save_records"]
