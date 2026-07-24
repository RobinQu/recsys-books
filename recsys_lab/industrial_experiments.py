"""Backward-compatible entry points for chapter-local experiments.

The tutorial used to hide every algorithm in this module. Implementations now
live beside their chapters; this file only provides stable cross-chapter routing
for older notebooks, summary APIs and external callers.
"""
from __future__ import annotations

from importlib import import_module
from typing import Callable

from .runtime import save_records


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


def run_dssm(epochs: int = 24) -> dict: return load_runner("run_dssm")(epochs)
def run_mind(epochs: int = 26) -> dict: return load_runner("run_mind")(epochs)
def run_sasrec(epochs: int = 30) -> dict: return load_runner("run_sasrec")(epochs)
def run_deepfm(epochs: int = 28) -> dict: return load_runner("run_deepfm")(epochs)
def run_din(epochs: int = 26) -> dict: return load_runner("run_din")(epochs)
def run_dien(epochs: int = 30) -> dict: return load_runner("run_dien")(epochs)
def run_mmoe(epochs: int = 28) -> dict: return load_runner("run_mmoe")(epochs)
def run_ple(epochs: int = 28) -> dict: return load_runner("run_ple")(epochs)
def run_openonerec(epochs: int = 32, cpu_smoke: bool = False) -> dict:
    return load_runner("run_openonerec")(epochs, cpu_smoke=cpu_smoke)


def run_hstu(epochs: int = 26, cpu_smoke: bool = False) -> dict:
    return load_runner("run_hstu")(epochs, cpu_smoke=cpu_smoke)


__all__ = [*CHAPTER_RUNNERS, "load_runner", "save_records"]
