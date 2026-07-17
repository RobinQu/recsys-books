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
    "run_dssm": "3_2_1_dssm",
    "run_mind": "3_2_2_mind",
    "run_sasrec": "3_2_3_sasrec",
    "run_deepfm": "3_3_1_deepfm",
    "run_din": "3_3_2_din",
    "run_dien": "3_3_3_dien",
    "run_mmoe": "3_4_1_mmoe",
    "run_ple": "3_4_2_ple",
    "run_openonerec": "4_2_openonerec_practice",
    "run_hstu": "4_3_dlrm_hstu_practice",
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
def run_openonerec(epochs: int = 32) -> dict: return load_runner("run_openonerec")(epochs)
def run_hstu(epochs: int = 26) -> dict: return load_runner("run_hstu")(epochs)


__all__ = [*CHAPTER_RUNNERS, "load_runner", "save_records"]
