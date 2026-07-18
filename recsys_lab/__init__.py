"""Small deterministic workloads shared by the app notebooks and tests.

Experiment imports are lazy so the resource initializer can run with only the
Python standard library before the scientific environment is installed.
"""

from __future__ import annotations

from typing import Any

__all__ = ["run_classic", "run_retrieval", "run_ranking", "run_multitask", "run_generative"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    from . import experiments

    return getattr(experiments, name)
