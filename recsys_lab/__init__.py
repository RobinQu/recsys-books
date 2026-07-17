"""Small deterministic workloads shared by the app notebooks and tests."""

from .experiments import run_classic, run_generative, run_multitask, run_ranking, run_retrieval

__all__ = ["run_classic", "run_retrieval", "run_ranking", "run_multitask", "run_generative"]

