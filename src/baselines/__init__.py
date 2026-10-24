"""Baseline NAS algorithms for head-to-head comparison with LLM-ENAS-PKB.

Each baseline reuses the same ground-truth oracle
(``llm_enas_pkb.nasbench201.NASBench201``) and spends the same query budget, so
its results are directly comparable to ``llm_enas_pkb.evolution.EvolutionSearch``.

Convention
----------
Every baseline exposes a class whose constructor takes ``(config, bench)`` and a
``run() -> dict`` method returning the same result shape as ``EvolutionSearch.run``:

    {"best_arch", "best_valid", "best_test", "queries", "history"}

so a single comparison harness can aggregate any baseline and the evolutionary
method without special-casing either.
"""

# Re-export lazily via ``__getattr__`` so ``python -m baselines.random_search``
# does not trigger the runpy "found in sys.modules" warning: an eager
# ``from .random_search import RandomSearch`` here would import the submodule as a
# side effect of importing the package, before runpy executes it as ``__main__``.
__all__ = ["RandomSearch"]


def __getattr__(name: str):
    if name == "RandomSearch":
        from .random_search import RandomSearch

        return RandomSearch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
