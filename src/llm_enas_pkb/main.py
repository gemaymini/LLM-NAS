"""Command-line entry point for the LLM-ENAS-PKB regression surrogate on NAS-Bench-201."""

from __future__ import annotations

import argparse
import logging
import sys

from .config import Config
from .evolution import EvolutionSearch
from .nasbench201 import NASBench201
from .predictor import RegressionPredictor
from .search_space import arch_to_str


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser with one flag per overridable config knob.

    Every argument defaults to ``None`` so the caller can distinguish "not
    provided" (leave the env-var / config default untouched) from an explicit
    value passed on the command line.
    """
    p = argparse.ArgumentParser(
        prog="llm-enas-pkb",
        description="LLM-ENAS-PKB regression surrogate (DeepSeek API) on NAS-Bench-201.",
    )
    p.add_argument("--dataset", choices=["cifar10", "cifar100", "ImageNet16-120"], default=None)
    p.add_argument("--model", default=None, help="DeepSeek model (deepseek-v4-pro / deepseek-flash)")
    p.add_argument("--iterations", type=int, default=None, dest="max_iterations")
    p.add_argument("--population-size", type=int, default=None)
    p.add_argument("--offspring-size", type=int, default=None)
    p.add_argument("--query-budget", type=int, default=None)
    p.add_argument("--n-init", type=int, default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--k-perf", type=int, default=None)
    p.add_argument("--k-div", type=int, default=None)
    p.add_argument("--selected-size", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--data-path", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--offline", action="store_true", help="Use a deterministic pseudo-LLM (no API)")
    p.add_argument("--no-projection", action="store_true", dest="no_projection",
                   help="Disable the Eq. (6) projection in the encoder")
    p.add_argument("--log-level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # Start from env-var configuration, then apply explicit CLI overrides on top.
    cfg = Config.from_env()
    # CLI overrides (only when explicitly provided).
    overrides = {
        "dataset": args.dataset,
        "model": args.model,
        "max_iterations": args.max_iterations,
        "population_size": args.population_size,
        "offspring_size": args.offspring_size,
        "query_budget": args.query_budget,
        "n_init": args.n_init,
        "alpha": args.alpha,
        "k_perf": args.k_perf,
        "k_div": args.k_div,
        "selected_size": args.selected_size,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "data_path": args.data_path,
        "seed": args.seed,
        "log_level": args.log_level,
    }
    for name, value in overrides.items():
        if value is not None:
            setattr(cfg, name, value)
    # Boolean flags handled separately: ``store_true`` yields False by default, so
    # we only act when the flag was actually passed.
    if args.offline:
        cfg.offline = True
    if args.no_projection:
        cfg.use_projection = False

    # Map the level name to the logging module's constant and set a shared format.
    logging.basicConfig(
        level=getattr(logging, (cfg.log_level or "INFO").upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if cfg.offline:
        logging.getLogger().warning("Running in OFFLINE mode (deterministic pseudo-LLM, no API call).")

    # Wire the three components: benchmark (ground truth), predictor (encoder +
    # PKB + LLM surrogate), and the evolutionary search loop that drives them.
    bench = NASBench201(cfg)
    predictor = RegressionPredictor(cfg)
    search = EvolutionSearch(cfg, bench, predictor)
    result = search.run()

    # Pretty-print the final result summary.
    print("\n" + "=" * 60)
    print("LLM-ENAS-PKB (regression) — NAS-Bench-201 /", cfg.dataset)
    print("=" * 60)
    print(f"model           : {cfg.model}" + ("  [OFFLINE]" if cfg.offline else ""))
    print(f"best arch       : {arch_to_str(result['best_arch'])}")
    print(f"best valid acc  : {result['best_valid'] * 100:.2f}%")
    print(f"best test acc   : {result['best_test'] * 100:.2f}%")
    print(f"queries used    : {result['queries']} / {cfg.query_budget}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
