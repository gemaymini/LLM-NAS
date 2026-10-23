"""Command-line entry point for the LLM-ENAS-PKB regression surrogate on NAS-Bench-201."""

from __future__ import annotations

import argparse
import logging
import os
import statistics
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
    p.add_argument("--run", type=int, default=1, dest="num_runs",
                   help="Number of repeated runs with different seeds (reports mean performance)")
    p.add_argument("--offline", action="store_true", help="Use a deterministic pseudo-LLM (no API)")
    p.add_argument("--no-projection", action="store_true", dest="no_projection",
                   help="Disable the Eq. (6) projection in the encoder")
    p.add_argument("--thinking", action="store_true",
                   help="Enable DeepSeek thinking (reasoning) mode (default: disabled)")
    p.add_argument("--reasoning-effort", default=None, choices=["low", "medium", "high"],
                   help="Reasoning intensity, only used together with --thinking")
    p.add_argument("--log-level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--prompt-log-path", default=None,
                   help="Append LLM prompts/responses to this file (empty string disables)")
    p.add_argument("--search-log-path", default=None,
                   help="Append per-generation search log to this file (empty string disables)")
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
        "prompt_log_path": args.prompt_log_path,
        "search_log_path": args.search_log_path,
        "reasoning_effort": args.reasoning_effort,
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
    if args.thinking:
        cfg.thinking = True

    # If the prompt-log path was not explicitly set (via CLI or env), auto-generate
    # a timestamped, hyperparameter-tagged filename so separate runs don't share one
    # log file.
    # Whether the log paths were left to auto-generation (so each repeated run
    # gets its own seed-tagged file) vs. explicitly fixed by the caller.
    auto_prompt = args.prompt_log_path is None and "LLMENAS_PROMPT_LOG_PATH" not in os.environ
    auto_search = args.search_log_path is None and "LLMENAS_SEARCH_LOG_PATH" not in os.environ

    # Map the level name to the logging module's constant and set a shared format.
    logging.basicConfig(
        level=getattr(logging, (cfg.log_level or "INFO").upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if cfg.offline:
        logging.getLogger().warning("Running in OFFLINE mode (deterministic pseudo-LLM, no API call).")

    # Wire the benchmark (ground truth) and predictor (encoder + PKB + LLM
    # surrogate) once; both are seed-independent, so they are shared across
    # repeated runs.  Only the search loop (and its RNG) is re-created per seed.
    bench = NASBench201(cfg)
    predictor = RegressionPredictor(cfg)

    num_runs = max(1, args.num_runs)
    # Resolve the per-run seeds: ``--seed`` fixes the base and each subsequent
    # run adds 1, so a whole batch is reproducible from a single ``--seed`` value.
    # Without ``--seed`` the base is Config's random seed (printed in each run's
    # log header and in the per-run summary below).
    base_seed = cfg.seed
    seeds = [base_seed + i for i in range(num_runs)]

    rows: list[dict] = []
    for i, seed in enumerate(seeds, start=1):
        cfg.seed = seed
        if auto_prompt:
            cfg.prompt_log_path = cfg.auto_prompt_log_path()
        if auto_search:
            cfg.search_log_path = cfg.auto_search_log_path()
        result = EvolutionSearch(cfg, bench, predictor).run()
        rows.append(
            {
                "run": i,
                "seed": seed,
                "best_valid": result["best_valid"],
                "best_test": result["best_test"],
                "queries": result["queries"],
                "best_arch": result["best_arch"],
            }
        )

    # Pretty-print the final result summary.
    print("\n" + "=" * 60)
    print("LLM-ENAS-PKB (regression) — NAS-Bench-201 /", cfg.dataset)
    print("=" * 60)
    print(f"model           : {cfg.model}" + ("  [OFFLINE]" if cfg.offline else ""))
    if num_runs > 1:
        print(f"runs            : {num_runs} (one seed each)")
        print("-" * 60)
        for r in rows:
            print(
                f"run {r['run']:>2} | seed {r['seed']:>10} | "
                f"best valid {r['best_valid'] * 100:6.2f}% | "
                f"best test  {r['best_test'] * 100:6.2f}% | "
                f"queries {r['queries']}"
            )
        print("-" * 60)
        valid = [r["best_valid"] * 100 for r in rows]
        test = [r["best_test"] * 100 for r in rows]
        queries = [r["queries"] for r in rows]
        print(
            f"average (mean)  : best valid {statistics.mean(valid):6.2f}% "
            f"± {statistics.stdev(valid):.2f}% | "
            f"best test {statistics.mean(test):6.2f}% "
            f"± {statistics.stdev(test):.2f}% | "
            f"queries {statistics.mean(queries):.1f}"
        )
    else:
        r = rows[0]
        print(f"best arch       : {arch_to_str(r['best_arch'])}")
        print(f"best valid acc  : {r['best_valid'] * 100:.2f}%")
        print(f"best test acc   : {r['best_test'] * 100:.2f}%")
        print(f"queries used    : {r['queries']} / {cfg.query_budget}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
