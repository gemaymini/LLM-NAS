"""Random-search baseline for NAS-Bench-201.

Random search is the standard sanity-check baseline in NAS: it samples
architectures uniformly at random from the search space and ground-truth evaluates
each one, tracking the best found.  It spends exactly the same query budget as the
LLM-ENAS-PKB pipeline (``query_budget``, 100 in the paper) so the two can be
compared head-to-head on validation/test accuracy at equal cost.

Unlike the evolutionary pipeline there is no surrogate model and no genetic
operators: the number of benchmark queries equals the number of architectures
sampled, and every sampled architecture is distinct (duplicates would waste budget
on an already-known answer).
"""

from __future__ import annotations

import argparse
import logging
import os
import statistics
import sys
from datetime import datetime

import numpy as np

from llm_enas_pkb.config import Config
from llm_enas_pkb.nasbench201 import NASBench201
from llm_enas_pkb.search_space import arch_to_str

logger = logging.getLogger(__name__)

_SEP = "=" * 78
_SUB = "-" * 78


class RandomSearch:
    """Sample ``query_budget`` distinct random architectures and keep the best.

    The constructor signature ``(config, bench)`` and the ``run()`` result shape
    mirror ``llm_enas_pkb.evolution.EvolutionSearch`` so both algorithms can be
    driven by the same comparison harness (see the ``baselines`` package docstring).
    """

    def __init__(self, config: Config, bench: NASBench201):
        self.config = config
        self.bench = bench

    # ------------------------------------------------------------- logging
    def _write(self, block: str) -> None:
        """Append one block to the search log, flushing so it stays visible in real time."""
        if not self.config.search_log_path:
            return
        try:
            with open(self.config.search_log_path, "a", encoding="utf-8") as fh:
                fh.write(block + "\n")
                fh.flush()
        except OSError as exc:
            logger.warning("Failed to write search log %s: %s", self.config.search_log_path, exc)

    def _log_header(self, budget: int) -> None:
        cfg = self.config
        self._write(
            "\n".join(
                [
                    _SEP,
                    f"[{datetime.now().isoformat(timespec='seconds')}] Random-search baseline start",
                    (
                        f"dataset={cfg.dataset} seed={cfg.seed} "
                        f"query_budget={cfg.query_budget} sampled={budget}"
                    ),
                    _SUB,
                ]
            )
        )

    def _log_eval(self, i: int, arch, value: float, best_valid: float) -> None:
        self._write(f"  {i:>4}. {arch_to_str(arch)}  val={value:.4f}  best={best_valid:.4f}")

    def _log_best(self, arch, value: float) -> None:
        self._write("\n".join([_SUB, f"Best: {arch_to_str(arch)}  val={value:.4f}", _SEP]))

    # ------------------------------------------------------------ main loop
    def run(self) -> dict:
        cfg = self.config
        # Seeded RNG makes the sequence of sampled architectures reproducible.
        rng = np.random.default_rng(cfg.seed)
        # Cap the sample count at the search-space size so ``bench.sample`` cannot
        # loop forever if a caller asks for more architectures than exist.
        budget = min(cfg.query_budget, len(self.bench))
        logger.info(
            "seed=%d (pass --seed to reproduce) | sampling %d random architectures",
            cfg.seed,
            budget,
        )

        archs = self.bench.sample(rng, budget)

        best_arch = None
        best_valid = float("-inf")
        history: list[dict] = []
        self._log_header(budget)

        for i, a in enumerate(archs, start=1):
            v = self.bench.valid_acc(a)
            if v > best_valid:
                best_valid = v
                best_arch = a
            history.append({"queries": i, "best_valid": best_valid})
            self._log_eval(i, a, v, best_valid)
            if i % 10 == 0 or i == budget:
                logger.info("evaluated %3d/%3d | best %.4f", i, budget, best_valid)

        self._log_best(best_arch, best_valid)

        return {
            "best_arch": best_arch,
            "best_valid": best_valid,
            "best_test": self.bench.test_acc(best_arch),
            "queries": budget,
            "history": history,
        }


def _auto_log_path(cfg: Config) -> str:
    """Timestamped, hyperparameter-tagged search-log filename for a baseline run."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"random_search_{cfg.dataset}_budget{cfg.query_budget}_seed{cfg.seed}_{ts}.log"


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser for the random-search baseline CLI."""
    p = argparse.ArgumentParser(
        prog="random-search",
        description="Random-search baseline on NAS-Bench-201 (equal query budget).",
    )
    p.add_argument("--dataset", choices=["cifar10", "cifar100", "ImageNet16-120"], default=None)
    p.add_argument("--query-budget", type=int, default=None)
    p.add_argument("--data-path", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--run",
        type=int,
        default=1,
        dest="num_runs",
        help="Number of repeated runs with different seeds (reports mean performance)",
    )
    p.add_argument("--log-level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument(
        "--search-log-path",
        default=None,
        help="Append per-evaluation search log to this file (empty string disables)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # Start from env-var configuration, then apply explicit CLI overrides on top.
    cfg = Config.from_env()
    overrides = {
        "dataset": args.dataset,
        "query_budget": args.query_budget,
        "data_path": args.data_path,
        "seed": args.seed,
        "log_level": args.log_level,
        "search_log_path": args.search_log_path,
    }
    for name, value in overrides.items():
        if value is not None:
            setattr(cfg, name, value)

    logging.basicConfig(
        level=getattr(logging, (cfg.log_level or "INFO").upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Auto-generate a baseline-specific log filename unless the caller fixed one
    # (via CLI or env), so separate runs don't share a single log file.
    auto_log = args.search_log_path is None and "LLMENAS_SEARCH_LOG_PATH" not in os.environ

    # Ground truth is seed-independent, so the benchmark is shared across runs;
    # only the sampling RNG (inside RandomSearch.run) is re-created per seed.
    bench = NASBench201(cfg)

    num_runs = max(1, args.num_runs)
    # ``--seed`` fixes the base and each subsequent run adds 1, mirroring the
    # evolutionary CLI, so a whole batch is reproducible from one ``--seed`` value.
    base_seed = cfg.seed
    seeds = [base_seed + i for i in range(num_runs)]

    rows: list[dict] = []
    for i, seed in enumerate(seeds, start=1):
        cfg.seed = seed
        if auto_log:
            cfg.search_log_path = _auto_log_path(cfg)
        result = RandomSearch(cfg, bench).run()
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

    # Pretty-print the final result summary (same shape as the evolutionary CLI).
    print("\n" + "=" * 60)
    print("Random search (baseline) — NAS-Bench-201 /", cfg.dataset)
    print("=" * 60)
    print(f"query budget    : {cfg.query_budget}")
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
