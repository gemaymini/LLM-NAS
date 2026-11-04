"""Grid-search temperature × top_p on cifar10 with repeated same-seed runs.

Goal
----
Measure how much the LLM's *sampling* (temperature / top_p) makes otherwise
identical searches differ.  Every repetition uses the same seed, so the
evolutionary RNG stream (initial population, crossover/mutation, exploration
draws) is byte-for-byte identical and the only source of run-to-run variance is
the LLM's response — which is exactly what temperature / top_p control.

For each (temperature, top_p) cell the search runs ``--repeats`` times on
``--seed`` and the script aggregates best valid/test accuracy: mean, sample
standard deviation (ddof=1), min-max range, and how often the global optimum
(or the same best architecture) is found.  Cells run sequentially; each
repetition costs roughly ``--iterations`` LLM calls (one prediction batch per
generation, plus retries).

Resume
------
By default the script *resumes*: it scans ``--log-dir`` for ``search_*.log``
files whose filename encodes (dataset, temperature, top_p, seed, rep), treats a
rep as finished only if its log reached the full generation/budget count (a
torn/interrupted log is re-run), reconstructs the finished reps' results from
their logs, and only runs the missing or incomplete reps.  Pass ``--fresh`` to
ignore existing logs and re-run the whole grid.

Usage (from the repository root)
--------------------------------
    .venv/bin/python tool/grid_temperature_top_p.py                  # defaults, auto-resumes
    .venv/bin/python tool/grid_temperature_top_p.py --fresh          # ignore logs, run everything
    .venv/bin/python tool/grid_temperature_top_p.py --dry-run        # plan only (shows resume plan)
    .venv/bin/python tool/grid_temperature_top_p.py --offline        # deterministic smoke test

Outputs (under ``--log-dir``):
    search_*.log            one per repetition (full per-generation state)
    grid_summary_*.log      human-readable aggregation
    grid_results_*.json     machine-readable aggregation
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_enas_pkb.config import Config
from llm_enas_pkb.evolution import EvolutionSearch
from llm_enas_pkb.experiment_log import experiment_time
from llm_enas_pkb.nasbench201 import NASBench201
from llm_enas_pkb.predictor import RegressionPredictor
from llm_enas_pkb.search_space import arch_to_str

logger = logging.getLogger(__name__)

# search_{dataset}_t{temp}_p{top_p}_seed{seed}_rep{rep}_{YYYYMMDD-HHMMSS}.log
_FILENAME_RE = re.compile(
    r"search_(?P<dataset>[A-Za-z0-9-]+)_t(?P<t>[0-9.]+)_p(?P<p>[0-9.]+)"
    r"_seed(?P<seed>\d+)_rep(?P<rep>\d+)_(?P<ts>\d{8}-\d{6})\.log$"
)
_HEADER_RE = re.compile(
    r"Hyperparameters \(effective Config, JSON\):\n(.*?)\nEnd hyperparameters", re.S)
_GEN_RE = re.compile(r"Generation (\d+) \| queries (\d+)/")
_ARCH_RE = re.compile(r"code=\[([0-9, ]+)\].*\[GT\]")


def _parse_float_list(values: list[str], name: str, lo: float, hi: float) -> list[float]:
    """Parse repeatable float CLI arguments and validate each against [lo, hi]."""
    out: list[float] = []
    for raw in values:
        try:
            v = float(raw)
        except ValueError:
            raise SystemExit(f"{name} must be numeric, got {raw!r}")
        if not lo <= v <= hi:
            raise SystemExit(f"{name} must be in [{lo}, {hi}], got {v}")
        out.append(v)
    if not out:
        raise SystemExit(f"at least one --{name} value is required")
    return list(dict.fromkeys(out))  # de-duplicate, preserve order


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="grid_temperature_top_p.py",
        description="Grid search temperature × top_p; repeated same-seed runs on cifar10.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", default="cifar10",
                   choices=["cifar10", "cifar100", "ImageNet16-120"])
    p.add_argument("--seed", type=int, default=42,
                   help="Fixed base seed used for every repetition (default: 42)")
    p.add_argument("--repeats", type=int, default=5,
                   help="Number of same-seed repetitions per (temperature, top_p) cell (default: 5)")
    p.add_argument("--temperature", action="append", dest="temperatures",
                   help="temperature value; repeat the flag to add more "
                        "(default: 0.1 0.3 0.6 0.8 1.0)")
    p.add_argument("--top-p", action="append", dest="top_ps",
                   help="top_p value; repeat the flag to add more "
                        "(default: 0.1 0.3 0.5 0.7 0.9)")
    p.add_argument("--iterations", type=int, default=None,
                   help="Override max_iterations (default: keep Config default, 30)")
    p.add_argument("--query-budget", type=int, default=None,
                   help="Override query_budget (default: keep Config default, 100)")
    p.add_argument("--log-dir", default="logs/temperature_top_p_grid",
                   help="Directory for search logs and aggregations (created if missing)")
    p.add_argument("--prompt-log", action="store_true",
                   help="Also write per-repetition LLM prompt/response logs (large)")
    p.add_argument("--offline", action="store_true",
                   help="Deterministic pseudo-LLM (no API); smoke-tests the harness only")
    p.add_argument("--fresh", action="store_true",
                   help="Ignore existing logs and re-run the whole grid (no resume)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the grid (and resume) plan and exit without running anything")
    return p


def _fmt(v: float) -> str:
    """Render a float compactly for filenames (0.0 -> '0', 0.3 -> '0.3')."""
    return f"{v:g}"


def _parse_header_config(text: str) -> dict | None:
    """Recover the recorded Config JSON block from a search-log header."""
    m = _HEADER_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def _is_complete(text: str) -> bool:
    """A rep is finished only if its log ran to the full generation + budget count."""
    cfg = _parse_header_config(text)
    if cfg is None:
        return False
    iterations = cfg.get("max_iterations") or cfg.get("iterations")
    budget = cfg.get("query_budget")
    gens = _GEN_RE.findall(text)
    if not gens or not iterations or not budget:
        return False
    last_gen, last_queries = int(gens[-1][0]), int(gens[-1][1])
    return last_gen >= int(iterations) and last_queries >= int(budget)


def _reconstruct_result(text: str, bench: NASBench201) -> dict:
    """Rebuild a completed rep's outcome from its search log (best over [GT] rows)."""
    archs = set()
    for line in text.splitlines():
        m = _ARCH_RE.search(line)
        if m:
            archs.add(tuple(int(x) for x in m.group(1).replace(" ", "").split(",")))
    if not archs:
        raise ValueError("no [GT] architecture rows found in search log")
    best = max(archs, key=bench.valid_acc)
    queries = int(_GEN_RE.findall(text)[-1][1])
    return {
        "best_arch": list(best),
        "best_arch_str": arch_to_str(best),
        "best_valid": bench.valid_acc(best),
        "best_test": bench.test_acc(best),
        "queries": queries,
    }


def _scan_log_dir(out_dir: Path, dataset: str, seed: int) -> dict[tuple[float, float], dict[int, Path]]:
    """Map (temperature, top_p) -> {rep: newest matching search-log path}."""
    found: dict[tuple[float, float], dict[int, Path]] = {}
    if not out_dir.is_dir():
        return found
    for path in out_dir.glob("search_*.log"):
        m = _FILENAME_RE.match(path.name)
        if not m:
            continue
        g = m.groupdict()
        if g["dataset"] != dataset or int(g["seed"]) != seed:
            continue
        key = (float(g["t"]), float(g["p"]))
        rep = int(g["rep"])
        cell = found.setdefault(key, {})
        # A rep may have several logs (e.g. an interrupted one then a re-run);
        # keep the one with the newest timestamp.
        prev = cell.get(rep)
        if prev is None or g["ts"] > _FILENAME_RE.match(prev.name).group("ts"):
            cell[rep] = path
    return found


def _run_one(cell_cfg: Config, bench: NASBench201, predictor: RegressionPredictor,
             rep: int) -> dict:
    """Run one same-seed search; a fresh EvolutionSearch re-seeds its RNG each call."""
    result = EvolutionSearch(cell_cfg, bench, predictor).run()
    return {
        "rep": rep,
        "seed": cell_cfg.seed,
        "best_valid": result["best_valid"],
        "best_test": result["best_test"],
        "queries": result["queries"],
        "best_arch": list(result["best_arch"]),
        "best_arch_str": arch_to_str(result["best_arch"]),
    }


def _aggregate(runs: list[dict]) -> dict:
    """Summarise a cell's successful repetitions with percent accuracies."""
    ok = [r for r in runs if "error" not in r]
    valid = [r["best_valid"] * 100 for r in ok]
    test = [r["best_test"] * 100 for r in ok]
    n = len(ok)

    def _stats(values):
        if not values:
            return {"mean": None, "sd": None, "min": None, "max": None}
        return {
            "mean": statistics.fmean(values),
            "sd": statistics.stdev(values) if n >= 2 else 0.0,
            "min": min(values),
            "max": max(values),
        }

    return {"n_reps": n, "n_errors": len(runs) - n,
            "valid_pct": _stats(valid), "test_pct": _stats(test)}


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    temperatures = _parse_float_list(
        args.temperatures or ["0.1", "0.3", "0.6", "0.8"], "temperature", 0.0, 2.0)
    top_ps = _parse_float_list(args.top_ps or ["0.1", "0.3", "0.5", "0.7", "0.9"],
                               "top_p", 0.0, 1.0)
    grid = [(t, p) for t in temperatures for p in top_ps]
    total_runs = len(grid) * args.repeats

    out_dir = Path(args.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- resume plan: which reps are already done / missing / torn ----
    found: dict[tuple[float, float], dict[int, Path]] = ({}
        if args.fresh else _scan_log_dir(out_dir, args.dataset, args.seed))
    plan = []  # (cell_idx, temperature, top_p, [reps to run], {rep: resumed log path})
    for cell_idx, (t, p) in enumerate(grid, start=1):
        cell_found = {k: v for k, v in found.get((t, p), {}).items()
                      if _is_complete(v.read_text())}
        torn = sorted(set(found.get((t, p), {})) - set(cell_found))
        to_run = [rep for rep in range(1, args.repeats + 1) if rep not in cell_found]
        plan.append((cell_idx, t, p, to_run, cell_found, torn))
    resumed = sum(len(cf) for *_, cf, _ in plan)
    remaining = sum(len(tr) for _, _, _, tr, _, _ in plan)
    torn_count = sum(len(tn) for *_, _, _, tn in plan)

    print("=" * 70)
    print("Grid search: temperature x top_p (same-seed repetitions)")
    print("=" * 70)
    print(f"dataset      : {args.dataset}")
    print(f"seed         : {args.seed}  (identical every repetition)")
    print(f"repeats/cell : {args.repeats}")
    print(f"temperatures : {temperatures}")
    print(f"top_p        : {top_ps}")
    print(f"cells x reps : {len(grid)} x {args.repeats} = {total_runs} searches")
    if args.offline:
        print("mode         : OFFLINE (deterministic pseudo-LLM; zero variance expected)")
    print("-" * 70)
    print(f"resume       : {resumed} rep(s) already finished (reused from logs),")
    print(f"               {remaining} rep(s) to run; {torn_count} "
          f"torn log(s) will be re-run.")
    if not args.fresh and resumed:
        print("               (pass --fresh to re-run everything from scratch)")
    print("=" * 70)

    if args.dry_run:
        print("dry-run: nothing executed.", flush=True)
        return 0

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Base config from environment (.env / LLMENAS_*), then pin the dataset.
    base = Config.from_env()
    base.dataset = args.dataset
    base.offline = args.offline
    if args.iterations is not None:
        base.max_iterations = args.iterations
    if args.query_budget is not None:
        base.query_budget = args.query_budget

    run_ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Ground truth is dataset-only, so one benchmark serves every cell and rebuild
    # of resumed results.
    bench = NASBench201(base)
    vmax = max(v[bench._dataset_index] for v in bench._valid.values())
    tmax = max(v[bench._dataset_index] for v in bench._test.values())

    cells: list[dict] = []
    for cell_idx, temperature, top_p, to_run, cell_found, _torn in plan:
        print(f"\n[{cell_idx}/{len(grid)}] temperature={temperature} top_p={top_p} "
              f"({len(cell_found)} done, {len(to_run)} to run)", flush=True)
        cell_cfg = replace(
            base,
            temperature=temperature,
            top_p=top_p,
            seed=args.seed,
            search_log_path=None,
            prompt_log_path=None,
            summary_log_path="",
        )
        runs: list[dict] = []
        # Reconstruct already-finished reps from disk so the aggregate stays complete.
        for rep in sorted(cell_found):
            try:
                rec = _reconstruct_result(cell_found[rep].read_text(), bench)
            except (ValueError, KeyError, IndexError) as exc:
                logger.warning("could not reconstruct rep %d for (%s,%s): %s; will re-run",
                               rep, temperature, top_p, exc)
                to_run.append(rep)
                continue
            runs.append({"rep": rep, "seed": args.seed, "resumed": True,
                         "log": str(cell_found[rep]), **rec})

        if to_run:
            # A fresh predictor binds the cell's temperature/top_p into the API client.
            predictor = RegressionPredictor(cell_cfg)
            for rep in sorted(to_run):
                stamp = f"t{_fmt(temperature)}_p{_fmt(top_p)}_seed{args.seed}_rep{rep}_{run_ts}"
                cell_cfg.search_log_path = str(
                    out_dir / f"search_{args.dataset}_{stamp}.log")
                if args.prompt_log:
                    cell_cfg.prompt_log_path = str(
                        out_dir / f"llm_prompts_{args.dataset}_{stamp}.log")
                try:
                    run = _run_one(cell_cfg, bench, predictor, rep)
                    runs.append(run)
                    print(
                        f"    rep {rep}/{args.repeats} | best valid "
                        f"{run['best_valid'] * 100:.4f}% | best test "
                        f"{run['best_test'] * 100:.4f}% | queries {run['queries']}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001 - keep the grid alive across cells
                    logger.exception("rep %d failed for temperature=%s top_p=%s",
                                     rep, temperature, top_p)
                    runs.append({"rep": rep, "seed": args.seed,
                                 "error": f"{type(exc).__name__}: {exc}"})

        runs.sort(key=lambda r: r["rep"])
        agg = _aggregate(runs)
        best_archs = {r["best_arch_str"] for r in runs if "error" not in r}
        cells.append({
            "temperature": temperature,
            "top_p": top_p,
            "runs": runs,
            "aggregate": agg,
            "validation_global_hits": sum(
                1 for r in runs if "error" not in r and abs(r["best_valid"] - vmax) < 1e-9),
            "test_global_hits": sum(
                1 for r in runs if "error" not in r and abs(r["best_test"] - tmax) < 1e-9),
            "distinct_best_archs": len(best_archs),
        })

    # ------------------------------------------------------------------ report
    summary_json = out_dir / f"grid_results_{run_ts}.json"
    summary_log = out_dir / f"grid_summary_{run_ts}.log"
    payload = {
        "generated_at": experiment_time(),
        "dataset": args.dataset,
        "model": base.model,
        "offline": args.offline,
        "seed": args.seed,
        "repeats": args.repeats,
        "temperatures": temperatures,
        "top_ps": top_ps,
        "iterations": base.max_iterations,
        "query_budget": base.query_budget,
        "validation_global_optimum": vmax,
        "test_global_optimum": tmax,
        "resumed_reps": resumed,
        "cells": cells,
    }
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    # Human-readable table.
    header = (f"{'temp':>5} {'top_p':>5} {'n':>2} "
              f"{'best valid mean ± sd (min–max)':>34} "
              f"{'best test mean ± sd':>22}  {'hit_v':>5} {'hit_t':>5} {'archs':>5}")
    lines = [
        "=" * 110,
        f"[{experiment_time()}] Grid summary: temperature x top_p ({args.dataset})",
        f"seed={args.seed} repeats={args.repeats} iterations={base.max_iterations} "
        f"query_budget={base.query_budget} offline={args.offline} resumed={resumed}",
        f"validation global optimum = {vmax * 100:.4f}% | "
        f"test global optimum = {tmax * 100:.4f}%",
        "Accuracies are percentages; sd is the sample standard deviation (ddof=1)",
        "across the same-seed repetitions. Larger sd/range means the LLM sampling",
        "noise (i.e. that temperature/top_p) moves the search outcome more.",
        "=" * 110,
        header,
        "-" * 110,
    ]
    for c in cells:
        a = c["aggregate"]
        v, t = a["valid_pct"], a["test_pct"]
        lines.append(
            f"{c['temperature']:>5.2f} {c['top_p']:>5.2f} {a['n_reps']:>2} "
            f"{v['mean']:>8.4f} ± {v['sd']:>5.3f} "
            f"({v['min']:>6.3f}–{v['max']:<6.3f}) "
            f"{t['mean']:>8.4f} ± {t['sd']:>5.3f}  "
            f"{c['validation_global_hits']:>5} {c['test_global_hits']:>5} "
            f"{c['distinct_best_archs']:>5}"
        )
        if a["n_errors"]:
            lines.append(f"      ^ {a['n_errors']} repetition(s) failed; see JSON/logs.")
    lines.append("=" * 110)
    summary_log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "\n".join(lines))
    print(f"\nsummary json : {summary_json}")
    print(f"summary log  : {summary_log}")
    print(f"search logs  : {out_dir}/search_*.log")
    return 0


if __name__ == "__main__":
    sys.exit(main())