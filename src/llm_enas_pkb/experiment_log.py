"""Record effective configuration and a timezone-aware start for every run."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
import statistics
from uuid import uuid4

from .config import Config
from .search_space import arch_to_str

logger = logging.getLogger(__name__)
_TIMEZONE = timezone(timedelta(hours=8), name="Asia/Shanghai")


def experiment_time() -> str:
    """Use explicit Beijing time even when experiments run on a UTC server."""
    return datetime.now(_TIMEZONE).isoformat(timespec="milliseconds")


def log_experiment_start(config: Config, *, algorithm: str,
                         include_prompt_log: bool = True) -> None:
    """Append the same run header to enabled logs before sampling/querying.

    Serialize declared configuration only. api_key_env is the variable's name;
    its secret value and other process environment variables are never read.
    Each invocation snapshots the current per-run seed and resolved log paths.
    """
    started_at = experiment_time()
    block = "\n".join([
        "=" * 78,
        f"[{started_at}] Experiment start",
        f"experiment_id={uuid4().hex}",
        f"algorithm={algorithm}",
        f"started_at={started_at}",
        "timezone=Asia/Shanghai (UTC+08:00)",
        "Hyperparameters (effective Config, JSON):",
        json.dumps(asdict(config), ensure_ascii=False, indent=2),
        "End hyperparameters",
        "=" * 78,
    ])
    logger.info("\n%s", block)
    paths = [config.search_log_path]
    if include_prompt_log:
        paths.append(config.prompt_log_path)
    written = set()
    for path in paths:
        if not path:
            continue
        try:
            destination = Path(path).resolve()
            if destination in written:
                continue
            written.add(destination)
            with destination.open("a", encoding="utf-8") as stream:
                stream.write(block + "\n")
                stream.flush()
        except OSError as exc:
            logger.warning("Failed to write experiment header to %s: %s", path, exc)


def log_experiment_summary(config: Config, rows: list[dict], *, algorithm: str) -> Path | None:
    """Persist completed batch results, with percentage accuracy and sample SD.

    Called once after all runs. Explicit destinations append; automatically named
    files are unique to the batch. Write failures propagate so missing results
    cannot be mistaken for successfully saved results.
    """
    if config.summary_log_path == "":
        return None
    if config.summary_log_path is None:
        directory = Path(config.search_log_path).parent if config.search_log_path else Path(".")
        stamp = datetime.now(_TIMEZONE).strftime("%Y%m%d-%H%M%S-%f")
        destination = directory / (
            f"summary_{algorithm}_{config.dataset}_seed{rows[0]['seed']}"
            f"_runs{len(rows)}_{stamp}_{uuid4().hex[:8]}.log"
        )
    else:
        destination = Path(config.summary_log_path)

    # A summary must not overwrite or append to an existing search/prompt log.
    source_paths = [config.search_log_path, config.prompt_log_path]
    source_paths.extend(row.get("search_log_path") for row in rows)
    if any(path and Path(path).resolve() == destination.resolve() for path in source_paths):
        raise ValueError("summary_log_path must differ from search/prompt log paths")

    lines = [
        "=" * 78,
        f"[{experiment_time()}] Experiment summary (completed)",
        f"algorithm={algorithm} dataset={config.dataset} runs={len(rows)}",
        f"query_budget={config.query_budget}",
    ]
    if algorithm == "llm-enas-pkb":
        lines.extend([
            f"model={config.model} offline={config.offline}",
            f"pkb_error_feedback={config.pkb_error_feedback} "
            f"architecture_features={config.architecture_features}",
        ])
    for row in rows:
        lines.extend([
            f"run {row['run']:>2} | seed {row['seed']} | "
            f"best valid {row['best_valid'] * 100:.6f}% | "
            f"best test {row['best_test'] * 100:.6f}% | queries {row['queries']}",
            f"  best arch: {arch_to_str(row['best_arch'])}",
            f"  search log: {row.get('search_log_path') or '(disabled)'}",
        ])
    if len(rows) > 1:
        valid = [row["best_valid"] * 100 for row in rows]
        test = [row["best_test"] * 100 for row in rows]
        queries = [row["queries"] for row in rows]
        lines.extend([
            "average (mean)  : "
            f"best valid {statistics.mean(valid):.6f}% ± {statistics.stdev(valid):.6f}% | "
            f"best test {statistics.mean(test):.6f}% ± {statistics.stdev(test):.6f}% | "
            f"queries {statistics.mean(queries):.6f}",
            "± denotes sample standard deviation across runs (ddof=1).",
        ])
    lines.append("=" * 78)
    destination.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if config.summary_log_path is None else "a"
    with destination.open(mode, encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    return destination.resolve()
