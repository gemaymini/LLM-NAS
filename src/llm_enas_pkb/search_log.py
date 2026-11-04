"""Per-generation search log (Sec. 3.1, Algorithm 1).

Writes one human-readable section per generation recording:

1. the incoming population ``P_t`` (scores/tags snapshotted before this generation),
2. the PKB exemplars selected from the archive for this generation's prompt,
3. the offspring produced by crossover + mutation (arch + LLM-predicted fitness),
4. pre-selection queries, their reasons, predictions, truths, and signed errors,
5. measured elites and the next population ``P_{t+1}``, plus survival counts.

Each architecture is printed as six categorical operation IDs in canonical edge
order, followed by its official NAS-Bench-201 string encoding (see
``search_space.arch_to_str``), so the prompt encoding and benchmark lookup can
both be checked.  Fitness values are tagged ``[GT]`` when the individual was
ground-truth evaluated (i.e. lives in the archive) and ``[pred]`` when its value
is an LLM prediction.

The file is opened/appended/flushed on every write so the log stays visible in
real time (``tail -f``) and survives a kill; write failures are non-fatal
(warn + continue), mirroring the prompt log in ``llm.py``.
"""

from __future__ import annotations

import logging

from .architecture_cost import feature_description, format_architecture_features
from .config import Config
from .experiment_log import experiment_time, log_experiment_start
from .search_space import arch_to_str

logger = logging.getLogger(__name__)

_SEP = "=" * 78
_SUB = "-" * 78


class SearchLogger:
    def __init__(self, config: Config):
        self.config = config
        self.path = config.search_log_path

    # ------------------------------------------------------------- helpers
    def _write(self, block: str) -> None:
        """Append one block to the search log, flushing so it is visible in real time."""
        if not self.path:
            return
        try:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(block + "\n")
                fh.flush()
        except OSError as exc:
            logger.warning("Failed to write search log %s: %s", self.path, exc)

    def _arch(self, arch, value, tag: str) -> str:
        """Render the direct six-ID code, official string, and tagged fitness."""
        row = f"code={list(arch)}  {arch_to_str(arch)}"
        features = format_architecture_features(
            arch, dataset=self.config.dataset, mode=self.config.architecture_features,
        )
        return row + ("  " + features if features else "") + f"  val={float(value):.4f}  [{tag}]"

    # ------------------------------------------------------------ sections
    def log_start(self) -> None:
        """Persist settings and start time before initialization can fail."""
        log_experiment_start(self.config, algorithm="LLM-ENAS-PKB")

    def log_header(self, population, fitness) -> None:
        """Record the run's config and the ground-truth-seeded initial population."""
        if not self.path:
            return
        cfg = self.config
        lines = [
            _SEP,
            f"[{experiment_time()}] Initial population ready",
            (
                f"dataset={cfg.dataset} model={cfg.model} seed={cfg.seed} "
                f"iterations={cfg.max_iterations} pop={cfg.population_size} "
                f"elite_ratio={cfg.elite_ratio} "
                f"offspring={cfg.offspring_size} query_budget={cfg.query_budget}"
            ),
            (
                "selection_policy=measured_elites query_timing=before_selection "
                f"query_explore_ratio={cfg.query_explore_ratio} "
                f"refresh_parent_predictions={cfg.refresh_parent_predictions}"
            ),
            f"pkb_error_feedback={cfg.pkb_error_feedback}",
            f"architecture_features={cfg.architecture_features}",
            (
                "Population scores/tags are from generation start; next-population scores "
                "are after queries. mean-pop-valid averages measured next-population members."
            ),
            (
                "query error = prediction - truth (positive means overestimated); "
                "prediction_gen identifies the latest prediction used for fitness."
            ),
            (
                "encoding=6 categorical operation IDs; "
                "edge order=(0->1, 0->2, 1->2, 0->3, 1->3, 2->3)"
            ),
            _SUB,
            f"Initial population (seeded, {len(population)} individuals, all ground-truth queried):",
        ]
        if cfg.architecture_features != "none":
            lines.insert(-2, feature_description(cfg.architecture_features, cfg.dataset))
        for i, a in enumerate(population, 1):
            lines.append(f"  {i:>3}. {self._arch(a, fitness[a], 'GT')}")
        lines.append(_SUB)
        self._write("\n".join(lines))

    def log_generation(
        self,
        *,
        generation: int,
        population,
        offspring,
        preds,
        next_pop,
        query_records,
        fitness,
        archive,
        population_fitness,
        archive_before,
        parent_prediction_generations,
        prediction_generations,
        predictions,
        elites,
        survival,
        pkb_archs,
        queries: int,
        best_valid: float,
        mean_pop_valid: float,
        pkb_feedback=None,
    ) -> None:
        """Record one generation's full search state (see module docstring)."""
        if not self.path:
            return
        cfg = self.config
        lines = [
            _SEP,
            (
                f"[{experiment_time()}] "
                f"Generation {generation} | queries {queries}/{cfg.query_budget} | "
                f"best-valid {best_valid:.4f} | mean-pop-valid {mean_pop_valid:.4f}"
            ),
            _SEP,
        ]

        lines.append(f"Population P_{generation} ({len(population)} individuals):")
        for i, a in enumerate(population, 1):
            tag = "GT" if a in archive_before else "pred"
            age = (f"  prediction_gen={parent_prediction_generations[a]}"
                   if tag == "pred" else "")
            lines.append(f"  {i:>3}. {self._arch(a, population_fitness[a], tag)}{age}")

        lines.append(_SUB)
        lines.append(
            f"PKB exemplars ({len(pkb_archs)} selected from {len(archive_before)} archived):"
        )
        for i, a in enumerate(pkb_archs, 1):
            entry = pkb_feedback[i - 1] if pkb_feedback is not None else None
            detail = " " + entry.describe(archive_before[a]) if entry is not None else ""
            lines.append(f"  {i:>3}. {self._arch(a, archive_before[a], 'GT')}{detail}")
        if pkb_feedback is not None:
            lines.append(f"PKB feedback coverage: {sum(e is not None for e in pkb_feedback)}"
                         f"/{len(pkb_archs)}")

        lines.append(_SUB)
        lines.append(
            f"Offspring ({len(offspring)} after de-duplication, LLM-predicted fitness):"
        )
        for i, (a, p) in enumerate(zip(offspring, preds), 1):
            lines.append(f"  {i:>3}. {self._arch(a, p, 'pred')}  prediction_gen={generation}")

        offspring_set = set(offspring)
        refreshed_parents = [a for a in predictions if a not in offspring_set]
        if refreshed_parents:
            lines.append(_SUB)
            lines.append(f"Refreshed unmeasured parents ({len(refreshed_parents)}):")
            for i, a in enumerate(refreshed_parents, 1):
                lines.append(
                    f"  {i:>3}. {self._arch(a, predictions[a], 'pred')} "
                    f" prediction_gen={generation}"
                )

        lines.append(_SUB)
        lines.append(f"Ground-truth queried this generation ({len(query_records)}):")
        for i, record in enumerate(query_records, 1):
            a = record["arch"]
            lines.append(
                f"  {i:>3}. {self._arch(a, record['true_valid'], 'GT')} "
                f" reason={record['reason']} prediction={record['prediction']:.4f} "
                f" prediction_gen={record['prediction_generation']} "
                f" error={record['error']:+.4f} survived={record['survived']}"
            )

        lines.append(_SUB)
        quota = int(cfg.population_size * cfg.elite_ratio)
        lines.append(f"Ground-truth elites ({len(elites)}/{quota} protected slots):")
        for i, a in enumerate(elites, 1):
            lines.append(f"  {i:>3}. {self._arch(a, archive[a], 'GT')}")

        lines.append(_SUB)
        lines.append(
            f"Next population P_{generation + 1} "
            f"(ground-truth elitism + binary tournament, {len(next_pop)}):"
        )
        for i, a in enumerate(next_pop, 1):
            tag = "GT" if a in archive else "pred"
            age = f"  prediction_gen={prediction_generations[a]}" if tag == "pred" else ""
            lines.append(f"  {i:>3}. {self._arch(a, fitness[a], tag)}{age}")

        lines.append(_SUB)
        lines.append("Survival (unique architectures): " + " ".join(
            f"{name}={value}" for name, value in survival.items()
        ))
        lines.append(
            "parents_retained + offspring_selected - overlap_selected = next population size; "
            "new_offspring_selected excludes architectures already in the parent population."
        )

        lines.append(_SUB)
        self._write("\n".join(lines))
