"""Per-generation search log (Sec. 3.1, Algorithm 1).

Writes one human-readable section per generation recording:

1. the current population ``P_t`` (arch + fitness, tagged ground-truth vs predicted),
2. the PKB exemplars selected from the archive for this generation's prompt,
3. the offspring produced by crossover + mutation (arch + LLM-predicted fitness),
4. the tournament-selected next population ``P_{t+1}``,
5. the individuals that were ground-truth evaluated ("queried") this generation.

Each architecture is printed in its official NAS-Bench-201 string encoding (see
``search_space.arch_to_str``), so entries can be pasted directly into the
benchmark API.  Fitness values are tagged ``[GT]`` when the individual was
ground-truth evaluated (i.e. lives in the archive) and ``[pred]`` when its value
is an LLM prediction.

The file is opened/appended/flushed on every write so the log stays visible in
real time (``tail -f``) and survives a kill; write failures are non-fatal
(warn + continue), mirroring the prompt log in ``llm.py``.
"""

from __future__ import annotations

import logging
from datetime import datetime

from .config import Config
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

    @staticmethod
    def _arch(arch, value, tag: str) -> str:
        """Render one architecture as ``<arch-str>  val=<value>  [<tag>]``."""
        return f"{arch_to_str(arch)}  val={float(value):.4f}  [{tag}]"

    # ------------------------------------------------------------ sections
    def log_header(self, population, fitness) -> None:
        """Record the run's config and the ground-truth-seeded initial population."""
        cfg = self.config
        lines = [
            _SEP,
            f"[{datetime.now().isoformat(timespec='seconds')}] Search start",
            (
                f"dataset={cfg.dataset} model={cfg.model} seed={cfg.seed} "
                f"iterations={cfg.max_iterations} pop={cfg.population_size} "
                f"offspring={cfg.offspring_size} query_budget={cfg.query_budget}"
            ),
            _SUB,
            f"Initial population (seeded, {len(population)} individuals, all ground-truth queried):",
        ]
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
        queried,
        fitness,
        archive,
        pkb_archs,
        queries: int,
        best_valid: float,
        mean_pop_valid: float,
    ) -> None:
        """Record one generation's full search state (see module docstring)."""
        cfg = self.config
        lines = [
            _SEP,
            (
                f"[{datetime.now().isoformat(timespec='seconds')}] "
                f"Generation {generation} | queries {queries}/{cfg.query_budget} | "
                f"best-valid {best_valid:.4f} | mean-pop-valid {mean_pop_valid:.4f}"
            ),
            _SEP,
        ]

        lines.append(f"Population P_{generation} ({len(population)} individuals):")
        for i, a in enumerate(population, 1):
            tag = "GT" if a in archive else "pred"
            lines.append(f"  {i:>3}. {self._arch(a, fitness[a], tag)}")

        lines.append(_SUB)
        lines.append(f"PKB exemplars ({len(pkb_archs)} selected from {len(archive)} archived):")
        for i, a in enumerate(pkb_archs, 1):
            lines.append(f"  {i:>3}. {self._arch(a, archive[a], 'GT')}")

        lines.append(_SUB)
        lines.append(
            f"Offspring ({len(offspring)} after de-duplication, LLM-predicted fitness):"
        )
        for i, (a, p) in enumerate(zip(offspring, preds), 1):
            lines.append(f"  {i:>3}. {self._arch(a, p, 'pred')}")

        lines.append(_SUB)
        lines.append(
            f"Next population P_{generation + 1} (tournament selection, {len(next_pop)}):"
        )
        for i, a in enumerate(next_pop, 1):
            tag = "GT" if a in archive else "pred"
            lines.append(f"  {i:>3}. {self._arch(a, fitness[a], tag)}")

        lines.append(_SUB)
        lines.append(f"Ground-truth queried this generation ({len(queried)}):")
        for i, a in enumerate(queried, 1):
            lines.append(f"  {i:>3}. {self._arch(a, archive[a], 'GT')}")

        lines.append(_SUB)
        self._write("\n".join(lines))
