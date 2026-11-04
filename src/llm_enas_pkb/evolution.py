"""Evolutionary search loop (Sec. 3.1, Algorithm 1).

Each generation:

1. generate candidate offspring via crossover + mutation (line 5),
2. select a compact, diverse PKB from the historical archive (line 7, Alg. 2),
3. encode PKB + offspring and build the regression prompt (lines 9-13),
4. predict offspring fitness with the LLM (lines 14-18),
5. query unmeasured merged candidates using predicted rank and random exploration,
6. retain measured elites using corrected ground truth, then fill the remaining
   next-population slots by binary tournament (extends line 24).

Ground-truth benchmark lookups ("queries") are the expensive resource the
surrogate is meant to save.  They are spent only to (a) seed the archive with the
initial population and (b) evaluate candidates before survivor selection, up to
``query_budget`` (100 in the paper). Unmeasured individuals use their latest
available prediction; measured values are never overwritten by predictions.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from .config import Config
from .nasbench201 import NASBench201
from .pkb import PredictionFeedback, select_pkb
from .predictor import RegressionPredictor
from .search_log import SearchLogger
from .search_space import NUM_EDGES, NUM_OPS

logger = logging.getLogger(__name__)


class EvolutionSearch:
    def __init__(self, config: Config, bench: NASBench201, predictor: RegressionPredictor):
        if not 0.0 <= config.elite_ratio <= 1.0:
            raise ValueError("elite_ratio must be between 0 and 1")
        if not 0.0 <= config.query_explore_ratio <= 1.0:
            raise ValueError("query_explore_ratio must be between 0 and 1")
        if config.population_size <= 0:
            raise ValueError("population_size must be positive")
        if config.max_iterations < 0 or config.offspring_size < 0:
            raise ValueError("max_iterations and offspring_size must be non-negative")
        n_init = config.n_init if config.n_init and config.n_init > 0 else config.population_size
        if config.query_budget < n_init:
            raise ValueError("query_budget must cover the initial population (n_init)")
        self.config = config
        self.bench = bench
        self.predictor = predictor

    # --------------------------------------------------------- genetic ops
    def _crossover(self, p1, p2, rng: np.random.Generator) -> list[int]:
        """Uniform crossover: with prob ``crossover_prob``, mix the genes of ``p1``/``p2``.

        Each of the six edge slots is taken from ``p2`` with 50% probability and
        from ``p1`` otherwise.  With prob ``1 - crossover_prob`` the first parent
        is copied unchanged (no recombination occurs).
        """
        if rng.random() < self.config.crossover_prob:
            mask = rng.random(NUM_EDGES) < 0.5
            return [int(p2[i] if mask[i] else p1[i]) for i in range(NUM_EDGES)]
        return list(p1)

    def _mutate(self, child: list[int], rng: np.random.Generator) -> tuple[int, ...]:
        """Per-gene mutation: each edge swaps to a *different* op with prob ``mutation_prob``.

        The replacement is drawn uniformly from the other four operations, so a
        mutation is always a real change (never "mutates" to the same op).
        """
        for i in range(NUM_EDGES):
            if rng.random() < self.config.mutation_prob:
                opts = [o for o in range(NUM_OPS) if o != child[i]]
                child[i] = int(rng.choice(opts))
        return tuple(child)

    def _make_offspring(self, parents, rng: np.random.Generator) -> list[tuple[int, ...]]:
        """Generate ``offspring_size`` children via select -> crossover -> mutate.

        Parents are two independent uniform draws (with replacement) from the
        current population.  Duplicate children are dropped with ``dict.fromkeys``
        so the candidate pool stays diverse.
        """
        offspring = []
        for _ in range(self.config.offspring_size):
            p1 = parents[int(rng.integers(len(parents)))]
            p2 = parents[int(rng.integers(len(parents)))]
            child = self._crossover(p1, p2, rng)
            offspring.append(self._mutate(child, rng))
        return list(dict.fromkeys(offspring))  # de-duplicate

    @staticmethod
    def _tournament(candidates, fitness, n: int, rng: np.random.Generator) -> list:
        """Binary tournament selection returning ``n`` distinct winners.

        Repeatedly samples two random candidates and keeps the fitter one until
        ``n`` distinct winners are found (a guard cap prevents an infinite loop
        when the pool is small).  A final safety net appends the globally-best
        remaining candidates so exactly ``n`` are always returned.
        """
        cand = list(candidates)
        if len(cand) <= n:
            return sorted(cand, key=fitness, reverse=True)[:n]
        chosen: list = []
        m = len(cand)
        guard = 0
        while len(chosen) < n and guard < 10 * n:
            guard += 1
            ia, ib = rng.integers(0, m, size=2)
            a, b = cand[int(ia)], cand[int(ib)]
            win = a if fitness(a) >= fitness(b) else b
            if win not in chosen:
                chosen.append(win)
        for a in sorted(cand, key=fitness, reverse=True):  # safety net
            if len(chosen) >= n:
                break
            if a not in chosen:
                chosen.append(a)
        return chosen[:n]

    def _select_elites(self, candidates, archive) -> list:
        """Only measured members of the current pool qualify for protected slots."""
        n_elite = int(self.config.population_size * self.config.elite_ratio)
        measured = [a for a in dict.fromkeys(candidates) if a in archive]
        return sorted(measured, key=archive.__getitem__, reverse=True)[:n_elite]

    def _select_next_population(self, candidates, fitness, rng: np.random.Generator,
                                *, archive) -> list:
        """Keep true elites, then tournament-select distinct remaining survivors.

        Unfilled elite slots become tournament slots. Tournament fitness is the
        ground truth when available, otherwise the latest surrogate prediction.
        A small candidate pool is kept whole rather than padded with duplicates.
        """
        cfg = self.config
        candidates = list(dict.fromkeys(candidates))
        target_size = min(cfg.population_size, len(candidates))
        elites = self._select_elites(candidates, archive)
        elite_set = set(elites)
        remaining = [a for a in candidates if a not in elite_set]
        return elites + self._tournament(remaining, fitness, target_size - len(elites), rng)

    def _select_queries(self, candidates, fitness, archive, n_queries: int,
                        queried_count: int, explored_count: int,
                        rng: np.random.Generator) -> list[tuple]:
        """Rank for exploitation, then sample the rest uniformly without replacement.

        Counts exclude initialization. The cumulative floor carries fractional
        exploration slots across generations and counts only available queries,
        so empty/short candidate pools do not consume either budget or quota.
        """
        unmeasured = [a for a in dict.fromkeys(candidates) if a not in archive]
        n = min(max(0, n_queries), len(unmeasured))
        if n == 0:
            return []
        target = math.floor((queried_count + n) * self.config.query_explore_ratio + 1e-12)
        n_explore = min(n, max(0, target - explored_count))
        ranked = sorted(unmeasured, key=fitness, reverse=True)
        n_exploit = n - n_explore
        selected = [(a, "exploit") for a in ranked[:n_exploit]]
        remaining = ranked[n_exploit:]
        if n_explore:
            indices = rng.choice(len(remaining), size=n_explore, replace=False)
            selected.extend((remaining[int(i)], "explore") for i in indices)
        return selected

    # ------------------------------------------------------------ main loop
    def run(self) -> dict:
        cfg = self.config
        search_log = SearchLogger(cfg)
        search_log.log_start()
        # Seeded RNG makes the entire evolutionary trajectory reproducible.
        rng = np.random.default_rng(cfg.seed)
        logger.info("seed=%d (pass --seed to reproduce this run)", cfg.seed)

        # archive: arch -> true valid accuracy (ground-truth evaluated only).
        # fitness: arch -> current fitness (true value if archived, else LLM prediction).
        archive: dict[tuple[int, ...], float] = {}
        fitness: dict[tuple[int, ...], float] = {}
        prediction_generations: dict[tuple[int, ...], int] = {}
        feedback_archive: dict[tuple[int, ...], PredictionFeedback] = {}

        # Seed the initial population and pay the ground-truth cost for each.
        n_init = cfg.n_init if cfg.n_init and cfg.n_init > 0 else cfg.population_size
        population = list(dict.fromkeys(self.bench.sample(rng, n_init)))
        for a in population:
            v = self.bench.valid_acc(a)
            archive[a] = v
            fitness[a] = v
        queries = len(population)
        queried_count = 0  # post-initialization queries only
        explored_count = 0
        # No fixed per-generation quota.  Floor division (``//``) would waste up
        # to ``max_iterations - 1`` queries (here 80 // 30 == 2 spends only 60 of
        # the 80 left over), so the quota is re-derived each generation as the
        # *ceiling* of the remaining budget over the remaining generations.
        # Ceiling aims to spend the budget and self-corrects: if a generation
        # yields fewer new members than its quota, the unspent queries roll into
        # the next generation automatically.

        search_log.log_header(population, fitness)

        history: list[dict] = []
        for gen in range(cfg.max_iterations):
            # Preserve the incoming population's scores/tags, not just membership.
            population_fitness = {a: fitness[a] for a in population}
            archive_before = dict(archive)
            parent_prediction_generations = dict(prediction_generations)
            # 1. offspring
            offspring = self._make_offspring(population, rng)
            candidates = list(dict.fromkeys(list(population) + offspring))

            # 2. PKB from archive (Algorithm 2)
            pkb_archs = list(archive.keys())
            pkb_vectors = self.predictor.encoder.encode_batch(pkb_archs)
            pkb_values = np.array([archive[a] for a in pkb_archs], dtype=np.float64)
            pkb_idx = select_pkb(
                pkb_vectors,
                pkb_values,
                alpha=cfg.alpha,
                k_perf=cfg.k_perf,
                k_div=cfg.k_div,
                selected_size=cfg.selected_size,
            )
            pkb_selected = [pkb_archs[i] for i in pkb_idx]
            pkb_feedback = ([feedback_archive.get(a) for a in pkb_selected]
                            if cfg.pkb_error_feedback else None)
            feedback_count = sum(entry is not None for entry in (pkb_feedback or []))

            # 3-4. predict offspring fitness with the LLM surrogate
            prediction_archs = list(offspring)
            if cfg.refresh_parent_predictions:
                prediction_archs = list(dict.fromkeys(
                    offspring + [a for a in population if a not in archive]
                ))
            cand_vectors = self.predictor.encoder.encode_batch(prediction_archs)
            prediction_values = self.predictor.predict_values(
                pkb_vectors[pkb_idx], pkb_values[pkb_idx], cand_vectors,
                generation=gen + 1,
                **({"pkb_feedback": pkb_feedback} if cfg.pkb_error_feedback else {}),
            )
            predictions = dict(zip(prediction_archs, map(float, prediction_values)))
            preds = [predictions[a] for a in offspring]
            for a, p in predictions.items():
                if a in archive:
                    fitness[a] = archive[a]
                else:
                    fitness[a] = p
                    prediction_generations[a] = gen + 1

            # 5. query from the full merged pool BEFORE any survivor selection.
            # Ceiling of the remaining budget over the remaining generations;
            # see the note above on why this is not a fixed floor-divided quota.
            generations_left = cfg.max_iterations - gen
            remaining = max(0, cfg.query_budget - queries)
            n_query_this_gen = (remaining + generations_left - 1) // generations_left
            query_plan = self._select_queries(
                candidates, fitness.__getitem__, archive, n_query_this_gen,
                queried_count, explored_count, rng,
            )
            query_records = []
            for a, reason in query_plan:
                prediction = fitness[a]
                v = self.bench.valid_acc(a)
                if cfg.pkb_error_feedback:
                    feedback_archive[a] = PredictionFeedback(
                        prediction, prediction_generations[a], gen + 1,
                    )
                query_records.append({
                    "arch": a, "reason": reason, "prediction": prediction,
                    "prediction_generation": prediction_generations[a],
                    "true_valid": v, "error": prediction - v,
                })
                archive[a] = v
                fitness[a] = v
                queries += 1
                queried_count += 1
                explored_count += int(reason == "explore")

            # 6. select once, using corrected truth for measured elites/survivors.
            elites = self._select_elites(candidates, archive)
            next_pop = self._select_next_population(
                candidates, fitness.__getitem__, rng, archive=archive,
            )
            parent_set, offspring_set, next_set = set(population), set(offspring), set(next_pop)
            for record in query_records:
                record["survived"] = record["arch"] in next_set
            survival = {
                "parents_retained": len(next_set & parent_set),
                "offspring_selected": len(next_set & offspring_set),
                "overlap_selected": len(next_set & parent_set & offspring_set),
                "new_offspring_selected": len((next_set & offspring_set) - parent_set),
            }

            best_arch = max(archive, key=archive.get)
            known = [fitness[a] for a in next_pop if a in archive]
            mean_pop_true = float(np.mean(known)) if known else float("nan")
            history.append(
                {
                    "generation": gen + 1,
                    "queries": queries,
                    "best_valid": archive[best_arch],
                    "mean_pop_valid": mean_pop_true,
                    "query_records": query_records,
                    "pkb_feedback_count": feedback_count,
                    "elites": elites,
                    **survival,
                }
            )
            logger.info(
                "gen %2d | queries %3d | best %.4f | mean-pop %.4f",
                gen + 1, queries, archive[best_arch], mean_pop_true,
            )
            search_log.log_generation(
                generation=gen + 1,
                population=population,
                offspring=offspring,
                preds=preds,
                next_pop=next_pop,
                query_records=query_records,
                fitness=dict(fitness),
                archive=dict(archive),
                population_fitness=population_fitness,
                archive_before=archive_before,
                parent_prediction_generations=parent_prediction_generations,
                prediction_generations=dict(prediction_generations),
                predictions=predictions,
                elites=elites,
                survival=survival,
                pkb_archs=pkb_selected,
                pkb_feedback=pkb_feedback,
                queries=queries,
                best_valid=archive[best_arch],
                mean_pop_valid=mean_pop_true,
            )
            # Keep P_t available for logging before advancing to P_{t+1}.
            population = next_pop

        best_arch = max(archive, key=archive.get)
        return {
            "best_arch": best_arch,
            "best_valid": archive[best_arch],
            "best_test": self.bench.test_acc(best_arch),
            "queries": queries,
            "history": history,
        }
