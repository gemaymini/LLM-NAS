"""Evolutionary search loop (Sec. 3.1, Algorithm 1).

Each generation:

1. generate candidate offspring via crossover + mutation (line 5),
2. select a compact, diverse PKB from the historical archive (line 7, Alg. 2),
3. encode PKB + offspring and build the regression prompt (lines 9-13),
4. predict offspring fitness with the LLM (lines 14-18),
5. tournament-select the next population from ``P_t`` + offspring (line 24).

Ground-truth benchmark lookups ("queries") are the expensive resource the
surrogate is meant to save.  They are spent only to (a) seed the archive with the
initial population and (b) re-label a small number of the highest-predicted
selected offspring each generation, up to ``query_budget`` (100 in the paper).
Individuals that are never ground-truth evaluated keep their LLM-predicted
fitness, so a persistent ``fitness`` cache is carried across generations.
"""

from __future__ import annotations

import logging

import numpy as np

from .config import Config
from .nasbench201 import NASBench201
from .pkb import select_pkb
from .predictor import RegressionPredictor
from .search_log import SearchLogger
from .search_space import NUM_EDGES, NUM_OPS

logger = logging.getLogger(__name__)


class EvolutionSearch:
    def __init__(self, config: Config, bench: NASBench201, predictor: RegressionPredictor):
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

    # ------------------------------------------------------------ main loop
    def run(self) -> dict:
        cfg = self.config
        # Seeded RNG makes the entire evolutionary trajectory reproducible.
        rng = np.random.default_rng(cfg.seed)

        # archive: arch -> true valid accuracy (ground-truth evaluated only).
        # fitness: arch -> current fitness (true value if archived, else LLM prediction).
        archive: dict[tuple[int, ...], float] = {}
        fitness: dict[tuple[int, ...], float] = {}

        # Seed the initial population and pay the ground-truth cost for each.
        n_init = cfg.n_init if cfg.n_init and cfg.n_init > 0 else cfg.population_size
        population = self.bench.sample(rng, n_init)
        for a in population:
            v = self.bench.valid_acc(a)
            archive[a] = v
            fitness[a] = v
        queries = len(population)
        # Spread the remaining query budget evenly across the remaining generations.
        n_query_per_gen = max(0, (cfg.query_budget - queries) // max(1, cfg.max_iterations))

        search_log = SearchLogger(cfg)
        search_log.log_header(population, fitness)

        history: list[dict] = []
        for gen in range(cfg.max_iterations):
            # 1. offspring
            offspring = self._make_offspring(population, rng)

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

            # 3-4. predict offspring fitness with the LLM surrogate
            cand_vectors = self.predictor.encoder.encode_batch(offspring)
            preds = self.predictor.predict_values(
                pkb_vectors[pkb_idx], pkb_values[pkb_idx], cand_vectors
            )
            for a, p in zip(offspring, preds):
                fitness.setdefault(a, float(p))  # keep archived true value if present

            # 5. tournament selection -> next population
            union = list(dict.fromkeys(list(population) + offspring))
            next_pop = self._tournament(union, fitness.__getitem__, cfg.population_size, rng)

            # 6. ground-truth re-label a few of the best selected offspring
            new_members = [a for a in next_pop if a not in archive]
            new_members.sort(key=fitness.__getitem__, reverse=True)
            queried_this_gen: list[tuple[int, ...]] = []
            for a in new_members[:n_query_per_gen]:
                if queries < cfg.query_budget:
                    v = self.bench.valid_acc(a)
                    archive[a] = v
                    fitness[a] = v
                    queries += 1
                    queried_this_gen.append(a)

            population = next_pop
            best_arch = max(archive, key=archive.get)
            known = [fitness[a] for a in population if a in archive]
            mean_pop_true = float(np.mean(known)) if known else float("nan")
            history.append(
                {
                    "generation": gen + 1,
                    "queries": queries,
                    "best_valid": archive[best_arch],
                    "mean_pop_valid": mean_pop_true,
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
                queried=queried_this_gen,
                fitness=fitness,
                archive=archive,
                pkb_archs=pkb_selected,
                queries=queries,
                best_valid=archive[best_arch],
                mean_pop_valid=mean_pop_true,
            )

        best_arch = max(archive, key=archive.get)
        return {
            "best_arch": best_arch,
            "best_valid": archive[best_arch],
            "best_test": self.bench.test_acc(best_arch),
            "queries": queries,
            "history": history,
        }
