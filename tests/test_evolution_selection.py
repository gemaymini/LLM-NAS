"""Regression checks for elitism and the remaining binary-tournament slots."""

from __future__ import annotations

import io
import itertools
import unittest
from unittest.mock import Mock, patch

import numpy as np

from llm_enas_pkb.config import Config
from llm_enas_pkb.encoder import ArchitectureInformationEncoder
from llm_enas_pkb.evolution import EvolutionSearch
from llm_enas_pkb.main import main


class EliteSelectionTests(unittest.TestCase):
    def make_search(self, **overrides):
        return EvolutionSearch(Config(**overrides), Mock(), Mock())

    def test_default_retains_top_five_and_tournaments_the_other_fifteen(self):
        search = self.make_search()
        self.assertEqual(search.config.population_size, 20)
        self.assertEqual(search.config.elite_ratio, 0.25)
        candidates = list(range(40))
        # Deliberately keep the next-best non-elites out of every tournament.
        # This distinguishes binary tournaments from filling all slots by rank.
        rng = Mock()
        rng.integers.side_effect = [np.array([i, i + 1]) for i in range(0, 30, 2)]

        with patch.object(search, "_tournament", wraps=search._tournament) as tournament:
            selected = search._select_next_population(candidates, float, rng)

        self.assertEqual(selected, [39, 38, 37, 36, 35] + list(range(1, 30, 2)))
        self.assertEqual(len(set(selected)), 20)
        tournament.assert_called_once()
        pool, _, count, passed_rng = tournament.call_args.args
        self.assertEqual(list(pool), list(range(35)))
        self.assertEqual(count, 15)
        self.assertIs(passed_rng, rng)
        self.assertEqual(rng.integers.call_count, 15)

    def test_elite_count_is_rounded_down_from_target_population_size(self):
        for size, ratio, elite_count in ((7, 0.25, 1), (3, 0.25, 0), (20, 0.35, 7)):
            with self.subTest(size=size, ratio=ratio):
                search = self.make_search(population_size=size, elite_ratio=ratio)
                with patch.object(search, "_tournament", wraps=search._tournament) as tournament:
                    selected = search._select_next_population(
                        list(range(40)), float, np.random.default_rng(23),
                    )
                self.assertEqual(selected[:elite_count], list(range(39, 39 - elite_count, -1)))
                self.assertEqual(len(selected), size)
                pool, _, count, _ = tournament.call_args.args
                self.assertEqual(len(pool), 40 - elite_count)
                self.assertEqual(count, size - elite_count)

    def test_duplicate_candidates_and_small_pools_cannot_duplicate_survivors(self):
        for candidates in ([], [1], [1, 3, 2, 3, 1], list(range(8)) * 3):
            with self.subTest(candidates=candidates):
                selected = self.make_search()._select_next_population(
                    candidates, float, np.random.default_rng(11),
                )
                self.assertEqual(len(selected), len(set(candidates)))
                self.assertEqual(set(selected), set(candidates))

    def test_zero_ratio_preserves_existing_tournament_selection(self):
        search = self.make_search(elite_ratio=0.0)
        candidates = list(range(60))
        expected = search._tournament(candidates, float, 20, np.random.default_rng(17))
        selected = search._select_next_population(candidates, float, np.random.default_rng(17))
        self.assertEqual(selected, expected)

    def test_full_elitism_selects_only_the_best_without_random_draws(self):
        search = self.make_search(elite_ratio=1.0)
        rng = Mock()
        selected = search._select_next_population(list(range(40)), float, rng)
        self.assertEqual(selected, list(range(39, 19, -1)))
        rng.integers.assert_not_called()

    def test_seed_reproduces_selection_and_tournament_keeps_randomness(self):
        search = self.make_search()
        candidates = list(range(100))
        first = search._select_next_population(candidates, float, np.random.default_rng(42))
        repeated = search._select_next_population(candidates, float, np.random.default_rng(42))
        different = search._select_next_population(candidates, float, np.random.default_rng(43))
        self.assertEqual(first, repeated)
        self.assertEqual(first[:5], different[:5])
        self.assertNotEqual(first[5:], different[5:])

    def test_invalid_ratios_are_rejected_before_search(self):
        for ratio in (-0.01, 1.01, float("nan"), float("inf"), float("-inf")):
            with self.subTest(ratio=ratio), self.assertRaisesRegex(ValueError, "elite_ratio"):
                self.make_search(elite_ratio=ratio)

    def test_run_selects_elites_from_both_parents_and_offspring(self):
        archs = list(itertools.islice(itertools.product(range(5), repeat=6), 16))
        parents, children = archs[:8], archs[8:]
        parent_scores = {arch: 0.4 + i / 100 for i, arch in enumerate(parents)}
        parent_scores[parents[0]] = 0.99
        # An existing parent also appears among the offspring; its archived
        # measurement must win over the deliberately low new prediction.
        offspring = [parents[0]] + children
        predictor = Mock()
        predictor.encoder = ArchitectureInformationEncoder()
        predictor.predict_values.return_value = np.array([0.01, 0.98] + [0.6] * 7)
        bench = Mock()
        bench.sample.return_value = parents
        bench.valid_acc.side_effect = parent_scores.__getitem__
        bench.test_acc.return_value = 0.97
        cfg = Config(population_size=8, n_init=8, max_iterations=1, query_budget=8,
                     seed=7, search_log_path=None, prompt_log_path=None)
        search = EvolutionSearch(cfg, bench, predictor)

        with patch.object(search, "_make_offspring", return_value=offspring), \
                patch.object(search, "_select_next_population",
                             wraps=search._select_next_population) as select, \
                patch("llm_enas_pkb.evolution.SearchLogger") as logger:
            result = search.run()

        select.assert_called_once()
        self.assertEqual(set(select.call_args.args[0]), set(archs))
        generation = logger.return_value.log_generation.call_args.kwargs
        self.assertEqual(generation["next_pop"][:2], [parents[0], children[0]])
        self.assertEqual(len(set(generation["next_pop"])), 8)
        self.assertEqual(generation["fitness"][parents[0]], 0.99)
        self.assertEqual(result["queries"], 8)
        self.assertEqual(bench.valid_acc.call_count, 8)


class EliteConfigTests(unittest.TestCase):
    def test_environment_parses_elite_ratio_as_float(self):
        with patch.dict("os.environ", {"LLMENAS_ELITE_RATIO": "0.4"}, clear=True), \
                patch("llm_enas_pkb.config.load_dotenv"):
            cfg = Config.from_env()
        self.assertIsInstance(cfg.elite_ratio, float)
        self.assertEqual(cfg.elite_ratio, 0.4)

    def test_cli_override_reaches_search_and_omitted_flag_preserves_environment(self):
        for flags, expected in (([], 0.75), (["--elite-ratio", "0"], 0.0),
                                (["--elite-ratio", "0.5"], 0.5)):
            with self.subTest(flags=flags), \
                    patch.dict("os.environ", {"LLMENAS_ELITE_RATIO": "0.75"}, clear=True), \
                    patch("llm_enas_pkb.config.load_dotenv"), \
                    patch("llm_enas_pkb.main.NASBench201"), \
                    patch("llm_enas_pkb.main.RegressionPredictor"), \
                    patch("llm_enas_pkb.main.EvolutionSearch") as search, \
                    patch("llm_enas_pkb.main.logging.basicConfig"), \
                    patch("sys.stdout", new_callable=io.StringIO):
                search.return_value.run.return_value = {
                    "best_valid": 0.9, "best_test": 0.89, "queries": 20,
                    "best_arch": (0, 0, 0, 0, 0, 0),
                }
                self.assertEqual(main(flags + ["--prompt-log-path", "", "--search-log-path", ""]), 0)
                self.assertEqual(search.call_args.args[0].elite_ratio, expected)


if __name__ == "__main__":
    unittest.main()
