"""Search-level regressions for prediction correction, query allocation and logs."""

from __future__ import annotations

import itertools
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from llm_enas_pkb.config import Config
from llm_enas_pkb.encoder import ArchitectureInformationEncoder
from llm_enas_pkb.evolution import EvolutionSearch
from llm_enas_pkb.search_log import SearchLogger


ARCHS = list(itertools.islice(itertools.product(range(5), repeat=6), 20))


class QueryAllocationTests(unittest.TestCase):
    def make_search(self, ratio=1 / 3):
        return EvolutionSearch(Config(query_explore_ratio=ratio), Mock(), Mock())

    def test_single_query_generations_accumulate_exploration_slots(self):
        search = self.make_search()
        archive = {}
        explored = 0
        reasons = []
        for count in range(9):
            plan = search._select_queries(
                list(range(20)), float, archive, 1, count, explored, np.random.default_rng(count),
            )
            self.assertEqual(len(plan), 1)
            a, reason = plan[0]
            self.assertNotIn(a, archive)
            archive[a] = float(a)
            explored += int(reason == "explore")
            reasons.append(reason)
            self.assertEqual(explored, math.floor((count + 1) / 3))
        self.assertEqual(reasons, ["exploit", "exploit", "explore"] * 3)

    def test_short_and_empty_pools_only_allocate_actual_queries(self):
        search = self.make_search()
        rng = np.random.default_rng(5)
        self.assertEqual(search._select_queries([], float, {}, 5, 0, 0, rng), [])
        self.assertEqual(search._select_queries([1], float, {1: 1.0}, 5, 0, 0, rng), [])
        first = search._select_queries([1, 1, 2, 2, 3], float, {3: 3.0}, 5, 0, 0, rng)
        self.assertEqual(first, [(2, "exploit"), (1, "exploit")])
        second = search._select_queries([4, 5], float, {}, 1, 2, 0, rng)
        self.assertEqual(second[0][1], "explore")
        self.assertEqual(search._select_queries([1], float, {}, 0, 3, 1, rng), [])

    def test_ratio_endpoints_and_seeded_sampling_without_replacement(self):
        candidates = list(range(30)) * 2
        archive = {29: 29.0}
        for ratio in (0.0, 1 / 3, 1.0):
            search = self.make_search(ratio)
            first = search._select_queries(
                candidates, float, archive, 6, 0, 0, np.random.default_rng(3),
            )
            repeated = search._select_queries(
                candidates, float, archive, 6, 0, 0, np.random.default_rng(3),
            )
            self.assertEqual(first, repeated)
            self.assertEqual(len({a for a, _ in first}), 6)
            self.assertNotIn(29, [a for a, _ in first])
            self.assertEqual(sum(r == "explore" for _, r in first), int(6 * ratio))
            n_exploit = 6 - int(6 * ratio)
            self.assertEqual(first[:n_exploit], [(a, "exploit") for a in
                                              range(28, 28 - n_exploit, -1)])
        # In exploration-only mode even the lowest predicted candidate is eligible.
        rng = Mock()
        rng.choice.return_value = np.array([28])
        self.assertEqual(self.make_search(1)._select_queries(
            candidates, float, archive, 1, 0, 0, rng,
        ), [(0, "explore")])
        rng.choice.assert_called_once_with(29, size=1, replace=False)

    def test_invalid_configuration_rejected_before_any_benchmark_query(self):
        invalid = [{"query_explore_ratio": ratio} for ratio in
                   (-0.1, 1.1, float("nan"), float("inf"))]
        invalid += [{"query_budget": 19}, {"population_size": 0},
                    {"max_iterations": -1}, {"offspring_size": -1},
                    {"n_init": 0, "population_size": 5, "query_budget": 4}]
        for overrides in invalid:
            with self.subTest(overrides=overrides):
                bench = Mock()
                with self.assertRaises(ValueError):
                    EvolutionSearch(Config(**overrides), bench, Mock())
                bench.sample.assert_not_called()
                bench.valid_acc.assert_not_called()


class SearchCorrectionTests(unittest.TestCase):
    def run_search(self, parents, batches, predictions, scores, **overrides):
        options = dict(population_size=len(parents), n_init=len(parents),
                       max_iterations=len(batches), query_budget=len(scores),
                       elite_ratio=1.0, query_explore_ratio=0.0, seed=7,
                       search_log_path=None, prompt_log_path=None)
        options.update(overrides)
        cfg = Config(**options)
        bench = Mock()
        bench.sample.return_value = parents
        bench.valid_acc.side_effect = scores.__getitem__
        bench.test_acc.return_value = 0.9
        predictor = Mock()
        predictor.encoder = ArchitectureInformationEncoder()
        predictor.predict_values.side_effect = [np.array(values) for values in predictions]
        search = EvolutionSearch(cfg, bench, predictor)
        log = SearchLogger(cfg)
        with patch.object(search, "_make_offspring", side_effect=batches), \
                patch.object(search, "_select_next_population",
                             wraps=search._select_next_population) as select, \
                patch("llm_enas_pkb.evolution.SearchLogger", return_value=log), \
                patch.object(log, "log_generation", wraps=log.log_generation) as write:
            result = search.run()
        self.assertEqual(select.call_count, len(batches))
        self.assertLessEqual(result["queries"], cfg.query_budget)
        self.assertEqual(bench.valid_acc.call_count, result["queries"])
        queried_archs = [call.args[0] for call in bench.valid_acc.call_args_list]
        self.assertEqual(len(queried_archs), len(set(queried_archs)))
        frames = [call.kwargs for call in write.call_args_list]
        for frame in frames:
            self.assertTrue(all(a in frame["archive"] for a in frame["elites"]))
            self.assertEqual(frame["next_pop"][:len(frame["elites"])], frame["elites"])
            # PKB uses only labels known before this generation's prediction.
            self.assertTrue(all(a in frame["archive_before"] for a in frame["pkb_archs"]))
            coverage = frame["pkb_diagnostics"]["coverage"]
            self.assertTrue(all(tuple(item["architecture"]) not in frame["archive_before"]
                                for item in coverage["per_candidate"]))
        return result, frames, predictor

    def test_overestimated_offspring_is_corrected_before_final_selection(self):
        p, child = ARCHS[:2]
        result, frames, _ = self.run_search([p], [[child]], [[0.899]], {p: 0.85, child: 0.746})
        frame = frames[0]
        self.assertEqual(frame["next_pop"], [p])
        self.assertEqual(frame["elites"], [p])
        record = frame["query_records"][0]
        self.assertEqual(record["arch"], child)
        self.assertEqual(record["reason"], "exploit")
        self.assertAlmostEqual(record["error"], 0.153)
        self.assertEqual(record["prediction_generation"], 1)
        self.assertIs(record["survived"], False)
        self.assertEqual(result["best_arch"], p)

    def test_mixed_queries_rescue_low_prediction_and_reject_overestimate_together(self):
        p1, p2, high, low = ARCHS[:4]
        result, frames, _ = self.run_search(
            [p1, p2], [[high, low]], [[0.99, 0.1]],
            {p1: 0.8, p2: 0.7, high: 0.2, low: 0.95}, query_explore_ratio=0.5,
        )
        frame = frames[0]
        self.assertEqual([(r["arch"], r["reason"]) for r in frame["query_records"]],
                         [(high, "exploit"), (low, "explore")])
        self.assertEqual(frame["next_pop"], [low, p1])
        self.assertEqual(result["queries"], 4)

    def test_underestimated_offspring_can_be_queried_and_become_true_elite(self):
        p, child = ARCHS[:2]
        result, frames, _ = self.run_search(
            [p], [[child]], [[0.1]], {p: 0.8, child: 0.95}, query_explore_ratio=1.0,
        )
        self.assertEqual(frames[0]["query_records"][0]["reason"], "explore")
        self.assertEqual(frames[0]["elites"], [child])
        self.assertEqual(frames[0]["next_pop"], [child])
        self.assertEqual(result["best_arch"], child)

    def test_latest_prediction_replaces_first_but_never_ground_truth(self):
        p, child = ARCHS[:2]
        _, frames, _ = self.run_search(
            [p], [[p, child], [p, child]], [[0.01, 0.1], [0.02, 0.99]],
            {p: 0.8}, query_budget=1,
        )
        self.assertEqual(frames[0]["fitness"][child], 0.1)
        self.assertEqual(frames[1]["fitness"][child], 0.99)
        self.assertEqual(frames[1]["prediction_generations"][child], 2)
        for frame in frames:
            self.assertEqual(frame["fitness"][p], 0.8)
            self.assertNotIn(p, frame["prediction_generations"])
            self.assertEqual(frame["query_records"], [])

    def test_predicted_parent_is_eligible_for_query_even_without_new_offspring(self):
        p, high, stale = ARCHS[:3]
        _, frames, _ = self.run_search(
            [p], [[high, stale], []], [[0.99, 0.95], []],
            {p: 0.5, high: 0.1, stale: 0.9}, population_size=3, elite_ratio=0.0,
        )
        self.assertIn(stale, frames[0]["next_pop"])
        self.assertEqual(frames[1]["query_records"][0]["arch"], stale)
        self.assertEqual(frames[1]["query_records"][0]["prediction_generation"], 1)
        self.assertEqual(frames[1]["population_fitness"][stale], 0.95)
        self.assertNotIn(stale, frames[1]["archive_before"])
        self.assertEqual(frames[1]["fitness"][stale], 0.9)

    def test_parent_prediction_refresh_is_optional_and_stays_in_one_batch(self):
        p, child, other = ARCHS[:3]
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                _, frames, predictor = self.run_search(
                    [p], [[child], [other]], [[0.9], [0.2, 0.3] if enabled else [0.2]],
                    {p: 0.5}, population_size=2, refresh_parent_predictions=enabled,
                )
                expected = [other, child] if enabled else [other]
                self.assertEqual(predictor.predict_values.call_count, 2)
                self.assertEqual(predictor.predict_values.call_args.args[2].tolist(),
                                 [list(a) for a in expected])
                self.assertEqual(frames[1]["fitness"][child], 0.3 if enabled else 0.9)
                self.assertEqual(frames[1]["prediction_generations"][child], 2 if enabled else 1)
                self.assertEqual(frames[1]["population_fitness"][child], 0.9)
                targets = frames[1]["pkb_diagnostics"]["coverage"]["per_candidate"]
                self.assertEqual([item["architecture"] for item in targets],
                                 [list(a) for a in expected])

    def test_query_budget_rolls_over_when_candidates_are_exhausted(self):
        p, a, b, c = ARCHS[:4]
        result, frames, _ = self.run_search(
            [p], [[p], [a], [b, c]], [[0.1], [0.9], [0.8, 0.7]],
            {p: 0.5, a: 0.6, b: 0.7, c: 0.8}, query_explore_ratio=1 / 3,
        )
        self.assertEqual([len(f["query_records"]) for f in frames], [0, 1, 2])
        self.assertEqual(result["queries"], 4)
        reasons = [r["reason"] for f in frames for r in f["query_records"]]
        self.assertEqual(reasons, ["exploit", "exploit", "explore"])

    def test_overlap_counts_and_small_candidate_pool_are_consistent(self):
        p, child = ARCHS[:2]
        _, frames, _ = self.run_search(
            [p], [[p, child]], [[0.01, 0.9]], {p: 0.8}, population_size=4,
        )
        frame = frames[0]
        self.assertEqual(len(frame["next_pop"]), 2)
        self.assertEqual(frame["survival"], {
            "parents_retained": 1, "offspring_selected": 2,
            "overlap_selected": 1, "new_offspring_selected": 1,
        })

    def test_real_log_reports_query_error_and_queried_but_rejected_candidate(self):
        p, child = ARCHS[:2]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "search.log"
            self.run_search([p], [[child]], [[0.899]], {p: 0.85, child: 0.746},
                            search_log_path=str(path))
            content = path.read_text()
        self.assertIn("selection_policy=measured_elites query_timing=before_selection", content)
        self.assertIn("PKB exemplars (1 selected from 1 archived)", content)
        self.assertIn("pkb_strategy=candidate_aware selected_size=20 pkb_global_size=6", content)
        self.assertIn("source=global", content)
        self.assertIn("unknown_candidates=1 neighbors=2", content)
        self.assertIn("Final PKB coverage: mean_nearest=0.8333", content)
        self.assertIn("topk_mean=0.4167", content)
        self.assertIn("reason=exploit prediction=0.8990", content)
        self.assertIn("error=+0.1530 survived=False", content)
        self.assertIn("Ground-truth elites (1/1 protected slots)", content)
        self.assertIn("parents_retained=1 offspring_selected=0", content)

    def test_real_log_preserves_predicted_parent_snapshot_before_query(self):
        p, high, stale = ARCHS[:3]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "search.log"
            self.run_search(
                [p], [[high, stale], []], [[0.99, 0.95], []],
                {p: 0.5, high: 0.1, stale: 0.9}, population_size=3, elite_ratio=0.0,
                search_log_path=str(path),
            )
            content = path.read_text().split("Generation 2", 1)[1]
        parents = content.split("Population P_2", 1)[1].split("PKB exemplars", 1)[0]
        survivors = content.split("Next population P_3", 1)[1]
        self.assertIn("val=0.9500  [pred]  prediction_gen=1", parents)
        self.assertIn("val=0.9000  [GT]", survivors)
        self.assertIn("prediction_gen=1  error=+0.0500 survived=True", content)

    def test_zero_iterations_keeps_initial_result_without_prediction(self):
        p = ARCHS[0]
        result, frames, predictor = self.run_search([p], [], [], {p: 0.8})
        self.assertEqual(result["best_arch"], p)
        self.assertEqual(result["queries"], 1)
        self.assertEqual(frames, [])
        predictor.predict_values.assert_not_called()

    def test_refresh_log_shows_new_prediction_separately_from_parent_snapshot(self):
        p, child, other = ARCHS[:3]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "search.log"
            self.run_search(
                [p], [[child], [other]], [[0.9], [0.2, 0.3]], {p: 0.5},
                population_size=2, refresh_parent_predictions=True, search_log_path=str(path),
            )
            content = path.read_text().split("Generation 2", 1)[1]
        parents = content.split("Population P_2", 1)[1].split("PKB exemplars", 1)[0]
        refreshed = content.split("Refreshed unmeasured parents (1):", 1)[1]
        self.assertIn("val=0.9000  [pred]  prediction_gen=1", parents)
        self.assertIn("val=0.3000  [pred]  prediction_gen=2", refreshed)


if __name__ == "__main__":
    unittest.main()
