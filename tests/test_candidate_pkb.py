"""Candidate-conditioned PKB selection, coverage accounting and configuration."""

from __future__ import annotations

import io
import itertools
import unittest
from unittest.mock import Mock, patch

import numpy as np

from llm_enas_pkb.categorical import categorical_similarity_matrix
from llm_enas_pkb.config import Config
from llm_enas_pkb.encoder import ArchitectureInformationEncoder
from llm_enas_pkb.evolution import EvolutionSearch
from llm_enas_pkb.main import main
from llm_enas_pkb.pkb import select_pkb, select_pkb_for_candidates


SPACE = np.array(list(itertools.product(range(5), repeat=6)), dtype=np.int64)


class CandidatePKBTests(unittest.TestCase):
    def example(self):
        rng = np.random.default_rng(17)
        codes = SPACE[rng.choice(len(SPACE), 60, replace=False)]
        return codes[:40], rng.uniform(0.3, 0.9, 40), codes[40:]

    def test_default_keeps_six_rescaled_global_references_and_fourteen_local(self):
        archive, values, candidates = self.example()
        selected = select_pkb_for_candidates(archive, values, candidates)
        global_indices = select_pkb(archive, values, k_perf=3, k_div=3, selected_size=6)
        np.testing.assert_array_equal(selected.indices[:6], global_indices)
        self.assertEqual(selected.sources[:6], ["global"] * 6)
        self.assertEqual(len(selected.indices), 20)
        self.assertEqual(len(set(selected.indices)), 20)
        self.assertEqual(sum(s in ("coverage", "relevance_fill") for s in selected.sources), 14)
        self.assertEqual(selected.coverage["neighbors"], 2)
        self.assertGreaterEqual(selected.coverage["mean_topk"],
                                selected.global_coverage["mean_topk"])

    def test_coverage_serves_two_distinct_groups_instead_of_redundant_high_scores(self):
        archive = np.array([[0, 0, 0, 0, 0, i] for i in (1, 2, 3)] +
                           [[4, 4, 4, 4, 4, i] for i in (1, 2, 3)])
        values = np.array([0.99, 0.98, 0.97, 0.1, 0.09, 0.08])
        candidates = np.array([[0] * 6, [4] * 6])
        result = select_pkb_for_candidates(archive, values, candidates,
                                           selected_size=4, global_size=0)
        self.assertEqual(sum(i < 3 for i in result.indices), 2)
        self.assertEqual(sum(i >= 3 for i in result.indices), 2)
        self.assertAlmostEqual(result.coverage["min_topk"], 5 / 6)
        # Independent relevance ranking ties here; ranking those ties by high
        # accuracy wastes a third example on group zero and leaves group four
        # with only one close reference.
        sim = categorical_similarity_matrix(archive, candidates)
        independent = sorted(range(6), key=lambda i: (-sim[i].mean(), -values[i]))[:4]
        independent_top2 = np.sort(sim[independent].T, axis=1)[:, -2:].mean(axis=1)
        self.assertGreater(result.coverage["min_topk"], float(independent_top2.min()))

    def test_changing_the_prediction_batch_changes_local_selection(self):
        archive = np.array([[0, 0, 0, 0, 0, i] for i in (1, 2, 3)] +
                           [[4, 4, 4, 4, 4, i] for i in (1, 2, 3)])
        values = np.full(6, 0.8)
        zero = select_pkb_for_candidates(archive, values, [[0] * 6],
                                         selected_size=2, global_size=0)
        four = select_pkb_for_candidates(archive, values, [[4] * 6],
                                         selected_size=2, global_size=0)
        self.assertTrue(all(i < 3 for i in zero.indices))
        self.assertTrue(all(i >= 3 for i in four.indices))

    def test_every_greedy_step_maximizes_the_actual_top_two_objective(self):
        archive, values, candidates = self.example()
        result = select_pkb_for_candidates(archive, values, candidates)
        sim = categorical_similarity_matrix(archive, candidates)

        def objective(indices):
            return np.sort(sim[indices].T, axis=1)[:, -2:].mean()

        for end in range(6, len(result.indices)):
            previous = result.indices[:end].tolist()
            best_possible = max(objective(previous + [i]) for i in range(len(archive))
                                if i not in previous)
            self.assertAlmostEqual(objective(result.indices[:end + 1]), best_possible)

    def test_duplicate_and_already_measured_candidates_have_no_extra_weight(self):
        archive, values, candidates = self.example()
        expected = select_pkb_for_candidates(archive, values, candidates)
        repeated = np.concatenate((candidates, candidates[:1].repeat(20, axis=0), archive[:5]))
        actual = select_pkb_for_candidates(archive, values, repeated)
        np.testing.assert_array_equal(actual.indices, expected.indices)
        self.assertEqual(actual.coverage, expected.coverage)

    def test_original_and_empty_unknown_batch_preserve_the_original_selector(self):
        archive, values, candidates = self.example()
        expected = select_pkb(archive, values)
        original = select_pkb_for_candidates(archive, values, candidates, strategy="original")
        np.testing.assert_array_equal(original.indices, expected)
        self.assertEqual(original.sources, ["original"] * 20)
        self.assertEqual(original.coverage["candidate_count"], 20)
        for batch in ([], archive[:3]):
            fallback = select_pkb_for_candidates(archive, values, batch)
            np.testing.assert_array_equal(fallback.indices, expected)
            self.assertEqual(fallback.mode, "original_no_unknown_candidates")
            self.assertEqual(fallback.coverage["candidate_count"], 0)
            self.assertIsNone(fallback.coverage["mean_topk"])

    def test_small_empty_archives_and_missing_neighbors_are_handled(self):
        result = select_pkb_for_candidates([[0, 0, 0, 0, 0, 1]], [0.8], [[0] * 6])
        self.assertEqual(result.indices.tolist(), [0])
        self.assertAlmostEqual(result.coverage["mean_nearest"], 5 / 6)
        self.assertAlmostEqual(result.coverage["mean_topk"], 5 / 12)
        empty = select_pkb_for_candidates([], [], [[0] * 6])
        self.assertEqual(empty.indices.size, 0)
        self.assertEqual(empty.coverage["mean_topk"], 0.0)
        archive, values, candidates = self.example()
        small = select_pkb_for_candidates(archive[:4], values[:4], candidates)
        self.assertEqual(set(small.indices), set(range(4)))
        # A smaller selected_size caps the global quota without padding.
        capped = select_pkb_for_candidates(archive, values, candidates, selected_size=3)
        self.assertEqual(len(capped.indices), 3)
        self.assertEqual(capped.sources, ["global"] * 3)

    def test_zero_gain_fills_by_relevance_without_duplicates(self):
        archive = [[i] * 6 for i in (1, 2, 3)]
        selected = select_pkb_for_candidates(archive, [0.4, 0.5, 0.6], [[0] * 6],
                                             selected_size=3, global_size=0)
        self.assertEqual(set(selected.indices), {0, 1, 2})
        self.assertEqual(selected.sources, ["relevance_fill"] * 3)
        self.assertEqual(selected.coverage["mean_topk"], 0.0)

    def test_selection_is_deterministic_and_invariant_to_operation_id_relabeling(self):
        archive, values, candidates = self.example()
        before = (archive.copy(), values.copy(), candidates.copy())
        selected = select_pkb_for_candidates(archive, values, candidates)
        repeat = select_pkb_for_candidates(archive, values, candidates[::-1])
        labels = np.array([4, 2, 0, 1, 3])
        relabeled = select_pkb_for_candidates(labels[archive], values, labels[candidates])
        np.testing.assert_array_equal(selected.indices, repeat.indices)
        np.testing.assert_array_equal(selected.indices, relabeled.indices)
        for actual, original in zip((archive, values, candidates), before):
            np.testing.assert_array_equal(actual, original)

    def test_global_quota_rescaling_preserves_both_routes(self):
        archive, values, candidates = self.example()
        # Verify the original selector is used with resized routes rather than
        # taking six entries from the original 10-performance/12-diversity merge.
        for k_perf, k_div, expected in ((10, 12, (3, 3)), (0, 12, (0, 6)),
                                      (10, 0, (6, 0)), (100, 1, (5, 1))):
            with self.subTest(k_perf=k_perf, k_div=k_div), \
                    patch("llm_enas_pkb.pkb.select_pkb", wraps=select_pkb) as original:
                select_pkb_for_candidates(archive, values, candidates, k_perf=k_perf, k_div=k_div)
            args = original.call_args.kwargs
            self.assertEqual((args["k_perf"], args["k_div"]), expected)

    def test_invalid_configuration_and_archive_labels_are_rejected(self):
        archive, values, candidates = self.example()
        for overrides in ({"strategy": "unknown"}, {"global_size": -1},
                          {"coverage_neighbors": 0}, {"coverage_neighbors": 1.5},
                          {"selected_size": 0}, {"k_perf": -1}, {"k_div": -1},
                          {"alpha": float("nan")}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                select_pkb_for_candidates(archive, values, candidates, **overrides)
        for labels in ([0.8], [float("nan")] * 40, [1.1] * 40):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                select_pkb_for_candidates(archive, labels, candidates)
        with self.assertRaisesRegex(ValueError, "unique"):
            select_pkb_for_candidates(archive[:1].repeat(2, axis=0), [0.8, 0.8], candidates)


class PKBConfigTests(unittest.TestCase):
    def test_defaults_and_environment_types(self):
        cfg = Config()
        self.assertEqual((cfg.pkb_strategy, cfg.selected_size, cfg.pkb_global_size,
                          cfg.pkb_coverage_neighbors), ("candidate_aware", 20, 6, 2))
        with patch.dict("os.environ", {"LLMENAS_PKB_STRATEGY": "original",
                                       "LLMENAS_PKB_GLOBAL_SIZE": "4",
                                       "LLMENAS_PKB_COVERAGE_NEIGHBORS": "3"}, clear=True), \
                patch("llm_enas_pkb.config.load_dotenv"):
            cfg = Config.from_env()
        self.assertEqual((cfg.pkb_strategy, cfg.pkb_global_size, cfg.pkb_coverage_neighbors),
                         ("original", 4, 3))

    def test_cli_preserves_environment_unless_explicitly_overridden(self):
        for flags, expected in (
            ([], ("original", 4, 3)),
            (["--pkb-strategy", "candidate_aware", "--pkb-global-size", "0",
              "--pkb-coverage-neighbors", "2"], ("candidate_aware", 0, 2)),
        ):
            with self.subTest(flags=flags), \
                    patch.dict("os.environ", {"LLMENAS_PKB_STRATEGY": "original",
                                              "LLMENAS_PKB_GLOBAL_SIZE": "4",
                                              "LLMENAS_PKB_COVERAGE_NEIGHBORS": "3"}, clear=True), \
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
                cfg = search.call_args.args[0]
                self.assertEqual((cfg.pkb_strategy, cfg.pkb_global_size, cfg.pkb_coverage_neighbors),
                                 expected)

    def test_bad_options_fail_before_search_queries(self):
        for options in ({"pkb_strategy": "typo"}, {"pkb_global_size": -1},
                        {"pkb_coverage_neighbors": 0}, {"selected_size": 0}):
            bench = Mock()
            with self.subTest(options=options), self.assertRaises(ValueError):
                EvolutionSearch(Config(**options), bench, Mock())
            bench.sample.assert_not_called()


class PKBSearchIntegrationTests(unittest.TestCase):
    def test_selected_examples_reach_one_prediction_batch_without_extra_truth_queries(self):
        rng = np.random.default_rng(5)
        codes = SPACE[rng.choice(len(SPACE), 60, replace=False)]
        parents = [tuple(row) for row in codes[:40]]
        offspring = [parents[0]] + [tuple(row) for row in codes[40:]]
        truth = {tuple(row): 0.4 + i / 200 for i, row in enumerate(codes)}
        for strategy in ("original", "candidate_aware"):
            with self.subTest(strategy=strategy):
                cfg = Config(n_init=40, query_budget=42, max_iterations=1, seed=5,
                             pkb_strategy=strategy, search_log_path=None, prompt_log_path=None)
                bench = Mock()
                bench.sample.return_value = parents
                bench.valid_acc.side_effect = truth.__getitem__
                bench.test_acc.return_value = 0.8
                predictor = Mock()
                predictor.encoder = ArchitectureInformationEncoder()
                expected = select_pkb_for_candidates(
                    codes[:40], [truth[a] for a in parents], offspring, strategy=strategy,
                )

                def predict(pkb_vectors, pkb_values, candidate_vectors, *, generation):
                    self.assertEqual(bench.valid_acc.call_count, 40)
                    np.testing.assert_array_equal(pkb_vectors, codes[expected.indices])
                    np.testing.assert_allclose(pkb_values, [truth[tuple(a)] for a in pkb_vectors])
                    np.testing.assert_array_equal(candidate_vectors, offspring)
                    self.assertEqual(generation, 1)
                    return np.linspace(0.5, 0.9, len(offspring))

                predictor.predict_values.side_effect = predict
                search = EvolutionSearch(cfg, bench, predictor)
                with patch.object(search, "_make_offspring", return_value=offspring):
                    result = search.run()
                predictor.predict_values.assert_called_once()
                self.assertEqual(result["queries"], 42)
                self.assertEqual(bench.valid_acc.call_count, 42)
                diagnostic = result["history"][0]["pkb"]
                self.assertEqual(diagnostic["coverage"]["candidate_count"], 20)
                self.assertEqual(diagnostic["sources"], expected.sources)
                self.assertTrue(all(a in parents for a in diagnostic["architectures"]))


if __name__ == "__main__":
    unittest.main()
