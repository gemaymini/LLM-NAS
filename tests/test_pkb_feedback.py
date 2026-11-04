"""Prediction feedback contract, ablation isolation, and run-local state."""

import unittest
from unittest.mock import Mock, patch

import numpy as np

from llm_enas_pkb.config import Config
from llm_enas_pkb.evolution import EvolutionSearch
from llm_enas_pkb.main import main
from llm_enas_pkb.pkb import PredictionFeedback
from llm_enas_pkb.predictor import RegressionPredictor
from llm_enas_pkb.prompts import build_regression_prompt


class FeedbackPromptTests(unittest.TestCase):
    def test_directions_scale_and_missing_history(self):
        codes = [[i] * 6 for i in range(4)]
        feedback = [PredictionFeedback(v, 1, 2) for v in (0.9, 0.7, 0.8)] + [None]
        prompt = build_regression_prompt(codes, [0.8] * 4, [codes[0]],
                                         generation=3, pkb_feedback=feedback)
        self.assertIn("<Prediction Error>: +0.100000; <Direction>: overestimated", prompt)
        self.assertIn("<Prediction Error>: -0.100000; <Direction>: underestimated", prompt)
        self.assertIn("<Prediction Error>: +0.000000; <Direction>: matched", prompt)
        rows = [line for line in prompt.splitlines() if line.startswith("PKB ")]
        self.assertNotIn("Historical Prediction", rows[-1])
        self.assertTrue(all("<Value>: 0.800000;" in row for row in rows))
        self.assertIn("2 percentage points", prompt)

    def test_no_feedback_preserves_prompt_exactly_and_bad_evidence_is_rejected(self):
        args = ([[0] * 6], [0.8], [[1] * 6])
        baseline = build_regression_prompt(*args, generation=1)
        self.assertEqual(baseline, build_regression_prompt(
            *args, generation=1, pkb_feedback=[None]))
        for feedback in ([], [0.2], [PredictionFeedback(0.9, 1, 1)]):
            with self.subTest(feedback=feedback), self.assertRaises(ValueError):
                build_regression_prompt(*args, generation=1, pkb_feedback=feedback)
        for prediction, pgen, qgen in [(float("nan"), 1, 1), (1.1, 1, 1),
                                      (0.5, 2, 1), (0.5, 0, 1)]:
            with self.assertRaises(ValueError):
                PredictionFeedback(prediction, pgen, qgen)

    def test_predictor_switch_controls_actual_api_prompt(self):
        for enabled in (False, True):
            cfg = Config(offline=True, pkb_error_feedback=enabled)
            predictor = RegressionPredictor(cfg)
            cfg.offline = False
            predictor.client = Mock()
            predictor.client.chat.return_value = '{"Value": [0.75]}'
            values = predictor.predict_archs(
                [[1] * 6], [[0] * 6], [0.8], generation=2,
                pkb_feedback=[PredictionFeedback(0.9, 1, 1)],
            )
            np.testing.assert_equal(values, [0.75])
            prompt = predictor.client.chat.call_args.args[0][0]["content"]
            self.assertEqual("<Historical Prediction>" in prompt, enabled)

    def test_cli_and_environment_override_both_directions(self):
        for flags, expected in [([], True), (["--no-pkb-error-feedback"], False),
                                (["--pkb-error-feedback"], True)]:
            with patch.dict("os.environ", {"LLMENAS_PKB_ERROR_FEEDBACK": "true"}, clear=True), \
                    patch("llm_enas_pkb.config.load_dotenv"), \
                    patch("llm_enas_pkb.main.NASBench201"), \
                    patch("llm_enas_pkb.main.RegressionPredictor"), \
                    patch("llm_enas_pkb.main.EvolutionSearch") as search, \
                    patch("llm_enas_pkb.main.logging.basicConfig"), patch("builtins.print"):
                search.return_value.run.return_value = dict(
                    best_arch=(0,) * 6, best_valid=0.8, best_test=0.8, queries=20)
                main(flags + ["--prompt-log-path", "", "--search-log-path", ""])
                self.assertIs(search.call_args.args[0].pkb_error_feedback, expected)

    def test_offline_ablation_keeps_trajectory_and_each_run_starts_without_feedback(self):
        results = []
        for enabled in (False, True):
            cfg = Config(offline=True, pkb_error_feedback=enabled, seed=42,
                         n_init=3, population_size=3, offspring_size=5,
                         max_iterations=3, query_budget=9,
                         prompt_log_path=None, search_log_path=None)
            bench = Mock()
            bench.sample.return_value = [(i,) * 6 for i in range(3)]
            bench.valid_acc.side_effect = lambda a: sum(a) / 24
            bench.test_acc.side_effect = bench.valid_acc.side_effect
            search = EvolutionSearch(cfg, bench, RegressionPredictor(cfg))
            first = search.run()
            self.assertEqual(first, search.run())
            self.assertEqual(first["history"][0]["pkb_feedback_count"], 0)
            for frame in first["history"]:
                frame.pop("pkb_feedback_count")
            results.append(first)
        self.assertEqual(results[0], results[1])


if __name__ == "__main__":
    unittest.main()
