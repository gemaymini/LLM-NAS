"""Independent cost checks, feature ablations, and prediction/log integration."""

import io
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch

from llm_enas_pkb.architecture_cost import (
    COST_PROTOCOL,
    FEATURE_MODES,
    _measure_cost,
    architecture_cost,
    format_architecture_features,
)
from llm_enas_pkb.config import Config
from llm_enas_pkb.evolution import EvolutionSearch
from llm_enas_pkb.main import main
from llm_enas_pkb.network import arch_to_network
from llm_enas_pkb.pkb import PredictionFeedback
from llm_enas_pkb.predictor import RegressionPredictor
from llm_enas_pkb.prompts import build_regression_prompt
from llm_enas_pkb.search_log import SearchLogger


def expected_counts(code, C, N, size, classes):
    """Closed-form oracle for the fixed macro network, independent of hooks."""
    params = 3 * C * 9 + 2 * C  # stem Conv + BN
    macs = size**2 * 3 * C * 9
    for stage in range(3):
        channels = C * 2**stage
        spatial = size // 2**stage
        if stage:
            # Two main 3x3 convolutions plus the shortcut 1x1 convolution.
            weights = 9 * (channels // 2) * channels + 9 * channels**2
            weights += (channels // 2) * channels
            params += weights + 4 * channels
            macs += spatial**2 * weights
        n1, n3 = code.count(2), code.count(3)
        params += N * (channels**2 * (n1 + 9*n3) + 2*channels*(n1+n3))
        macs += N * spatial**2 * channels**2 * (n1 + 9*n3)
    params += 2 * (4*C) + (4*C)*classes + classes  # final BN and linear head
    macs += (4*C)*classes
    return params, 2*macs


class ArchitectureCostTests(unittest.TestCase):
    def test_counts_match_formula_and_real_parameters_for_all_operations_and_datasets(self):
        codes = [[i]*6 for i in range(5)] + [[0, 1, 2, 3, 4, 1]]
        for dataset, size, classes in (("cifar10", 32, 10), ("cifar100", 32, 100),
                                        ("ImageNet16-120", 16, 120)):
            for C, N in ((2, 1), (16, 5)):
                for code in codes:
                    with self.subTest(dataset=dataset, C=C, N=N, code=code):
                        cost = architecture_cost(code, dataset=dataset, C=C, N=N)
                        self.assertEqual((cost.params, cost.flops),
                                         expected_counts(code, C, N, size, classes))
                        model = arch_to_network(code, dataset=dataset, C=C, N=N)
                        self.assertEqual(cost.params, sum(p.numel() for p in model.parameters()))

    def test_topology_permutations_can_have_equal_cost(self):
        first = architecture_cost([3, 0, 2, 1, 4, 0])
        second = architecture_cost([0, 1, 4, 3, 0, 2])
        self.assertEqual(first, second)

    def test_cache_keys_and_random_state_preservation(self):
        _measure_cost.cache_clear()
        torch_state = torch.get_rng_state().clone()
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        first = architecture_cost([0, 1, 2, 3, 4, 1])
        duplicate = architecture_cost(np.array([0, 1, 2, 3, 4, 1]))
        self.assertIs(first, duplicate)
        self.assertEqual(_measure_cost.cache_info().misses, 1)
        self.assertEqual(_measure_cost.cache_info().hits, 1)
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        self.assertEqual(python_state, random.getstate())
        for before, after in zip(numpy_state, np.random.get_state()):
            np.testing.assert_equal(before, after)
        other_dataset = architecture_cost([0, 1, 2, 3, 4, 1], dataset="cifar100")
        other_width = architecture_cost([0, 1, 2, 3, 4, 1], C=8)
        other_depth = architecture_cost([0, 1, 2, 3, 4, 1], N=1)
        for other in (other_dataset, other_width, other_depth):
            self.assertNotEqual(first, other)
        self.assertEqual(_measure_cost.cache_info().misses, 4)

    def test_params_only_never_runs_forward_and_disabled_mode_never_profiles(self):
        _measure_cost.cache_clear()
        with patch("llm_enas_pkb.network.TinyNetwork.forward", side_effect=AssertionError):
            cost = architecture_cost([2]*6, include_flops=False)
        self.assertIsNone(cost.flops)
        with patch("llm_enas_pkb.architecture_cost.architecture_cost",
                   side_effect=AssertionError):
            prompt = build_regression_prompt([[0]*6], [0.8], [[1]*6],
                                             architecture_features="none")
        self.assertNotIn("<Params>", prompt)
        self.assertNotIn("<FLOPs>", prompt)
        self.assertNotIn("Architecture features:", prompt)

    def test_invalid_inputs_rejected_even_after_equal_valid_key_is_cached(self):
        architecture_cost([1]*6, C=1, N=1)
        for code, kwargs in (([True]*6, {"C": 1, "N": 1}),
                             ([1.0]*6, {"C": 1, "N": 1}), ([5]*6, {}),
                             ([1]*5, {}), ([1]*6, {"C": True, "N": 1}),
                             ([1]*6, {"C": 1.0}), ([1]*6, {"N": 0}),
                             ([1]*6, {"dataset": "unknown"}),
                             ([1]*6, {"include_flops": 1})):
            with self.subTest(code=code, kwargs=kwargs), self.assertRaises(ValueError):
                architecture_cost(code, **kwargs)
        with self.assertRaises(ValueError):
            RegressionPredictor(Config(offline=True, architecture_features="invalid"))


class ArchitectureFeatureIntegrationTests(unittest.TestCase):
    def test_actual_api_prompt_has_identical_features_for_pkb_and_repeated_candidates(self):
        code = [0, 1, 2, 3, 4, 1]
        for mode in FEATURE_MODES:
            with self.subTest(mode=mode):
                cfg = Config(offline=True, architecture_features=mode, pkb_error_feedback=True)
                predictor = RegressionPredictor(cfg)
                cfg.offline = False
                predictor.client = Mock()
                predictor.client.chat.return_value = '{"Value": [0.8, 0.8]}'
                result = predictor.predict_archs(
                    [code, code], [code], [0.8], generation=2,
                    pkb_feedback=[PredictionFeedback(0.75, 1, 1)],
                )
                np.testing.assert_equal(result, [0.8, 0.8])
                prompt = predictor.client.chat.call_args.args[0][0]["content"]
                rows = [line for line in prompt.splitlines()
                        if line.startswith(("PKB 1:", "Candidate 1:", "Candidate 2:"))]
                self.assertEqual(len(rows), 3)
                fields = format_architecture_features(code, dataset="cifar10", mode=mode)
                for row in rows:
                    self.assertEqual("<Params>" in row, mode in ("params", "both"))
                    self.assertEqual("<FLOPs>" in row, mode in ("flops", "both"))
                    if fields:
                        self.assertIn(fields, row)
                self.assertIn("<Prediction Error>: -0.050000;", rows[0])
                self.assertNotIn("<Historical Prediction>", rows[1])
                if mode in ("flops", "both"):
                    self.assertIn("1 multiply-accumulate = 2 FLOPs", prompt)
                    self.assertIn("no graph pruning", prompt)

    def test_cli_environment_precedence_and_invalid_env_fail_before_benchmark(self):
        for flags, expected in (([], "params"), (["--architecture-features", "none"], "none"),
                                (["--architecture-features", "both"], "both"),
                                (["--architecture-features", "flops"], "flops")):
            with patch.dict("os.environ", {"LLMENAS_ARCHITECTURE_FEATURES": "params"},
                            clear=True), patch("llm_enas_pkb.config.load_dotenv"), \
                    patch("llm_enas_pkb.main.NASBench201"), \
                    patch("llm_enas_pkb.main.RegressionPredictor"), \
                    patch("llm_enas_pkb.main.EvolutionSearch") as search, \
                    patch("llm_enas_pkb.main.logging.basicConfig"), \
                    patch("sys.stdout", new_callable=io.StringIO):
                search.return_value.run.return_value = dict(
                    best_arch=(0,)*6, best_valid=0.8, best_test=0.8, queries=20)
                main(flags + ["--prompt-log-path", "", "--search-log-path", ""])
                self.assertEqual(search.call_args.args[0].architecture_features, expected)
        with patch.dict("os.environ", {"LLMENAS_ARCHITECTURE_FEATURES": "invalid"}), \
                patch("llm_enas_pkb.config.load_dotenv"), \
                patch("llm_enas_pkb.main.NASBench201") as bench:
            with self.assertRaises(ValueError):
                main([])
            bench.assert_not_called()

    def test_search_log_matches_prompt_features(self):
        code = (0, 1, 2, 3, 4, 1)
        for mode in FEATURE_MODES:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "search.log"
                cfg = Config(dataset="ImageNet16-120", architecture_features=mode,
                             search_log_path=str(path))
                SearchLogger(cfg).log_header([code], {code: 0.8})
                content = path.read_text()
            self.assertIn(f"architecture_features={mode}", content)
            self.assertEqual("<Params>" in content, mode in ("params", "both"))
            self.assertEqual("<FLOPs>" in content, mode in ("flops", "both"))
            if mode != "none":
                self.assertIn(COST_PROTOCOL, content)
                self.assertIn(format_architecture_features(
                    code, dataset=cfg.dataset, mode=mode), content)
            if mode in ("flops", "both"):
                self.assertIn("16x16", content)

    def test_offline_search_and_query_counts_unchanged_across_modes(self):
        results = []
        for mode in FEATURE_MODES:
            cfg = Config(offline=True, architecture_features=mode, pkb_error_feedback=True,
                         refresh_parent_predictions=True, seed=42, n_init=3, population_size=3,
                         offspring_size=5, max_iterations=3, query_budget=9,
                         prompt_log_path=None, search_log_path=None)
            bench = Mock()
            bench.sample.return_value = [(i,)*6 for i in range(3)]
            bench.valid_acc.side_effect = lambda code: sum(code)/24
            bench.test_acc.side_effect = bench.valid_acc.side_effect
            result = EvolutionSearch(cfg, bench, RegressionPredictor(cfg)).run()
            self.assertEqual(bench.valid_acc.call_count, result["queries"])
            self.assertEqual(result["queries"], 9)
            results.append(result)
        for result in results[1:]:
            self.assertEqual(result, results[0])


if __name__ == "__main__":
    unittest.main()
