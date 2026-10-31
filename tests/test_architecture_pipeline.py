"""Regression checks for categorical codes, executable prompts, and the search loop."""

from __future__ import annotations

import copy
import itertools
import json
import re
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn

from llm_enas_pkb.categorical import (
    categorical_similarity_matrix,
    validate_architecture_vectors,
)
from llm_enas_pkb.config import Config
from llm_enas_pkb.encoder import ArchitectureInformationEncoder
from llm_enas_pkb.evolution import EvolutionSearch
from llm_enas_pkb.network import (
    NASBench201Cell,
    ResNetBasicblock,
    arch_to_network,
    make_operation,
    network_to_arch,
    network_to_code,
)
from llm_enas_pkb.pkb import select_pkb
from llm_enas_pkb.predictor import RegressionPredictor
from llm_enas_pkb.prompts import build_regression_prompt
from llm_enas_pkb.search_space import arch_to_str, random_arch, str_to_arch


class EncodingTests(unittest.TestCase):
    def test_entire_search_space_is_lossless(self):
        encoder = ArchitectureInformationEncoder()
        for arch in itertools.product(range(5), repeat=6):
            encoded = encoder.encode(arch)
            self.assertEqual(encoded.dtype, np.dtype("int64"))
            self.assertEqual(tuple(encoded), arch)
            self.assertEqual(str_to_arch(arch_to_str(arch)), arch)
        self.assertEqual(encoder.output_dim, 6)
        self.assertEqual(encoder.encode_batch([]).shape, (0, 6))

    def test_invalid_codes_are_never_silently_coerced(self):
        for invalid in ([0] * 5, [0] * 7, [0, 0, 0, 0, 0, 5],
                        [0, 0, 0, 0, 0, -1], [0, 0, 0, 0, 0, True],
                        [0, 0, 0, 0, 0, 1.0]):
            with self.subTest(code=invalid):
                with self.assertRaises(ValueError):
                    ArchitectureInformationEncoder().encode(invalid)
                with self.assertRaises(ValueError):
                    validate_architecture_vectors([invalid])

    def test_id_relabeling_cannot_change_similarity_or_selection(self):
        codes = np.array([[0] * 6, [1] * 6, [4] * 6, [0, 1, 2, 3, 4, 1]])
        relabeling = np.array([4, 2, 0, 1, 3])
        labels = np.array([0.2, 0.3, 0.6, 0.8])
        similarity = categorical_similarity_matrix(codes)
        np.testing.assert_allclose(similarity, categorical_similarity_matrix(relabeling[codes]))
        np.testing.assert_allclose(similarity.diagonal(), 1)
        self.assertEqual(similarity[0, 1], 0)
        np.testing.assert_array_equal(
            select_pkb(codes, labels, selected_size=3),
            select_pkb(relabeling[codes], labels, selected_size=3),
        )
        predictor = RegressionPredictor(Config(offline=True))
        np.testing.assert_allclose(
            predictor.predict_values(codes, labels, codes[::-1]),
            predictor.predict_values(relabeling[codes], labels, relabeling[codes[::-1]]),
        )

    def test_zero_codes_and_unmatched_or_empty_pkb_are_well_defined(self):
        predictor = RegressionPredictor(Config(offline=True))
        self.assertEqual(predictor.predict_values([[0] * 6], [0.4], [[0] * 6])[0], 0.4)
        self.assertAlmostEqual(predictor.predict_values([[1] * 6, [2] * 6], [0.4, 0.8],
                                                       [[0] * 6])[0], 0.6)
        np.testing.assert_array_equal(predictor.predict_values([], [], [[0] * 6]), [0])
        self.assertEqual(predictor.predict_values([], [], []).size, 0)


class NetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_cell_topology_and_pooling_border_semantics(self):
        inputs = torch.ones(2, 2, 8, 8)
        torch.testing.assert_close(NASBench201Cell([0] * 6, 2)(inputs), inputs * 0)
        torch.testing.assert_close(NASBench201Cell([1] * 6, 2)(inputs), inputs * 4)
        torch.testing.assert_close(make_operation(4, 2)(inputs), inputs)

    def test_full_networks_roundtrip_and_classify_all_datasets(self):
        code = [0, 1, 2, 3, 4, 1]
        for dataset, classes, size in (("cifar10", 10, 32), ("cifar100", 100, 32),
                                       ("ImageNet16-120", 120, 16)):
            with self.subTest(dataset=dataset):
                network = arch_to_network(code, dataset=dataset).eval()
                self.assertEqual(network_to_code(network), code)
                normal = [cell for cell in network.cells if isinstance(cell, NASBench201Cell)]
                self.assertEqual(len(normal), 15)
                self.assertEqual(sum(isinstance(c, ResNetBasicblock) for c in network.cells), 2)
                self.assertIsNot(normal[0].ops[2][1].weight, normal[1].ops[2][1].weight)
                with torch.no_grad():
                    features, logits = network(torch.ones(2, 3, size, size))
                self.assertEqual(tuple(features.shape), (2, 64))
                self.assertEqual(tuple(logits.shape), (2, classes))

    def test_inverse_conversion_reads_modules_and_rejects_invalid_operations(self):
        network = arch_to_network([1] * 6, C=2, N=1)
        network.cells[0].ops[0] = make_operation(3, 2)
        with self.assertRaises(ValueError):
            network_to_arch(network)
        for cell in network.cells:
            if isinstance(cell, NASBench201Cell):
                cell.ops[0] = make_operation(3, 2)
        self.assertEqual(network_to_code(network), [3, 1, 1, 1, 1, 1])
        cell = NASBench201Cell([2] * 6, 2)
        cell.ops[0][1].padding = (1, 1)
        with self.assertRaises(ValueError):
            network_to_arch(cell)
        cell = NASBench201Cell([4] * 6, 2)
        cell.ops[0].divisor_override = 1
        with self.assertRaises(ValueError):
            network_to_arch(cell)


class PromptAndSearchTests(unittest.TestCase):
    def test_prompt_code_is_executable_and_describes_this_batch(self):
        code = [0, 1, 2, 3, 4, 1]
        prompt = build_regression_prompt([code], [0.75], [code, code, [0] * 6],
                                         dataset="cifar100", generation=7)
        self.assertIn("Generation: 7.", prompt)
        self.assertIn("cifar100 (100 classes", prompt)
        self.assertIn('"Value" must be a list of exactly 3 numbers', prompt)
        candidate_rows = re.findall(r"Candidate \d+: <Architecture Code>: (\[.*?\]);", prompt)
        self.assertEqual([json.loads(row) for row in candidate_rows], [code, code, [0] * 6])
        blocks = re.findall(r"```python\n(.*?)\n```", prompt, re.DOTALL)
        self.assertEqual(len(blocks), 2)
        namespace = {}
        exec("\n\n".join(blocks), namespace)
        self.assertEqual(namespace["network_to_code"](namespace["network"]), code)
        self.assertEqual(namespace["network"].classifier.out_features, 100)

    def test_invalid_pkb_or_legacy_float_vectors_are_rejected(self):
        for pkb, values, candidates in (
            ([[0] * 6], [], [[1] * 6]),
            ([[0] * 6], [float("nan")], [[1] * 6]),
            ([[0] * 6], [75.0], [[1] * 6]),
            ([[0.0] * 30], [0.75], [[1] * 6]),
        ):
            with self.assertRaises(ValueError):
                build_regression_prompt(pkb, values, candidates)

    def test_single_user_http_request_and_count_correction(self):
        payloads = []
        replies = iter(({"Value": [0.5]}, {"Value": [0.6, 0.7, 0.8]}))

        def post(url, **kwargs):
            payloads.append(copy.deepcopy(kwargs["json"]))
            return Mock(status_code=200, json=lambda: {
                "choices": [{"message": {"content": json.dumps(next(replies))}}]
            })

        cfg = Config(api_key_env="LLMENAS_TEST_KEY", api_base="https://example.invalid",
                     dataset="cifar100", max_retries=1, prompt_log_path=None)
        with patch.dict("os.environ", {"LLMENAS_TEST_KEY": "dummy-only"}):
            with patch("llm_enas_pkb.llm.requests.post", side_effect=post):
                predictor = RegressionPredictor(cfg)
                predictions = predictor.predict_archs(
                    [[0] * 6, [1] * 6, [2] * 6], [[3] * 6], [0.75], generation=2,
                )
        np.testing.assert_allclose(predictions, [0.6, 0.7, 0.8])
        self.assertEqual([m["role"] for m in payloads[0]["messages"]], ["user"])
        self.assertEqual([m["role"] for m in payloads[1]["messages"]],
                         ["user", "assistant", "user"])
        self.assertIn("Generation: 2.", payloads[0]["messages"][0]["content"])
        self.assertIn("cifar100", payloads[0]["messages"][0]["content"])

    def test_response_constraints_reject_invalid_or_incomplete_predictions(self):
        predictor = RegressionPredictor(Config(offline=True))
        for response in ('{"Value": [0.5]}', '{"Value": [true, 0.5]}',
                         '{"Value": [1.1, 0.5]}', '{"Value": [NaN, 0.5]}',
                         '{"Value": ["0.5", 0.5]}', '{"Value": [0.5, 0.5], "extra": 1}'):
            with self.subTest(response=response), self.assertRaises(ValueError):
                predictor._parse_predictions(response, 2)

    def test_multigeneration_search_passes_direct_codes_and_generation(self):
        class SyntheticBenchmark:
            def sample(self, rng, n):
                result = []
                while len(result) < n:
                    arch = random_arch(rng)
                    if arch not in result:
                        result.append(arch)
                return result

            def valid_acc(self, arch):
                return 0.45 + 0.05 * sum(op == 3 for op in arch)

            test_acc = valid_acc

        cfg = Config(offline=True, max_iterations=3, population_size=6, offspring_size=6,
                     n_init=6, query_budget=12, seed=27, search_log_path=None,
                     prompt_log_path=None)
        predictor = RegressionPredictor(cfg)
        with patch.object(predictor, "predict_values", wraps=predictor.predict_values) as predict:
            first = EvolutionSearch(cfg, SyntheticBenchmark(), predictor).run()
        self.assertEqual([call.kwargs["generation"] for call in predict.call_args_list], [1, 2, 3])
        for call in predict.call_args_list:
            for codes in (call.args[0], call.args[2]):
                self.assertEqual(codes.shape[1], 6)
                self.assertEqual(codes.dtype, np.dtype("int64"))
        self.assertLessEqual(first["queries"], cfg.query_budget)
        self.assertEqual(first, EvolutionSearch(cfg, SyntheticBenchmark(), predictor).run())


if __name__ == "__main__":
    unittest.main()
