"""Run metadata must precede work and snapshot the actual per-run settings."""

from dataclasses import asdict
from datetime import datetime, timedelta
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import Mock, patch

from llm_enas_pkb.config import Config
from llm_enas_pkb.evolution import EvolutionSearch
from llm_enas_pkb.experiment_log import log_experiment_start


def snapshots(content):
    return [json.loads(value) for value in re.findall(
        r"Hyperparameters \(effective Config, JSON\):\n(.*?)\nEnd hyperparameters",
        content, re.DOTALL,
    )]


class ExperimentLogTests(unittest.TestCase):
    def test_full_config_shared_across_logs_and_repeated_runs_snapshot_current_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            search = Path(tmp) / "search.log"
            prompt = Path(tmp) / "prompts.log"
            cfg = Config(search_log_path=str(search), prompt_log_path=str(prompt),
                         seed=42, mutation_prob=0.15, temperature=0.6,
                         alpha=0.7, pkb_error_feedback=True,
                         api_key_env="LLMENAS_TEST_SECRET")
            before = asdict(cfg)
            with patch.dict("os.environ", {"LLMENAS_TEST_SECRET": "never-log-this-value"}):
                log_experiment_start(cfg, algorithm="LLM-ENAS-PKB")
                cfg.seed = 43
                cfg.architecture_features = "params"
                log_experiment_start(cfg, algorithm="LLM-ENAS-PKB")
            content = search.read_text()
            self.assertEqual(content, prompt.read_text())
        self.assertEqual(snapshots(content), [before, asdict(cfg)])
        self.assertNotIn("never-log-this-value", content)
        ids = re.findall(r"^experiment_id=(.+)$", content, re.MULTILINE)
        self.assertEqual(len(ids), 2)
        self.assertNotEqual(ids[0], ids[1])
        for stamp in re.findall(r"^started_at=(.+)$", content, re.MULTILINE):
            self.assertEqual(datetime.fromisoformat(stamp).utcoffset(), timedelta(hours=8))
        self.assertIn("timezone=Asia/Shanghai", content)

    def test_start_is_persisted_before_initial_sampling_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "search.log"
            cfg = Config(search_log_path=str(path), prompt_log_path=None, seed=7)
            bench = Mock()

            def fail_sampling(*args):
                self.assertEqual(snapshots(path.read_text()), [asdict(cfg)])
                raise RuntimeError("sampling failed")

            bench.sample.side_effect = fail_sampling
            with self.assertRaisesRegex(RuntimeError, "sampling failed"):
                EvolutionSearch(cfg, bench, Mock()).run()
            bench.valid_acc.assert_not_called()
            self.assertEqual(path.read_text().count("Experiment start"), 1)

    def test_shared_path_is_written_once_and_random_search_omits_prompt_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp) / "shared.log"
            cfg = Config(search_log_path=str(shared), prompt_log_path=str(shared))
            log_experiment_start(cfg, algorithm="LLM-ENAS-PKB")
            self.assertEqual(shared.read_text().count("Experiment start"), 1)
            cfg.prompt_log_path = str(Path(tmp) / "unused.log")
            log_experiment_start(cfg, algorithm="random-search", include_prompt_log=False)
            self.assertFalse(Path(cfg.prompt_log_path).exists())
            self.assertIn("algorithm=random-search", shared.read_text())

    def test_disabled_and_unwritable_logs_do_not_abort_search_start(self):
        cfg = Config(search_log_path=None, prompt_log_path="")
        with patch("llm_enas_pkb.experiment_log.Path.open") as open_file:
            log_experiment_start(cfg, algorithm="LLM-ENAS-PKB")
            open_file.assert_not_called()
        with tempfile.TemporaryDirectory() as tmp:
            cfg.search_log_path = tmp  # A directory cannot be opened as a log file.
            cfg.prompt_log_path = str(Path(tmp) / "prompt.log")
            with self.assertLogs("llm_enas_pkb.experiment_log", level="WARNING"):
                log_experiment_start(cfg, algorithm="LLM-ENAS-PKB")
            self.assertIn("Experiment start", Path(cfg.prompt_log_path).read_text())


if __name__ == "__main__":
    unittest.main()
