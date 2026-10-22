"""NAS-Bench-201 benchmark access.

Wraps the ``nas_201_api`` package: loads the pre-computed ``.pth`` archive and
answers ground-truth performance lookups (the paper's "queries") for candidate
architectures.  Validation accuracy (normalised to [0, 1]) is the fitness metric,
matching Sec. 4.1 ("the entire search process is based solely on architecture
information from the validation set").
"""

from __future__ import annotations

import logging
import os

import numpy as np
import torch

from .config import Config
from .search_space import arch_to_str, is_valid_arch, random_arch

logger = logging.getLogger(__name__)

# Canonical archive filename (the recommended v1.1 benchmark file).
DEFAULT_FILE = "NAS-Bench-201-v1_1-096897.pth"
DEFAULT_DIR = "data"


class NASBench201:
    def __init__(self, config: Config):
        self.config = config
        self.dataset = config.dataset
        path = self._resolve_path()
        self.api = self._load_api(path)
        # Memoise ground-truth lookups: the same architecture is queried many
        # times across generations, and each ``get_more_info`` call is not free.
        self._cache: dict[tuple[int, ...], float] = {}
        logger.info("Loaded NAS-Bench-201 (%s) with %d architectures", config.dataset, len(self.api))

    # ------------------------------------------------------------- loading
    def _resolve_path(self) -> str:
        """Locate the archive: explicit path, local ``data/`` dir, legacy TORCH_HOME, else download."""
        if self.config.data_path and os.path.isfile(self.config.data_path):
            return self.config.data_path
        candidate = os.path.join(DEFAULT_DIR, DEFAULT_FILE)
        if os.path.isfile(candidate):
            return candidate
        # Also check the legacy TORCH_HOME location used by nas_201_api.
        torch_home = os.environ.get("TORCH_HOME")
        if torch_home:
            legacy = os.path.join(torch_home, DEFAULT_FILE)
            if os.path.isfile(legacy):
                return legacy
        return self._download(candidate)

    @staticmethod
    def _download(target: str) -> str:
        from .download import download_nasbench201

        logger.warning("NAS-Bench-201 archive not found; downloading to %s", target)
        return download_nasbench201(target)

    @staticmethod
    def _load_api(path: str):
        # torch>=2.6 defaults weights_only=True which cannot load this pickled
        # benchmark (it stores custom objects), so pass weights_only=False.
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            # Older torch releases do not accept the weights_only kwarg at all.
            data = torch.load(path, map_location="cpu")
        from nas_201_api import NASBench201API

        return NASBench201API(data, verbose=False)

    # --------------------------------------------------------------- queries
    def valid_acc(self, arch: tuple[int, ...]) -> float:
        """Ground-truth validation accuracy of ``arch`` in [0, 1]."""
        if arch not in self._cache:
            index = self.api.query_index_by_arch(arch_to_str(arch))
            info = self.api.get_more_info(index, self.dataset, hp="200", is_random=False)
            # The API reports percentages (0-100); normalise to [0, 1] to match
            # the value range documented in the regression prompt.
            self._cache[arch] = float(info["valid-accuracy"]) / 100.0
        return self._cache[arch]

    def test_acc(self, arch: tuple[int, ...]) -> float:
        """Ground-truth test accuracy of ``arch`` in [0, 1]."""
        index = self.api.query_index_by_arch(arch_to_str(arch))
        info = self.api.get_more_info(index, self.dataset, hp="200", is_random=False)
        return float(info["test-accuracy"]) / 100.0

    def valid_acc_batch(self, archs) -> np.ndarray:
        return np.array([self.valid_acc(a) for a in archs])

    def sample(self, rng: np.random.Generator, n: int) -> list[tuple[int, ...]]:
        """Sample ``n`` distinct random architectures."""
        out: list[tuple[int, ...]] = []
        seen: set[tuple[int, ...]] = set()
        while len(out) < n:
            a = random_arch(rng)
            if a not in seen:
                seen.add(a)
                out.append(a)
        return out

    def __len__(self) -> int:
        return len(self.api)
