"""NAS-Bench-201 benchmark access.

Wraps the ``nas_201_api`` package: loads the pre-computed ``.pth`` archive and
answers ground-truth performance lookups (the paper's "queries") for candidate
architectures.  Validation accuracy (normalised to [0, 1]) is the fitness metric,
matching Sec. 4.1 ("the entire search process is based solely on architecture
information from the validation set").

The archive is ~4.7 GB and almost entirely trained model *weights*; unpickling it
plus ``NASBench201API``'s object construction takes minutes, even though the search
only ever needs one number per architecture (validation/test accuracy).  To avoid
paying that cost on every run, the first load extracts those accuracies into a
small pickle (``<archive>.acc_cache.pkl``, a few MB) beside the archive.  Later
runs read that cache in milliseconds and never touch the heavy ``.pth`` file
(unless the archive changes, in which case the cache is rebuilt automatically).
"""

from __future__ import annotations

import logging
import os
import pickle

import numpy as np

from .config import Config
from .search_space import arch_to_str, random_arch

logger = logging.getLogger(__name__)

# Canonical archive filename (the recommended v1.1 benchmark file).
DEFAULT_FILE = "NAS-Bench-201-v1_1-096897.pth"
DEFAULT_DIR = "data"

# The three datasets the benchmark was trained on.  The cache stores valid/test
# accuracy for all three, so one cache file serves any ``--dataset`` choice.
DATASETS: tuple[str, ...] = ("cifar10", "cifar100", "ImageNet16-120")
CACHE_VERSION = 1


class NASBench201:
    def __init__(self, config: Config):
        self.config = config
        self.dataset = config.dataset
        path = self._resolve_path()
        cache = self._load_or_build_cache(path)
        self._valid = cache["valid"]  # arch_str -> (valid_cifar10, valid_cifar100, valid_img)
        self._test = cache["test"]  # arch_str -> (test_cifar10, test_cifar100, test_img)
        self._datasets = cache["datasets"]
        self._dataset_index = self._datasets.index(self.dataset)
        logger.info(
            "Loaded NAS-Bench-201 (%s) with %d architectures",
            self.dataset,
            len(self._valid),
        )

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
    def _load_api(path: str):
        # Import lazily: the cache-hit path never needs torch, so keeping this out
        # of module scope shaves ~1-2s off every cached startup.
        import torch

        # torch>=2.6 defaults weights_only=True which cannot load this pickled
        # benchmark (it stores custom objects), so pass weights_only=False.
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            # Older torch releases do not accept the weights_only kwarg at all.
            data = torch.load(path, map_location="cpu")
        from nas_201_api import NASBench201API

        return NASBench201API(data, verbose=False)

    # ------------------------------------------------------------ accuracy cache
    def _load_or_build_cache(self, path: str) -> dict:
        """Return the accuracy table, reading it from disk if present else building it."""
        cache_path = os.path.splitext(path)[0] + ".acc_cache.pkl"
        cache = self._read_cache(cache_path, path)
        if cache is not None:
            logger.info("Read accuracy cache from %s", cache_path)
            return cache

        logger.info("No usable accuracy cache; loading the full benchmark (one-time, slow) ...")
        api = self._load_api(path)
        return self._build_cache(api, path, cache_path)

    @staticmethod
    def _read_cache(cache_path: str, source_path: str):
        """Load a valid cache, or return None if missing, stale, or unreadable."""
        if not os.path.isfile(cache_path):
            return None
        try:
            with open(cache_path, "rb") as fh:
                cache = pickle.load(fh)
        except Exception as exc:  # noqa: BLE001 - any corruption should just trigger a rebuild
            logger.warning("Ignoring unreadable cache %s (%s); rebuilding", cache_path, exc)
            return None
        src = cache.get("source", {})
        if (
            cache.get("version") != CACHE_VERSION
            or src.get("file") != os.path.basename(source_path)
            or src.get("size") != os.path.getsize(source_path)
        ):
            logger.info("Cache %s is stale (source changed); rebuilding", cache_path)
            return None
        return cache

    @staticmethod
    def _build_cache(api, source_path: str, cache_path: str) -> dict:
        """Extract per-architecture valid/test accuracies for all datasets into a small dict."""
        valid: dict[str, tuple[float, float, float]] = {}
        test: dict[str, tuple[float, float, float]] = {}
        total = len(api)
        for arch_str, index in api.archstr2index.items():
            v: list[float] = []
            t: list[float] = []
            for ds in DATASETS:
                if ds == "cifar10":
                    # cifar10 validation accuracy lives in the "cifar10-valid" split;
                    # its test accuracy is the "ori-test" of the "cifar10" split.
                    vinfo = api.get_more_info(index, "cifar10-valid", hp="200", is_random=False)
                    tinfo = api.get_more_info(index, "cifar10", hp="200", is_random=False)
                    v.append(float(vinfo["valid-accuracy"]) / 100.0)
                    t.append(float(tinfo["test-accuracy"]) / 100.0)
                else:
                    info = api.get_more_info(index, ds, hp="200", is_random=False)
                    v.append(float(info["valid-accuracy"]) / 100.0)
                    t.append(float(info["test-accuracy"]) / 100.0)
            valid[arch_str] = tuple(v)
            test[arch_str] = tuple(t)
            if len(valid) % 2000 == 0:
                logger.info("  cached %d/%d architectures", len(valid), total)

        cache = {
            "version": CACHE_VERSION,
            "source": {"file": os.path.basename(source_path), "size": os.path.getsize(source_path)},
            "datasets": list(DATASETS),
            "valid": valid,
            "test": test,
        }
        try:
            with open(cache_path, "wb") as fh:
                pickle.dump(cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Saved accuracy cache to %s (%d architectures)", cache_path, len(valid))
        except OSError as exc:
            # Best-effort: a read-only data dir must not abort the run; the in-memory
            # table is still returned and used.
            logger.warning("Could not save cache %s (%s); run proceeds uncached", cache_path, exc)
        return cache

    # --------------------------------------------------------------- queries
    def valid_acc(self, arch: tuple[int, ...]) -> float:
        """Ground-truth validation accuracy of ``arch`` in [0, 1]."""
        return self._valid[arch_to_str(arch)][self._dataset_index]

    def test_acc(self, arch: tuple[int, ...]) -> float:
        """Ground-truth test accuracy of ``arch`` in [0, 1]."""
        return self._test[arch_to_str(arch)][self._dataset_index]

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
        return len(self._valid)
