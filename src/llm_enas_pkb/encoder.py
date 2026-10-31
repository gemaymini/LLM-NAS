"""Encode NAS-Bench-201 cells as six categorical operation IDs.

Positions follow the canonical edges (0,1), (0,2), (1,2), (0,3), (1,3), (2,3).
Each position stores the corresponding ID in ``search_space.OPS`` directly.
"""

from __future__ import annotations

import numpy as np

from .search_space import INPUT_DIM, NUM_EDGES, NUM_OPS, is_valid_arch


class ArchitectureInformationEncoder:
    """Lossless architecture -> six-integer operation vector encoder."""

    @property
    def output_dim(self) -> int:
        return INPUT_DIM

    def encode(self, arch: tuple[int, ...]) -> np.ndarray:
        """Return a fresh integer array containing the six operation IDs."""
        if not is_valid_arch(arch):
            raise ValueError(
                f"Expected {NUM_EDGES} integer operation IDs in [0, {NUM_OPS - 1}]"
            )
        return np.array(arch, dtype=np.int64, copy=True)

    def encode_batch(self, archs) -> np.ndarray:
        """Encode architectures into an integer (N, 6) matrix, including N=0."""
        encoded = [self.encode(arch) for arch in archs]
        if not encoded:
            return np.empty((0, INPUT_DIM), dtype=np.int64)
        return np.stack(encoded)
