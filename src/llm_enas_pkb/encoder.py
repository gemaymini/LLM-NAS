"""Architecture information encoder (Sec. 3.4).

The architecture is modelled as a directed acyclic graph ``G = (V, E)`` whose
edges carry operator types.  Each edge operation is one-hot encoded and,
optionally, projected into a dense feature space (Eq. 6):

    E_op = O W,   O in {0,1}^{m x l},  W in R^{l x d}

For NAS-Bench-201 ``m = 6`` edges and ``l = 5`` operators, so with ``d = l`` the
edge embedding ``E_op`` has shape (6, 5) and flattening the six operation slots
yields the 30-dimensional input vector reported in Table 1.

Topological information is injected through the degree-normalised neighbourhood
aggregation of Eq. (7):

    h_v^(k+1) = h_v^(k) + sum_{u in N(v)} h_u^(k) / d_u

run for ``k`` iterations over the operation-slot graph (two slots are neighbours
when their cell edges share a vertex).  After ``k`` iterations the node features
are arranged in topological order and flattened into a fixed-length vector.

The encoder is *training-free*: the projection matrix ``W`` is a fixed
seed-initialised matrix and is never updated, keeping the encoding identical
across runs so the LLM always observes a consistent feature space.
"""

from __future__ import annotations

import numpy as np

from .search_space import INPUT_DIM, NUM_EDGES, NUM_OPS, one_hot_ops, slot_adjacency


class ArchitectureInformationEncoder:
    """Deterministic, training-free encoder: architecture -> 30-dim vector."""

    def __init__(self, *, depth: int = 3, use_projection: bool = True, seed: int = 0):
        self.depth = depth
        self.use_projection = use_projection
        # Seeded RNG makes the fixed projection matrix reproducible across runs.
        rng = np.random.default_rng(seed)
        # Fixed (never-trained) linear map of Eq. (6): R^{NUM_OPS x NUM_OPS}.
        self._W = rng.standard_normal((NUM_OPS, NUM_OPS))
        # Adjacency of the six operation slots (the line graph of the cell DAG).
        self._adj = slot_adjacency()
        # Degree of each slot node (sum of in- and out-degrees; undirected here).
        self._degree = self._adj.sum(axis=1)
        # Avoid division-by-zero for any isolated node (defensive: none occur for
        # the NAS-Bench-201 cell, but keeps the aggregation numerically safe).
        self._degree[self._degree == 0] = 1.0

    @property
    def output_dim(self) -> int:
        return INPUT_DIM

    def encode(self, arch: tuple[int, ...]) -> np.ndarray:
        """Encode ``arch`` (6 op indices) into a 30-dim float vector."""
        # Eq. (6): one-hot operators -> (optional) dense edge features.
        e_op = one_hot_ops(arch)  # (6, 5)
        if self.use_projection:
            e_op = e_op @ self._W  # (6, 5)

        # Eq. (7): degree-normalised neighbourhood aggregation over the slots.
        # h_v^(k+1) = h_v^(k) + sum_{u in N(v)} h_u^(k) / d_u, run `depth` times.
        h = e_op
        for _ in range(self.depth):
            # ``self._adj @ h`` gathers neighbour features into each node; the
            # ``/ self._degree[:, None]`` broadcasts the per-node degree down each
            # column to normalise the incoming sum.
            h = h + (self._adj @ h) / self._degree[:, None]

        # Flatten in topological (canonical edge) order -> fixed-length vector.
        return h.reshape(-1).astype(np.float64)

    def encode_batch(self, archs) -> np.ndarray:
        """Encode many architectures into an (N, 30) matrix."""
        return np.stack([self.encode(a) for a in archs])
