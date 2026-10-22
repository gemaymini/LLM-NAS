"""NAS-Bench-201 search-space primitives.

NAS-Bench-201 is a cell-based search space.  A cell contains four nodes (node 0
is the input / stem, nodes 1-3 are intermediate feature maps) connected by six
*directed* edges.  Each edge selects one of five candidate operations, giving
5^6 = 15,625 architectures in total.

Internally an architecture is a 6-tuple of operation indices in *canonical edge
order* (node 1's input first, then node 2's, then node 3's):

    arch = (op_01, op_02, op_12, op_03, op_13, op_23)

matching the official string encoding ``|op~0|+|op~0|op~1|+|op~0|op~1|op~2|``.
"""

from __future__ import annotations

import numpy as np

# Candidate operations (index -> name).  Note ``none`` (index 0) is the zeroise
# operation that effectively removes the edge, which gives the space its name.
OPS: tuple[str, ...] = (
    "none",
    "skip_connect",
    "nor_conv_1x1",
    "nor_conv_3x3",
    "avg_pool_3x3",
)
NUM_OPS: int = len(OPS)

# Directed edges of the cell, in the canonical (arch-string) order.
EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),              # node 1 <- node 0
    (0, 2), (1, 2),      # node 2 <- nodes 0, 1
    (0, 3), (1, 3), (2, 3),  # node 3 <- nodes 0, 1, 2
)
NUM_EDGES: int = len(EDGES)
NUM_NODES: int = 4  # node 0 = input, nodes 1-3 = intermediate (output is implicit)

# Encoder input dimension d = NUM_EDGES * NUM_OPS = 30 (matches Table 1).
INPUT_DIM: int = NUM_EDGES * NUM_OPS


def random_arch(rng: np.random.Generator) -> tuple[int, ...]:
    """Sample a uniformly random architecture (6-tuple of op indices)."""
    return tuple(int(i) for i in rng.integers(0, NUM_OPS, size=NUM_EDGES))


def is_valid_arch(arch) -> bool:
    """Return True if ``arch`` is a well-formed 6-tuple of valid op indices."""
    return (
        isinstance(arch, (tuple, list))
        and len(arch) == NUM_EDGES
        and all(isinstance(i, (int, np.integer)) and 0 <= i < NUM_OPS for i in arch)
    )


def arch_to_str(arch: tuple[int, ...]) -> str:
    """Convert to the official NAS-Bench-201 string encoding (``|op~i|+...``)."""
    ops = [OPS[i] for i in arch]
    return (
        f"|{ops[0]}~0|"
        f"+|{ops[1]}~0|{ops[2]}~1|"
        f"+|{ops[3]}~0|{ops[4]}~1|{ops[5]}~2|"
    )


def str_to_arch(arch_str: str) -> tuple[int, ...]:
    """Parse the official NAS-Bench-201 string back into the 6-tuple."""
    ops: list[int] = []
    # The string is ``|op~0|+|op~0|op~1|+...``: split on ``+`` to get node blocks,
    # then on ``|`` within each block, and finally on ``~`` to get (op, node).
    for node_str in arch_str.split("+"):
        for tok in filter(None, node_str.split("|")):
            op_name, _node = tok.split("~")
            ops.append(op_index_from_name(op_name))
    if len(ops) != NUM_EDGES:
        raise ValueError(f"Expected {NUM_EDGES} ops, got {len(ops)} from {arch_str!r}")
    return tuple(ops)


def one_hot_ops(arch: tuple[int, ...]) -> np.ndarray:
    """One-hot operator matrix ``O`` of shape (NUM_EDGES, NUM_OPS) (Eq. 6 input)."""
    oh = np.zeros((NUM_EDGES, NUM_OPS), dtype=np.float64)
    oh[np.arange(NUM_EDGES), list(arch)] = 1.0
    return oh


def slot_adjacency() -> np.ndarray:
    """Adjacency matrix over the six operation *slots* (line graph).

    Two slots are adjacent iff their edges share a vertex in the cell DAG.  This
    is the graph over which the Eq. (7) neighbourhood aggregation runs; treating
    each operation slot as a node yields exactly ``NUM_EDGES x NUM_OPS = 30``
    features, the input dimension ``d`` reported in Table 1.
    """
    adj = np.zeros((NUM_EDGES, NUM_EDGES), dtype=np.float64)
    for i, (a, b) in enumerate(EDGES):
        for j, (c, d) in enumerate(EDGES):
            if i != j and ({a, b} & {c, d}):
                adj[i, j] = 1.0
    return adj


def op_index_from_name(name: str) -> int:
    return OPS.index(name)
