"""Population knowledge base selection (Sec. 3.3, Algorithm 2).

Given the historical archive ``{(x_i, V_i)}`` of architecture vectors and their
true performance values, the PKB selects a compact, representative and diverse
subset to inject into the LLM prompt:

  * normalize performance to [0, 1]                       (Eq. 3)
  * categorical similarity: fraction of matching edge operations
  * combined score = alpha * perf + (1-alpha) * diversity (Eq. 4)
  * crowding distance over the (perf, diversity) plane    (Eq. 5)
  * merge top-K_perf (score) and top-K_div (crowding), fill to ``selected_size``.
"""

from __future__ import annotations

import numpy as np

from .categorical import categorical_similarity_matrix, validate_architecture_vectors


def normalize_performance(values: np.ndarray) -> np.ndarray:
    """Min-max normalise performance to [0, 1] (Eq. 3).

    Returns zeros when the range is degenerate (all values identical) so the
    downstream score does not divide by zero.
    """
    values = np.asarray(values, dtype=np.float64)
    vmin, vmax = values.min(), values.max()
    if vmax - vmin < 1e-12:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def crowding_distance(perf: np.ndarray, div: np.ndarray) -> np.ndarray:
    """NSGA-II style crowding distance over the two objectives ``perf`` and ``div``.

    For each objective: sort the individuals, assign +inf to the boundary points,
    and give each interior point the normalised distance between its two sorted
    neighbours, summed across both objectives (Eq. 5).  Higher distance = more
    isolated = more "diverse" in the (perf, diversity) plane.
    """
    n = len(perf)
    dist = np.zeros(n, dtype=np.float64)
    for obj in (np.asarray(perf, dtype=np.float64), np.asarray(div, dtype=np.float64)):
        order = np.argsort(obj)
        o = obj[order]
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        denom = o[-1] - o[0]
        if denom < 1e-12:
            continue
        dist[order[1:-1]] += (o[2:] - o[:-2]) / denom
    return dist


def select_pkb(
    arch_vectors: np.ndarray,
    perf_values: np.ndarray,
    *,
    alpha: float = 0.6,
    k_perf: int = 10,
    k_div: int = 12,
    selected_size: int = 20,
) -> np.ndarray:
    """Return indices (into ``arch_vectors``) of the selected PKB individuals.

    ``arch_vectors`` contain six categorical operation IDs per architecture.
    Diversity depends on matching positions, never on numeric ID magnitudes.
    """
    arch_vectors = validate_architecture_vectors(arch_vectors)
    perf_values = np.asarray(perf_values, dtype=np.float64)
    if perf_values.ndim != 1 or len(arch_vectors) != len(perf_values):
        raise ValueError("Expected one performance value per architecture")
    n = len(perf_values)
    if n == 0:
        return np.array([], dtype=int)
    # Clamp the selection sizes to the available pool size.
    selected_size = min(selected_size, n)
    k_perf = min(k_perf, n)
    k_div = min(k_div, n)

    sim = categorical_similarity_matrix(arch_vectors)
    perf_norm = normalize_performance(perf_values)  # Eq. (3)
    # Diversity of individual i = 1 - its mean similarity to the rest (Eq. 4 term).
    diversity = 1.0 - sim.mean(axis=1)  # Eq. (4) diversity term
    score = alpha * perf_norm + (1.0 - alpha) * diversity  # Eq. (4)
    crowd = crowding_distance(perf_norm, diversity)  # Eq. (5)

    # Top-K_perf by combined score, top-K_div by crowding distance.
    top_perf = np.argsort(-score)[:k_perf]
    top_div = np.argsort(-crowd)[:k_div]
    global_sorted = np.argsort(-score)  # fallback ordering used to fill the set

    # Merge the two top-K sets (dedup, preserving order), then fill any remaining
    # slots from the globally-best-scoring individuals.
    selected: list[int] = []
    for idx in np.concatenate([top_perf, top_div]):
        if int(idx) not in selected:
            selected.append(int(idx))
    if len(selected) < selected_size:
        for idx in global_sorted:
            if int(idx) not in selected:
                selected.append(int(idx))
            if len(selected) >= selected_size:
                break
    return np.asarray(selected[:selected_size], dtype=int)
