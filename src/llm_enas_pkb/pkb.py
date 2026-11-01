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

from dataclasses import dataclass

import numpy as np

from .categorical import categorical_similarity_matrix, validate_architecture_vectors
from .search_space import NUM_EDGES


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


@dataclass
class PKBSelection:
    """Selected archive indices and diagnostics, kept out of the LLM examples."""

    indices: np.ndarray
    sources: list[str]
    mode: str
    coverage: dict
    global_coverage: dict | None = None


def validate_pkb_options(*, strategy, selected_size, global_size, coverage_neighbors,
                         alpha, k_perf, k_div) -> None:
    """Validate public selection options before search or direct selection."""
    if strategy not in ("original", "candidate_aware"):
        raise ValueError("pkb_strategy must be original or candidate_aware")
    for name, value, minimum in (
        ("selected_size", selected_size, 1), ("pkb_global_size", global_size, 0),
        ("pkb_coverage_neighbors", coverage_neighbors, 1),
        ("k_perf", k_perf, 0), ("k_div", k_div, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")


def _coverage_summary(similarities, selected, candidates, neighbors: int) -> dict:
    """Report nearest/top-k similarities, zero-padding missing reference slots."""
    if len(candidates) == 0:
        return {
            "candidate_count": 0, "neighbors": neighbors,
            "mean_nearest": None, "min_nearest": None,
            "mean_topk": None, "min_topk": None, "per_candidate": [],
        }
    nearest = np.zeros((len(candidates), neighbors), dtype=np.float64)
    count = min(neighbors, len(selected))
    if count:
        nearest[:, -count:] = np.sort(similarities[selected].T, axis=1)[:, -count:]
    best = nearest[:, -1]
    topk = nearest.mean(axis=1)
    return {
        "candidate_count": len(candidates), "neighbors": neighbors,
        "mean_nearest": float(best.mean()), "min_nearest": float(best.min()),
        "mean_topk": float(topk.mean()), "min_topk": float(topk.min()),
        "per_candidate": [
            {"architecture": code.tolist(), "nearest": float(first), "topk_mean": float(mean)}
            for code, first, mean in zip(candidates, best, topk)
        ],
    }


def select_pkb_for_candidates(
    arch_vectors: np.ndarray,
    perf_values: np.ndarray,
    candidate_vectors: np.ndarray,
    *,
    strategy: str = "candidate_aware",
    selected_size: int = 20,
    global_size: int = 6,
    coverage_neighbors: int = 2,
    alpha: float = 0.6,
    k_perf: int = 10,
    k_div: int = 12,
) -> PKBSelection:
    """Select global references, then greedily cover the unknown prediction batch.

    Examples remain architecture/truth pairs. Unknown candidates are de-duplicated
    and measured candidates receive no coverage weight. Global quotas rescale
    k_perf:k_div to global_size (10:12 becomes 3:3 for six slots). Remaining slots
    maximize the mean top-k candidate/reference similarity, padding missing
    neighbors with zeros. Ties favor novelty to the selected set, the original
    combined score, then the earlier archive index. Once coverage saturates,
    mean candidate similarity precedes those tie-breakers.

    Only categorical matches are used. Integer match counts make coverage ties
    exact rather than dependent on floating-point summation order. No RNG, new
    truth queries, predictions, or error cases are used by this selector.
    """
    validate_pkb_options(
        strategy=strategy, selected_size=selected_size, global_size=global_size,
        coverage_neighbors=coverage_neighbors, alpha=alpha, k_perf=k_perf, k_div=k_div,
    )
    arch_vectors = validate_architecture_vectors(arch_vectors)
    candidates = validate_architecture_vectors(candidate_vectors)
    values = np.asarray(perf_values, dtype=np.float64)
    if values.shape != (len(arch_vectors),) or not np.all(np.isfinite(values)):
        raise ValueError("Expected one finite performance value per archived architecture")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("Archived accuracy values must lie in [0, 1]")
    archive_codes = {tuple(row) for row in arch_vectors}
    if len(archive_codes) != len(arch_vectors):
        raise ValueError("Archived architectures must be unique")
    unknown = list(dict.fromkeys(tuple(row) for row in candidates if tuple(row) not in archive_codes))
    candidates = np.asarray(unknown, dtype=np.int64).reshape(-1, NUM_EDGES)
    sim = categorical_similarity_matrix(arch_vectors, candidates)

    if strategy == "original" or len(candidates) == 0:
        selected = select_pkb(arch_vectors, values, alpha=alpha, k_perf=k_perf,
                              k_div=k_div, selected_size=selected_size)
        return PKBSelection(
            selected, ["original"] * len(selected),
            "original" if strategy == "original" else "original_no_unknown_candidates",
            _coverage_summary(sim, selected, candidates, coverage_neighbors),
        )

    target_size = min(selected_size, len(arch_vectors))
    global_count = min(global_size, target_size)
    total_quota = k_perf + k_div
    perf_quota = int(np.floor(global_count * k_perf / total_quota + 0.5)) if total_quota else 0
    if k_perf and k_div and global_count >= 2:
        perf_quota = min(global_count - 1, max(1, perf_quota))
    div_quota = global_count - perf_quota if total_quota else 0
    selected = select_pkb(
        arch_vectors, values, alpha=alpha, k_perf=perf_quota,
        k_div=div_quota, selected_size=global_count,
    ).tolist()
    sources = ["global"] * len(selected)
    global_coverage = _coverage_summary(sim, selected, candidates, coverage_neighbors)
    if target_size == 0:
        return PKBSelection(np.array([], dtype=int), sources, "candidate_aware",
                            global_coverage, global_coverage)

    archive_sim = categorical_similarity_matrix(arch_vectors)
    score = alpha * normalize_performance(values) + (1 - alpha) * (1 - archive_sim.mean(axis=1))
    matches = np.rint(sim * NUM_EDGES).astype(np.int64)
    archive_matches = np.rint(archive_sim * NUM_EDGES).astype(np.int64)
    # Sorted ascending: column zero is the reference displaced by a better match.
    top = np.zeros((len(candidates), coverage_neighbors), dtype=np.int64)
    for idx in selected:
        top = np.sort(np.column_stack((top, matches[idx])), axis=1)[:, -coverage_neighbors:]
    available = [i for i in range(len(arch_vectors)) if i not in selected]
    while len(selected) < target_size:
        gains = np.maximum(matches[available] - top[:, 0], 0).sum(axis=1)
        max_gain = int(gains.max())
        pool = [i for i, gain in zip(available, gains) if gain == max_gain]

        def priority(i):
            novelty = NUM_EDGES - int(archive_matches[i, selected].max()) if selected else NUM_EDGES
            relevance = int(matches[i].sum()) if max_gain == 0 else 0
            return relevance, novelty, float(score[i]), -i

        chosen = max(pool, key=priority)
        selected.append(chosen)
        sources.append("coverage" if max_gain else "relevance_fill")
        available.remove(chosen)
        top = np.sort(np.column_stack((top, matches[chosen])), axis=1)[:, -coverage_neighbors:]

    indices = np.asarray(selected, dtype=int)
    return PKBSelection(indices, sources, "candidate_aware",
                        _coverage_summary(sim, indices, candidates, coverage_neighbors),
                        global_coverage)
