"""Validation and equality-based similarity for categorical architecture IDs."""

from __future__ import annotations

import numpy as np

from .search_space import NUM_EDGES, NUM_OPS, is_valid_arch


def validate_architecture_vectors(vectors) -> np.ndarray:
    """Return an (N, 6) integer matrix without interpreting IDs as magnitudes."""
    # Preserve Python booleans and floats in mixed lists until they have been
    # checked; a normal np.asarray could silently turn True into integer ID 1.
    vectors = np.asarray(vectors, dtype=object)
    if vectors.ndim == 1 and vectors.shape == (0,):
        return np.empty((0, NUM_EDGES), dtype=np.int64)
    if vectors.ndim != 2 or vectors.shape[1] != NUM_EDGES:
        raise ValueError(f"Architecture vectors must have shape (N, {NUM_EDGES})")
    if any(not is_valid_arch(row) for row in vectors):
        raise ValueError(
            f"Architecture operation IDs must be integers in [0, {NUM_OPS - 1}], "
            "not floats or booleans"
        )
    return vectors.astype(np.int64, copy=False)


def categorical_similarity_matrix(a, b=None) -> np.ndarray:
    """Return the fraction of identical edge operations for each pair of rows.

    This is one minus normalised Hamming distance. IDs represent categories:
    changing an operation from 0 to 1 is no closer than changing it from 0 to 4.
    The all-zero architecture is valid and has similarity 1 with itself.
    """
    a = validate_architecture_vectors(a)
    b = a if b is None else validate_architecture_vectors(b)
    return (a[:, None, :] == b[None, :, :]).mean(axis=2)
