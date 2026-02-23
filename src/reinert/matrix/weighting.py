"""Weighting helpers for term matrices."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix


def relative_frequencies(matrix: csr_matrix) -> np.ndarray:
    """Compute row-normalized frequencies from a sparse count matrix."""
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    return matrix.toarray() / row_sums[:, None]
