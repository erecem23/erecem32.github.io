"""Minimal CA-like projection utilities.

This module provides a lightweight approximation using TruncatedSVD. A full CA
implementation can be added in future versions.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


def project_first_axis(matrix: csr_matrix, random_state: int = 0) -> np.ndarray:
    """Project rows to one latent axis, as a pragmatic CA approximation."""
    if matrix.shape[0] == 0:
        return np.array([], dtype=float)
    if matrix.shape[0] == 1:
        return np.array([0.0], dtype=float)
    svd = TruncatedSVD(n_components=1, random_state=random_state)
    return svd.fit_transform(matrix).ravel()
