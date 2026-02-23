"""Split strategy implementations."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD


def split_kmeans_init(matrix: csr_matrix, random_state: int) -> np.ndarray:
    """Binary split with KMeans over sparse rows."""
    model = KMeans(n_clusters=2, n_init=10, random_state=random_state)
    return model.fit_predict(matrix)


def split_ca_projection(matrix: csr_matrix, random_state: int) -> np.ndarray:
    """Approximate CA split using sign of first SVD component."""
    if matrix.shape[0] < 2:
        return np.zeros(matrix.shape[0], dtype=int)

    n_components = 1
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    projection = svd.fit_transform(matrix).ravel()
    labels = (projection >= 0).astype(int)

    if labels.min() == labels.max():
        midpoint = matrix.shape[0] // 2
        labels = np.zeros(matrix.shape[0], dtype=int)
        labels[midpoint:] = 1
    return labels
