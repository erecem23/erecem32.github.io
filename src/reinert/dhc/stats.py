"""Statistical utilities for Reinert DHC."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import chi2_contingency


def split_chi2_gain(matrix: csr_matrix, mask: np.ndarray) -> float:
    """Aggregate chi-square gain for a binary partition across vocabulary terms."""
    if mask.sum() == 0 or mask.sum() == mask.size:
        return 0.0

    left = np.asarray(matrix[mask].sum(axis=0)).ravel()
    right = np.asarray(matrix[~mask].sum(axis=0)).ravel()

    gains = []
    for l_count, r_count in zip(left, right, strict=False):
        table = np.array(
            [[l_count, r_count], [left.sum() - l_count, right.sum() - r_count]]
        )
        if table.min() < 0 or table.sum() == 0:
            continue
        chi2_stat, _, _, _ = chi2_contingency(table, correction=False)
        gains.append(float(chi2_stat))
    return float(np.sum(gains))


def term_class_stats(
    matrix: csr_matrix,
    labels: np.ndarray,
    class_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute chi2, p-value and log(O/E) for each term in a class-vs-rest setup."""
    in_class = labels == class_id
    out_class = ~in_class

    class_counts = np.asarray(matrix[in_class].sum(axis=0)).ravel()
    rest_counts = np.asarray(matrix[out_class].sum(axis=0)).ravel()

    total_class = class_counts.sum()
    total_rest = rest_counts.sum()

    chi2_vals = np.zeros_like(class_counts, dtype=float)
    p_vals = np.ones_like(class_counts, dtype=float)
    effects = np.zeros_like(class_counts, dtype=float)

    for idx, (obs_in, obs_out) in enumerate(zip(class_counts, rest_counts, strict=False)):
        table = np.array([[obs_in, obs_out], [total_class - obs_in, total_rest - obs_out]])
        if table.min() < 0 or table.sum() == 0:
            continue
        chi2_stat, p_val, _, expected = chi2_contingency(table, correction=False)
        chi2_vals[idx] = float(chi2_stat)
        p_vals[idx] = float(p_val)
        expected_in = max(expected[0, 0], 1e-9)
        effects[idx] = float(np.log((obs_in + 1e-9) / expected_in))

    return chi2_vals, p_vals, effects
