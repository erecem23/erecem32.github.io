"""Keyword extraction for lexical classes."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from reinert.dhc.stats import term_class_stats


def build_keywords_df(
    matrix: csr_matrix,
    labels: np.ndarray,
    vocab: list[str],
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Build keyword table with chi2, p-value and log(O/E) effect for each class."""
    rows: list[dict[str, float | int | str]] = []

    for class_id in np.unique(labels):
        chi2_vals, p_vals, effects = term_class_stats(matrix, labels, int(class_id))
        for term, chi2_val, p_val, effect in zip(
            vocab,
            chi2_vals,
            p_vals,
            effects,
            strict=False,
        ):
            if p_val < alpha and chi2_val > 0:
                rows.append(
                    {
                        "term": term,
                        "class_id": int(class_id),
                        "chi2": float(chi2_val),
                        "p": float(p_val),
                        "effect": float(effect),
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["term", "class_id", "chi2", "p", "effect"])

    return (
        pd.DataFrame(rows)
        .sort_values(["class_id", "chi2"], ascending=[True, False])
        .reset_index(drop=True)
    )
