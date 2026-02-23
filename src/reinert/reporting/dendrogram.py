"""Dendrogram placeholders for future visualization API."""

from __future__ import annotations

import pandas as pd


def to_dendrogram_table(classes: pd.DataFrame) -> pd.DataFrame:
    """Return class table as a simple dendrogram-compatible structure."""
    cols = [col for col in ["class_id", "size", "depth", "chi2_gain"] if col in classes.columns]
    return classes[cols].copy()
