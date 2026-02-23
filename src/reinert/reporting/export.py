"""Export helpers for result tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_outputs(
    segments: pd.DataFrame,
    classes: pd.DataFrame,
    keywords: pd.DataFrame,
    outdir: str,
) -> None:
    """Write standard CSV outputs."""
    target = Path(outdir)
    target.mkdir(parents=True, exist_ok=True)
    segments.to_csv(target / "segments.csv", index=False)
    classes.to_csv(target / "classes.csv", index=False)
    keywords.to_csv(target / "keywords.csv", index=False)
