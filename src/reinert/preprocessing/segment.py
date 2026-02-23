"""Text segmentation methods for DHC."""

from __future__ import annotations

import re
from collections.abc import Callable

_SENTENCE_SPLIT_RE = re.compile(r"[\.\?\!;]+")


def _sentence_segments(text: str) -> list[str]:
    return [chunk.strip() for chunk in _SENTENCE_SPLIT_RE.split(text) if chunk.strip()]


def _fixed_segments(text: str, segment_size: int) -> list[str]:
    tokens = text.split()
    if segment_size <= 0:
        raise ValueError("segment_size must be > 0 for fixed segmentation")
    output: list[str] = []
    for i in range(0, len(tokens), segment_size):
        window = tokens[i : i + segment_size]
        if window:
            output.append(" ".join(window))
    return output


def segment_text(
    text: str,
    *,
    mode: str | Callable[[str], list[str]] = "sentence",
    segment_size: int = 40,
) -> list[str]:
    """Segment text using sentence boundaries, fixed token windows, or a custom callable."""
    if callable(mode):
        segments = mode(text)
        return [segment.strip() for segment in segments if segment.strip()]
    if mode == "sentence":
        return _sentence_segments(text)
    if mode == "fixed":
        return _fixed_segments(text, segment_size)
    raise ValueError(f"Unsupported segmenter mode: {mode}")
