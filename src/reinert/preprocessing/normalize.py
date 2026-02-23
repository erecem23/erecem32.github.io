"""Text normalization utilities."""

from __future__ import annotations

import re

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize whitespace and lowercase text."""
    cleaned = _WHITESPACE_RE.sub(" ", text.strip())
    return cleaned.lower()
