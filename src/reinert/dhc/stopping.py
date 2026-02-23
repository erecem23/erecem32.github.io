"""Stopping criteria utilities."""

from __future__ import annotations


def should_stop(size: int, depth: int, min_cluster_size: int, max_depth: int | None) -> bool:
    """Stop split recursion if depth or cluster size limits are reached."""
    if size < (2 * min_cluster_size):
        return True
    if max_depth is not None and depth >= max_depth:
        return True
    return False
