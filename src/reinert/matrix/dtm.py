"""Document-term matrix utilities."""

from __future__ import annotations

from collections.abc import Iterable

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer


def build_dtm(
    texts: list[str],
    *,
    min_df: int | float = 3,
    max_df: int | float = 0.9,
    stopwords: str | Iterable[str] | None = None,
) -> tuple[csr_matrix, CountVectorizer, list[str]]:
    """Build sparse count matrix and vocabulary."""
    vectorizer = CountVectorizer(min_df=min_df, max_df=max_df, stop_words=stopwords)
    matrix = vectorizer.fit_transform(texts)
    vocab = vectorizer.get_feature_names_out().tolist()
    return matrix.tocsr(), vectorizer, vocab
