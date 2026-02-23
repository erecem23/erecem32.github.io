"""Main Reinert DHC estimator with scikit-learn-like API."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from reinert.dhc.split import split_ca_projection, split_kmeans_init
from reinert.dhc.stats import split_chi2_gain
from reinert.dhc.stopping import should_stop
from reinert.matrix.dtm import build_dtm
from reinert.preprocessing.normalize import normalize_text
from reinert.preprocessing.segment import segment_text
from reinert.preprocessing.tokenize import tokenize_texts
from reinert.reporting.export import export_outputs
from reinert.reporting.keywords import build_keywords_df


@dataclass
class _ClusterNode:
    indices: np.ndarray
    depth: int


class ReinertDHC:
    """Descending Hierarchical Classification (Reinert method), v1 implementation."""

    def __init__(
        self,
        segmenter: str | Callable[[str], list[str]] = "sentence",
        segment_size: int = 40,
        min_df: int | float = 3,
        max_df: int | float = 0.9,
        stopwords: str | list[str] | None = "es",
        lemmatizer: str | Callable[[str], list[str]] | None = None,
        spacy_model: str = "es_core_news_sm",
        max_depth: int | None = None,
        min_cluster_size: int = 20,
        random_state: int = 0,
        splitter: str = "ca",
        criterion: str = "chi2_gain",
        alpha: float = 0.05,
    ) -> None:
        self.segmenter = segmenter
        self.segment_size = segment_size
        self.min_df = min_df
        self.max_df = max_df
        self.stopwords = stopwords
        self.lemmatizer = lemmatizer
        self.spacy_model = spacy_model
        self.max_depth = max_depth
        self.min_cluster_size = min_cluster_size
        self.random_state = random_state
        self.splitter = splitter
        self.criterion = criterion
        self.alpha = alpha

    def _segment_corpus(self, texts: list[str], doc_ids: list[str] | None) -> pd.DataFrame:
        rows: list[dict[str, str]] = []
        ids = doc_ids or [f"doc_{idx}" for idx in range(len(texts))]
        for doc_id, text in zip(ids, texts, strict=True):
            normalized = normalize_text(text)
            chunks = segment_text(normalized, mode=self.segmenter, segment_size=self.segment_size)
            for s_idx, chunk in enumerate(chunks):
                rows.append(
                    {
                        "doc_id": str(doc_id),
                        "segment_id": f"{doc_id}_{s_idx}",
                        "text": chunk,
                    }
                )
        return pd.DataFrame(rows)

    def _compute_split_labels(self, matrix: csr_matrix) -> np.ndarray:
        if self.splitter == "kmeans_init":
            return split_kmeans_init(matrix, random_state=self.random_state)
        if self.splitter == "ca":
            return split_ca_projection(matrix, random_state=self.random_state)
        raise ValueError(f"Unsupported splitter: {self.splitter}")

    def fit(self, texts: list[str], *, doc_ids: list[str] | None = None) -> ReinertDHC:
        """Fit DHC over text corpus."""
        if not texts:
            raise ValueError("texts cannot be empty")

        self.segments_ = self._segment_corpus(texts, doc_ids)
        processed = tokenize_texts(
            self.segments_["text"].tolist(),
            lemmatizer=self.lemmatizer,
            spacy_model=self.spacy_model,
        )
        self._dtm, self._vectorizer, self.vocab_ = build_dtm(
            processed,
            min_df=self.min_df,
            max_df=self.max_df,
            stopwords=self.stopwords,
        )

        n_segments = self._dtm.shape[0]
        labels = np.zeros(n_segments, dtype=int)
        classes: list[dict[str, float | int]] = []
        cluster_queue = [_ClusterNode(indices=np.arange(n_segments), depth=0)]
        next_class_id = 1

        while cluster_queue:
            node = cluster_queue.pop(0)
            if should_stop(
                size=node.indices.size,
                depth=node.depth,
                min_cluster_size=self.min_cluster_size,
                max_depth=self.max_depth,
            ):
                labels[node.indices] = next_class_id
                classes.append(
                    {
                        "class_id": next_class_id,
                        "size": int(node.indices.size),
                        "chi2_gain": 0.0,
                        "depth": int(node.depth),
                    }
                )
                next_class_id += 1
                continue

            sub = self._dtm[node.indices]
            local_split = self._compute_split_labels(sub)
            left_mask = local_split == 0

            if (
                left_mask.sum() < self.min_cluster_size
                or (~left_mask).sum() < self.min_cluster_size
            ):
                labels[node.indices] = next_class_id
                classes.append(
                    {
                        "class_id": next_class_id,
                        "size": int(node.indices.size),
                        "chi2_gain": 0.0,
                        "depth": int(node.depth),
                    }
                )
                next_class_id += 1
                continue

            gain = split_chi2_gain(sub, left_mask)
            if self.criterion == "chi2_gain" and gain <= 0:
                labels[node.indices] = next_class_id
                classes.append(
                    {
                        "class_id": next_class_id,
                        "size": int(node.indices.size),
                        "chi2_gain": float(gain),
                        "depth": int(node.depth),
                    }
                )
                next_class_id += 1
                continue

            left_indices = node.indices[left_mask]
            right_indices = node.indices[~left_mask]
            cluster_queue.append(_ClusterNode(indices=left_indices, depth=node.depth + 1))
            cluster_queue.append(_ClusterNode(indices=right_indices, depth=node.depth + 1))

        self.labels_ = labels
        self.classes_ = pd.DataFrame(classes).sort_values("class_id").reset_index(drop=True)
        self.keywords_ = build_keywords_df(self._dtm, self.labels_, self.vocab_, alpha=self.alpha)
        return self

    def fit_transform(
        self,
        texts: list[str],
        *,
        doc_ids: list[str] | None = None,
    ) -> dict[str, object]:
        """Fit and return labels and report tables."""
        self.fit(texts, doc_ids=doc_ids)
        return {
            "labels": self.labels_,
            "segments": self.segments_,
            "classes": self.classes_,
            "keywords": self.keywords_,
        }

    def get_keywords(self, class_id: int, top_k: int = 30) -> pd.DataFrame:
        """Return top keywords for a class."""
        if not hasattr(self, "keywords_"):
            raise ValueError("Model is not fitted")
        return (
            self.keywords_[self.keywords_["class_id"] == class_id]
            .sort_values("chi2", ascending=False)
            .head(top_k)
            .reset_index(drop=True)
        )

    def export_csv(self, outdir: str) -> None:
        """Export segments, classes and keywords CSV files."""
        if not all(hasattr(self, attr) for attr in ["segments_", "classes_", "keywords_"]):
            raise ValueError("Model is not fitted")
        export_outputs(self.segments_, self.classes_, self.keywords_, outdir)
