# reinert-dhc

`reinert-dhc` is a lightweight Python implementation of Descending Hierarchical Classification (DHC, método Reinert) for text corpora, with a scikit-learn-like API.

## Installation

Base install:

```bash
pip install reinert-dhc
```

With spaCy support (optional lemmatization):

```bash
pip install "reinert-dhc[spacy]"
```

Spanish extras (spaCy + lookups):

```bash
pip install "reinert-dhc[es]"
```

For local development:

```bash
pip install -e .[dev]
```

> spaCy models are **not** auto-downloaded by this package. Install manually, e.g.:
>
> ```bash
> python -m spacy download es_core_news_sm
> ```

## Quickstart

```python
from reinert import ReinertDHC

texts = [
    "La economía crece con exportaciones y empleo.",
    "Mercados financieros y tasas de interés suben.",
    "El equipo ganó el partido con gran defensa.",
    "Fútbol, goles y entrenamiento semanal.",
]

model = ReinertDHC(min_cluster_size=2, min_df=1, max_df=1.0, random_state=42)
result = model.fit_transform(texts)

print(result["classes"])
print(model.get_keywords(class_id=1, top_k=10))
model.export_csv("outputs")
```

## Conceptual overview

DHC/Reinert iteratively splits a corpus of text segments into lexically distinct classes. In this implementation:

1. Texts are segmented (`sentence` or fixed windows).
2. A document-term matrix is built with `CountVectorizer`.
3. Each node is split into two groups using either:
   - `ca`: sign of the first SVD axis (CA-like approximation), or
   - `kmeans_init`: `KMeans(n_clusters=2)`.
4. A split is accepted if it satisfies size constraints and positive chi-square gain.
5. Keywords per class are computed via 2x2 chi-square tests (class vs rest), including p-values and a log(O/E) effect size.

Outputs include:

- `labels_`: class label for each segment.
- `segments_`: `doc_id`, `segment_id`, and segment text.
- `classes_`: class-level metadata (`size`, `chi2_gain`, `depth`).
- `keywords_`: characteristic terms with statistics.
- `vocab_`: fitted vocabulary.

## Development

Run checks:

```bash
ruff check .
pytest
python -m build
```

See `docs/tutorial.md` for a short guide.
