"""Tokenization and optional lemmatization helpers."""

from __future__ import annotations

from collections.abc import Callable


def _spacy_tokenizer(model_name: str) -> Callable[[str], list[str]]:
    try:
        import spacy
    except ImportError as exc:  # pragma: no cover - exercised only when spacy missing
        raise ImportError(
            "spaCy is not installed. Use pip install 'reinert-dhc[spacy]' or '[es]'."
        ) from exc

    nlp = spacy.load(model_name, disable=["ner", "textcat", "parser"])

    def lemmatize(text: str) -> list[str]:
        doc = nlp(text)
        return [token.lemma_.lower() for token in doc if token.is_alpha and token.lemma_]

    return lemmatize


def tokenize_texts(
    texts: list[str],
    lemmatizer: str | Callable[[str], list[str]] | None = None,
    spacy_model: str = "es_core_news_sm",
) -> list[str]:
    """Return processed texts ready for vectorization."""
    if lemmatizer is None:
        return texts

    fn: Callable[[str], list[str]]
    if lemmatizer == "spacy":
        fn = _spacy_tokenizer(spacy_model)
    elif callable(lemmatizer):
        fn = lemmatizer
    else:
        raise ValueError("lemmatizer must be None, 'spacy' or callable")

    return [" ".join(fn(text)) for text in texts]
