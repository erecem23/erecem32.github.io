# Tutorial breve

## 1) Preparar entorno

```bash
pip install -e .[dev]
```

Opcional con spaCy:

```bash
pip install "reinert-dhc[es]"
python -m spacy download es_core_news_sm
```

## 2) Entrenar un modelo

```python
from reinert import ReinertDHC

texts = [
    "Economía y empleo formal en crecimiento.",
    "Inflación y bancos centrales ajustan tasas.",
    "Entrenamiento, goles y campeonato local.",
    "Defensa sólida y victoria del equipo.",
]

dhc = ReinertDHC(
    segmenter="sentence",
    min_df=1,
    max_df=1.0,
    min_cluster_size=2,
    splitter="ca",
)
dhc.fit(texts)

print(dhc.classes_)
print(dhc.get_keywords(1, top_k=10))
```

## 3) Exportar resultados

```python
dhc.export_csv("./out")
```

Se generan:
- `out/segments.csv`
- `out/classes.csv`
- `out/keywords.csv`
