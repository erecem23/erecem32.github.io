import numpy as np

from reinert import ReinertDHC


def toy_corpus() -> list[str]:
    return [
        "economía mercado inversión banco finanzas",
        "inflación tasas banco crédito mercado",
        "empleo salario industria fábrica trabajo",
        "equipo fútbol gol estadio entrenador",
        "partido liga defensa ataque gol",
        "tenis saque cancha torneo jugador",
        "baloncesto equipo defensa rebote liga",
        "economía exportación industria comercio",
    ]


def test_fit_produces_multiple_classes() -> None:
    model = ReinertDHC(min_df=1, max_df=1.0, min_cluster_size=2, random_state=42)
    model.fit(toy_corpus())

    assert hasattr(model, "labels_")
    assert len(np.unique(model.labels_)) >= 2


def test_keywords_not_empty() -> None:
    model = ReinertDHC(min_df=1, max_df=1.0, min_cluster_size=2, random_state=42)
    model.fit(toy_corpus())

    assert not model.keywords_.empty
