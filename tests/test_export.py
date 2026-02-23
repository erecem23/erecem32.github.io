from pathlib import Path

from reinert import ReinertDHC


def test_export_csv_creates_files(tmp_path: Path) -> None:
    texts = [
        "economía mercado finanzas",
        "banco crédito inversión",
        "equipo gol partido",
        "liga torneo defensa",
    ]
    model = ReinertDHC(min_df=1, max_df=1.0, min_cluster_size=2, random_state=0)
    model.fit(texts)

    model.export_csv(str(tmp_path))

    assert (tmp_path / "segments.csv").exists()
    assert (tmp_path / "classes.csv").exists()
    assert (tmp_path / "keywords.csv").exists()
