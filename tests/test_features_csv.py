from __future__ import annotations

import csv
from rotor_owl.cli import main


def test_features_csv_export(tmp_path, mini_owl):
    # Arrange: OWL-Datei schreiben
    owl_path = tmp_path / "mini.owl"
    owl_path.write_text(mini_owl, encoding="utf-8")

    out_csv = tmp_path / "features.csv"

    # Act: CLI ausführen
    rc = main(
        [
            "features",
            str(owl_path),
            "--assembly-iri",
            "http://ontology.innomotics.net/ims#C_WELLE_1",
            "--out",
            str(out_csv),
        ]
    )

    # Assert: Rückgabecode & Datei
    assert rc == 0
    assert out_csv.is_file()

    # Assert: CSV-Struktur
    with out_csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))

    # mindestens Header + eine Datenzeile
    assert len(rows) > 1

    header = rows[0]
    assert header == [
        "feature_name",
        "value",
        "unit",
        "type",
        "feature_iri",
        "feature_class_iri",
        "comment",
    ]
