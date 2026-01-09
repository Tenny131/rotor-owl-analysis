from __future__ import annotations

import csv

from rotor_owl.cli import main


def test_features_csv_export(tmp_path, mini_owl):
    owl_path = tmp_path / "mini.owl"
    owl_path.write_text(mini_owl, encoding="utf-8")

    out_csv = tmp_path / "features.csv"

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
    assert rc == 0
    assert out_csv.exists()

    with out_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))

    assert rows[0] == [
        "feature_name",
        "value",
        "unit",
        "type",
        "feature_iri",
        "feature_class_iri",
        "comment",
    ]
    assert len(rows) >= 2
    assert any("P_TEST_1" in r[4] for r in rows[1:])
