from __future__ import annotations

import csv
from pathlib import Path

from rotor_owl.cli import main


def test_generate_writes_csv_and_is_reproducible(tmp_path: Path):
    # Beispiel Parameter-CSV
    ref = tmp_path / "parameters.csv"
    ref.write_text(
        "Parameter_ID,Component_ID,Name,ParamType_ID,DataType,Unit,EnumDomain,Value,Definition\n"
        "P_A,C_WELLE,Durchmesser,GEOM,numeric,mm,,,x\n"
        'P_B,C_WELLE,Material,MTRL,enum,-,"{42CrMo4, C45}",,x\n'
        'P_C,C_WELLE,Kuehlung,REQ,boolean,-,"{ja, nein}",,x\n',
        encoding="utf-8",
    )

    outdir = tmp_path / "out"

    # Run 1
    rc1 = main(
        [
            "generate",
            "--n",
            "2",
            "--seed",
            "123",
            "--out",
            str(outdir),
            "--parameters-csv",
            str(ref),
            "--missing-rate",
            "0",
        ]
    )
    assert rc1 == 0
    csv_path = outdir / "instances.csv"
    assert csv_path.exists()

    content1 = csv_path.read_text(encoding="utf-8")

    # Zweiter Lauf mit gleichen Parametern sollte identische Datei erzeugen
    rc2 = main(
        [
            "generate",
            "--n",
            "2",
            "--seed",
            "123",
            "--out",
            str(outdir),
            "--parameters-csv",
            str(ref),
            "--missing-rate",
            "0",
        ]
    )
    assert rc2 == 0

    content2 = csv_path.read_text(encoding="utf-8")
    assert content1 == content2

    # Inhaltliche Prüfung der CSV
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))

    assert rows[0] == [
        "Design_ID",
        "Component_ID",
        "Parameter_ID",
        "ParamType_ID",
        "DataType",
        "Unit",
        "Value",
        "IsMissing",
    ]
    # 2 Designs x 3 Parameter
    assert len(rows) == 1 + 2 * 3
