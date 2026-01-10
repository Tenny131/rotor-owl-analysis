from __future__ import annotations

from pathlib import Path

from rotor_owl.similarity import top_k_jaccard


def test_top_k_jaccard_unweighted(tmp_path: Path):
    p = tmp_path / "instances.csv"
    p.write_text(
        "Design_ID,Component_ID,Parameter_ID,ParamType_ID,DataType,Unit,Value,IsMissing\n"
        # Query
        "D001,C_X,P_A,GEOM,numeric,mm,10,0\n"
        "D001,C_X,P_B,MTRL,enum,-,C45,0\n"
        # Identisch -> similarity 1.0
        "D002,C_X,P_A,GEOM,numeric,mm,10,0\n"
        "D002,C_X,P_B,MTRL,enum,-,C45,0\n"
        # Teilweise ähnlich -> Jaccard = 1/3
        "D003,C_X,P_B,MTRL,enum,-,C45,0\n"
        "D003,C_X,P_C,REQ,boolean,-,ja,0\n"
        # Keine Features (alles missing)
        "D004,C_X,P_A,GEOM,numeric,mm,,1\n",
        encoding="utf-8",
    )

    top = top_k_jaccard(p, "D001", k=3)
    assert top[0][0] == "D002"
    assert abs(top[0][1] - 1.0) < 1e-9

    # D003 sollte vor D004 kommen (D004 hat leere Feature-Menge)
    assert top[1][0] == "D003"
    assert abs(top[1][1] - (1 / 3)) < 1e-9


def test_top_k_jaccard_weighted(tmp_path: Path):
    p = tmp_path / "instances.csv"
    p.write_text(
        "Design_ID,Component_ID,Parameter_ID,ParamType_ID,DataType,Unit,Value,IsMissing\n"
        "D001,C_X,P_A,GEOM,numeric,mm,10,0\n"
        "D001,C_X,P_B,REQ,boolean,-,ja,0\n"
        "D002,C_X,P_A,GEOM,numeric,mm,10,0\n"
        "D002,C_X,P_B,REQ,boolean,-,ja,0\n"
        "D003,C_X,P_A,GEOM,numeric,mm,10,0\n"
        "D003,C_X,P_C,GEOM,numeric,mm,99,0\n",
        encoding="utf-8",
    )

    weights = {"GEOM": 1.0, "REQ": 0.1}
    top = top_k_jaccard(p, "D001", k=2, paramtype_weights=weights)

    assert top[0][0] == "D002"
    assert abs(top[0][1] - 1.0) < 1e-9
