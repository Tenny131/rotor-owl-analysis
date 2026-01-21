# tests/test_gesamt_aehnlichkeit.py
import math

from rotor_owl.methoden.regelbasierte_aehnlichkeit import berechne_topk_aehnlichkeiten


def test_gewichtete_gesamt_aehnlichkeit() -> None:
    features_by_rotor = {
        "Rotor_A": {
            "params": {
                ("C_WELLE", "P_WELLE_PAKETSITZ"): {"value": 150.0, "unit": "mm", "ptype": "GEOM"},
                ("C_WELLE", "P_WELLE_MATERIAL"): {"value": "42CrMo4", "unit": "-", "ptype": "MTRL"},
            }
        },
        "Rotor_B": {
            "params": {
                ("C_WELLE", "P_WELLE_PAKETSITZ"): {"value": 160.0, "unit": "mm", "ptype": "GEOM"},
                ("C_WELLE", "P_WELLE_MATERIAL"): {"value": "42CrMo4", "unit": "-", "ptype": "MTRL"},
            }
        },
    }

    rotor_ids = ["Rotor_A", "Rotor_B"]

    # WICHTIG: Stats so wählen, dass sim = 1 - 10/100 = 0.9
    parameter_schluessel = ("C_WELLE", "P_WELLE_PAKETSITZ")
    numerische_statistik = {parameter_schluessel: (100.0, 200.0)}

    # Gewichtung nach 3er-Kategorien (nicht nach ptypes!)
    gewichtung = {"K_GEOM_MECH": 2.0, "K_MTRL_PROC": 1.0, "K_REQ_ELEC": 0.0}

    ergebnisse = berechne_topk_aehnlichkeiten(
        query_rotor_id="Rotor_A",
        rotor_ids=rotor_ids,
        features_by_rotor=features_by_rotor,
        stats=numerische_statistik,  # type: ignore
        gewichtung_pro_kategorie=gewichtung,
        top_k=1,
    )

    rotor_id, gesamt_sim, similarity_pro_kat = ergebnisse[0]

    assert rotor_id == "Rotor_B"
    # GEOM -> K_GEOM_MECH
    assert "K_GEOM_MECH" in similarity_pro_kat
    assert math.isclose(similarity_pro_kat["K_GEOM_MECH"], 0.9, rel_tol=1e-9)
    # MTRL -> K_MTRL_PROC
    assert "K_MTRL_PROC" in similarity_pro_kat
    assert math.isclose(similarity_pro_kat["K_MTRL_PROC"], 1.0, rel_tol=1e-9)

    # gesamt = (2*0.9 + 1*1.0)/(2+1) = 0.933333...
    assert math.isclose(gesamt_sim, 0.9333333333333333, rel_tol=1e-9)
