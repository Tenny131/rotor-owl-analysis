import math

from rotor_owl.gesamt_aehnlichkeit import berechne_topk_aehnlichkeiten
from rotor_owl.numerische_aehnlichkeit import build_numeric_stats


def test_gewichtete_gesamt_aehnlichkeit() -> None:
    """
    Prüft:
    - numerische Ähnlichkeit > 0
    - kategoriale Ähnlichkeit = 1 bei Gleichheit
    - korrekte Gewichtung im Gesamtscore
    """

    features_by_rotor = {
        "Rotor_A": {
            "params": {
                ("C_WELLE", "P_WELLE_PAKETSITZ"): {
                    "value": 150.0,
                    "unit": "mm",
                    "ptype": "GEOM",
                },
                ("C_WELLE", "P_WELLE_MATERIAL"): {
                    "value": "42CrMo4",
                    "unit": "-",
                    "ptype": "MTRL",
                },
            }
        },
        "Rotor_B": {
            "params": {
                ("C_WELLE", "P_WELLE_PAKETSITZ"): {
                    "value": 160.0,
                    "unit": "mm",
                    "ptype": "GEOM",
                },
                ("C_WELLE", "P_WELLE_MATERIAL"): {
                    "value": "42CrMo4",
                    "unit": "-",
                    "ptype": "MTRL",
                },
            }
        },
    }

    rotor_ids = ["Rotor_A", "Rotor_B"]

    numerische_statistik = build_numeric_stats(features_by_rotor)

    gewichtung = {
        "GEOM": 2.0,
        "MTRL": 1.0,
        "UNKNOWN": 0.0,
    }

    ergebnisse = berechne_topk_aehnlichkeiten(
        query_rotor_id="Rotor_A",
        rotor_ids=rotor_ids,
        features_by_rotor=features_by_rotor,
        stats=numerische_statistik,
        gewichtung_pro_typ=gewichtung,
        k=1,
    )

    rotor_id, gesamt_sim, similarity_pro_typ = ergebnisse[0]

    assert rotor_id == "Rotor_B"
    assert math.isclose(similarity_pro_typ["GEOM"], 0.9, rel_tol=1e-9)
    assert math.isclose(similarity_pro_typ["MTRL"], 1.0, rel_tol=1e-9)

    # (2*0.9 + 1*1.0) / 3 = 0.9333...
    assert math.isclose(gesamt_sim, 0.9333333333333333, rel_tol=1e-9)
