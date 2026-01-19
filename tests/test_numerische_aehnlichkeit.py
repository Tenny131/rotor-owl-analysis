"""Tests für feature_fetcher.py (build_numeric_stats) und regelbasierte_aehnlichkeit.py"""

import pytest

from rotor_owl.daten.feature_fetcher import build_numeric_stats
from rotor_owl.methoden.regelbasierte_aehnlichkeit import berechne_numerische_parameter_aehnlichkeit


def test_numerische_similarity_identisch() -> None:
    """Identische Werte sollten Similarity 1.0 haben."""
    stats = {("C_WELLE", "P_DURCHMESSER"): (100.0, 200.0)}

    sim = berechne_numerische_parameter_aehnlichkeit(
        wert_a=150.0,
        wert_b=150.0,
        parameter_schluessel=("C_WELLE", "P_DURCHMESSER"),
        stats=stats,
    )

    assert sim == pytest.approx(1.0, abs=1e-9)


def test_numerische_similarity_maximal_unterschiedlich() -> None:
    """Werte an den Grenzen sollten Similarity 0.0 haben."""
    stats = {("C_WELLE", "P_DURCHMESSER"): (100.0, 200.0)}

    sim = berechne_numerische_parameter_aehnlichkeit(
        wert_a=100.0,
        wert_b=200.0,
        parameter_schluessel=("C_WELLE", "P_DURCHMESSER"),
        stats=stats,
    )

    assert sim == pytest.approx(0.0, abs=1e-9)


def test_numerische_similarity_mittlere_distanz() -> None:
    """Test mit Werten in der Mitte des Bereichs."""
    stats = {("C_WELLE", "P_DURCHMESSER"): (100.0, 200.0)}

    # Distanz = |150 - 160| = 10, Range = 100, Similarity = 1 - 10/100 = 0.9
    sim = berechne_numerische_parameter_aehnlichkeit(
        wert_a=150.0,
        wert_b=160.0,
        parameter_schluessel=("C_WELLE", "P_DURCHMESSER"),
        stats=stats,
    )

    assert sim == pytest.approx(0.9, abs=1e-9)


def test_numerische_similarity_keine_streuung() -> None:
    """Wenn min==max, ist Similarity 1.0 bei Gleichheit, sonst 0.0."""
    stats = {("C_WELLE", "P_KONSTANTE"): (42.0, 42.0)}

    # Beide gleich
    sim1 = berechne_numerische_parameter_aehnlichkeit(
        wert_a=42.0,
        wert_b=42.0,
        parameter_schluessel=("C_WELLE", "P_KONSTANTE"),
        stats=stats,
    )
    assert sim1 == 1.0

    # Unterschiedlich
    sim2 = berechne_numerische_parameter_aehnlichkeit(
        wert_a=42.0,
        wert_b=43.0,
        parameter_schluessel=("C_WELLE", "P_KONSTANTE"),
        stats=stats,
    )
    assert sim2 == 0.0


def test_numerische_similarity_clamping() -> None:
    """Similarity sollte auf [0, 1] begrenzt sein."""
    stats = {("C_WELLE", "P_DURCHMESSER"): (100.0, 200.0)}

    # Wert außerhalb des Bereichs -> sollte trotzdem geclampt werden
    sim = berechne_numerische_parameter_aehnlichkeit(
        wert_a=50.0,  # unter min
        wert_b=250.0,  # über max
        parameter_schluessel=("C_WELLE", "P_DURCHMESSER"),
        stats=stats,
    )

    # Distanz = 200, Range = 100, raw = 1 - 200/100 = -1, clamped = 0
    assert sim == 0.0


def test_build_numeric_stats() -> None:
    """Test für build_numeric_stats."""
    features_by_rotor = {
        "Rotor_1": {
            "params": {
                ("C_WELLE", "P_DURCHMESSER"): {"value": 100.0, "unit": "mm", "ptype": "GEOM"},
                ("C_WELLE", "P_LAENGE"): {"value": 500.0, "unit": "mm", "ptype": "GEOM"},
                ("C_WELLE", "P_MATERIAL"): {
                    "value": "42CrMo4",
                    "unit": "-",
                    "ptype": "MTRL",
                },  # kategorisch
            }
        },
        "Rotor_2": {
            "params": {
                ("C_WELLE", "P_DURCHMESSER"): {"value": 150.0, "unit": "mm", "ptype": "GEOM"},
                ("C_WELLE", "P_LAENGE"): {"value": 600.0, "unit": "mm", "ptype": "GEOM"},
            }
        },
    }

    stats = build_numeric_stats(features_by_rotor)

    assert stats[("C_WELLE", "P_DURCHMESSER")] == (100.0, 150.0)
    assert stats[("C_WELLE", "P_LAENGE")] == (500.0, 600.0)
    # P_MATERIAL ist kategorisch -> nicht in stats
    assert ("C_WELLE", "P_MATERIAL") not in stats
