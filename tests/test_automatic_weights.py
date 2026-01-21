"""Tests für automatische Gewichtsberechnung aus Dependency-Constraints."""

from rotor_owl.methoden.regelbasierte_aehnlichkeit import (
    berechne_automatische_gewichte,
    map_komponenten_zu_kategorie_gewichte,
)


def test_berechne_automatische_gewichte_normalization():
    """Prüft dass Komponenten-Gewichte auf sum=1.0 normalisiert werden."""
    dependencies = {
        ("Welle", "Blechpaket"): {"strength": "hoch", "percentage": 0.8},
        ("Aktivteil", "Welle"): {"strength": "mittel", "percentage": 0.6},
        ("Blechpaket", "Luefter"): {"strength": "niedrig", "percentage": 0.3},
    }

    weights = berechne_automatische_gewichte(dependencies)

    # Summe muss 1.0 sein
    assert abs(sum(weights.values()) - 1.0) < 1e-6

    # Alle Gewichte positiv
    for weight in weights.values():
        assert weight > 0


def test_berechne_automatische_gewichte_incoming_outgoing():
    """Prüft dass incoming 100% und outgoing 50% Gewicht bekommen, mit Stärke-Multiplikator."""
    dependencies = {
        ("Source", "Target"): {"strength": "hoch", "percentage": 1.0},
    }

    weights = berechne_automatische_gewichte(dependencies)

    # Target bekommt 100%, Source bekommt 50%, beide mit Stärke-Multiplikator 1.5
    # Target: 1.0 * 1.5 = 1.5, Source: 1.0 * 1.5 * 0.5 = 0.75
    # Summe = 2.25 -> normalisiert: Target = 1.5/2.25 = 0.667, Source = 0.75/2.25 = 0.333
    assert "Target" in weights
    assert "Source" in weights

    # Target sollte doppelt so viel Gewicht haben wie Source (unabhängig von Stärke)
    assert abs(weights["Target"] - 0.6666666666666666) < 1e-6
    assert abs(weights["Source"] - 0.3333333333333333) < 1e-6


def test_berechne_automatische_gewichte_strength_affects_weight():
    """Prüft dass unterschiedliche Stärken unterschiedliche Gewichte erzeugen."""
    # Gleiche Struktur, aber mit niedrig
    deps_niedrig = {
        ("A", "B"): {"strength": "niedrig", "percentage": 1.0},
    }
    deps_hoch = {
        ("A", "B"): {"strength": "hoch", "percentage": 1.0},
    }

    weights_niedrig = berechne_automatische_gewichte(deps_niedrig)
    weights_hoch = berechne_automatische_gewichte(deps_hoch)

    # Verhältnis bleibt gleich (2:1), aber absolute Werte vor Normalisierung unterschiedlich
    # Nach Normalisierung sind die Verhältnisse identisch
    assert abs(weights_niedrig["B"] - weights_hoch["B"]) < 1e-6
    assert abs(weights_niedrig["A"] - weights_hoch["A"]) < 1e-6

    # Aber bei mehreren Dependencies macht Stärke einen Unterschied
    deps_mixed = {
        ("A", "B"): {"strength": "hoch", "percentage": 1.0},
        ("C", "D"): {"strength": "niedrig", "percentage": 1.0},
    }
    weights_mixed = berechne_automatische_gewichte(deps_mixed)

    # B (hoch) sollte mehr Gewicht haben als D (niedrig)
    assert weights_mixed["B"] > weights_mixed["D"]


def test_berechne_automatische_gewichte_fallback():
    """Prüft Fallback bei leeren Dependencies."""
    dependencies = {}

    weights = berechne_automatische_gewichte(dependencies)

    # Fallback: gleichmäßige Verteilung auf 6 Standard-Komponenten
    assert len(weights) == 6
    assert abs(sum(weights.values()) - 1.0) < 1e-6

    # Jede Komponente sollte ~16.67% haben
    for weight in weights.values():
        assert abs(weight - 1.0 / 6) < 1e-6


def test_map_komponenten_zu_kategorie_gewichte_normalization():
    """Prüft dass Kategorie-Gewichte auf sum=1.0 normalisiert werden."""
    komponenten_gewichte = {
        "welle": 0.5,
        "blechpaket": 0.3,
        "aktivteil": 0.2,
    }

    # Mock features_by_rotor
    features_by_rotor = {
        "Rotor_D001": {
            "params": {
                ("C_WELLE", "P_DURCHMESSER"): {"value": 100.0, "unit": "mm", "ptype": "GEOM"},
                ("C_WELLE", "P_MATERIAL"): {"value": "42CrMo4", "unit": "-", "ptype": "MTRL"},
                ("C_BLECHPAKET", "P_LAENGE"): {"value": 200.0, "unit": "mm", "ptype": "GEOM"},
                ("C_AKTIVTEIL", "P_SPANNUNG"): {"value": 230.0, "unit": "V", "ptype": "ELEC"},
            }
        }
    }

    kategorie_weights = map_komponenten_zu_kategorie_gewichte(
        komponenten_gewichte, features_by_rotor
    )

    # Summe muss 1.0 sein
    assert abs(sum(kategorie_weights.values()) - 1.0) < 1e-6

    # Alle Gewichte positiv
    for weight in kategorie_weights.values():
        assert weight >= 0

    # Sollte 3 Kategorien haben
    assert len(kategorie_weights) == 3


def test_map_komponenten_zu_kategorie_gewichte_fallback():
    """Prüft Fallback bei fehlenden Features."""
    komponenten_gewichte = {
        "unknown_component": 1.0,  # Nicht in features
    }

    features_by_rotor = {
        "Rotor_D001": {
            "params": {
                ("C_WELLE", "P_DURCHMESSER"): {"value": 100.0, "unit": "mm", "ptype": "GEOM"},
            }
        }
    }

    kategorie_weights = map_komponenten_zu_kategorie_gewichte(
        komponenten_gewichte, features_by_rotor
    )

    # Sollte Fallback auf gleichmäßige Verteilung machen
    assert abs(sum(kategorie_weights.values()) - 1.0) < 1e-6
    assert len(kategorie_weights) == 3

    # Jede Kategorie sollte ~33.33% haben (Fallback)
    for weight in kategorie_weights.values():
        assert abs(weight - 1.0 / 3) < 1e-6


def test_komponenten_name_normalization():
    """Prüft dass Komponenten-Namen flexibel gematcht werden."""
    komponenten_gewichte = {
        "welle": 0.6,  # lowercase mit _
        "blechpaket": 0.4,  # lowercase ohne _
    }

    features_by_rotor = {
        "Rotor_D001": {
            "params": {
                ("C_WELLE", "P_DURCHMESSER"): {"value": 100.0, "unit": "mm", "ptype": "GEOM"},
                ("C_BLECHPAKET", "P_LAENGE"): {"value": 200.0, "unit": "mm", "ptype": "GEOM"},
            }
        }
    }

    kategorie_weights = map_komponenten_zu_kategorie_gewichte(
        komponenten_gewichte, features_by_rotor
    )

    # Sollte erfolgreich matchen trotz unterschiedlicher Schreibweisen
    assert abs(sum(kategorie_weights.values()) - 1.0) < 1e-6

    # K_GEOM_MECH sollte beide Welle und Blechpaket enthalten
    assert "K_GEOM_MECH" in kategorie_weights
    assert kategorie_weights["K_GEOM_MECH"] > 0
