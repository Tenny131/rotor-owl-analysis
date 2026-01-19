"""Tests für ontology_helpers.py"""

from rotor_owl.utils.ontology_helpers import komponente_aus_text_erkennen


def test_komponente_erkennen_welle() -> None:
    """Test Erkennung von Welle."""
    component_id, erkannt = komponente_aus_text_erkennen("Welle", 1)
    assert erkannt is True
    assert component_id == "C_WELLE_1"


def test_komponente_erkennen_aktivteil() -> None:
    """Test Erkennung von Aktivteil."""
    component_id, erkannt = komponente_aus_text_erkennen("Aktivteil", 1)
    assert erkannt is True
    assert component_id == "C_AKTIVTEIL_1"


def test_komponente_erkennen_luefter() -> None:
    """Test Erkennung von Lüfter."""
    component_id, erkannt = komponente_aus_text_erkennen("Lüfter", 1)
    assert erkannt is True
    assert component_id == "C_LUEFTER_1"


def test_komponente_erkennen_case_insensitive() -> None:
    """Groß-/Kleinschreibung sollte ignoriert werden."""
    component_id, erkannt = komponente_aus_text_erkennen("WELLE", 1)
    assert erkannt is True
    assert component_id == "C_WELLE_1"


def test_komponente_erkennen_mit_zusatztext() -> None:
    """Komponente sollte auch in Freitext erkannt werden."""
    component_id, erkannt = komponente_aus_text_erkennen("Die Welle hat...", 5)
    assert erkannt is True
    assert component_id == "C_WELLE_5"


def test_komponente_erkennen_unbekannt() -> None:
    """Unbekannte Komponenten sollten (None, False) zurückgeben."""
    component_id, erkannt = komponente_aus_text_erkennen("Unbekannte Komponente", 1)
    assert erkannt is False
    assert component_id is None


def test_komponente_erkennen_verschiedene_nummern() -> None:
    """Instanz-Nummer sollte korrekt angehängt werden."""
    _, _ = komponente_aus_text_erkennen("Welle", 1)
    component_id_2, erkannt_2 = komponente_aus_text_erkennen("Welle", 42)
    assert erkannt_2 is True
    assert component_id_2 == "C_WELLE_42"
