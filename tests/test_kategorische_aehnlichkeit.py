"""Tests für kategorische_aehnlichkeit.py (jetzt in regelbasierte_aehnlichkeit.py)"""

from rotor_owl.methoden.regelbasierte_aehnlichkeit import (
    berechne_kategorische_parameter_aehnlichkeit,
)


def test_kategorische_similarity_identisch() -> None:
    """Identische Werte sollten Similarity 1.0 haben."""
    sim = berechne_kategorische_parameter_aehnlichkeit("42CrMo4", "42CrMo4")
    assert sim == 1.0


def test_kategorische_similarity_unterschiedlich() -> None:
    """Unterschiedliche Werte sollten Similarity 0.0 haben."""
    sim = berechne_kategorische_parameter_aehnlichkeit("42CrMo4", "16MnCr5")
    assert sim == 0.0


def test_kategorische_similarity_whitespace_toleranz() -> None:
    """Whitespace sollte ignoriert werden."""
    sim1 = berechne_kategorische_parameter_aehnlichkeit("  42CrMo4  ", "42CrMo4")
    assert sim1 == 1.0

    sim2 = berechne_kategorische_parameter_aehnlichkeit("42CrMo4", "42CrMo4  ")
    assert sim2 == 1.0


def test_kategorische_similarity_case_sensitive() -> None:
    """Groß-/Kleinschreibung ist relevant (aktuelles Verhalten)."""
    sim = berechne_kategorische_parameter_aehnlichkeit("ABC", "abc")
    assert sim == 0.0


def test_kategorische_similarity_boolean_werte() -> None:
    """Test mit Boolean-ähnlichen String-Werten."""
    sim1 = berechne_kategorische_parameter_aehnlichkeit("ja", "ja")
    assert sim1 == 1.0

    sim2 = berechne_kategorische_parameter_aehnlichkeit("ja", "nein")
    assert sim2 == 0.0
