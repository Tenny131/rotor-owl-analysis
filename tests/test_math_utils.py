"""Tests für math_utils.py"""

import numpy as np
import pytest

from rotor_owl.utils.math_utils import cosine_similarity, berechne_gewichtete_gesamt_similarity


def test_cosine_similarity_identische_vektoren() -> None:
    """Identische Vektoren sollten Similarity 1.0 haben."""
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])

    assert cosine_similarity(a, b) == pytest.approx(1.0, abs=1e-9)


def test_cosine_similarity_orthogonale_vektoren() -> None:
    """Orthogonale Vektoren sollten Similarity 0.0 haben."""
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])

    assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-9)


def test_cosine_similarity_entgegengesetzte_vektoren() -> None:
    """Entgegengesetzte Vektoren sollten Similarity -1.0 haben."""
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([-1.0, -2.0, -3.0])

    assert cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-9)


def test_cosine_similarity_nullvektor() -> None:
    """Nullvektoren sollten Similarity 0.0 zurückgeben."""
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 2.0, 3.0])

    assert cosine_similarity(a, b) == 0.0
    assert cosine_similarity(b, a) == 0.0
    assert cosine_similarity(a, a) == 0.0


def test_gewichtete_gesamt_similarity_einfach() -> None:
    """Test mit einfachen Werten."""
    sim_pro_kat = {
        "K1": 0.8,
        "K2": 0.6,
        "K3": 0.4,
    }
    gewichtung = {
        "K1": 2.0,
        "K2": 1.0,
        "K3": 0.0,  # Gewicht 0 -> wird ignoriert
    }

    # Erwartung: (2*0.8 + 1*0.6) / (2+1) = 2.2 / 3 = 0.7333...
    erwartet = (2.0 * 0.8 + 1.0 * 0.6) / 3.0

    resultat = berechne_gewichtete_gesamt_similarity(sim_pro_kat, gewichtung)
    assert resultat == pytest.approx(erwartet, abs=1e-9)


def test_gewichtete_gesamt_similarity_nur_nullgewichte() -> None:
    """Wenn alle Gewichte 0 sind, sollte 0.0 zurückkommen."""
    sim_pro_kat = {"K1": 0.8, "K2": 0.6}
    gewichtung = {"K1": 0.0, "K2": 0.0}

    assert berechne_gewichtete_gesamt_similarity(sim_pro_kat, gewichtung) == 0.0


def test_gewichtete_gesamt_similarity_fehlende_kategorie() -> None:
    """Fehlende Kategorien sollten als 0.0 Similarity behandelt werden."""
    sim_pro_kat = {"K1": 0.8}  # K2 fehlt
    gewichtung = {"K1": 1.0, "K2": 1.0}

    # Erwartung: (1*0.8 + 1*0.0) / 2 = 0.4
    resultat = berechne_gewichtete_gesamt_similarity(sim_pro_kat, gewichtung)
    assert resultat == pytest.approx(0.4, abs=1e-9)
