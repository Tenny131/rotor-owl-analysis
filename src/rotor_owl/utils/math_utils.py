"""
Mathematische Hilfsfunktionen für Similarity-Berechnungen.
Zentrale Stelle für oft verwendete mathematische Operationen.
"""

from __future__ import annotations

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Berechnet Kosinus-Ähnlichkeit cos(θ) = (a·b) / (||a||·||b||).

    Args:
        a: Vektor 1
        b: Vektor 2

    Returns:
        Similarity [-1, 1], bei Nullvektor 0.0
    """
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU Aktivierungsfunktion: max(0, x).

    Args:
        x: Input-Array

    Returns:
        max(0, x) elementweise
    """
    return np.maximum(0.0, x)


def berechne_gewichtete_gesamt_similarity(
    similarity_pro_kategorie: dict[str, float],
    gewichtung_pro_kategorie: dict[str, float],
) -> float:
    """Gewichtetes Mittel: sum(w_i * sim_i) / sum(w_i).

    Args:
        similarity_pro_kategorie: Kategorie -> Similarity
        gewichtung_pro_kategorie: Kategorie -> Gewicht

    Returns:
        Gewichtetes Mittel, 0.0 bei sum(weights)=0
    """
    gewichtete_summe = 0.0
    gewicht_summe = 0.0

    for kategorie, gewicht in gewichtung_pro_kategorie.items():
        if gewicht <= 0.0:
            continue

        similarity = similarity_pro_kategorie.get(kategorie, 0.0)
        gewichtete_summe += gewicht * similarity
        gewicht_summe += gewicht

    return (gewichtete_summe / gewicht_summe) if gewicht_summe > 0 else 0.0
