from __future__ import annotations


def berechne_kategorische_parameter_aehnlichkeit(wert_a: str, wert_b: str) -> float:
    """
    Kategorische / Enum / String Similarity:
    - exakt gleich -> 1.0
    - sonst -> 0.0

    Hinweis:
    - Später kann man hier fuzzy matching / Synonyme / Ontologie-Distanzen einbauen.
    """
    return 1.0 if str(wert_a).strip() == str(wert_b).strip() else 0.0
