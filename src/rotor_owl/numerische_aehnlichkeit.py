from __future__ import annotations

from collections import defaultdict


def build_numeric_stats(
    features_by_rotor: dict[str, dict],
) -> dict[tuple[str, str], tuple[float, float]]:
    """
    Baut min/max Statistik pro Parameter-Key (component_basis, parameter_basis).

    Idee:
    - Für numerische Similarity brauchen wir eine Normierung
    - Dafür ist min/max über alle Rotoren eine einfache, robuste Basis

    Ergebnis:
      stats[(C_WELLE, P_WELLE_TIR)] = (min_wert, max_wert)
    """
    numerische_werte_pro_parameter: dict[tuple[str, str], list[float]] = defaultdict(list)

    for _, rotor_daten in features_by_rotor.items():
        for parameter_schluessel, parameter_datensatz in rotor_daten["params"].items():
            parameter_wert = parameter_datensatz.get("value")

            if isinstance(parameter_wert, (int, float)):
                numerische_werte_pro_parameter[parameter_schluessel].append(float(parameter_wert))

    stats: dict[tuple[str, str], tuple[float, float]] = {}

    for parameter_schluessel, werte_liste in numerische_werte_pro_parameter.items():
        if len(werte_liste) >= 2:
            stats[parameter_schluessel] = (min(werte_liste), max(werte_liste))
        elif len(werte_liste) == 1:
            stats[parameter_schluessel] = (werte_liste[0], werte_liste[0])

    return stats


def berechne_numerische_parameter_aehnlichkeit(
    wert_a: float,
    wert_b: float,
    parameter_schluessel: tuple[str, str],
    stats: dict[tuple[str, str], tuple[float, float]],
) -> float:
    """
    Numerische Similarity:
      sim = 1 - |a-b| / (max-min)

    Grenzen:
    - sim ist auf [0..1] gekappt

    Spezialfall:
    - Wenn max==min, dann gibt es keine Streuung -> Gleichheit = 1, sonst 0
    """
    minimum_wert, maximum_wert = stats.get(parameter_schluessel, (wert_a, wert_a))

    if maximum_wert == minimum_wert:
        return 1.0 if float(wert_a) == float(wert_b) else 0.0

    similarity = 1.0 - abs(float(wert_a) - float(wert_b)) / (maximum_wert - minimum_wert)
    return max(0.0, min(1.0, similarity))
