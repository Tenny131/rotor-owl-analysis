from __future__ import annotations

from collections import defaultdict
import math

from rotor_owl.numerische_aehnlichkeit import berechne_numerische_parameter_aehnlichkeit
from rotor_owl.kategorische_aehnlichkeit import berechne_kategorische_parameter_aehnlichkeit


def value_similarity(
    parameter_a: dict | None,
    parameter_b: dict | None,
    parameter_schluessel: tuple[str, str],
    stats: dict[tuple[str, str], tuple[float, float]],
) -> float:
    """
    Berechnet die Similarity für GENAU EINEN Parameter.

    Regeln (gleich wie in deinem bisherigen Code):
    - Wenn Parameter fehlt -> 0.0
    - Wenn beide Values None -> NaN (damit später ignoriert wird)
    - Wenn einer None -> 0.0
    - Wenn beide numerisch -> numerische Similarity (normiert)
    - Sonst -> kategorische Similarity (equal / not equal)
    """
    if parameter_a is None or parameter_b is None:
        return 0.0

    wert_a = parameter_a.get("value")
    wert_b = parameter_b.get("value")

    # Beide existieren formal, aber sind "Missing"
    if wert_a is None and wert_b is None:
        return float("nan")

    # Einer fehlt -> mismatch
    if wert_a is None or wert_b is None:
        return 0.0

    # Numerischer Vergleich
    if isinstance(wert_a, (int, float)) and isinstance(wert_b, (int, float)):
        return berechne_numerische_parameter_aehnlichkeit(
            wert_a=float(wert_a),
            wert_b=float(wert_b),
            parameter_schluessel=parameter_schluessel,
            stats=stats,
        )

    # Kategorischer Vergleich
    return berechne_kategorische_parameter_aehnlichkeit(
        wert_a=str(wert_a),
        wert_b=str(wert_b),
    )


def rotor_similarity(
    rotor_a_id: str,
    rotor_b_id: str,
    features_by_rotor: dict[str, dict],
    stats: dict[tuple[str, str], tuple[float, float]],
    gewichtung_pro_typ: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """
    Berechnet:
    - Gesamt-Similarity zwischen zwei Rotoren
    - Similarity pro Kategorie (GEOM, MTRL, ...)

    Vorgehen:
    1) Union aller Parameterkeys bilden (A ∪ B)
    2) Keys nach ptype gruppieren
    3) Pro ptype Mittelwert bilden
    4) Gewichteten Gesamtscore berechnen
    """
    rotor_a_parameter = features_by_rotor[rotor_a_id]["params"]
    rotor_b_parameter = features_by_rotor[rotor_b_id]["params"]

    alle_parameter_schluessel = set(rotor_a_parameter.keys()) | set(rotor_b_parameter.keys())

    # Parameter nach Kategorie gruppieren (GEOM, MTRL, ...)
    parameter_keys_pro_typ: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for parameter_schluessel in alle_parameter_schluessel:
        parameter_typ = (
            rotor_a_parameter.get(parameter_schluessel)
            or rotor_b_parameter.get(parameter_schluessel)
            or {}
        ).get("ptype") or "UNKNOWN"
        parameter_keys_pro_typ[parameter_typ].append(parameter_schluessel)

    similarity_pro_typ: dict[str, float] = {}
    anzahl_parameter_pro_typ: dict[str, int] = {}

    # Pro Kategorie mitteln
    for parameter_typ, parameter_schluessel_liste in parameter_keys_pro_typ.items():
        similarity_summe = 0.0
        anzahl_vergleichbare_parameter = 0

        for parameter_schluessel in parameter_schluessel_liste:
            sim = value_similarity(
                parameter_a=rotor_a_parameter.get(parameter_schluessel),
                parameter_b=rotor_b_parameter.get(parameter_schluessel),
                parameter_schluessel=parameter_schluessel,
                stats=stats,
            )

            # Beide missing -> ignorieren
            if isinstance(sim, float) and math.isnan(sim):
                continue

            similarity_summe += sim
            anzahl_vergleichbare_parameter += 1

        similarity_pro_typ[parameter_typ] = (
            (similarity_summe / anzahl_vergleichbare_parameter)
            if anzahl_vergleichbare_parameter > 0
            else 0.0
        )
        anzahl_parameter_pro_typ[parameter_typ] = anzahl_vergleichbare_parameter

    # Gewichtetes Gesamtmittel
    gewichtete_summe = 0.0
    gewicht_summe = 0.0

    for parameter_typ, sim_typ in similarity_pro_typ.items():
        gewicht = gewichtung_pro_typ.get(parameter_typ, 0.0)

        # Nur berücksichtigen, wenn:
        # - Gewicht > 0
        # - es überhaupt Parameter in dieser Kategorie gab
        if gewicht > 0 and anzahl_parameter_pro_typ.get(parameter_typ, 0) > 0:
            gewichtete_summe += gewicht * sim_typ
            gewicht_summe += gewicht

    gesamt_similarity = (gewichtete_summe / gewicht_summe) if gewicht_summe > 0 else 0.0
    return gesamt_similarity, similarity_pro_typ


def berechne_topk_aehnlichkeiten(
    query_rotor_id: str,
    rotor_ids: list[str],
    features_by_rotor: dict[str, dict],
    stats: dict[tuple[str, str], tuple[float, float]],
    gewichtung_pro_typ: dict[str, float],
    k: int,
) -> list[tuple[str, float, dict[str, float]]]:
    """
    Berechnet Similarity vom Query-Rotor zu allen anderen Rotoren und gibt Top-k zurück.

    Rückgabe:
      [(rotor_id, gesamt_similarity, similarity_pro_typ), ...]
    """
    ergebnisse: list[tuple[str, float, dict[str, float]]] = []

    for ziel_rotor_id in rotor_ids:
        if ziel_rotor_id == query_rotor_id:
            continue

        gesamt_sim, sim_pro_typ = rotor_similarity(
            rotor_a_id=query_rotor_id,
            rotor_b_id=ziel_rotor_id,
            features_by_rotor=features_by_rotor,
            stats=stats,
            gewichtung_pro_typ=gewichtung_pro_typ,
        )
        ergebnisse.append((ziel_rotor_id, gesamt_sim, sim_pro_typ))

    ergebnisse.sort(key=lambda x: x[1], reverse=True)
    return ergebnisse[:k]
