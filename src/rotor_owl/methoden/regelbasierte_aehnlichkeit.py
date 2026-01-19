"""
Regelbasierte Ähnlichkeitsberechnung (Methode A).

Diese Datei enthält alle Funktionen für die regelbasierte Similarity-Methode:
- Numerische Parameter-Ähnlichkeit (Min-Max-normierte Distanz)
- Kategorische Parameter-Ähnlichkeit (Exakte Übereinstimmung)
- Parameter-weise Similarity-Berechnung
- Gewichtete Gesamt-Similarity pro Kategorie
- Top-k Ähnlichste Rotoren
- Automatische Gewichtsberechnung aus Dependency-Constraints
"""

from __future__ import annotations

from collections import defaultdict
import math

from rotor_owl.config.kategorien import map_paramtype_to_kategorie, KATEGORIEN_3
from rotor_owl.utils.math_utils import berechne_gewichtete_gesamt_similarity


# ============================================================================
# Automatische Gewichtsberechnung aus Dependencies
# ============================================================================


def berechne_automatische_gewichte(
    dependencies: dict[tuple[str, str], dict], normalization: str = "sum"
) -> dict[str, float]:
    """
    Berechnet Kategorie-Gewichte automatisch aus Dependency-Constraints.

    Strategie:
    - Für jede Komponente: Summiere alle incoming DependencyPercentages
    - Komponenten mit hohen Dependencies bekommen höheres Gewicht
    - Normalisiere auf Summe = 1.0

    Args:
        dependencies: Dict mit (source, target) -> {"strength": str, "percentage": float}
        normalization: "sum" (Summe=1.0) oder "max" (Maximum=1.0)

    Returns:
        Dict mit Komponenten-Name -> Gewicht (0.0 - 1.0)
    """
    component_importance = defaultdict(float)

    # Sammle Dependencies pro Komponente (incoming + outgoing)
    for (source, target), dep_info in dependencies.items():
        percentage = dep_info["percentage"]

        # Target wird beeinflusst -> wichtig für Similarity
        component_importance[target] += percentage

        # Source beeinflusst andere -> auch wichtig (aber weniger)
        component_importance[source] += percentage * 0.5

    if not component_importance:
        # Fallback: gleichmäßige Verteilung
        return {
            comp: 1.0 / 6
            for comp in [
                "Welle",
                "Aktivteil",
                "Blechpaket",
                "Luefter",
                "Welleende",
                "Wuchtscheiben",
            ]
        }

    # Normalisierung
    if normalization == "sum":
        total = sum(component_importance.values())
        if total > 0:
            return {comp: weight / total for comp, weight in component_importance.items()}
    elif normalization == "max":
        max_weight = max(component_importance.values())
        if max_weight > 0:
            return {comp: weight / max_weight for comp, weight in component_importance.items()}

    return dict(component_importance)


def map_komponenten_zu_kategorie_gewichte(
    komponenten_gewichte: dict[str, float], features_by_rotor: dict[str, dict]
) -> dict[str, float]:
    """
    Mappt Komponenten-Gewichte zu Kategorie-Gewichten (GEOM_MECH, MTRL_PROC, REQ_ELEC).

    Strategie:
    - Analysiere für jede Komponente, welche Parameter-Typen sie hat
    - Verteile Komponenten-Gewicht proportional auf Kategorien
    - Summiere für finale Kategorie-Gewichte

    Args:
        komponenten_gewichte: Dict mit Komponenten-Name -> Gewicht
        features_by_rotor: Feature-Daten aller Rotoren

    Returns:
        Dict mit Kategorie -> Gewicht
    """
    from rotor_owl.config.kategorien import map_paramtype_to_kategorie, KATEGORIEN_3

    # Zähle Parameter pro Komponente und Kategorie
    komponente_kategorie_counts = defaultdict(lambda: defaultdict(int))

    # Sample first rotor to get component structure
    sample_rotor = next(iter(features_by_rotor.values()))

    for (component, param), param_data in sample_rotor["params"].items():
        ptype = param_data.get("ptype")
        kategorie = map_paramtype_to_kategorie(ptype)
        komponente_kategorie_counts[component][kategorie] += 1

    # Berechne Kategorie-Gewichte
    kategorie_gewichte = defaultdict(float)

    for komponente, gewicht in komponenten_gewichte.items():
        # Normalisiere Komponenten-Namen (Welle, Blechpaket, etc.)
        komponente_normalized = komponente.lower().replace("_", "")

        # Finde matching Komponente in counts
        total_params = 0
        komponente_verteilung = {}

        for comp_key, kat_counts in komponente_kategorie_counts.items():
            comp_key_normalized = comp_key.lower().replace("_", "")
            if (
                komponente_normalized in comp_key_normalized
                or comp_key_normalized in komponente_normalized
            ):
                total_params = sum(kat_counts.values())
                komponente_verteilung = kat_counts
                break

        if total_params > 0:
            # Verteile Gewicht proportional auf Kategorien
            for kategorie, count in komponente_verteilung.items():
                kategorie_gewichte[kategorie] += gewicht * (count / total_params)
        else:
            # Fallback: Verteile gleichmäßig
            for kat in KATEGORIEN_3:
                kategorie_gewichte[kat] += gewicht / len(KATEGORIEN_3)

    # Normalisierung auf Summe = 1.0
    total = sum(kategorie_gewichte.values())
    if total > 0:
        return {kat: w / total for kat, w in kategorie_gewichte.items()}

    # Fallback: Gleichverteilung
    return {kat: 1.0 / len(KATEGORIEN_3) for kat in KATEGORIEN_3}


# ============================================================================
# Numerische Ähnlichkeit
# ============================================================================


def berechne_numerische_parameter_aehnlichkeit(
    wert_a: float,
    wert_b: float,
    parameter_schluessel: tuple[str, str],
    stats: dict[tuple[str, str], tuple[float, float]],
) -> float:
    """
    Numerische Similarity:
      sim = 1 - |a-b| / (max-min)

    Args:
        wert_a: Numerischer Wert von Parameter A
        wert_b: Numerischer Wert von Parameter B
        parameter_schluessel: Tuple (component_basis, parameter_basis)
        stats: Min/Max-Statistik für Normierung

    Returns:
        Similarity-Wert im Bereich [0.0, 1.0]

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


# ============================================================================
# Kategorische Ähnlichkeit
# ============================================================================


def berechne_kategorische_parameter_aehnlichkeit(wert_a: str, wert_b: str) -> float:
    """
    Kategorische / Enum / String Similarity:
    - exakt gleich -> 1.0
    - sonst -> 0.0

    Args:
        wert_a: Kategorischer Wert von Parameter A
        wert_b: Kategorischer Wert von Parameter B

    Returns:
        1.0 bei exakter Übereinstimmung, sonst 0.0

    Hinweis:
        Später kann man hier fuzzy matching / Synonyme / Ontologie-Distanzen einbauen.
    """
    return 1.0 if str(wert_a).strip() == str(wert_b).strip() else 0.0


# ============================================================================
# Parameter-weise Similarity
# ============================================================================


def value_similarity(
    parameter_a: dict | None,
    parameter_b: dict | None,
    parameter_schluessel: tuple[str, str],
    stats: dict[tuple[str, str], tuple[float, float]],
) -> float:
    """
    Berechnet Similarity für GENAU EINEN Parameter.

    Args:
        parameter_a: Parameter-Dict von Rotor A (mit keys: value, unit, ptype)
        parameter_b: Parameter-Dict von Rotor B (mit keys: value, unit, ptype)
        parameter_schluessel: Tuple (component_basis, parameter_basis)
        stats: Min/Max-Statistik für numerische Normierung

    Returns:
        Similarity-Wert:
        - float im Bereich [0.0, 1.0] bei gültigem Vergleich
        - float('nan') wenn beide Werte fehlen (wird später ignoriert)
        - 0.0 wenn nur ein Wert fehlt (Mismatch)

    Methode:
        - Numerische Werte: Min-Max-normierte Distanz
        - Kategorische Werte: Exakte Übereinstimmung (0 oder 1)
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

    # Numerisch
    if isinstance(wert_a, (int, float)) and isinstance(wert_b, (int, float)):
        return berechne_numerische_parameter_aehnlichkeit(
            wert_a=float(wert_a),
            wert_b=float(wert_b),
            parameter_schluessel=parameter_schluessel,
            stats=stats,
        )

    # Kategorisch
    return berechne_kategorische_parameter_aehnlichkeit(
        wert_a=str(wert_a),
        wert_b=str(wert_b),
    )


# ============================================================================
# Rotor-Similarity (Gesamt)
# ============================================================================


def rotor_similarity(
    rotor_a_id: str,
    rotor_b_id: str,
    features_by_rotor: dict[str, dict],
    stats: dict[tuple[str, str], tuple[float, float]],
    gewichtung_pro_kategorie: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """
    Methode A (regelbasiert):
    - Parameterweise Similarity
    - Mittelwert pro Kategorie
    - Gewichtete Gesamtsimilarity

    Args:
        rotor_a_id: ID des ersten Rotors
        rotor_b_id: ID des zweiten Rotors
        features_by_rotor: Feature-Daten aller Rotoren
        stats: Min/Max-Statistik für numerische Parameter
        gewichtung_pro_kategorie: Gewichtung pro Kategorie (GEOM, MECH, ELEC)

    Returns:
        Tuple (gesamt_similarity, similarity_pro_kategorie)
    """

    rotor_a_parameter = features_by_rotor[rotor_a_id]["params"]
    rotor_b_parameter = features_by_rotor[rotor_b_id]["params"]

    alle_parameter_schluessel = set(rotor_a_parameter.keys()) | set(rotor_b_parameter.keys())

    # Parameter nach 3er Kategorie gruppieren
    keys_pro_kategorie: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for parameter_schluessel in alle_parameter_schluessel:
        parameter_typ = (
            rotor_a_parameter.get(parameter_schluessel)
            or rotor_b_parameter.get(parameter_schluessel)
            or {}
        ).get("ptype") or "UNKNOWN"

        kat = map_paramtype_to_kategorie(parameter_typ)
        keys_pro_kategorie[kat].append(parameter_schluessel)

    similarity_pro_kategorie: dict[str, float] = {k: 0.0 for k in KATEGORIEN_3}
    anzahl_pro_kategorie: dict[str, int] = {k: 0 for k in KATEGORIEN_3}

    # Pro Kategorie mitteln
    for kat, parameter_keys in keys_pro_kategorie.items():
        sim_summe = 0.0
        cnt = 0

        for parameter_schluessel in parameter_keys:
            sim = value_similarity(
                parameter_a=rotor_a_parameter.get(parameter_schluessel),
                parameter_b=rotor_b_parameter.get(parameter_schluessel),
                parameter_schluessel=parameter_schluessel,
                stats=stats,
            )

            # beide missing -> ignorieren
            if isinstance(sim, float) and math.isnan(sim):
                continue

            sim_summe += sim
            cnt += 1

        similarity_pro_kategorie[kat] = (sim_summe / cnt) if cnt > 0 else 0.0
        anzahl_pro_kategorie[kat] = cnt

    # Gewichtete Gesamt-Similarity (zentrale Funktion)
    gesamt_similarity = berechne_gewichtete_gesamt_similarity(
        similarity_pro_kategorie, gewichtung_pro_kategorie
    )

    return gesamt_similarity, similarity_pro_kategorie


# ============================================================================
# Top-k Berechnung
# ============================================================================


def berechne_topk_aehnlichkeiten(
    query_rotor_id: str,
    rotor_ids: list[str],
    features_by_rotor: dict[str, dict],
    stats: dict[tuple[str, str], tuple[float, float]],
    gewichtung_pro_kategorie: dict[str, float],
    k: int,
) -> list[tuple[str, float, dict[str, float]]]:
    """
    Methode A: Top-k nach regelbasierter Similarity

    Args:
        query_rotor_id: ID des Query-Rotors
        rotor_ids: Liste aller Rotor-IDs
        features_by_rotor: Feature-Daten aller Rotoren
        stats: Min/Max-Statistik für numerische Parameter
        gewichtung_pro_kategorie: Gewichtung pro Kategorie
        k: Anzahl der Top-k Ergebnisse

    Returns:
        Liste von Tupeln (rotor_id, gesamt_similarity, similarity_pro_kategorie)
        sortiert nach Similarity (absteigend)
    """
    ergebnisse: list[tuple[str, float, dict[str, float]]] = []

    for ziel_rotor_id in rotor_ids:
        if ziel_rotor_id == query_rotor_id:
            continue

        gesamt_sim, sim_pro_kat = rotor_similarity(
            rotor_a_id=query_rotor_id,
            rotor_b_id=ziel_rotor_id,
            features_by_rotor=features_by_rotor,
            stats=stats,
            gewichtung_pro_kategorie=gewichtung_pro_kategorie,
        )
        ergebnisse.append((ziel_rotor_id, gesamt_sim, sim_pro_kat))

    ergebnisse.sort(key=lambda x: x[1], reverse=True)
    return ergebnisse[:k]
