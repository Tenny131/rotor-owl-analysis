from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rotor_owl.config.kategorien import KATEGORIEN_3, map_paramtype_to_kategorie
from rotor_owl.utils.math_utils import cosine_similarity, berechne_gewichtete_gesamt_similarity


@dataclass(frozen=True)
class _FeatureSpec:
    """Feature-Spezifikation pro Kategorie.

    Attributes:
        numeric_keys: Liste numerischer Features (keys)
        categorical_values: Dict categorical key -> allowed values
        index: (key, kind, value) -> idx für Vektor-Position
        dim: Gesamtdimension des Feature-Vektors
    """

    numeric_keys: list[tuple[str, str]]
    categorical_values: dict[tuple[str, str], list[str]]
    index: dict[tuple[str, str, str], int]
    dim: int


@dataclass(frozen=True)
class _VektorEmbeddings:
    """Vektorbasierte Embeddings pro Kategorie.

    Attributes:
        specs: Kategorie -> Feature-Spezifikation
        vectors: Kategorie -> Rotor-ID -> Feature-Vektor
    """

    specs: dict[str, _FeatureSpec]
    vectors: dict[str, dict[str, np.ndarray]]


def _ist_numerischer_wert(wert: object) -> bool:
    """Prüft ob ein Wert numerisch ist (int oder float)."""
    return isinstance(wert, (int, float))


def _normalisiere_numerisch(
    wert: float,
    parameter_key: tuple[str, str],
    stats: dict[tuple[str, str], tuple[float, float]],
) -> float:
    """Normalisiert numerischen Wert auf [0, 1] Bereich."""
    minimum, maximum = stats.get(parameter_key, (wert, wert))
    if maximum <= minimum:
        return 0.5  # Keine Streuung -> neutral
    normalisiert = (wert - minimum) / (maximum - minimum)
    return float(max(0.0, min(1.0, normalisiert)))


def _build_feature_specs(
    features_by_rotor: dict[str, dict],
) -> dict[str, _FeatureSpec]:
    """
    Baut Feature-Spezifikationen pro Kategorie.

    Args:
        features_by_rotor (dict): Feature-Daten aller Rotoren

    Returns:
        dict: Kategorie -> _FeatureSpec
    """
    numerische_keys_pro_kat: dict[str, set[tuple[str, str]]] = {kat: set() for kat in KATEGORIEN_3}
    kategorische_werte_pro_kat: dict[str, dict[tuple[str, str], set[str]]] = {
        kat: {} for kat in KATEGORIEN_3
    }

    for _, rotor_daten in features_by_rotor.items():
        for parameter_key, parameter_daten in rotor_daten["params"].items():
            param_typ = parameter_daten.get("ptype") or "UNKNOWN"
            kategorie = map_paramtype_to_kategorie(param_typ)
            wert = parameter_daten.get("value")

            if wert is None:
                continue

            if _ist_numerischer_wert(wert):
                numerische_keys_pro_kat[kategorie].add(parameter_key)
            else:
                kategorische_werte_pro_kat[kategorie].setdefault(parameter_key, set()).add(
                    str(wert).strip()
                )

    specs: dict[str, _FeatureSpec] = {}

    for kategorie in KATEGORIEN_3:
        numerische_keys = sorted(numerische_keys_pro_kat[kategorie])
        kategorische_werte: dict[tuple[str, str], list[str]] = {}

        for param_key, werte in kategorische_werte_pro_kat[kategorie].items():
            werte_sortiert = sorted({w for w in werte if w})
            kategorische_werte[param_key] = werte_sortiert + ["<MISSING>"]

        # Index bauen: numerisch = 2 Dimensionen, kategorial = 1 pro Wert
        index: dict[tuple[str, str, str], int] = {}
        idx = 0

        for param_key in numerische_keys:
            index[(param_key[0], param_key[1], "__NUM_VALUE__")] = idx
            idx += 1
            index[(param_key[0], param_key[1], "__NUM_MISSING__")] = idx
            idx += 1

        for param_key, werte in kategorische_werte.items():
            for wert in werte:
                index[(param_key[0], param_key[1], f"__CAT__{wert}")] = idx
                idx += 1

        specs[kategorie] = _FeatureSpec(
            numeric_keys=numerische_keys,
            categorical_values=kategorische_werte,
            index=index,
            dim=idx,
        )

    return specs


def _vektorisiere_rotor_fuer_kategorie(
    rotor_id: str,
    features_by_rotor: dict[str, dict],
    stats: dict[tuple[str, str], tuple[float, float]],
    kategorie: str,
    spec: _FeatureSpec,
) -> np.ndarray:
    """
    Erstellt Feature-Vektor für einen Rotor in einer Kategorie.

    Args:
        rotor_id (str): Rotor-ID
        features_by_rotor (dict): Feature-Daten
        stats (dict): Min/Max-Statistiken
        kategorie (str): Kategorie-Name
        spec (_FeatureSpec): Feature-Spezifikation

    Returns:
        np.ndarray: Feature-Vektor
    """
    vektor = np.zeros(spec.dim, dtype=float)
    parameter = features_by_rotor[rotor_id]["params"]

    # Numerische Features
    for param_key in spec.numeric_keys:
        param_daten = parameter.get(param_key)
        wert = None if param_daten is None else param_daten.get("value")

        idx_wert = spec.index.get((param_key[0], param_key[1], "__NUM_VALUE__"))
        idx_missing = spec.index.get((param_key[0], param_key[1], "__NUM_MISSING__"))

        if idx_wert is None or idx_missing is None:
            continue

        if wert is None or not _ist_numerischer_wert(wert):
            vektor[idx_missing] = 1.0
            vektor[idx_wert] = 0.0
        else:
            vektor[idx_missing] = 0.0
            vektor[idx_wert] = _normalisiere_numerisch(float(wert), param_key, stats)

    # Kategorische Features
    for param_key, erlaubte_werte in spec.categorical_values.items():
        param_daten = parameter.get(param_key)
        wert = None if param_daten is None else param_daten.get("value")

        wert_str = "<MISSING>" if wert is None else str(wert).strip()
        if wert_str not in erlaubte_werte:
            wert_str = "<MISSING>"

        idx_kat = spec.index.get((param_key[0], param_key[1], f"__CAT__{wert_str}"))
        if idx_kat is not None:
            vektor[idx_kat] = 1.0

    return vektor


def build_vektor_embeddings(
    features_by_rotor: dict[str, dict],
    stats: dict[tuple[str, str], tuple[float, float]],
) -> _VektorEmbeddings:
    """
    Baut pro Kategorie Vektoren für alle Rotoren.

    Args:
        features_by_rotor (dict): Feature-Daten aller Rotoren
        stats (dict): Min/Max-Statistiken

    Returns:
        _VektorEmbeddings: Embedding-Objekt mit Vektoren
    """
    specs = _build_feature_specs(features_by_rotor)

    vektoren: dict[str, dict[str, np.ndarray]] = {}
    rotor_ids = list(features_by_rotor.keys())

    for kategorie in KATEGORIEN_3:
        spec = specs[kategorie]
        vektoren[kategorie] = {}
        for rotor_id in rotor_ids:
            vektoren[kategorie][rotor_id] = _vektorisiere_rotor_fuer_kategorie(
                rotor_id=rotor_id,
                features_by_rotor=features_by_rotor,
                stats=stats,
                kategorie=kategorie,
                spec=spec,
            )

    return _VektorEmbeddings(specs=specs, vectors=vektoren)


def rotor_similarity_vektorbasiert(
    rotor_a_id: str,
    rotor_b_id: str,
    embeddings: _VektorEmbeddings,
    gewichtung_pro_kategorie: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """
    Berechnet Rotor-Similarity mit vektorbasierter Cosine-Methode.

    Args:
        rotor_a_id (str): ID des ersten Rotors
        rotor_b_id (str): ID des zweiten Rotors
        embeddings (_VektorEmbeddings): Vorberechnete Feature-Vektoren
        gewichtung_pro_kategorie (dict): Gewichte für die 3 Kategorien

    Returns:
        tuple: (gesamt_similarity, similarity_pro_kategorie)
    """
    sim_pro_kat: dict[str, float] = {}

    for kategorie, vektor_dict in embeddings.vectors.items():
        vektor_a = vektor_dict[rotor_a_id]
        vektor_b = vektor_dict[rotor_b_id]
        # Cosine gibt [-1, 1], normalisieren auf [0, 1]
        raw_sim = cosine_similarity(vektor_a, vektor_b)
        sim_pro_kat[kategorie] = (raw_sim + 1.0) / 2.0

    # Gewichtete Aggregation (zentrale Funktion)
    total = berechne_gewichtete_gesamt_similarity(sim_pro_kat, gewichtung_pro_kategorie)

    return total, sim_pro_kat


def berechne_topk_aehnlichkeiten_vektorbasiert(
    query_rotor_id: str,
    rotor_ids: list[str],
    embeddings: _VektorEmbeddings,
    gewichtung_pro_kategorie: dict[str, float],
    top_k: int,
) -> list[tuple[str, float, dict[str, float]]]:
    """
    Berechnet die Top-k ähnlichsten Rotoren (vektorbasiert).

    Args:
        query_rotor_id (str): ID des Abfrage-Rotors
        rotor_ids (list): Liste aller Rotor-IDs
        embeddings (_VektorEmbeddings): Vorberechnete Embeddings
        gewichtung_pro_kategorie (dict): Kategorie-Gewichte
        top_k (int): Anzahl der Ergebnisse

    Returns:
        list: Top-k Ergebnisse als (rotor_id, similarity, sim_pro_kat)
    """
    ergebnisse: list[tuple[str, float, dict[str, float]]] = []

    for ziel_rotor_id in rotor_ids:
        if ziel_rotor_id == query_rotor_id:
            continue
        total, sim_pro_kat = rotor_similarity_vektorbasiert(
            rotor_a_id=query_rotor_id,
            rotor_b_id=ziel_rotor_id,
            embeddings=embeddings,
            gewichtung_pro_kategorie=gewichtung_pro_kategorie,
        )
        ergebnisse.append((ziel_rotor_id, total, sim_pro_kat))

    ergebnisse.sort(key=lambda x: x[1], reverse=True)
    return ergebnisse[:top_k]
