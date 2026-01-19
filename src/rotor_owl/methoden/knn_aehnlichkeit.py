from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rotor_owl.config.kategorien import KATEGORIEN_3, map_paramtype_to_kategorie
from rotor_owl.utils.math_utils import cosine_similarity, berechne_gewichtete_gesamt_similarity


@dataclass(frozen=True)
class _FeatureSpec:
    # Für jede Kategorie: Liste numerischer Features (keys) und dict categorical key -> allowed values
    numeric_keys: list[tuple[str, str]]
    categorical_values: dict[tuple[str, str], list[str]]
    index: dict[tuple[str, str, str], int]  # (key, kind, value) -> idx
    dim: int


@dataclass(frozen=True)
class _KNNEmbeddings:
    specs: dict[str, _FeatureSpec]  # kategorie -> spec
    vectors: dict[str, dict[str, np.ndarray]]  # kategorie -> rotor_id -> vector


def _is_numeric_value(v: object) -> bool:
    return isinstance(v, (int, float))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _normalize_numeric(
    x: float,
    key: tuple[str, str],
    stats: dict[tuple[str, str], tuple[float, float]],
) -> float:
    lo, hi = stats.get(key, (x, x))
    if hi <= lo:
        return 0.5  # keine Streuung -> neutral
    t = (x - lo) / (hi - lo)
    return float(max(0.0, min(1.0, t)))


def _build_feature_specs(
    features_by_rotor: dict[str, dict],
) -> dict[str, _FeatureSpec]:
    """
    Scannt alle Rotoren und baut pro 3er-Kategorie:
    - welche Parameter-Keys numerisch sind
    - welche Parameter-Keys kategorial sind + mögliche Values
    """

    numeric_keys_by_cat: dict[str, set[tuple[str, str]]] = {c: set() for c in KATEGORIEN_3}
    cat_values_by_cat: dict[str, dict[tuple[str, str], set[str]]] = {c: {} for c in KATEGORIEN_3}

    for _, rotor_daten in features_by_rotor.items():
        for key, pdata in rotor_daten["params"].items():
            ptype = pdata.get("ptype") or "UNKNOWN"
            cat = map_paramtype_to_kategorie(ptype)

            v = pdata.get("value")

            if v is None:
                # missing erstmal ignorieren (kommt als eigene OneHot "<MISSING>")
                continue

            if _is_numeric_value(v):
                numeric_keys_by_cat[cat].add(key)
            else:
                cat_values_by_cat[cat].setdefault(key, set()).add(str(v).strip())

    specs: dict[str, _FeatureSpec] = {}

    for cat in KATEGORIEN_3:
        num_keys = sorted(numeric_keys_by_cat[cat])
        cat_vals: dict[tuple[str, str], list[str]] = {}

        for k, vals in cat_values_by_cat[cat].items():
            # "<MISSING>" immer erlauben
            vals_sorted = sorted({v for v in vals if v})
            cat_vals[k] = vals_sorted + ["<MISSING>"]

        # Index bauen:
        # numerisch: pro key 2 dimensionen -> (value, missing_flag)
        # kategorial: pro value 1 dimension
        index: dict[tuple[str, str, str], int] = {}
        idx = 0

        for k in num_keys:
            index[(k[0], k[1], "__NUM_VALUE__")] = idx
            idx += 1
            index[(k[0], k[1], "__NUM_MISSING__")] = idx
            idx += 1

        for k, values in cat_vals.items():
            for vv in values:
                index[(k[0], k[1], f"__CAT__{vv}")] = idx
                idx += 1

        specs[cat] = _FeatureSpec(
            numeric_keys=num_keys,
            categorical_values=cat_vals,
            index=index,
            dim=idx,
        )

    return specs


def _vectorize_rotor_for_category(
    rotor_id: str,
    features_by_rotor: dict[str, dict],
    stats: dict[tuple[str, str], tuple[float, float]],
    category: str,
    spec: _FeatureSpec,
) -> np.ndarray:
    vec = np.zeros(spec.dim, dtype=float)

    params = features_by_rotor[rotor_id]["params"]

    # 1) numerisch
    for key in spec.numeric_keys:
        pdata = params.get(key)
        v = None if pdata is None else pdata.get("value")

        idx_value = spec.index.get((key[0], key[1], "__NUM_VALUE__"))
        idx_missing = spec.index.get((key[0], key[1], "__NUM_MISSING__"))

        if idx_value is None or idx_missing is None:
            continue

        if v is None or not _is_numeric_value(v):
            vec[idx_missing] = 1.0
            vec[idx_value] = 0.0
        else:
            vec[idx_missing] = 0.0
            vec[idx_value] = _normalize_numeric(float(v), key, stats)

    # 2) kategorial
    for key, allowed_values in spec.categorical_values.items():
        pdata = params.get(key)
        v = None if pdata is None else pdata.get("value")

        v_str = "<MISSING>" if v is None else str(v).strip()
        if v_str not in allowed_values:
            v_str = "<MISSING>"

        idx_cat = spec.index.get((key[0], key[1], f"__CAT__{v_str}"))
        if idx_cat is not None:
            vec[idx_cat] = 1.0

    return vec


def build_knn_embeddings(
    features_by_rotor: dict[str, dict],
    stats: dict[tuple[str, str], tuple[float, float]],
) -> _KNNEmbeddings:
    """
    Baut pro Kategorie Vektoren für alle Rotoren.
    """
    specs = _build_feature_specs(features_by_rotor)

    vectors: dict[str, dict[str, np.ndarray]] = {}
    rotor_ids = list(features_by_rotor.keys())

    for cat in KATEGORIEN_3:
        spec = specs[cat]
        vectors[cat] = {}
        for rid in rotor_ids:
            vectors[cat][rid] = _vectorize_rotor_for_category(
                rotor_id=rid,
                features_by_rotor=features_by_rotor,
                stats=stats,
                category=cat,
                spec=spec,
            )

    return _KNNEmbeddings(specs=specs, vectors=vectors)


def rotor_similarity_knn(
    rotor_a_id: str,
    rotor_b_id: str,
    embeddings: _KNNEmbeddings,
    gewichtung_pro_kategorie: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """
    Berechnet Rotor-Similarity mit kNN/Cosine-Methode.

    Args:
        rotor_a_id: ID des ersten Rotors
        rotor_b_id: ID des zweiten Rotors
        embeddings: Vorberechnete Feature-Vektoren pro Kategorie
        gewichtung_pro_kategorie: Gewichte für die 3 Kategorien

    Returns:
        Tuple aus (gesamt_similarity, similarity_pro_kategorie)
    """
    sim_pro_kat: dict[str, float] = {}

    for cat, vecs in embeddings.vectors.items():
        va = vecs[rotor_a_id]
        vb = vecs[rotor_b_id]
        sim_pro_kat[cat] = cosine_similarity(va, vb)

    # Gewichtete Aggregation (zentrale Funktion)
    total = berechne_gewichtete_gesamt_similarity(sim_pro_kat, gewichtung_pro_kategorie)

    return total, sim_pro_kat


def berechne_topk_aehnlichkeiten_knn(
    query_rotor_id: str,
    rotor_ids: list[str],
    embeddings: _KNNEmbeddings,
    gewichtung_pro_kategorie: dict[str, float],
    k: int,
) -> list[tuple[str, float, dict[str, float]]]:
    ergebnisse: list[tuple[str, float, dict[str, float]]] = []

    for ziel in rotor_ids:
        if ziel == query_rotor_id:
            continue
        total, sim_pro_kat = rotor_similarity_knn(
            rotor_a_id=query_rotor_id,
            rotor_b_id=ziel,
            embeddings=embeddings,
            gewichtung_pro_kategorie=gewichtung_pro_kategorie,
        )
        ergebnisse.append((ziel, total, sim_pro_kat))

    ergebnisse.sort(key=lambda x: x[1], reverse=True)
    return ergebnisse[:k]
