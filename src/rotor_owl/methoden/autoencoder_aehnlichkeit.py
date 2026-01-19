from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.neural_network import MLPRegressor

from rotor_owl.methoden.knn_aehnlichkeit import build_knn_embeddings
from rotor_owl.utils.math_utils import (
    cosine_similarity,
    relu,
    berechne_gewichtete_gesamt_similarity,
)


def _extract_latent_embeddings(
    model: MLPRegressor,
    X: np.ndarray,
    latent_layer_index: int,
) -> np.ndarray:
    """
    Berechnet die Aktivierungen einer bestimmten Hidden-Layer-Stufe.
    latent_layer_index:
      0 -> nach 1. Hidden Layer
      1 -> nach 2. Hidden Layer (bei Architektur: (h1, latent, h1) ist das der Latent Space)
    """
    A = X
    # Hidden layers liegen in indices 0..(n_hidden-1)
    n_hidden = len(model.hidden_layer_sizes)  # type: ignore

    for li in range(n_hidden):
        A = A @ model.coefs_[li] + model.intercepts_[li]
        A = relu(A)  # ReLU aus math_utils

        if li == latent_layer_index:
            return A

    return A


def _normalize_embeddings_structure(
    embeddings: Any,
    rotor_ids: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    """
    Robust gegen 2 mögliche Strukturen:
    (A) embeddings[rotor_id][kategorie] = vector
    (B) embeddings[kategorie][rotor_id] = vector

    Gibt immer zurück:
      normalized[rotor_id][kategorie] = np.ndarray
    """
    if not embeddings:
        return {}

    first_key = next(iter(embeddings.keys()))
    first_val = embeddings[first_key]

    # Fall (A): embeddings[rotor_id] ist ein dict von kategorien -> vector
    if isinstance(first_val, dict):
        # Heuristik: wenn first_key ein Rotor ist
        if first_key in rotor_ids:
            out: dict[str, dict[str, np.ndarray]] = {}
            for rid in rotor_ids:
                cat_map = embeddings.get(rid, {})
                out[rid] = {k: np.asarray(v, dtype=float) for k, v in cat_map.items()}
            return out

    # Fall (B): embeddings[kategorie][rotor_id] = vector
    out_b: dict[str, dict[str, np.ndarray]] = {rid: {} for rid in rotor_ids}
    for cat, rid_map in embeddings.items():
        if not isinstance(rid_map, dict):
            continue
        for rid, vec in rid_map.items():
            if rid in out_b:
                out_b[rid][cat] = np.asarray(vec, dtype=float)
    return out_b


def build_autoencoder_embeddings(
    features_by_rotor: dict[str, dict[str, Any]],
    stats: Any,
    latent_dim: int = 8,
    seed: int = 42,
    max_iter: int = 3000,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Option C2:
    1) Reuse build_knn_embeddings(...) => Feature-Vektoren pro Kategorie (gemischt numerisch+enum)
    2) Trainiere pro Kategorie einen Autoencoder (MLPRegressor: X -> X)
    3) Nutze Latent Layer als Embedding für Cosine Similarity / kNN

    Return-Format:
      embeddings_latent[rotor_id][kategorie] = latent_vector (np.ndarray)
    """
    rotor_ids = sorted(features_by_rotor.keys())

    # (1) Feature-Vektoren aus Methode B wiederverwenden
    base = build_knn_embeddings(features_by_rotor, stats)
    base_norm = _normalize_embeddings_structure(base.vectors, rotor_ids)

    # Kategorien aus dem ersten Rotor ableiten
    categories: list[str] = sorted({c for rid in base_norm for c in base_norm[rid].keys()})

    if not categories:
        return {}

    latent_embeddings: dict[str, dict[str, np.ndarray]] = {rid: {} for rid in rotor_ids}

    # (2) Autoencoder pro Kategorie trainieren
    for cat in categories:
        # Matrix X: (n_rotors, dim)
        X_list: list[np.ndarray] = []
        for rid in rotor_ids:
            vec = base_norm.get(rid, {}).get(cat)
            if vec is None:
                # falls Rotor in der Kategorie nichts hat -> Nullvektor der passenden Länge
                # wir versuchen Länge über irgendein anderes Beispiel zu finden
                fallback = None
                for rid2 in rotor_ids:
                    if cat in base_norm.get(rid2, {}):
                        fallback = base_norm[rid2][cat]
                        break
                if fallback is None:
                    continue
                vec = np.zeros_like(fallback)
            X_list.append(vec)

        if not X_list:
            continue

        X = np.vstack(X_list).astype(float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        input_dim = X.shape[1]
        if input_dim <= 1:
            # zu wenig Info -> einfach identisch übernehmen
            for i, rid in enumerate(rotor_ids):
                latent_embeddings[rid][cat] = X[i]
            continue

        # Hidden-Größe sinnvoll wählen
        h1 = max(16, 2 * latent_dim)

        # Architektur: input -> h1 -> latent_dim -> h1 -> output=input
        # => latent layer ist Hidden-Layer index 1
        model = MLPRegressor(
            hidden_layer_sizes=(h1, latent_dim, h1),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            max_iter=max_iter,
            random_state=seed,
            early_stopping=True,
            n_iter_no_change=30,
            verbose=False,
        )

        # Trainiere Autoencoder
        model.fit(X, X)

        # (3) Latent Embeddings extrahieren (nach 2. Hidden Layer)
        Z = _extract_latent_embeddings(model, X, latent_layer_index=1)

        for i, rid in enumerate(rotor_ids):
            latent_embeddings[rid][cat] = Z[i]

    return latent_embeddings


def berechne_topk_aehnlichkeiten_autoencoder(
    query_rotor_id: str,
    rotor_ids: list[str],
    embeddings: dict[str, dict[str, np.ndarray]],
    gewichtung_pro_kategorie: dict[str, float],
    k: int = 5,
) -> list[tuple[str, float, dict[str, float]]]:
    """
    Berechnet Top-k ähnliche Rotoren mit Autoencoder-Embeddings.

    Args:
        query_rotor_id: Query-Rotor ID
        rotor_ids: Liste aller Rotor-IDs
        embeddings: Autoencoder-Embeddings im Format {rotor_id: {kategorie: vector}}
        gewichtung_pro_kategorie: Kategorie-Gewichte
        k: Anzahl Top-Ergebnisse

    Returns:
        Liste von (rotor_id, gesamt_similarity, similarity_pro_kategorie)
        Sortiert nach gesamt_similarity (absteigend)

    Methode:
        kNN/Cosine im Autoencoder-Latent-Space mit Kategorie-Gewichtung
    """
    if query_rotor_id not in rotor_ids:
        raise ValueError(f"Query Rotor nicht gefunden: {query_rotor_id}")

    results: list[tuple[str, float, dict[str, float]]] = []

    q_map = embeddings.get(query_rotor_id, {})
    categories = sorted({*q_map.keys(), *gewichtung_pro_kategorie.keys()})

    for other in rotor_ids:
        if other == query_rotor_id:
            continue

        o_map = embeddings.get(other, {})
        sim_by_cat: dict[str, float] = {}

        for cat in categories:
            w = float(gewichtung_pro_kategorie.get(cat, 0.0))
            if w <= 0.0:
                sim_by_cat[cat] = 0.0
                continue

            qv = q_map.get(cat)
            ov = o_map.get(cat)

            if qv is None or ov is None:
                sim_by_cat[cat] = 0.0
            else:
                sim_by_cat[cat] = cosine_similarity(qv, ov)

        # Gewichtete Gesamt-Similarity (zentrale Funktion)
        sim_total = berechne_gewichtete_gesamt_similarity(sim_by_cat, gewichtung_pro_kategorie)
        results.append((other, sim_total, sim_by_cat))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:k]
