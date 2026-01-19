from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rotor_owl.methoden.knn_aehnlichkeit import build_knn_embeddings
from rotor_owl.utils.math_utils import cosine_similarity, berechne_gewichtete_gesamt_similarity


def _row_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return X / norms


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

    # Achtung: embeddings muss ein dict sein (bei uns: base.vectors)
    if not isinstance(embeddings, dict):
        return {}

    first_key = next(iter(embeddings.keys()))
    first_val = embeddings[first_key]

    # Fall (A): embeddings[rotor_id] = {cat: vec}
    if isinstance(first_val, dict) and first_key in rotor_ids:
        out: dict[str, dict[str, np.ndarray]] = {}
        for rid in rotor_ids:
            cat_map = embeddings.get(rid, {})
            out[rid] = {k: np.asarray(v, dtype=float) for k, v in cat_map.items()}
        return out

    # Fall (B): embeddings[cat] = {rid: vec}
    out_b: dict[str, dict[str, np.ndarray]] = {rid: {} for rid in rotor_ids}
    for cat, rid_map in embeddings.items():
        if not isinstance(rid_map, dict):
            continue
        for rid, vec in rid_map.items():
            if rid in out_b:
                out_b[rid][cat] = np.asarray(vec, dtype=float)
    return out_b


@dataclass(frozen=True)
class _KMeansResult:
    labels_by_rotor: dict[str, int]  # rotor_id -> cluster_id
    centroids: np.ndarray  # shape: (k, dim)


def _spherical_kmeans(
    X: np.ndarray,
    rotor_ids: list[str],
    k: int,
    seed: int = 42,
    max_iter: int = 50,
) -> _KMeansResult:
    """
    Custom "Spherical K-Means":
    - arbeitet auf L2-normalisierten Vektoren
    - Assignment über max(cosine) = max(dot) weil normiert
    - Update: centroid = normalisierter Mittelwert der Cluster-Vektoren
    """
    rng = np.random.default_rng(seed)

    n, dim = X.shape
    if n == 0:
        return _KMeansResult(labels_by_rotor={}, centroids=np.zeros((0, dim), dtype=float))

    k_eff = max(1, min(k, n))  # nicht mehr Cluster als Punkte
    Xn = _row_normalize(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0))

    # Init: zufällige k Punkte als Start-Zentroiden
    init_idx = rng.choice(n, size=k_eff, replace=False)
    C = Xn[init_idx].copy()  # (k, dim)

    labels = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        # Assignment: Cluster = argmax(dot(X, C))
        sims = Xn @ C.T  # (n, k)
        new_labels = np.argmax(sims, axis=1)

        if np.array_equal(new_labels, labels):
            break

        labels = new_labels

        # Update Zentroiden
        for j in range(k_eff):
            mask = labels == j
            if not np.any(mask):
                # leeres Cluster -> random re-init
                C[j] = Xn[rng.integers(0, n)]
            else:
                cj = np.mean(Xn[mask], axis=0)
                norm = np.linalg.norm(cj)
                C[j] = cj / norm if norm != 0.0 else cj

    labels_by_rotor = {rid: int(labels[i]) for i, rid in enumerate(rotor_ids)}
    return _KMeansResult(labels_by_rotor=labels_by_rotor, centroids=C)


def build_kmeans_models(
    features_by_rotor: dict[str, dict[str, Any]],
    stats: Any,
    n_clusters: int = 5,
    seed: int = 42,
    max_iter: int = 50,
) -> dict[str, _KMeansResult]:
    """
    Option D:
    - Reuse Feature-Vektoren aus build_knn_embeddings (Option B)
    - Pro Kategorie: Spherical K-Means clustern

    Return:
      models[category] = _KMeansResult(labels_by_rotor, centroids)
    """
    rotor_ids = sorted(features_by_rotor.keys())

    base = build_knn_embeddings(features_by_rotor, stats)
    base_norm = _normalize_embeddings_structure(base.vectors, rotor_ids)

    # Kategorien ableiten
    categories = sorted({c for rid in base_norm for c in base_norm[rid].keys()})
    if not categories:
        return {}

    models: dict[str, _KMeansResult] = {}

    for cat in categories:
        X_list: list[np.ndarray] = []
        for rid in rotor_ids:
            vec = base_norm.get(rid, {}).get(cat)
            if vec is None:
                # fallback = irgendein Vektor aus der Kategorie -> Nullvektor gleicher Länge
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
        if X.shape[1] <= 1:
            # zu wenig Dim -> Clustering bringt nix
            # => 1 Cluster für alle
            labels_by_rotor = {rid: 0 for rid in rotor_ids}
            centroids = _row_normalize(np.mean(X, axis=0, keepdims=True))
            models[cat] = _KMeansResult(labels_by_rotor=labels_by_rotor, centroids=centroids)
            continue

        models[cat] = _spherical_kmeans(
            X=X,
            rotor_ids=rotor_ids,
            k=n_clusters,
            seed=seed,
            max_iter=max_iter,
        )

    return models


def berechne_topk_aehnlichkeiten_kmeans(
    query_rotor_id: str,
    rotor_ids: list[str],
    features_by_rotor: dict[str, dict[str, Any]],
    stats: Any,
    gewichtung_pro_kategorie: dict[str, float],
    n_clusters: int = 5,
    k: int = 5,
    seed: int = 42,
    max_iter: int = 50,
) -> list[tuple[str, float, dict[str, float]]]:
    """
    Similarity-Scoring (Option D):
    - Clustering pro Kategorie mit Custom Spherical K-Means
    - pro Kategorie:
        wenn gleicher Cluster: cosine(feature_vector_query, feature_vector_other)
        sonst: cosine(centroid_cluster_query, centroid_cluster_other) * 0.5
    - Dann gewichtete Aggregation wie immer
    """
    if query_rotor_id not in rotor_ids:
        raise ValueError(f"Query Rotor nicht gefunden: {query_rotor_id}")

    # Reuse Feature-Vektoren aus Methode B
    base = build_knn_embeddings(features_by_rotor, stats)
    base_norm = _normalize_embeddings_structure(base.vectors, rotor_ids)

    # Modelle (Cluster) bauen
    models = build_kmeans_models(
        features_by_rotor=features_by_rotor,
        stats=stats,
        n_clusters=n_clusters,
        seed=seed,
        max_iter=max_iter,
    )

    q_map = base_norm.get(query_rotor_id, {})
    categories = sorted({*q_map.keys(), *gewichtung_pro_kategorie.keys()})

    results: list[tuple[str, float, dict[str, float]]] = []

    for other in rotor_ids:
        if other == query_rotor_id:
            continue

        o_map = base_norm.get(other, {})

        sim_by_cat: dict[str, float] = {}

        for cat in categories:
            w = float(gewichtung_pro_kategorie.get(cat, 0.0))
            if w <= 0.0:
                sim_by_cat[cat] = 0.0
                continue

            qv = q_map.get(cat)
            ov = o_map.get(cat)

            if qv is None or ov is None:
                s = 0.0
            else:
                model = models.get(cat)
                if model is None:
                    # fallback: direkt cosine
                    s = cosine_similarity(qv, ov)
                else:
                    q_label = model.labels_by_rotor.get(query_rotor_id, -1)
                    o_label = model.labels_by_rotor.get(other, -1)

                    if q_label == -1 or o_label == -1:
                        s = cosine_similarity(qv, ov)
                    elif q_label == o_label:
                        s = cosine_similarity(qv, ov)
                    else:
                        cq = model.centroids[q_label]
                        co = model.centroids[o_label]
                        s = 0.5 * max(0.0, cosine_similarity(cq, co))

            sim_by_cat[cat] = s

        # Gewichtete Gesamt-Similarity (zentrale Funktion)
        sim_total = berechne_gewichtete_gesamt_similarity(sim_by_cat, gewichtung_pro_kategorie)
        results.append((other, sim_total, sim_by_cat))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:k]
