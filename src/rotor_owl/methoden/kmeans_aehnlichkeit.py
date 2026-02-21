from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rotor_owl.methoden.vektorbasierte_aehnlichkeit import build_vektor_embeddings
from rotor_owl.utils.math_utils import (
    cosine_similarity,
    berechne_gewichtete_gesamt_similarity,
    normalisiere_embeddings_struktur,
)


def _zeilen_normalisieren(matrix: np.ndarray) -> np.ndarray:
    """
    L2-Normalisierung jeder Zeile.

    Args:
        matrix (np.ndarray): Eingabematrix

    Returns:
        np.ndarray: Zeilennormalisierte Matrix
    """
    normen = np.linalg.norm(matrix, axis=1, keepdims=True)
    normen = np.where(normen == 0.0, 1.0, normen)
    return matrix / normen


@dataclass(frozen=True)
class _KMeansResult:
    """K-Means Clustering-Ergebnis."""

    labels_by_rotor: dict[str, int]  # rotor_id -> cluster_id
    centroids: np.ndarray  # shape: (k, dim)


def _spherical_kmeans(
    eingabe_matrix: np.ndarray,
    rotor_ids: list[str],
    anzahl_cluster: int,
    seed: int = 42,
    max_iter: int = 50,
) -> _KMeansResult:
    """
    Spherical K-Means auf L2-normalisierten Vektoren.

    Args:
        eingabe_matrix (np.ndarray): Feature-Matrix (n_samples, dim)
        rotor_ids (list): Rotor-IDs
        anzahl_cluster (int): Anzahl Cluster
        seed (int): Random Seed
        max_iter (int): Maximale Iterationen

    Returns:
        _KMeansResult: Cluster-Labels und Zentroide
    """
    rng = np.random.default_rng(seed)

    anzahl_samples, dim = eingabe_matrix.shape
    if anzahl_samples == 0:
        return _KMeansResult(labels_by_rotor={}, centroids=np.zeros((0, dim), dtype=float))

    effektive_cluster = max(1, min(anzahl_cluster, anzahl_samples))
    normalisiert = _zeilen_normalisieren(
        np.nan_to_num(eingabe_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    )

    # Init: zufällige Punkte als Start-Zentroide
    init_indices = rng.choice(anzahl_samples, size=effektive_cluster, replace=False)
    zentroide = normalisiert[init_indices].copy()

    labels = np.zeros(anzahl_samples, dtype=int)

    for _ in range(max_iter):
        # Assignment: Cluster = argmax(dot(X, C))
        aehnlichkeiten = normalisiert @ zentroide.T
        neue_labels = np.argmax(aehnlichkeiten, axis=1)

        if np.array_equal(neue_labels, labels):
            break

        labels = neue_labels

        # Update Zentroide
        for cluster_idx in range(effektive_cluster):
            maske = labels == cluster_idx
            if not np.any(maske):
                # Leeres Cluster -> random re-init
                zentroide[cluster_idx] = normalisiert[rng.integers(0, anzahl_samples)]
            else:
                mittelwert = np.mean(normalisiert[maske], axis=0)
                norm = np.linalg.norm(mittelwert)
                zentroide[cluster_idx] = mittelwert / norm if norm != 0.0 else mittelwert

    labels_by_rotor = {rotor_id: int(labels[idx]) for idx, rotor_id in enumerate(rotor_ids)}
    return _KMeansResult(labels_by_rotor=labels_by_rotor, centroids=zentroide)


def build_kmeans_models(
    features_by_rotor: dict[str, dict[str, Any]],
    stats: Any,
    n_clusters: int = 5,
    seed: int = 42,
    max_iter: int = 50,
) -> dict[str, _KMeansResult]:
    """
    Baut K-Means Modelle pro Kategorie.

    Args:
        features_by_rotor (dict): Feature-Daten aller Rotoren
        stats: Min/Max-Statistiken
        n_clusters (int): Anzahl Cluster
        seed (int): Random Seed
        max_iter (int): Maximale Iterationen

    Returns:
        dict: {kategorie: _KMeansResult}
    """
    rotor_ids = sorted(features_by_rotor.keys())

    base = build_vektor_embeddings(features_by_rotor, stats)
    base_norm = normalisiere_embeddings_struktur(base.vectors, rotor_ids)

    kategorien = sorted({kat for rotor_id in base_norm for kat in base_norm[rotor_id].keys()})
    if not kategorien:
        return {}

    modelle: dict[str, _KMeansResult] = {}

    for kategorie in kategorien:
        vektoren_liste: list[np.ndarray] = []
        for rotor_id in rotor_ids:
            vektor = base_norm.get(rotor_id, {}).get(kategorie)
            if vektor is None:
                # Fallback: Nullvektor der passenden Länge
                fallback = None
                for anderer_rotor in rotor_ids:
                    if kategorie in base_norm.get(anderer_rotor, {}):
                        fallback = base_norm[anderer_rotor][kategorie]
                        break
                if fallback is None:
                    continue
                vektor = np.zeros_like(fallback)
            vektoren_liste.append(vektor)

        if not vektoren_liste:
            continue

        eingabe_matrix = np.vstack(vektoren_liste).astype(float)
        if eingabe_matrix.shape[1] <= 1:
            # Zu wenig Dimensionen -> 1 Cluster für alle
            labels_by_rotor = {rotor_id: 0 for rotor_id in rotor_ids}
            zentroide = _zeilen_normalisieren(np.mean(eingabe_matrix, axis=0, keepdims=True))
            modelle[kategorie] = _KMeansResult(labels_by_rotor=labels_by_rotor, centroids=zentroide)
            continue

        modelle[kategorie] = _spherical_kmeans(
            eingabe_matrix=eingabe_matrix,
            rotor_ids=rotor_ids,
            anzahl_cluster=n_clusters,
            seed=seed,
            max_iter=max_iter,
        )

    return modelle


def berechne_topk_aehnlichkeiten_kmeans(
    query_rotor_id: str,
    rotor_ids: list[str],
    features_by_rotor: dict[str, dict[str, Any]],
    stats: Any,
    gewichtung_pro_kategorie: dict[str, float],
    n_clusters: int = 5,
    top_k: int = 5,
    seed: int = 42,
    max_iter: int = 50,
) -> list[tuple[str, float, dict[str, float]]]:
    """
    Berechnet Top-k ähnliche Rotoren mit K-Means Clustering.

    Args:
        query_rotor_id (str): Query-Rotor ID
        rotor_ids (list): Liste aller Rotor-IDs
        features_by_rotor (dict): Feature-Daten
        stats: Min/Max-Statistiken
        gewichtung_pro_kategorie (dict): Kategorie-Gewichte
        n_clusters (int): Anzahl Cluster
        top_k (int): Anzahl Top-Ergebnisse
        seed (int): Random Seed
        max_iter (int): Maximale Iterationen

    Returns:
        list: (rotor_id, gesamt_similarity, similarity_pro_kategorie), sortiert
    """
    if query_rotor_id not in rotor_ids:
        raise ValueError(f"Query Rotor nicht gefunden: {query_rotor_id}")

    base = build_vektor_embeddings(features_by_rotor, stats)
    base_norm = normalisiere_embeddings_struktur(base.vectors, rotor_ids)

    modelle = build_kmeans_models(
        features_by_rotor=features_by_rotor,
        stats=stats,
        n_clusters=n_clusters,
        seed=seed,
        max_iter=max_iter,
    )

    query_vektoren = base_norm.get(query_rotor_id, {})
    kategorien = sorted({*query_vektoren.keys(), *gewichtung_pro_kategorie.keys()})

    ergebnisse: list[tuple[str, float, dict[str, float]]] = []

    for ziel_rotor_id in rotor_ids:
        if ziel_rotor_id == query_rotor_id:
            continue

        ziel_vektoren = base_norm.get(ziel_rotor_id, {})
        sim_pro_kat: dict[str, float] = {}

        for kategorie in kategorien:
            gewicht = float(gewichtung_pro_kategorie.get(kategorie, 0.0))
            if gewicht <= 0.0:
                sim_pro_kat[kategorie] = 0.0
                continue

            query_vektor = query_vektoren.get(kategorie)
            ziel_vektor = ziel_vektoren.get(kategorie)

            if query_vektor is None or ziel_vektor is None:
                similarity = 0.0
            else:
                modell = modelle.get(kategorie)
                if modell is None:
                    # Fallback: direkt Cosine, normalisiert auf [0, 1]
                    similarity = (cosine_similarity(query_vektor, ziel_vektor) + 1.0) / 2.0
                else:
                    query_label = modell.labels_by_rotor.get(query_rotor_id, -1)
                    ziel_label = modell.labels_by_rotor.get(ziel_rotor_id, -1)

                    if query_label == -1 or ziel_label == -1:
                        similarity = (cosine_similarity(query_vektor, ziel_vektor) + 1.0) / 2.0
                    elif query_label == ziel_label:
                        # Gleicher Cluster: direkte Similarity
                        similarity = (cosine_similarity(query_vektor, ziel_vektor) + 1.0) / 2.0
                    else:
                        # Verschiedene Cluster: halbe Zentroid-Similarity
                        query_zentroid = modell.centroids[query_label]
                        ziel_zentroid = modell.centroids[ziel_label]
                        similarity = 0.5 * (
                            (cosine_similarity(query_zentroid, ziel_zentroid) + 1.0) / 2.0
                        )

            sim_pro_kat[kategorie] = similarity

        sim_total = berechne_gewichtete_gesamt_similarity(sim_pro_kat, gewichtung_pro_kategorie)
        ergebnisse.append((ziel_rotor_id, sim_total, sim_pro_kat))

    ergebnisse.sort(key=lambda x: x[1], reverse=True)
    return ergebnisse[:top_k]
