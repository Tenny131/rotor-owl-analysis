from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rotor_owl.config.kategorien import KATEGORIEN_3
from rotor_owl.methoden.vektorbasierte_aehnlichkeit import (
    build_vektor_embeddings,
    _VektorEmbeddings,
)
from rotor_owl.utils.math_utils import cosine_similarity, berechne_gewichtete_gesamt_similarity


@dataclass(frozen=True)
class _PCAEmbeddings:
    """PCA-reduzierte Embeddings pro Kategorie."""

    vectors_latent: dict[str, dict[str, np.ndarray]]
    latent_dim: int


def _pca_fit_transform(eingabe_matrix: np.ndarray, latent_dim: int) -> np.ndarray:
    """
    PCA via SVD.

    Args:
        eingabe_matrix (np.ndarray): Shape (n_samples, n_features)
        latent_dim (int): Zieldimension

    Returns:
        np.ndarray: Reduzierte Matrix mit Shape (n_samples, latent_dim)
    """
    if eingabe_matrix.size == 0:
        return np.zeros((eingabe_matrix.shape[0], 0), dtype=float)

    # Zentrieren
    zentriert = eingabe_matrix - eingabe_matrix.mean(axis=0, keepdims=True)

    # SVD: zentriert = U S Vt
    komponenten_u, singular_werte, komponenten_vt = np.linalg.svd(zentriert, full_matrices=False)

    # Begrenzung auf verfügbare Dimensionen
    dim = min(latent_dim, komponenten_vt.shape[0])
    if dim <= 0:
        return np.zeros((eingabe_matrix.shape[0], 0), dtype=float)

    # Projektion auf erste dim Komponenten
    latent = zentriert @ komponenten_vt[:dim].T
    return latent


def build_pca_embeddings(
    features_by_rotor: dict[str, dict],
    stats: dict[tuple[str, str], tuple[float, float]],
    latent_dim: int = 16,
) -> _PCAEmbeddings:
    """
    Baut PCA-reduzierte Embeddings für alle Rotoren.

    Args:
        features_by_rotor (dict): Feature-Daten aller Rotoren
        stats (dict): Min/Max-Statistiken
        latent_dim (int): Zieldimension für PCA

    Returns:
        _PCAEmbeddings: PCA-reduzierte Vektoren pro Kategorie
    """
    vektor_embeddings: _VektorEmbeddings = build_vektor_embeddings(features_by_rotor, stats)

    rotor_ids = sorted(features_by_rotor.keys())
    vectors_latent: dict[str, dict[str, np.ndarray]] = {}

    for kategorie in KATEGORIEN_3:
        # Matrix bauen: (n_rotors x dim)
        vektoren_liste = []
        for rotor_id in rotor_ids:
            vektoren_liste.append(vektor_embeddings.vectors[kategorie][rotor_id])
        eingabe_matrix = (
            np.vstack(vektoren_liste) if vektoren_liste else np.zeros((0, 0), dtype=float)
        )

        latent_matrix = _pca_fit_transform(eingabe_matrix, latent_dim=latent_dim)

        vectors_latent[kategorie] = {}
        for idx, rotor_id in enumerate(rotor_ids):
            vectors_latent[kategorie][rotor_id] = latent_matrix[idx]

    return _PCAEmbeddings(vectors_latent=vectors_latent, latent_dim=latent_dim)


def rotor_similarity_pca(
    rotor_a_id: str,
    rotor_b_id: str,
    embeddings: _PCAEmbeddings,
    gewichtung_pro_kategorie: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """
    Berechnet Rotor-Similarity mit PCA-Latent-Space.

    Args:
        rotor_a_id (str): ID des ersten Rotors
        rotor_b_id (str): ID des zweiten Rotors
        embeddings (_PCAEmbeddings): PCA-reduzierte Embeddings
        gewichtung_pro_kategorie (dict): Gewichte für die 3 Kategorien

    Returns:
        tuple: (gesamt_similarity, similarity_pro_kategorie)
    """
    sim_pro_kat: dict[str, float] = {}

    for kategorie, vektor_dict in embeddings.vectors_latent.items():
        vektor_a = vektor_dict[rotor_a_id]
        vektor_b = vektor_dict[rotor_b_id]
        # Cosine gibt [-1, 1], normalisieren auf [0, 1]
        raw_sim = cosine_similarity(vektor_a, vektor_b)
        sim_pro_kat[kategorie] = (raw_sim + 1.0) / 2.0

    total = berechne_gewichtete_gesamt_similarity(sim_pro_kat, gewichtung_pro_kategorie)

    return total, sim_pro_kat


def berechne_topk_aehnlichkeiten_pca(
    query_rotor_id: str,
    rotor_ids: list[str],
    embeddings: _PCAEmbeddings,
    gewichtung_pro_kategorie: dict[str, float],
    top_k: int,
) -> list[tuple[str, float, dict[str, float]]]:
    """
    Berechnet die Top-k ähnlichsten Rotoren mit PCA.

    Args:
        query_rotor_id (str): ID des Abfrage-Rotors
        rotor_ids (list): Liste aller Rotor-IDs
        embeddings (_PCAEmbeddings): PCA-Embeddings
        gewichtung_pro_kategorie (dict): Kategorie-Gewichte
        top_k (int): Anzahl der Ergebnisse

    Returns:
        list: Top-k Ergebnisse als (rotor_id, similarity, sim_pro_kat)
    """
    ergebnisse: list[tuple[str, float, dict[str, float]]] = []

    for ziel_rotor_id in rotor_ids:
        if ziel_rotor_id == query_rotor_id:
            continue

        total, sim_pro_kat = rotor_similarity_pca(
            rotor_a_id=query_rotor_id,
            rotor_b_id=ziel_rotor_id,
            embeddings=embeddings,
            gewichtung_pro_kategorie=gewichtung_pro_kategorie,
        )
        ergebnisse.append((ziel_rotor_id, total, sim_pro_kat))

    ergebnisse.sort(key=lambda x: x[1], reverse=True)
    return ergebnisse[:top_k]
