from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rotor_owl.config.kategorien import KATEGORIEN_3
from rotor_owl.methoden.knn_aehnlichkeit import build_knn_embeddings, _KNNEmbeddings
from rotor_owl.utils.math_utils import cosine_similarity, berechne_gewichtete_gesamt_similarity


@dataclass(frozen=True)
class _PCAEmbeddings:
    # Kategorie -> RotorID -> PCA-Latentvektor
    vectors_latent: dict[str, dict[str, np.ndarray]]
    latent_dim: int


def _pca_fit_transform(X: np.ndarray, latent_dim: int) -> np.ndarray:
    """
    PCA via SVD:
    X: shape (n_samples, n_features)
    Return: Z shape (n_samples, latent_dim)
    """
    if X.size == 0:
        return np.zeros((X.shape[0], 0), dtype=float)

    # Zentrieren
    Xc = X - X.mean(axis=0, keepdims=True)

    # SVD
    # Xc = U S Vt
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

    # Begrenzung
    d = min(latent_dim, Vt.shape[0])
    if d <= 0:
        return np.zeros((X.shape[0], 0), dtype=float)

    # Projektion auf erste d Komponenten
    Z = Xc @ Vt[:d].T
    return Z


def build_pca_embeddings(
    features_by_rotor: dict[str, dict],
    stats: dict[tuple[str, str], tuple[float, float]],
    latent_dim: int = 8,
) -> _PCAEmbeddings:
    """
    Option C:
    - nutzt Vektorisierung aus Option B (mixed numeric + enum)
    - reduziert je Kategorie mit PCA auf latent space
    - Similarity später über Cosine im latent space
    """
    knn_embeddings: _KNNEmbeddings = build_knn_embeddings(features_by_rotor, stats)

    rotor_ids = sorted(features_by_rotor.keys())
    vectors_latent: dict[str, dict[str, np.ndarray]] = {}

    for cat in KATEGORIEN_3:
        # Matrix bauen: (n_rotors x dim)
        mats = []
        for rid in rotor_ids:
            mats.append(knn_embeddings.vectors[cat][rid])
        X = np.vstack(mats) if mats else np.zeros((0, 0), dtype=float)

        Z = _pca_fit_transform(X, latent_dim=latent_dim)

        vectors_latent[cat] = {}
        for i, rid in enumerate(rotor_ids):
            vectors_latent[cat][rid] = Z[i]

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
        rotor_a_id: ID des ersten Rotors
        rotor_b_id: ID des zweiten Rotors
        embeddings: PCA-reduzierte Embeddings pro Kategorie
        gewichtung_pro_kategorie: Gewichte für die 3 Kategorien

    Returns:
        Tuple aus (gesamt_similarity, similarity_pro_kategorie)

    Methode:
        - Pro Kategorie: Cosine-Similarity im PCA-Latent-Space
        - Gesamt: Gewichtetes Mittel
    """
    sim_pro_kat: dict[str, float] = {}

    for cat, vecs in embeddings.vectors_latent.items():
        va = vecs[rotor_a_id]
        vb = vecs[rotor_b_id]
        sim_pro_kat[cat] = cosine_similarity(va, vb)

    # Gewichtete Aggregation (zentrale Funktion)
    total = berechne_gewichtete_gesamt_similarity(sim_pro_kat, gewichtung_pro_kategorie)

    return total, sim_pro_kat


def berechne_topk_aehnlichkeiten_pca(
    query_rotor_id: str,
    rotor_ids: list[str],
    embeddings: _PCAEmbeddings,
    gewichtung_pro_kategorie: dict[str, float],
    k: int,
) -> list[tuple[str, float, dict[str, float]]]:
    ergebnisse: list[tuple[str, float, dict[str, float]]] = []

    for ziel in rotor_ids:
        if ziel == query_rotor_id:
            continue

        total, sim_pro_kat = rotor_similarity_pca(
            rotor_a_id=query_rotor_id,
            rotor_b_id=ziel,
            embeddings=embeddings,
            gewichtung_pro_kategorie=gewichtung_pro_kategorie,
        )
        ergebnisse.append((ziel, total, sim_pro_kat))

    ergebnisse.sort(key=lambda x: x[1], reverse=True)
    return ergebnisse[:k]
