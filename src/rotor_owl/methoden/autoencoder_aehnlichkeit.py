from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.neural_network import MLPRegressor

from rotor_owl.methoden.knn_aehnlichkeit import build_knn_embeddings
from rotor_owl.utils.math_utils import (
    cosine_similarity,
    relu,
    berechne_gewichtete_gesamt_similarity,
    normalisiere_embeddings_struktur,
)


def _extract_latent_embeddings(
    model: MLPRegressor,
    eingabe: np.ndarray,
    latent_layer_index: int,
) -> np.ndarray:
    """
    Berechnet Aktivierungen einer bestimmten Hidden-Layer-Stufe.

    Args:
        model (MLPRegressor): Trainiertes Autoencoder-Modell
        eingabe (np.ndarray): Eingabe-Matrix
        latent_layer_index (int): 0 = 1. Layer, 1 = 2. Layer (Latent Space)

    Returns:
        np.ndarray: Latent-Aktivierungen (OHNE ReLU für volle Repräsentation)
    """
    aktivierung = eingabe
    anzahl_hidden = len(model.hidden_layer_sizes)  # type: ignore

    for layer_idx in range(anzahl_hidden):
        aktivierung = aktivierung @ model.coefs_[layer_idx] + model.intercepts_[layer_idx]

        if layer_idx == latent_layer_index:
            return aktivierung  # Latent Space OHNE ReLU zurückgeben!

        aktivierung = relu(aktivierung)

    return aktivierung


# _normalize_embeddings_structure wurde nach math_utils.normalisiere_embeddings_struktur verschoben


def build_autoencoder_embeddings(
    features_by_rotor: dict[str, dict[str, Any]],
    stats: Any,
    latent_dim: int = 16,
    seed: int = 42,
    max_iter: int = 3000,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Baut Autoencoder-Embeddings für alle Rotoren.

    Args:
        features_by_rotor (dict): Feature-Daten aller Rotoren
        stats: Min/Max-Statistiken
        latent_dim (int): Dimensionalität des Latent Space
        seed (int): Random Seed
        max_iter (int): Maximale Trainingsiterationen

    Returns:
        dict: {rotor_id: {kategorie: latent_vector}}
    """
    rotor_ids = sorted(features_by_rotor.keys())

    base = build_knn_embeddings(features_by_rotor, stats)
    base_norm = normalisiere_embeddings_struktur(base.vectors, rotor_ids)

    kategorien: list[str] = sorted(
        {kat for rotor_id in base_norm for kat in base_norm[rotor_id].keys()}
    )

    if not kategorien:
        return {}

    latent_embeddings: dict[str, dict[str, np.ndarray]] = {rotor_id: {} for rotor_id in rotor_ids}

    for kategorie in kategorien:
        # Matrix erstellen: (n_rotors, dim)
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
        eingabe_matrix = np.nan_to_num(eingabe_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        input_dim = eingabe_matrix.shape[1]
        if input_dim <= 1:
            # Zu wenig Dimensionen -> identisch übernehmen
            for idx, rotor_id in enumerate(rotor_ids):
                latent_embeddings[rotor_id][kategorie] = eingabe_matrix[idx]
            continue

        # Hidden-Größe für gute Informationserhaltung
        hidden_size = max(64, 4 * latent_dim)

        # Architektur: input -> hidden -> latent -> hidden -> output
        model = MLPRegressor(
            hidden_layer_sizes=(hidden_size, latent_dim, hidden_size),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            max_iter=max_iter,
            random_state=seed,
            early_stopping=True,
            n_iter_no_change=30,
            verbose=False,
        )

        model.fit(eingabe_matrix, eingabe_matrix)

        latent_matrix = _extract_latent_embeddings(model, eingabe_matrix, latent_layer_index=1)

        for idx, rotor_id in enumerate(rotor_ids):
            latent_embeddings[rotor_id][kategorie] = latent_matrix[idx]

    return latent_embeddings


def berechne_topk_aehnlichkeiten_autoencoder(
    query_rotor_id: str,
    rotor_ids: list[str],
    embeddings: dict[str, dict[str, np.ndarray]],
    gewichtung_pro_kategorie: dict[str, float],
    top_k: int = 5,
) -> list[tuple[str, float, dict[str, float]]]:
    """
    Berechnet Top-k ähnliche Rotoren mit Autoencoder-Embeddings.

    Args:
        query_rotor_id (str): Query-Rotor ID
        rotor_ids (list): Liste aller Rotor-IDs
        embeddings (dict): Autoencoder-Embeddings {rotor_id: {kategorie: vector}}
        gewichtung_pro_kategorie (dict): Kategorie-Gewichte
        top_k (int): Anzahl Top-Ergebnisse

    Returns:
        list: (rotor_id, gesamt_similarity, similarity_pro_kategorie), sortiert
    """
    if query_rotor_id not in rotor_ids:
        raise ValueError(f"Query Rotor nicht gefunden: {query_rotor_id}")

    ergebnisse: list[tuple[str, float, dict[str, float]]] = []

    query_vektoren = embeddings.get(query_rotor_id, {})
    kategorien = sorted({*query_vektoren.keys(), *gewichtung_pro_kategorie.keys()})

    for ziel_rotor_id in rotor_ids:
        if ziel_rotor_id == query_rotor_id:
            continue

        ziel_vektoren = embeddings.get(ziel_rotor_id, {})
        sim_pro_kat: dict[str, float] = {}

        for kategorie in kategorien:
            gewicht = float(gewichtung_pro_kategorie.get(kategorie, 0.0))
            if gewicht <= 0.0:
                sim_pro_kat[kategorie] = 0.0
                continue

            query_vektor = query_vektoren.get(kategorie)
            ziel_vektor = ziel_vektoren.get(kategorie)

            if query_vektor is None or ziel_vektor is None:
                sim_pro_kat[kategorie] = 0.0
            else:
                # Cosine gibt [-1, 1], normalisieren auf [0, 1]
                raw_sim = cosine_similarity(query_vektor, ziel_vektor)
                sim_pro_kat[kategorie] = (raw_sim + 1.0) / 2.0

        sim_total = berechne_gewichtete_gesamt_similarity(sim_pro_kat, gewichtung_pro_kategorie)
        ergebnisse.append((ziel_rotor_id, sim_total, sim_pro_kat))

    ergebnisse.sort(key=lambda x: x[1], reverse=True)
    return ergebnisse[:top_k]
