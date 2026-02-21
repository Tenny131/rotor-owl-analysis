"""Modulare Hybrid-Similarity: Kombiniert beliebige Methoden mit konfigurierbaren Gewichten."""

from __future__ import annotations

from typing import Any

from rotor_owl.methoden.regelbasierte_aehnlichkeit import berechne_topk_aehnlichkeiten
from rotor_owl.methoden.vektorbasierte_aehnlichkeit import (
    build_vektor_embeddings,
    berechne_topk_aehnlichkeiten_vektorbasiert,
)
from rotor_owl.methoden.pca_aehnlichkeit import (
    build_pca_embeddings,
    berechne_topk_aehnlichkeiten_pca,
)
from rotor_owl.methoden.autoencoder_aehnlichkeit import (
    build_autoencoder_embeddings,
    berechne_topk_aehnlichkeiten_autoencoder,
)
from rotor_owl.methoden.kmeans_aehnlichkeit import berechne_topk_aehnlichkeiten_kmeans


# Typ-Alias für Methoden-Ergebnisse: Rotor-ID -> (similarity, sim_pro_kategorie)
MethodenErgebnis = dict[str, tuple[float, dict[str, float]]]


def _berechne_einzelne_methode(
    methoden_name: str,
    query_rotor_id: str,
    rotor_ids: list[str],
    features_by_rotor: dict[str, Any],
    stats: dict[tuple[str, str], tuple[float, float]],
    gewichtung_pro_kategorie: dict[str, float],
    latent_dim: int,
    n_clusters: int,
) -> MethodenErgebnis:
    """
    Berechnet Similarity-Ergebnisse für eine einzelne Methode.

    Args:
        methoden_name (str): Name der Methode
        query_rotor_id (str): Query-Rotor ID
        rotor_ids (list[str]): Alle Rotor-IDs
        features_by_rotor (dict): Feature-Daten pro Rotor
        stats (dict): Statistiken für Normierung
        gewichtung_pro_kategorie (dict): Kategorie-Gewichte
        latent_dim (int): Latent-Dimension für PCA/Autoencoder
        n_clusters (int): Cluster-Anzahl für K-Means

    Returns:
        MethodenErgebnis: Rotor-ID -> (similarity, sim_pro_kategorie)
    """
    anzahl_rotoren = len(rotor_ids)
    ergebnisse: list[tuple[str, float, dict[str, float]]] = []

    if methoden_name == "Regelbasiert (kein ML)":
        ergebnisse = berechne_topk_aehnlichkeiten(
            query_rotor_id=query_rotor_id,
            rotor_ids=rotor_ids,
            features_by_rotor=features_by_rotor,
            stats=stats,
            gewichtung_pro_kategorie=gewichtung_pro_kategorie,
            top_k=anzahl_rotoren,
        )

    elif methoden_name == "Vektorbasiert (ML)":
        embeddings = build_vektor_embeddings(features_by_rotor, stats)
        ergebnisse = berechne_topk_aehnlichkeiten_vektorbasiert(
            query_rotor_id=query_rotor_id,
            rotor_ids=rotor_ids,
            embeddings=embeddings,
            gewichtung_pro_kategorie=gewichtung_pro_kategorie,
            top_k=anzahl_rotoren,
        )

    elif methoden_name == "PCA-Embedding (ML)":
        pca_embeddings = build_pca_embeddings(features_by_rotor, stats, latent_dim)
        ergebnisse = berechne_topk_aehnlichkeiten_pca(
            query_rotor_id=query_rotor_id,
            rotor_ids=rotor_ids,
            embeddings=pca_embeddings,
            gewichtung_pro_kategorie=gewichtung_pro_kategorie,
            top_k=anzahl_rotoren,
        )

    elif methoden_name == "Autoencoder (ML)":
        ae_embeddings = build_autoencoder_embeddings(features_by_rotor, stats, latent_dim)
        ergebnisse = berechne_topk_aehnlichkeiten_autoencoder(
            query_rotor_id=query_rotor_id,
            rotor_ids=rotor_ids,
            embeddings=ae_embeddings,
            gewichtung_pro_kategorie=gewichtung_pro_kategorie,
            top_k=anzahl_rotoren,
        )

    elif methoden_name == "K-Means Clustering (ML)":
        ergebnisse = berechne_topk_aehnlichkeiten_kmeans(
            query_rotor_id=query_rotor_id,
            rotor_ids=rotor_ids,
            features_by_rotor=features_by_rotor,
            stats=stats,
            gewichtung_pro_kategorie=gewichtung_pro_kategorie,
            n_clusters=n_clusters,
            top_k=anzahl_rotoren,
        )

    else:
        return {}

    return {rotor_id: (sim_total, sim_pro_kat) for rotor_id, sim_total, sim_pro_kat in ergebnisse}


def berechne_topk_aehnlichkeiten_hybrid(
    query_rotor_id: str,
    rotor_ids: list[str],
    features_by_rotor: dict[str, Any],
    stats: dict[tuple[str, str], tuple[float, float]],
    gewichtung_pro_kategorie: dict[str, float],
    methoden_gewichte: dict[str, float],
    latent_dim: int = 16,
    n_clusters: int = 5,
    top_k: int = 5,
) -> list[tuple[str, float, dict[str, float], dict[str, float]]]:
    """
    Kombiniert mehrere Similarity-Methoden zu einer gewichteten Hybrid-Similarity.

    Args:
        query_rotor_id (str): Rotor-ID für die Suche
        rotor_ids (list): Liste aller Rotor-IDs
        features_by_rotor (dict): Feature-Daten pro Rotor
        stats (dict): Statistiken für numerische Normierung
        gewichtung_pro_kategorie (dict): Gewichte für Kategorien
        methoden_gewichte (dict): Gewichte für jede Methode
        latent_dim (int): Latent Dimension für PCA/Autoencoder
        n_clusters (int): Anzahl Cluster für K-Means
        top_k (int): Anzahl der Top-Ergebnisse

    Returns:
        list: (rotor_id, sim_total, sim_pro_kategorie, sim_pro_methode)
    """
    # Berechne nur aktivierte Methoden
    methoden_ergebnisse: dict[str, MethodenErgebnis] = {}

    for methoden_name, gewicht in methoden_gewichte.items():
        if gewicht <= 0:
            continue

        methoden_ergebnisse[methoden_name] = _berechne_einzelne_methode(
            methoden_name=methoden_name,
            query_rotor_id=query_rotor_id,
            rotor_ids=rotor_ids,
            features_by_rotor=features_by_rotor,
            stats=stats,
            gewichtung_pro_kategorie=gewichtung_pro_kategorie,
            latent_dim=latent_dim,
            n_clusters=n_clusters,
        )

    # Kombiniere Ergebnisse gewichtet
    hybrid_scores: dict[str, tuple[float, dict[str, float], dict[str, float]]] = {}

    for rotor_id in rotor_ids:
        if rotor_id == query_rotor_id:
            continue

        gewichtete_similarity = 0.0
        kombinierte_kategorie_sim: dict[str, float] = {}
        similarity_pro_methode: dict[str, float] = {}

        for methoden_name, gewicht in methoden_gewichte.items():
            if gewicht <= 0 or methoden_name not in methoden_ergebnisse:
                continue

            if rotor_id in methoden_ergebnisse[methoden_name]:
                sim_total, sim_pro_kat = methoden_ergebnisse[methoden_name][rotor_id]
                gewichtete_similarity += sim_total * gewicht
                similarity_pro_methode[methoden_name] = sim_total

                for kategorie, sim in sim_pro_kat.items():
                    if kategorie not in kombinierte_kategorie_sim:
                        kombinierte_kategorie_sim[kategorie] = 0.0
                    kombinierte_kategorie_sim[kategorie] += sim * gewicht

        hybrid_scores[rotor_id] = (
            gewichtete_similarity,
            kombinierte_kategorie_sim,
            similarity_pro_methode,
        )

    # Sortiere nach Gesamt-Similarity
    sortierte_ergebnisse = sorted(
        [
            (rotor_id, score, sim_kat, sim_meth)
            for rotor_id, (score, sim_kat, sim_meth) in hybrid_scores.items()
        ],
        key=lambda x: x[1],
        reverse=True,
    )

    return sortierte_ergebnisse[:top_k]
