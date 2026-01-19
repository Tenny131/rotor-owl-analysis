"""Hybrid-Similarity: Kombiniert mehrere Methoden mit konfigurierbaren Gewichten."""

from __future__ import annotations

from typing import Any


from rotor_owl.methoden.regelbasierte_aehnlichkeit import berechne_topk_aehnlichkeiten
from rotor_owl.methoden.knn_aehnlichkeit import (
    build_knn_embeddings,
    berechne_topk_aehnlichkeiten_knn,
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
from rotor_owl.methoden.graph_embedding_aehnlichkeit import (
    berechne_topk_aehnlichkeiten_graph_embedding,
)


def berechne_topk_aehnlichkeiten_hybrid(
    query_rotor_id: str,
    rotor_ids: list[str],
    features_by_rotor: dict[str, Any],
    stats: dict[tuple[str, str], tuple[float, float]],
    gewichtung_pro_kategorie: dict[str, float],
    methoden_gewichte: dict[str, float],
    latent_dim: int = 8,
    n_clusters: int = 5,
    k: int = 5,
    ontologie_graph: Any = None,
    embedding_dim: int = 64,
    num_walks: int = 5,
    walk_length: int = 30,
    dependencies: dict[tuple[str, str], dict] | None = None,
) -> list[tuple[str, float, dict[str, float], dict[str, float]]]:
    """
    Kombiniert mehrere Similarity-Methoden zu einer gewichteten Hybrid-Similarity.

    Args:
        query_rotor_id: Rotor-ID für die Suche
        rotor_ids: Liste aller Rotor-IDs
        features_by_rotor: Feature-Daten pro Rotor
        stats: Statistiken für numerische Normierung
        gewichtung_pro_kategorie: Gewichte für Kategorien
        methoden_gewichte: Gewichte für jede Methode (z.B. {"Regelbasiert": 0.4, "k-NN": 0.3, ...})
        latent_dim: Latent Dimension für PCA/Autoencoder
        n_clusters: Anzahl Cluster für K-Means
        k: Anzahl der Top-Ergebnisse
        ontologie_graph: RDFlib Graph für Graph-Embeddings (optional)
        embedding_dim: Dimensionalität für Graph-Embeddings
        num_walks: Anzahl Random Walks für Graph-Embeddings
        walk_length: Länge der Walks für Graph-Embeddings

    Returns:
        Liste von (rotor_id, sim_total, similarity_pro_kategorie, similarity_pro_methode) sortiert nach sim_total
    """
    # Sammle Ergebnisse von allen aktivierten Methoden
    methoden_ergebnisse: dict[str, dict[str, tuple[float, dict[str, float]]]] = {}

    # 1. Regelbasiert
    if methoden_gewichte.get("Regelbasiert", 0.0) > 0:
        results = berechne_topk_aehnlichkeiten(
            query_rotor_id=query_rotor_id,
            rotor_ids=rotor_ids,
            features_by_rotor=features_by_rotor,
            stats=stats,
            gewichtung_pro_kategorie=gewichtung_pro_kategorie,
            k=len(rotor_ids),  # Alle Rotoren
        )
        methoden_ergebnisse["Regelbasiert"] = {
            rotor_id: (sim_total, sim_pro_kat) for rotor_id, sim_total, sim_pro_kat in results
        }

    # 2. k-NN
    if methoden_gewichte.get("k-Nearest Neighbors", 0.0) > 0:
        embeddings = build_knn_embeddings(features_by_rotor, stats)
        results = berechne_topk_aehnlichkeiten_knn(
            query_rotor_id=query_rotor_id,
            rotor_ids=rotor_ids,
            embeddings=embeddings,
            gewichtung_pro_kategorie=gewichtung_pro_kategorie,
            k=len(rotor_ids),
        )
        methoden_ergebnisse["k-Nearest Neighbors"] = {
            rotor_id: (sim_total, sim_pro_kat) for rotor_id, sim_total, sim_pro_kat in results
        }

    # 3. PCA-Embedding
    if methoden_gewichte.get("PCA-Embedding", 0.0) > 0:
        pca_embeddings = build_pca_embeddings(features_by_rotor, stats, latent_dim)
        results = berechne_topk_aehnlichkeiten_pca(
            query_rotor_id=query_rotor_id,
            rotor_ids=rotor_ids,
            embeddings=pca_embeddings,
            gewichtung_pro_kategorie=gewichtung_pro_kategorie,
            k=len(rotor_ids),
        )
        methoden_ergebnisse["PCA-Embedding"] = {
            rotor_id: (sim_total, sim_pro_kat) for rotor_id, sim_total, sim_pro_kat in results
        }

    # 4. Autoencoder
    if methoden_gewichte.get("Autoencoder", 0.0) > 0:
        ae_embeddings = build_autoencoder_embeddings(features_by_rotor, stats, latent_dim)
        results = berechne_topk_aehnlichkeiten_autoencoder(
            query_rotor_id=query_rotor_id,
            rotor_ids=rotor_ids,
            embeddings=ae_embeddings,
            gewichtung_pro_kategorie=gewichtung_pro_kategorie,
            k=len(rotor_ids),
        )
        methoden_ergebnisse["Autoencoder"] = {
            rotor_id: (sim_total, sim_pro_kat) for rotor_id, sim_total, sim_pro_kat in results
        }

    # 5. K-Means Clustering
    if methoden_gewichte.get("K-Means Clustering", 0.0) > 0:
        results = berechne_topk_aehnlichkeiten_kmeans(
            query_rotor_id=query_rotor_id,
            rotor_ids=rotor_ids,
            features_by_rotor=features_by_rotor,
            stats=stats,
            gewichtung_pro_kategorie=gewichtung_pro_kategorie,
            n_clusters=n_clusters,
            k=len(rotor_ids),
        )
        methoden_ergebnisse["K-Means Clustering"] = {
            rotor_id: (sim_total, sim_pro_kat) for rotor_id, sim_total, sim_pro_kat in results
        }

    # 6. Graph-Embeddings (Node2Vec)
    if methoden_gewichte.get("Graph-Embeddings (Node2Vec)", 0.0) > 0:
        if ontologie_graph is not None:
            results = berechne_topk_aehnlichkeiten_graph_embedding(
                query_rotor_id=query_rotor_id,
                alle_rotor_ids=rotor_ids,
                ontologie_graph=ontologie_graph,
                kategorie_gewichte=gewichtung_pro_kategorie,
                embedding_dimensions=embedding_dim,
                num_walks=num_walks,
                walk_length=walk_length,
                k=len(rotor_ids),
                dependencies=dependencies,
            )
            methoden_ergebnisse["Graph-Embeddings (Node2Vec)"] = {
                rotor_id: (sim_total, sim_pro_kat) for rotor_id, sim_total, sim_pro_kat in results
            }

    # Kombiniere Ergebnisse gewichtet
    hybrid_scores: dict[str, tuple[float, dict[str, float], dict[str, float]]] = {}

    for rotor_id in rotor_ids:
        if rotor_id == query_rotor_id:
            continue

        weighted_sim_total = 0.0
        combined_sim_pro_kat: dict[str, float] = {}
        sim_pro_methode: dict[str, float] = {}

        for methode, gewicht in methoden_gewichte.items():
            if gewicht <= 0 or methode not in methoden_ergebnisse:
                continue

            if rotor_id in methoden_ergebnisse[methode]:
                sim_total, sim_pro_kat = methoden_ergebnisse[methode][rotor_id]
                weighted_sim_total += sim_total * gewicht

                # Speichere Similarity pro Methode (ungewichtet für Heatmap)
                sim_pro_methode[methode] = sim_total

                # Kombiniere auch Kategorie-Similarities
                for kat, sim in sim_pro_kat.items():
                    if kat not in combined_sim_pro_kat:
                        combined_sim_pro_kat[kat] = 0.0
                    combined_sim_pro_kat[kat] += sim * gewicht

        hybrid_scores[rotor_id] = (weighted_sim_total, combined_sim_pro_kat, sim_pro_methode)

    # Sortiere nach Gesamt-Similarity und gebe Top-K zurück
    sorted_results = sorted(
        [
            (rid, score, sim_kat, sim_meth)
            for rid, (score, sim_kat, sim_meth) in hybrid_scores.items()
        ],
        key=lambda x: x[1],
        reverse=True,
    )

    return sorted_results[:k]
