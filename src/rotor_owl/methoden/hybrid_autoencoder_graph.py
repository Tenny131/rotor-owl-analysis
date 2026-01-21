"""
Hybrid-Methode: Autoencoder + k-NN.

Kombiniert parametrische (Autoencoder - Pattern-basiert) und attributive (k-NN - Distanz-basiert) Ähnlichkeit
für optimale Rotor-Vergleiche.

WARUM k-NN statt Graph-Embeddings?
- Alle Rotoren haben identische Graphstruktur (gleiche composed_of Beziehungen)
- Graph-Embeddings zeigt nur 6.1% Range (zu schwach für Diskriminierung)
- k-NN zeigt 54.7% Range (exzellente Diskriminierung)
- Autoencoder (93.1% Range) und k-NN (54.7% Range) sind komplementär:
  * Autoencoder: Findet Pattern-basierte Ähnlichkeiten (gleiche Feature-Korrelationen)
  * k-NN: Findet Attribut-basierte Ähnlichkeiten (gleiche Absolutwerte)

HINWEIS: Diese Datei heißt "hybrid_autoencoder_graph.py" aus historischen Gründen,
verwendet aber k-NN statt Graph-Embeddings (siehe Begründung oben).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rotor_owl.methoden.autoencoder_aehnlichkeit import (
    build_autoencoder_embeddings,
    berechne_topk_aehnlichkeiten_autoencoder,
)
from rotor_owl.methoden.knn_aehnlichkeit import (
    build_knn_embeddings,
    berechne_topk_aehnlichkeiten_knn,
)

if TYPE_CHECKING:
    from rdflib import Graph


def berechne_topk_aehnlichkeiten_hybrid_ae_graph(
    query_rotor_id: str,
    rotor_ids: list[str],
    features_by_rotor: dict[str, Any],
    stats: dict[tuple[str, str], tuple[float, float]],
    ontologie_graph: Graph,  # Wird nicht mehr verwendet, aber beibehalten für Kompatibilität
    gewichtung_pro_kategorie: dict[str, float],
    gewicht_autoencoder: float = 0.5,
    gewicht_graph: float = 0.5,  # Jetzt k-NN Gewicht
    latent_dim: int = 8,
    embedding_dim: int = 32,  # Nicht mehr verwendet
    num_walks: int = 3,  # Nicht mehr verwendet
    walk_length: int = 20,  # Nicht mehr verwendet
    top_k: int = 5,
) -> list[tuple[str, float, dict[str, float], dict[str, float]]]:
    """
    Hybrid-Methode: Kombiniert Autoencoder (Pattern-basiert) + k-NN (Attribut-basiert).

    Args:
        query_rotor_id (str): Query Rotor ID
        rotor_ids (list): Alle Rotor IDs
        features_by_rotor (dict): Feature-Daten
        stats (dict): Statistiken für Normierung
        ontologie_graph: NICHT MEHR VERWENDET (nur für API-Kompatibilität)
        gewichtung_pro_kategorie (dict): Kategorie-Gewichte
        gewicht_autoencoder (float): Gewicht für Autoencoder-Ähnlichkeit
        gewicht_graph (float): Gewicht für k-NN-Ähnlichkeit
        latent_dim (int): Latent Dimension für Autoencoder
        embedding_dim (int): NICHT MEHR VERWENDET
        num_walks (int): NICHT MEHR VERWENDET
        walk_length (int): NICHT MEHR VERWENDET
        top_k (int): Top-K Ergebnisse

    Returns:
        list: (rotor_id, sim_total, sim_pro_kategorie, sim_pro_methode)
    """
    # Autoencoder-Ähnlichkeit (Pattern-basiert)
    ae_results = berechne_topk_aehnlichkeiten_autoencoder(
        query_rotor_id=query_rotor_id,
        rotor_ids=rotor_ids,
        embeddings=build_autoencoder_embeddings(features_by_rotor, stats, latent_dim),
        gewichtung_pro_kategorie=gewichtung_pro_kategorie,
        top_k=len(rotor_ids),
    )

    ae_scores = {rotor_id: (sim, sim_kat) for rotor_id, sim, sim_kat in ae_results}

    # k-NN-Ähnlichkeit (Attribut-basiert)
    knn_results = berechne_topk_aehnlichkeiten_knn(
        query_rotor_id=query_rotor_id,
        rotor_ids=rotor_ids,
        embeddings=build_knn_embeddings(features_by_rotor, stats),
        gewichtung_pro_kategorie=gewichtung_pro_kategorie,
        top_k=len(rotor_ids),
    )

    knn_scores = {rotor_id: (sim, sim_kat) for rotor_id, sim, sim_kat in knn_results}

    # Gewichtete Kombination
    hybrid_results = []

    for rotor_id in rotor_ids:
        if rotor_id == query_rotor_id:
            continue

        ae_sim, ae_sim_kat = ae_scores.get(rotor_id, (0.0, {}))
        knn_sim, knn_sim_kat = knn_scores.get(rotor_id, (0.0, {}))

        # Gewichtete Gesamt-Similarity
        hybrid_sim = gewicht_autoencoder * ae_sim + gewicht_graph * knn_sim

        # Kombinierte Kategorie-Similarities
        combined_sim_kat = {}
        for kategorie in gewichtung_pro_kategorie.keys():
            ae_kat_sim = ae_sim_kat.get(kategorie, 0.0)
            knn_kat_sim = knn_sim_kat.get(kategorie, 0.0)
            combined_sim_kat[kategorie] = (
                gewicht_autoencoder * ae_kat_sim + gewicht_graph * knn_kat_sim
            )

        # Similarity pro Methode
        sim_pro_methode = {
            "Autoencoder": ae_sim,
            "k-NN": knn_sim,
        }

        hybrid_results.append((rotor_id, hybrid_sim, combined_sim_kat, sim_pro_methode))

    hybrid_results.sort(key=lambda x: x[1], reverse=True)

    return hybrid_results[:top_k]
