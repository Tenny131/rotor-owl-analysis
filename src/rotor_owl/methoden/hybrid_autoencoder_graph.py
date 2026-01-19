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
    k: int = 5,
) -> list[tuple[str, float, dict[str, float], dict[str, float]]]:
    """
    Hybrid-Methode: Kombiniert Autoencoder (Pattern-basiert) + k-NN (Attribut-basiert).

    ÄNDERUNG: Graph-Embeddings wurde durch k-NN ersetzt, weil:
    - Alle Rotoren haben identische Graphstruktur (95%+ Similarity bei Graph-Embeddings)
    - k-NN zeigt 41% Range vs Graph-Embeddings 4.5% Range
    - Autoencoder (76% Range) + k-NN (41% Range) sind komplementär

    Args:
        query_rotor_id: Query Rotor ID
        rotor_ids: Alle Rotor IDs
        features_by_rotor: Feature-Daten
        stats: Statistiken für Normierung
        ontologie_graph: NICHT MEHR VERWENDET (nur für API-Kompatibilität)
        gewichtung_pro_kategorie: Kategorie-Gewichte
        gewicht_autoencoder: Gewicht für Autoencoder-Ähnlichkeit (0-1)
        gewicht_graph: Gewicht für k-NN-Ähnlichkeit (0-1) - UMBENENNUNG geplant
        latent_dim: Latent Dimension für Autoencoder
        embedding_dim: NICHT MEHR VERWENDET
        num_walks: NICHT MEHR VERWENDET
        walk_length: NICHT MEHR VERWENDET
        k: Top-K Ergebnisse

    Returns:
        Liste von (rotor_id, sim_total, sim_pro_kategorie, sim_pro_methode)
    """
    # 1. Autoencoder-Ähnlichkeit (Pattern-basiert)
    ae_results = berechne_topk_aehnlichkeiten_autoencoder(
        query_rotor_id=query_rotor_id,
        rotor_ids=rotor_ids,
        embeddings=build_autoencoder_embeddings(features_by_rotor, stats, latent_dim),
        gewichtung_pro_kategorie=gewichtung_pro_kategorie,
        k=len(rotor_ids),
    )

    ae_scores = {rotor_id: (sim, sim_kat) for rotor_id, sim, sim_kat in ae_results}

    # 2. k-NN-Ähnlichkeit (Attribut-basiert) - ERSETZT Graph-Embeddings
    knn_results = berechne_topk_aehnlichkeiten_knn(
        query_rotor_id=query_rotor_id,
        rotor_ids=rotor_ids,
        embeddings=build_knn_embeddings(features_by_rotor, stats),
        gewichtung_pro_kategorie=gewichtung_pro_kategorie,
        k=len(rotor_ids),
    )

    knn_scores = {rotor_id: (sim, sim_kat) for rotor_id, sim, sim_kat in knn_results}

    # 3. Gewichtete Kombination
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
        for kat in gewichtung_pro_kategorie.keys():
            ae_kat_sim = ae_sim_kat.get(kat, 0.0)
            knn_kat_sim = knn_sim_kat.get(kat, 0.0)
            combined_sim_kat[kat] = gewicht_autoencoder * ae_kat_sim + gewicht_graph * knn_kat_sim

        # Similarity pro Methode (für Heatmap)
        sim_pro_methode = {
            "Autoencoder": ae_sim,
            "k-NN": knn_sim,  # GEÄNDERT von "Graph-Embeddings"
        }

        hybrid_results.append((rotor_id, hybrid_sim, combined_sim_kat, sim_pro_methode))

    # Sortiere nach Gesamt-Similarity
    hybrid_results.sort(key=lambda x: x[1], reverse=True)

    return hybrid_results[:k]
