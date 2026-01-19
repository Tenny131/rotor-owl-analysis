"""
Graph-Embedding-basierte Ähnlichkeitsberechnung (Node2Vec).

Diese Methode nutzt die ontologische Graphstruktur direkt:
- Konvertiert RDF-Graph zu NetworkX
- Trainiert Node2Vec-Embeddings auf Ontologie-Entitäten
- Berechnet Cosine-Similarity zwischen Rotor-Embeddings
- Aggregiert Embeddings von allen Rotor-Komponenten/Properties

Vorteile:
- Berücksichtigt semantische Relationen (rdfs:subClassOf, owl:sameAs, etc.)
- Robust gegen fehlende Werte
- Erfasst implizite Ähnlichkeiten durch Graphstruktur
"""

from __future__ import annotations

import multiprocessing
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

from rotor_owl.config.kategorien import KATEGORIEN_3

if TYPE_CHECKING:
    from rdflib import Graph

# Optimale Worker-Anzahl: CPU-Kerne, aber maximal 8
_OPTIMAL_WORKERS = min(multiprocessing.cpu_count(), 8)


# ============================================================================
# Node2Vec Random Walk
# ============================================================================


def generate_random_walks(
    G: nx.Graph,
    num_walks: int = 10,
    walk_length: int = 80,
    p: float = 1.0,
    q: float = 1.0,
) -> list[list[str]]:
    """
    Generiert Random Walks auf dem Graph (Node2Vec-Strategie).

    Args:
        G: NetworkX Graph
        num_walks: Anzahl Walks pro Knoten
        walk_length: Länge jedes Walks
        p: Return parameter (Wahrscheinlichkeit zurück zu gehen)
        q: In-Out parameter (BFS vs DFS)

    Returns:
        Liste von Walks (jeder Walk ist Liste von Node-IDs)
    """
    walks = []
    nodes = list(G.nodes())

    for _ in range(num_walks):
        np.random.shuffle(nodes)
        for node in nodes:
            walks.append(_node2vec_walk(G, node, walk_length, p, q))

    return walks


def _node2vec_walk(G: nx.Graph, start_node: str, walk_length: int, p: float, q: float) -> list[str]:
    """
    Führt einen biased Random Walk durch (Node2Vec).

    Args:
        G: NetworkX Graph
        start_node: Startknoten
        walk_length: Länge des Walks
        p: Return parameter
        q: In-Out parameter

    Returns:
        Walk als Liste von Knoten
    """
    walk = [start_node]

    while len(walk) < walk_length:
        cur = walk[-1]
        neighbors = list(G.neighbors(cur))

        if len(neighbors) == 0:
            break

        if len(walk) == 1:
            # Erster Schritt: uniform random
            walk.append(np.random.choice(neighbors))
        else:
            prev = walk[-2]
            probabilities = []

            for neighbor in neighbors:
                if neighbor == prev:
                    # Zurück zum vorherigen Knoten
                    probabilities.append(1.0 / p)
                elif G.has_edge(neighbor, prev):
                    # Nachbar von prev (lokale Exploration)
                    probabilities.append(1.0)
                else:
                    # Neuer Knoten (globale Exploration)
                    probabilities.append(1.0 / q)

            # Normieren
            probabilities = np.array(probabilities)
            probabilities = probabilities / probabilities.sum()

            walk.append(np.random.choice(neighbors, p=probabilities))

    return walk


# ============================================================================
# Graph-Embedding Training
# ============================================================================


def train_graph_embeddings(
    ontologie_graph: Graph,
    dimensions: int = 128,
    window_size: int = 10,
    num_walks: int = 10,
    walk_length: int = 80,
    p: float = 1.0,
    q: float = 1.0,
    min_count: int = 1,
    workers: int = 4,
    dependencies: dict[tuple[str, str], dict] | None = None,
) -> Word2Vec:
    """
    Trainiert Node2Vec-Embeddings auf der Ontologie mit gewichteten Kanten.

    Args:
        ontologie_graph: RDFlib Graph mit Ontologie
        dimensions: Dimensionalität der Embeddings
        window_size: Context-Window für Word2Vec
        num_walks: Anzahl Random Walks pro Knoten
        walk_length: Länge jedes Walks
        p: Return parameter (1.0 = unbiased)
        q: In-Out parameter (1.0 = unbiased)
        min_count: Minimale Frequenz für Vokabular
        workers: Anzahl paralleler Threads
        dependencies: Optional dict mit Dependency-Informationen für gewichtete Kanten

    Returns:
        Trainiertes Word2Vec-Modell mit Node-Embeddings
    """
    # Konvertiere RDF zu NetworkX (mit gewichteten Kanten)
    G = _rdf_to_networkx(ontologie_graph, dependencies)

    # Generiere Random Walks
    walks = generate_random_walks(G, num_walks, walk_length, p, q)

    # Trainiere Word2Vec auf Walks
    model = Word2Vec(
        sentences=walks,
        vector_size=dimensions,
        window=window_size,
        min_count=min_count,
        workers=workers,
        sg=1,  # Skip-gram
        epochs=1,  # Reduziert von 5 für schnellere Berechnung
        negative=5,  # Negative sampling für Effizienz
        hs=0,  # Deaktiviere Hierarchical Softmax
    )

    return model


def _rdf_to_networkx(
    rdf_graph: Graph, dependencies: dict[tuple[str, str], dict] | None = None
) -> nx.Graph:
    """
    Konvertiert RDFlib Graph zu NetworkX Graph mit gewichteten Kanten.

    Args:
        rdf_graph: RDFlib Graph
        dependencies: Optional dict mit (source, target) -> {"strength": str, "percentage": float}

    Returns:
        NetworkX Graph (ungerichtet, mit gewichteten Kanten wenn dependencies gegeben)
    """
    G = nx.Graph()

    # Füge alle Tripel als Kanten hinzu
    for s, p, o in rdf_graph:
        subject = str(s)
        obj = str(o)
        predicate = str(p)

        # Füge Knoten hinzu
        G.add_node(subject)
        if not obj.startswith("http://www.w3.org/2001/XMLSchema#"):
            # Nur URIs, keine Literale als Knoten
            G.add_node(obj)

            # Bestimme Kantengewicht aus Dependencies
            weight = 1.0  # Default
            if dependencies:
                # Extrahiere Komponenten-Namen aus URIs
                # z.B. "http://...#C_BLECHPAKET_D001" -> "Blechpaket"
                source_comp = _extract_component_name(subject)
                target_comp = _extract_component_name(obj)

                if source_comp and target_comp:
                    dep_key = (source_comp, target_comp)
                    if dep_key in dependencies:
                        weight = dependencies[dep_key]["percentage"]
                    # Auch umgekehrte Richtung prüfen (bidirektional)
                    elif (target_comp, source_comp) in dependencies:
                        weight = dependencies[(target_comp, source_comp)]["percentage"]

            G.add_edge(subject, obj, predicate=predicate, weight=weight)

    return G


def _extract_component_name(uri: str) -> str | None:
    """
    Extrahiert Komponenten-Name aus URI.

    "http://...#C_BLECHPAKET_D001" -> "blechpaket"
    "http://...#C_WELLE_1" -> "welle"

    WICHTIG: Verwendet lower() für konsistentes Matching mit map_komponenten_zu_kategorie_gewichte()
    """
    if "#" not in uri:
        return None

    fragment = uri.split("#")[-1]

    # C_BLECHPAKET_D001 -> BLECHPAKET -> blechpaket
    if fragment.startswith("C_"):
        parts = fragment[2:].split("_")
        # Entferne Suffix (_D001, _1, etc.)
        component = "_".join([p for p in parts if not (p.startswith("D") or p.isdigit())])
        # Lowercase für konsistentes Matching
        return component.lower() if component else None

    return None


# ============================================================================
# Rotor-Embedding Aggregation
# ============================================================================


def get_rotor_embedding(
    rotor_uri: str,
    ontologie_graph: Graph,
    embedding_model: Word2Vec,
) -> np.ndarray:
    """
    Erzeugt Embedding für einen Rotor durch Aggregation.

    Strategie:
    - Finde alle mit Rotor verbundenen Entitäten (Komponenten, Properties, Werte)
    - Mittele deren Embeddings

    Args:
        rotor_uri: URI oder ID des Rotors (z.B. "Rotor_D001" oder vollständige URI)
        ontologie_graph: RDFlib Graph
        embedding_model: Trainiertes Word2Vec-Modell

    Returns:
        Embedding-Vektor für Rotor
    """
    embeddings = []

    # Normalisiere rotor_uri:
    # Fall 1: "Rotor_D001" → extrahiere "D001"
    # Fall 2: "http://...#Rotor_001" → extrahiere "001"
    # Fall 3: "http://...#C_WELLE_D001" → extrahiere "D001"

    if rotor_uri.startswith("http://"):
        # Vollständige URI: Extrahiere Fragment nach #
        rotor_id = rotor_uri.split("#")[-1]
        # Entferne Präfixe wie "Rotor_", "C_WELLE_", etc.
        if "_" in rotor_id:
            # Nimm nur den Teil nach dem letzten Unterstrich (z.B. D001, 001)
            parts = rotor_id.split("_")
            rotor_id = parts[-1]  # D001, 001, etc.
    else:
        # Kurze ID: "Rotor_D001" → "D001"
        rotor_id = rotor_uri.replace("Rotor_", "")

    # Finde alle URIs die zu diesem Rotor gehören (z.B. C_WELLE_D001, C_LUEFTER_D001, etc.)
    for s, p, o in ontologie_graph.triples((None, None, None)):
        s_str = str(s)
        o_str = str(o)

        # Check ob Subject zu diesem Rotor gehört (endet mit _D001, _D010, _001, etc.)
        if rotor_id in s_str:
            # Füge Subject Embedding hinzu
            if s_str in embedding_model.wv:
                embeddings.append(embedding_model.wv[s_str])

            # Füge Object Embedding hinzu (wenn es eine URI ist, kein Literal)
            if not o_str.startswith("http://www.w3.org/2001/XMLSchema#"):
                if o_str in embedding_model.wv:
                    embeddings.append(embedding_model.wv[o_str])

    if len(embeddings) == 0:
        # Fallback: Zero-Vektor
        return np.zeros(embedding_model.vector_size)

    # Mittelwert über alle Embeddings
    return np.mean(embeddings, axis=0)


# ============================================================================
# Top-K Ähnlichkeitsberechnung
# ============================================================================


def berechne_topk_aehnlichkeiten_graph_embedding(
    query_rotor_id: str,
    alle_rotor_ids: list[str],
    ontologie_graph: Graph,
    kategorie_gewichte: dict[str, float],
    embedding_dimensions: int = 128,
    num_walks: int = 10,
    walk_length: int = 80,
    k: int = 5,
    dependencies: dict[tuple[str, str], dict] | None = None,
) -> list[tuple[str, float, dict[str, float]]]:
    """
    Berechnet Top-k ähnlichste Rotoren mittels Graph-Embeddings mit gewichteten Kanten.

    Args:
        query_rotor_id: ID des Query-Rotors
        alle_rotor_ids: Liste aller Rotor-IDs
        ontologie_graph: RDFlib Graph mit Ontologie
        kategorie_gewichte: Gewichte pro Kategorie (nicht verwendet, für API-Konsistenz)
        embedding_dimensions: Dimensionalität der Embeddings
        num_walks: Anzahl Random Walks
        walk_length: Länge der Walks
        k: Anzahl zurückzugebender Top-Matches
        dependencies: Optional dict mit Dependency-Informationen für gewichtete Kanten

    Returns:
        Liste von (rotor_id, similarity_gesamt, similarity_pro_kategorie)
        Sortiert nach similarity_gesamt (absteigend)
    """
    # 1. Trainiere Graph-Embeddings mit gewichteten Kanten
    embedding_model = train_graph_embeddings(
        ontologie_graph,
        dimensions=embedding_dimensions,
        num_walks=num_walks,
        walk_length=walk_length,
        workers=_OPTIMAL_WORKERS,
        dependencies=dependencies,  # NEU: Nutze Dependency-Gewichte
    )

    # 2. Query-Embedding berechnen
    query_embedding = get_rotor_embedding(query_rotor_id, ontologie_graph, embedding_model)

    # 3. Batch-Berechnung aller Rotor-Embeddings
    rotor_embeddings = {}
    for rotor_id in alle_rotor_ids:
        if rotor_id != query_rotor_id:
            rotor_embeddings[rotor_id] = get_rotor_embedding(
                rotor_id, ontologie_graph, embedding_model
            )

    # 4. Batch Cosine-Similarity
    ergebnisse = []
    for rotor_id, rotor_embedding in rotor_embeddings.items():
        # Cosine-Similarity
        similarity = cosine_similarity(
            query_embedding.reshape(1, -1), rotor_embedding.reshape(1, -1)
        )[0, 0]

        # Normiere auf [0, 1]
        similarity = max(0.0, min(1.0, (similarity + 1.0) / 2.0))

        # Für Kategorie-Konsistenz: Verteile auf alle 3 Kategorien gleichmäßig
        similarity_pro_kategorie = {kat: similarity for kat in KATEGORIEN_3}

        ergebnisse.append((rotor_id, similarity, similarity_pro_kategorie))

    # Sortiere nach Gesamt-Similarity
    ergebnisse.sort(key=lambda x: x[1], reverse=True)

    return ergebnisse[:k]
