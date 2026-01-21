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
    graph: nx.Graph,
    num_walks: int = 10,
    walk_length: int = 80,
    param_p: float = 1.0,
    param_q: float = 1.0,
) -> list[list[str]]:
    """
    Generiert Random Walks auf dem Graph (Node2Vec-Strategie).

    Args:
        graph (nx.Graph): NetworkX Graph
        num_walks (int): Anzahl Walks pro Knoten
        walk_length (int): Länge jedes Walks
        param_p (float): Return parameter (Wahrscheinlichkeit zurück zu gehen)
        param_q (float): In-Out parameter (BFS vs DFS)

    Returns:
        list: Liste von Walks (jeder Walk ist Liste von Node-IDs)
    """
    # Vorberechnete Nachbarlisten für Performance
    nachbar_cache = {node: list(graph.neighbors(node)) for node in graph.nodes()}

    walks = []
    nodes = list(graph.nodes())

    for _ in range(num_walks):
        np.random.shuffle(nodes)
        for node in nodes:
            walks.append(
                _node2vec_walk_fast(graph, node, walk_length, param_p, param_q, nachbar_cache)
            )

    return walks


def _node2vec_walk_fast(
    graph: nx.Graph,
    start_node: str,
    walk_length: int,
    param_p: float,
    param_q: float,
    nachbar_cache: dict[str, list[str]],
) -> list[str]:
    """
    Schneller biased Random Walk mit gecachten Nachbarn.

    Args:
        graph (nx.Graph): NetworkX Graph
        start_node (str): Startknoten
        walk_length (int): Walk-Länge
        param_p (float): Return parameter
        param_q (float): In-Out parameter
        nachbar_cache (dict): Gecachte Nachbarlisten

    Returns:
        list: Walk als Liste von Knoten-IDs
    """
    walk = [start_node]
    rng = np.random.default_rng()

    for _ in range(walk_length - 1):
        aktueller_knoten = walk[-1]
        nachbarn = nachbar_cache.get(aktueller_knoten, [])

        if not nachbarn:
            break

        # Vereinfacht: uniform random für Geschwindigkeit
        walk.append(rng.choice(nachbarn))

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
    param_p: float = 1.0,
    param_q: float = 1.0,
    min_count: int = 1,
    workers: int = 4,
    dependencies: dict[tuple[str, str], dict] | None = None,
) -> Word2Vec:
    """
    Trainiert Node2Vec-Embeddings mit gewichteten Kanten.

    Args:
        ontologie_graph (Graph): RDFlib Graph
        dimensions (int): Embedding-Dimensionalität
        window_size (int): Word2Vec Context-Window
        num_walks (int): Random Walks pro Knoten
        walk_length (int): Walk-Länge
        param_p (float): Return parameter
        param_q (float): In-Out parameter
        min_count (int): Min. Vokabular-Frequenz
        workers (int): Parallele Threads
        dependencies (dict): (source, target) -> {"percentage": float}

    Returns:
        Word2Vec: Modell mit Node-Embeddings
    """
    networkx_graph = _rdf_to_networkx(ontologie_graph, dependencies)

    walks = generate_random_walks(networkx_graph, num_walks, walk_length, param_p, param_q)

    model = Word2Vec(
        sentences=walks,
        vector_size=dimensions,
        window=window_size,
        min_count=min_count,
        workers=workers,
        sg=1,  # Skip-gram
        epochs=2,
        negative=5,
        hs=0,
    )

    print(f"[DEBUG] Word2Vec trained: {len(model.wv)} nodes in vocabulary")

    return model


def _rdf_to_networkx(
    rdf_graph: Graph, dependencies: dict[tuple[str, str], dict] | None = None
) -> nx.Graph:
    """
    Konvertiert RDFlib Graph zu NetworkX Graph.

    Args:
        rdf_graph (Graph): RDFlib Graph
        dependencies (dict): (source, target) -> {"strength": str, "percentage": float}

    Returns:
        nx.Graph: NetworkX Graph (ungerichtet, gewichtet)
    """
    networkx_graph = nx.Graph()

    for subjekt, praedikat, objekt in rdf_graph:
        subjekt_str = str(subjekt)
        objekt_str = str(objekt)
        praedikat_str = str(praedikat)

        # Nur URIs, keine Literale
        if not objekt_str.startswith("http://") and not objekt_str.startswith("https://"):
            continue

        networkx_graph.add_node(subjekt_str)
        networkx_graph.add_node(objekt_str)

        # Kantengewicht aus Dependencies
        gewicht = 1.0
        if dependencies:
            source_komponente = _extract_component_name(subjekt_str)
            target_komponente = _extract_component_name(objekt_str)

            if source_komponente and target_komponente:
                dep_key = (source_komponente, target_komponente)
                if dep_key in dependencies:
                    gewicht = dependencies[dep_key]["percentage"]
                elif (target_komponente, source_komponente) in dependencies:
                    gewicht = dependencies[(target_komponente, source_komponente)]["percentage"]

        networkx_graph.add_edge(subjekt_str, objekt_str, predicate=praedikat_str, weight=gewicht)

    return networkx_graph


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


# Cache für Rotor-Komponenten (wird einmal berechnet, dann wiederverwendet)
_rotor_components_cache: dict[str, list[str]] = {}


def _build_rotor_components_index(ontologie_graph: Graph) -> dict[str, list[str]]:
    """
    Baut einmalig Index: rotor_id -> [component_uris].
    Vermeidet O(n) Iteration pro Rotor.
    """
    global _rotor_components_cache

    if _rotor_components_cache:
        return _rotor_components_cache

    index: dict[str, list[str]] = {}

    for s, p, o in ontologie_graph:
        s_str = str(s)

        # Extrahiere Rotor-ID aus URI (z.B. "...#C_WELLE_D001" -> "D001")
        if "_" in s_str and "#" in s_str:
            fragment = s_str.split("#")[-1]
            parts = fragment.split("_")
            # Letzter Teil ist oft die Rotor-ID (D001, D002, etc.)
            rotor_id = parts[-1]
            if rotor_id.startswith("D") or rotor_id.isdigit():
                if rotor_id not in index:
                    index[rotor_id] = []
                if s_str not in index[rotor_id]:
                    index[rotor_id].append(s_str)

    _rotor_components_cache = index
    return index


def get_rotor_embedding(
    rotor_uri: str,
    ontologie_graph: Graph,
    embedding_model: Word2Vec,
    components_index: dict[str, list[str]] | None = None,
) -> np.ndarray:
    """
    Erzeugt Embedding für einen Rotor durch Aggregation.
    Nutzt vorberechneten Index für O(1) Lookup statt O(n) Iteration.
    """
    # Normalisiere rotor_uri:
    if rotor_uri.startswith("http://"):
        rotor_id = rotor_uri.split("#")[-1]
        if "_" in rotor_id:
            parts = rotor_id.split("_")
            rotor_id = parts[-1]
    else:
        rotor_id = rotor_uri.replace("Rotor_", "")

    # Nutze Index statt Iteration
    if components_index is None:
        components_index = _build_rotor_components_index(ontologie_graph)

    component_uris = components_index.get(rotor_id, [])

    # Sammle Embeddings
    embeddings = [embedding_model.wv[uri] for uri in component_uris if uri in embedding_model.wv]

    if len(embeddings) == 0:
        return np.zeros(embedding_model.vector_size)

    return np.mean(embeddings, axis=0)


# ============================================================================
# Top-K Ähnlichkeitsberechnung
# ============================================================================


def berechne_topk_aehnlichkeiten_graph_embedding(
    query_rotor_id: str,
    alle_rotor_ids: list[str],
    ontologie_graph: Graph,
    kategorie_gewichte: dict[str, float],
    embedding_dimensions: int = 32,
    num_walks: int = 1,
    walk_length: int = 10,
    top_k: int = 5,
    dependencies: dict[tuple[str, str], dict] | None = None,
) -> list[tuple[str, float, dict[str, float]]]:
    """
    Berechnet Top-k ähnlichste Rotoren mittels Graph-Embeddings.

    Args:
        query_rotor_id (str): ID des Query-Rotors
        alle_rotor_ids (list): Liste aller Rotor-IDs
        ontologie_graph (Graph): RDFlib Graph mit Ontologie
        kategorie_gewichte (dict): Gewichte pro Kategorie
        embedding_dimensions (int): Dimensionalität der Embeddings
        num_walks (int): Anzahl Random Walks
        walk_length (int): Länge der Walks
        top_k (int): Anzahl zurückzugebender Top-Matches
        dependencies (dict): Dependency-Informationen für gewichtete Kanten

    Returns:
        list: (rotor_id, similarity_gesamt, similarity_pro_kategorie), sortiert
    """
    global _rotor_components_cache
    _rotor_components_cache = {}

    embedding_model = train_graph_embeddings(
        ontologie_graph,
        dimensions=embedding_dimensions,
        num_walks=num_walks,
        walk_length=walk_length,
        workers=_OPTIMAL_WORKERS,
        dependencies=dependencies,
    )

    komponenten_index = _build_rotor_components_index(ontologie_graph)

    query_embedding = get_rotor_embedding(
        query_rotor_id, ontologie_graph, embedding_model, komponenten_index
    )

    rotor_embeddings = {}
    for rotor_id in alle_rotor_ids:
        if rotor_id != query_rotor_id:
            rotor_embeddings[rotor_id] = get_rotor_embedding(
                rotor_id, ontologie_graph, embedding_model, komponenten_index
            )

    ergebnisse = []
    for rotor_id, rotor_embedding in rotor_embeddings.items():
        similarity = cosine_similarity(
            query_embedding.reshape(1, -1), rotor_embedding.reshape(1, -1)
        )[0, 0]

        # Normiere auf [0, 1]
        similarity = max(0.0, min(1.0, (similarity + 1.0) / 2.0))

        # Für Kategorie-Konsistenz: gleichmäßig auf alle 3 Kategorien
        similarity_pro_kategorie = {kat: similarity for kat in KATEGORIEN_3}

        ergebnisse.append((rotor_id, similarity, similarity_pro_kategorie))

    ergebnisse.sort(key=lambda x: x[1], reverse=True)

    return ergebnisse[:top_k]
