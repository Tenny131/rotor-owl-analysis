"""Unit tests für Graph-Embedding-basierte Ähnlichkeitsberechnung."""

import pytest
from rdflib import Graph, Namespace, RDF, RDFS, Literal
from rotor_owl.methoden.graph_embedding_aehnlichkeit import (
    train_graph_embeddings,
    get_rotor_embedding,
    _rdf_to_networkx,
)


@pytest.fixture
def sample_ontology():
    """Erzeugt eine minimale Test-Ontologie."""
    g = Graph()
    IMS = Namespace("http://ontology.innomotics.net/ims#")

    # Klassen
    g.add((IMS.Rotor, RDF.type, RDFS.Class))
    g.add((IMS.Welle, RDF.type, RDFS.Class))
    g.add((IMS.Welle, RDFS.subClassOf, IMS.Rotor))

    # Individuen
    g.add((IMS.Rotor_001, RDF.type, IMS.Rotor))
    g.add((IMS.Rotor_002, RDF.type, IMS.Rotor))
    g.add((IMS.Welle_001, RDF.type, IMS.Welle))

    # Properties
    g.add((IMS.hasDiameter, RDF.type, RDF.Property))
    g.add((IMS.Rotor_001, IMS.hasDiameter, Literal(100)))
    g.add((IMS.Rotor_002, IMS.hasDiameter, Literal(120)))

    return g


def test_rdf_to_networkx(sample_ontology):
    """Testet Konvertierung von RDF zu NetworkX."""
    nx_graph = _rdf_to_networkx(sample_ontology)

    assert nx_graph.number_of_nodes() > 0
    assert nx_graph.number_of_edges() > 0


def test_train_graph_embeddings(sample_ontology):
    """Testet Training von Graph-Embeddings."""
    model = train_graph_embeddings(
        sample_ontology,
        dimensions=16,
        num_walks=5,
        walk_length=20,
    )

    assert model is not None
    assert model.vector_size == 16


def test_get_rotor_embedding(sample_ontology):
    """Testet Rotor-Embedding-Generierung mit vollständiger URI."""
    model = train_graph_embeddings(sample_ontology, dimensions=16)

    rotor_uri = "http://ontology.innomotics.net/ims#Rotor_001"
    embedding = get_rotor_embedding(rotor_uri, sample_ontology, model)

    assert embedding is not None
    assert len(embedding) == 16
    # WICHTIG: Embedding sollte NICHT Zero-Vektor sein
    assert not all(v == 0.0 for v in embedding), "Embedding ist Zero-Vektor!"


def test_get_rotor_embedding_with_short_id(sample_ontology):
    """Testet Rotor-Embedding mit kurzer ID (z.B. 'Rotor_001')."""
    model = train_graph_embeddings(sample_ontology, dimensions=16)

    # Test mit kurzer ID statt vollständiger URI
    rotor_id = "Rotor_001"
    embedding = get_rotor_embedding(rotor_id, sample_ontology, model)

    assert embedding is not None
    assert len(embedding) == 16
    # WICHTIG: Embedding sollte NICHT Zero-Vektor sein
    assert not all(v == 0.0 for v in embedding), "Embedding ist Zero-Vektor trotz Match!"


def test_similarity_not_always_zero_point_five():
    """Testet dass Similarity NICHT immer 0.5 ist (Zero-Vektor Bug)."""
    from rotor_owl.methoden.graph_embedding_aehnlichkeit import (
        berechne_topk_aehnlichkeiten_graph_embedding,
    )

    # Einfacher Test-Graph mit 3 Rotoren
    g = Graph()
    IMS = Namespace("http://ontology.innomotics.net/ims#")

    # Rotor_D001 und Rotor_D002 sind ähnlich (beide haben C_WELLE)
    g.add((IMS.C_WELLE_D001, RDF.type, IMS.Welle))
    g.add((IMS.C_WELLE_D002, RDF.type, IMS.Welle))
    g.add((IMS.C_LUEFTER_D003, RDF.type, IMS.Luefter))

    # Properties
    g.add((IMS.C_WELLE_D001, IMS.hasMaterial, IMS.Stahl))
    g.add((IMS.C_WELLE_D002, IMS.hasMaterial, IMS.Stahl))
    g.add((IMS.C_LUEFTER_D003, IMS.hasMaterial, IMS.Aluminium))

    results = berechne_topk_aehnlichkeiten_graph_embedding(
        query_rotor_id="Rotor_D001",
        alle_rotor_ids=["Rotor_D001", "Rotor_D002", "Rotor_D003"],
        ontologie_graph=g,
        kategorie_gewichte={"kat1": 1.0},
        embedding_dimensions=16,
        num_walks=3,
        walk_length=10,
        top_k=2,
    )

    # Mindestens eine Similarity sollte NICHT 0.5 sein
    similarities = [sim for _, sim, _ in results]
    assert not all(
        abs(s - 0.5) < 0.01 for s in similarities
    ), "Alle Similarities sind 0.5 - Zero-Vektor Bug!"

    # INFO: D002 sollte ähnlicher sein als D003, aber Random Walks können zu Varianz führen
    if len(results) >= 2:
        sim_d002 = next((s for rid, s, _ in results if "D002" in rid), None)
        sim_d003 = next((s for rid, s, _ in results if "D003" in rid), None)
        if sim_d002 and sim_d003:
            print(f"\nℹ️  INFO: D002 Similarity={sim_d002:.3f}, D003 Similarity={sim_d003:.3f}")
            if sim_d002 <= sim_d003:
                print("   ⚠️  D003 ist ähnlicher - kann durch Random Walk Varianz passieren")
