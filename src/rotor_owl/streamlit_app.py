from __future__ import annotations

import time
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from SPARQLWrapper import SPARQLWrapper, XML

from rotor_owl.config.konfiguration import FUSEKI_ENDPOINT_STANDARD
from rotor_owl.daten.feature_fetcher import fetch_all_features, build_numeric_stats
from rotor_owl.methoden.pca_aehnlichkeit import (
    build_pca_embeddings,
    berechne_topk_aehnlichkeiten_pca,
)
from rotor_owl.methoden.regelbasierte_aehnlichkeit import berechne_topk_aehnlichkeiten

from rotor_owl.config.kategorien import (
    KAT_GEOM_MECH,
    KAT_MTRL_PROC,
    KAT_REQ_ELEC,
    KATEGORIE_LABEL,
    KATEGORIE_BESCHREIBUNG,
)

from rotor_owl.methoden.knn_aehnlichkeit import (
    build_knn_embeddings,
    berechne_topk_aehnlichkeiten_knn,
)

# Autoencoder (C2)
from rotor_owl.methoden.autoencoder_aehnlichkeit import (
    build_autoencoder_embeddings,
    berechne_topk_aehnlichkeiten_autoencoder,
)

# NEU: Custom K-Means (D)
from rotor_owl.methoden.kmeans_aehnlichkeit import berechne_topk_aehnlichkeiten_kmeans

# Hybrid
from rotor_owl.methoden.hybrid_aehnlichkeit import berechne_topk_aehnlichkeiten_hybrid

# Graph-Embeddings
from rotor_owl.methoden.graph_embedding_aehnlichkeit import (
    berechne_topk_aehnlichkeiten_graph_embedding,
)
from rotor_owl.methoden.hybrid_autoencoder_graph import (
    berechne_topk_aehnlichkeiten_hybrid_ae_graph,
)
from rdflib import Graph


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Rotor-Ähnlichkeitsanalyse", layout="wide")
st.title("🔍 Rotor-Ähnlichkeitsanalyse")


with st.sidebar:
    st.header("Einstellungen")

    endpoint_url = st.text_input("Fuseki SPARQL Endpoint", value=FUSEKI_ENDPOINT_STANDARD)

    st.divider()
    st.subheader("Methode")

    methode = st.selectbox(
        "Similarity-Methode",
        options=[
            "Regelbasiert",
            "k-Nearest Neighbors",
            "PCA-Embedding",
            "Autoencoder",
            "K-Means Clustering",
            "Graph-Embeddings (Node2Vec)",
            "Hybrid-Methode (Autoencoder + Graph)",
        ],
        index=6,  # Hybrid-Methode als Standard
    )

    # Latent Dimension für PCA / Autoencoder
    latent_dim = 8
    if methode == "PCA-Embedding":
        latent_dim = st.slider("PCA Latent Dimension", 2, 32, 8, 1)

    if methode == "Autoencoder":
        latent_dim = st.slider("Autoencoder Latent Dimension", 2, 32, 8, 1)

    # K-Means: Anzahl Cluster
    n_clusters = 5
    if methode == "K-Means Clustering":
        n_clusters = st.slider("K-Means Cluster (k)", 2, 30, 5, 1)

    # Graph-Embeddings: Embedding-Dimensionen
    embedding_dim = 32
    num_walks = 3
    walk_length = 20
    if methode == "Graph-Embeddings (Node2Vec)":
        embedding_dim = st.slider("Embedding-Dimension", 16, 128, 32, 16)
        num_walks = st.slider("Anzahl Random Walks pro Knoten", 2, 10, 3, 1)
        walk_length = st.slider("Walk-Länge", 10, 50, 20, 10)

    # Hybrid-Methode: Gewichte
    gewicht_autoencoder = 0.5
    gewicht_graph = 0.5
    if methode == "Hybrid-Methode (Autoencoder + Graph)":
        st.caption("**Gewichtung**")

        gewicht_autoencoder = st.slider(
            "Autoencoder (Pattern-basiert)",
            0.0,
            1.0,
            0.5,
            0.05,
            help="Gewicht für Pattern-basierte Ähnlichkeit (Feature-Korrelationen, globale Muster)",
        )
        gewicht_graph = st.slider(
            "k-NN (Attribut-basiert)",
            0.0,
            1.0,
            0.5,
            0.05,
            help="Gewicht für Attribut-basierte Ähnlichkeit (Distanz zwischen Werten)",
        )
        summe = gewicht_autoencoder + gewicht_graph
        if abs(summe - 1.0) > 0.01:
            st.warning(f"⚠️ Gewichte-Summe: {summe:.2f} (empfohlen: 1.0)")
        else:
            st.success(f"✓ Gewichte-Summe: {summe:.2f}")

        # Parameter für Hybrid-Komponenten
        latent_dim = st.slider("Autoencoder Latent Dimension", 2, 32, 8, 1)
        embedding_dim = 32
        num_walks = 3
        walk_length = 20

    st.divider()

    # Vergleich aller Methoden (für Analyse)
    vergleich_aktiv = st.checkbox(
        "📊 Vergleich aller Methoden",
        help="Berechnet ALLE Methoden gleichzeitig für Analyse-Zwecke (dauert länger)",
    )

    st.divider()
    st.subheader("Top-k")
    top_k = st.slider("k", min_value=1, max_value=20, value=5, step=1)

    st.divider()
    st.subheader("Gewichte (3 Kategorien)")

    gewicht_geom_mech = st.slider(KATEGORIE_LABEL[KAT_GEOM_MECH], 0.0, 5.0, 2.0, 0.1)
    gewicht_mtrl_proc = st.slider(KATEGORIE_LABEL[KAT_MTRL_PROC], 0.0, 5.0, 1.0, 0.1)
    gewicht_req_elec = st.slider(KATEGORIE_LABEL[KAT_REQ_ELEC], 0.0, 5.0, 0.5, 0.1)

    gewichtung_pro_kategorie = {
        KAT_GEOM_MECH: gewicht_geom_mech,
        KAT_MTRL_PROC: gewicht_mtrl_proc,
        KAT_REQ_ELEC: gewicht_req_elec,
    }

    st.divider()
    daten_neuladen = st.button("Daten neu laden (Fuseki erneut abfragen)")


if daten_neuladen:
    fetch_all_features.clear()


# Daten laden
try:
    with st.spinner("Lade Features aus Fuseki..."):
        features_by_rotor = fetch_all_features(endpoint_url)
except Exception as fehler:
    st.error(f"Fuseki-Abfrage fehlgeschlagen: {fehler}")
    st.stop()


rotor_ids = sorted(features_by_rotor.keys())
if not rotor_ids:
    st.warning("Keine Rotoren gefunden. Prüfe Dataset in Fuseki und den Endpoint.")
    st.stop()


linke_spalte, rechte_spalte = st.columns([2, 1])

with linke_spalte:
    query_rotor_id = st.selectbox("Query Rotor", options=rotor_ids, index=0)

with rechte_spalte:
    st.write("")
    st.write("")
    starte_berechnung = st.button("Similarity berechnen", type="primary")


if starte_berechnung:
    startzeit = time.perf_counter()

    # Stats für numerische Normierung
    stats = build_numeric_stats(features_by_rotor)

    # Vergleich aller Methoden (für Analyse)
    if vergleich_aktiv:
        # Setze alle Gewichte gleichmäßig
        methoden_gewichte = {
            "Regelbasiert": 1 / 6,
            "k-Nearest Neighbors": 1 / 6,
            "PCA-Embedding": 1 / 6,
            "Autoencoder": 1 / 6,
            "K-Means Clustering": 1 / 6,
            "Graph-Embeddings (Node2Vec)": 1 / 6,
        }

        # Lade Ontologie-Graph
        with st.spinner("Lade Ontologie-Graph für Vergleich..."):
            ontologie_graph = Graph()
            construct_query = """
            CONSTRUCT { ?s ?p ?o }
            WHERE { ?s ?p ?o }
            LIMIT 5000
            """
            sparql = SPARQLWrapper(endpoint_url)
            sparql.setQuery(construct_query)
            sparql.setReturnFormat(XML)
            result_data = sparql.query().convert()
            if isinstance(result_data, Graph):
                ontologie_graph = result_data
            else:
                ontologie_graph = Graph()
                if isinstance(result_data, bytes):
                    ontologie_graph.parse(data=result_data, format="xml")
                else:
                    ontologie_graph.parse(data=str(result_data), format="xml")

        topk_ergebnisse = berechne_topk_aehnlichkeiten_hybrid(
            query_rotor_id=query_rotor_id,
            rotor_ids=rotor_ids,
            features_by_rotor=features_by_rotor,
            stats=stats,
            gewichtung_pro_kategorie=gewichtung_pro_kategorie,
            methoden_gewichte=methoden_gewichte,
            latent_dim=latent_dim,
            n_clusters=n_clusters,
            k=top_k,
            ontologie_graph=ontologie_graph,
            embedding_dim=embedding_dim,
            num_walks=num_walks,
            walk_length=walk_length,
        )

    elif methode == "Hybrid-Methode (Autoencoder + Graph)":
        # Lade Ontologie-Graph
        with st.spinner("Lade Ontologie-Graph für Hybrid-Methode..."):
            ontologie_graph = Graph()
            construct_query = """
            CONSTRUCT { ?s ?p ?o }
            WHERE { ?s ?p ?o }
            LIMIT 5000
            """
            sparql = SPARQLWrapper(endpoint_url)
            sparql.setQuery(construct_query)
            sparql.setReturnFormat(XML)
            result_data = sparql.query().convert()
            if isinstance(result_data, Graph):
                ontologie_graph = result_data
            else:
                ontologie_graph = Graph()
                if isinstance(result_data, bytes):
                    ontologie_graph.parse(data=result_data, format="xml")
                else:
                    ontologie_graph.parse(data=str(result_data), format="xml")

        # Hybrid-Methode: Autoencoder + Graph-Embeddings
        topk_ergebnisse = berechne_topk_aehnlichkeiten_hybrid_ae_graph(
            query_rotor_id=query_rotor_id,
            rotor_ids=rotor_ids,
            features_by_rotor=features_by_rotor,
            stats=stats,
            ontologie_graph=ontologie_graph,
            gewichtung_pro_kategorie=gewichtung_pro_kategorie,
            gewicht_autoencoder=gewicht_autoencoder,
            gewicht_graph=gewicht_graph,
            latent_dim=latent_dim,
            embedding_dim=embedding_dim,
            num_walks=num_walks,
            walk_length=walk_length,
            k=top_k,
        )

    elif methode == "Regelbasiert":
        # Regelbasierte Similarity
        topk_ergebnisse = berechne_topk_aehnlichkeiten(
            query_rotor_id=query_rotor_id,
            rotor_ids=rotor_ids,
            features_by_rotor=features_by_rotor,
            stats=stats,
            gewichtung_pro_kategorie=gewichtung_pro_kategorie,
            k=top_k,
        )

    elif methode == "k-Nearest Neighbors":
        # kNN Cosine auf Feature-Vektoren
        embeddings = build_knn_embeddings(features_by_rotor, stats)

        topk_ergebnisse = berechne_topk_aehnlichkeiten_knn(
            query_rotor_id=query_rotor_id,
            rotor_ids=rotor_ids,
            embeddings=embeddings,
            gewichtung_pro_kategorie=gewichtung_pro_kategorie,
            k=top_k,
        )

    elif methode == "Autoencoder":
        # Autoencoder-Embedding + kNN/Cosine
        ae_embeddings = build_autoencoder_embeddings(
            features_by_rotor=features_by_rotor,
            stats=stats,
            latent_dim=latent_dim,
        )

        topk_ergebnisse = berechne_topk_aehnlichkeiten_autoencoder(
            query_rotor_id=query_rotor_id,
            rotor_ids=rotor_ids,
            embeddings=ae_embeddings,
            gewichtung_pro_kategorie=gewichtung_pro_kategorie,
            k=top_k,
        )

    elif methode == "K-Means Clustering":
        # Custom K-Means Similarity
        topk_ergebnisse = berechne_topk_aehnlichkeiten_kmeans(
            query_rotor_id=query_rotor_id,
            rotor_ids=rotor_ids,
            features_by_rotor=features_by_rotor,
            stats=stats,
            gewichtung_pro_kategorie=gewichtung_pro_kategorie,
            n_clusters=n_clusters,
            k=top_k,
        )

    elif methode == "Graph-Embeddings (Node2Vec)":
        # Lade RDF-Graph von Fuseki
        with st.spinner("Lade Ontologie-Graph von Fuseki..."):
            construct_query = """
            CONSTRUCT { ?s ?p ?o }
            WHERE { ?s ?p ?o }
            LIMIT 5000
            """
            sparql = SPARQLWrapper(endpoint_url)
            sparql.setQuery(construct_query)
            sparql.setReturnFormat(XML)
            result_data = sparql.query().convert()
            # SPARQLWrapper with XML returns rdflib.Graph directly
            if isinstance(result_data, Graph):
                ontologie_graph = result_data
            else:
                # Fallback: parse if it's bytes/string
                ontologie_graph = Graph()
                if isinstance(result_data, bytes):
                    ontologie_graph.parse(data=result_data, format="xml")
                else:
                    ontologie_graph.parse(data=str(result_data), format="xml")

        # Graph-Embeddings (Node2Vec)
        topk_ergebnisse = berechne_topk_aehnlichkeiten_graph_embedding(
            query_rotor_id=query_rotor_id,
            alle_rotor_ids=rotor_ids,
            ontologie_graph=ontologie_graph,
            kategorie_gewichte=gewichtung_pro_kategorie,
            embedding_dimensions=embedding_dim,
            num_walks=num_walks,
            walk_length=walk_length,
            k=top_k,
        )

    else:
        # PCA-Embedding + kNN/Cosine
        pca_embeddings = build_pca_embeddings(
            features_by_rotor=features_by_rotor,
            stats=stats,
            latent_dim=latent_dim,
        )

        topk_ergebnisse = berechne_topk_aehnlichkeiten_pca(
            query_rotor_id=query_rotor_id,
            rotor_ids=rotor_ids,
            embeddings=pca_embeddings,
            gewichtung_pro_kategorie=gewichtung_pro_kategorie,
            k=top_k,
        )

    laufzeit = time.perf_counter() - startzeit

    # Methoden-Info anzeigen
    if vergleich_aktiv:
        st.caption(
            f"Vergleich aller Methoden berechnet in {laufzeit:.3f} s | Vergleiche: {len(rotor_ids) - 1}"
        )
    elif methode == "Hybrid-Methode (Autoencoder + Graph)":
        st.caption(
            f"Hybrid-Methode ({gewicht_autoencoder:.0%} Autoencoder + {gewicht_graph:.0%} k-NN) berechnet in {laufzeit:.3f} s | Vergleiche: {len(rotor_ids) - 1}"
        )
    else:
        st.caption(f"Similarity berechnet in {laufzeit:.3f} s | Vergleiche: {len(rotor_ids) - 1}")

    # Tabelle: Top-k Übersicht
    tabellen_zeilen = []

    if vergleich_aktiv or (
        methode == "Hybrid-Methode (Autoencoder + Graph)"
        and len(topk_ergebnisse) > 0
        and len(topk_ergebnisse[0]) == 4
    ):
        # Hybrid/Vergleich-Modus: Zeige Similarity pro Methode
        for item in topk_ergebnisse:
            if len(item) == 4:  # Hybrid gibt 4 Elemente zurück
                rotor_id, sim_total, _, similarity_pro_methode = item
                zeile: dict[str, str | float] = {"Rotor": rotor_id}

                # DEBUG: Zeige verfügbare Keys
                # st.write(f"DEBUG {rotor_id}: {list(similarity_pro_methode.keys())}")

                # Spalten für jede Methode
                if vergleich_aktiv:
                    # Keys müssen mit hybrid_aehnlichkeit.py übereinstimmen
                    methoden_mapping = {
                        "Regelbasiert": "Regelbasiert",
                        "k-NN": "k-Nearest Neighbors",
                        "PCA": "PCA-Embedding",
                        "Autoencoder": "Autoencoder",
                        "K-Means": "K-Means Clustering",
                        "Graph-Embed": "Graph-Embeddings (Node2Vec)",
                    }
                    for display_name, key_name in methoden_mapping.items():
                        zeile[display_name] = float(
                            f"{similarity_pro_methode.get(key_name, 0.0):.4f}"
                        )
                else:
                    # Hybrid-Methode: Nur 2 Methoden
                    zeile["Autoencoder"] = float(
                        f"{similarity_pro_methode.get('Autoencoder', 0.0):.4f}"
                    )
                    zeile["k-NN"] = float(
                        f"{similarity_pro_methode.get('k-NN', similarity_pro_methode.get('Graph-Embeddings', 0.0)):.4f}"
                    )  # Fallback für alte Daten

                zeile["S_ges"] = float(f"{sim_total:.4f}")
                tabellen_zeilen.append(zeile)
    else:
        # Standard-Modus: Zeige Similarity pro Kategorie
        for item in topk_ergebnisse:
            rotor_id = item[0]
            sim_total = item[1]
            similarity_pro_kategorie = item[2]
            zeile: dict[str, str | float] = {
                "Rotor": rotor_id,
                "S_G": float(f"{similarity_pro_kategorie.get(KAT_GEOM_MECH, 0.0):.4f}"),
                "S_M": float(f"{similarity_pro_kategorie.get(KAT_MTRL_PROC, 0.0):.4f}"),
                "S_A": float(f"{similarity_pro_kategorie.get(KAT_REQ_ELEC, 0.0):.4f}"),
                "S_ges": float(f"{sim_total:.4f}"),
            }
            tabellen_zeilen.append(zeile)

    df_topk = pd.DataFrame(tabellen_zeilen)

    st.subheader(f"Top-{top_k} ähnlich zu {query_rotor_id}")
    st.dataframe(df_topk, width="stretch")

    # Heatmap: Methoden-Ähnlichkeit (Hybrid/Vergleich) oder Kategorie-Ähnlichkeit (Standard)
    if vergleich_aktiv or (
        methode == "Hybrid-Methode (Autoencoder + Graph)"
        and len(topk_ergebnisse) > 0
        and len(topk_ergebnisse[0]) == 4
    ):
        st.subheader("📊 Methoden-Ähnlichkeit Heatmap")

        # Daten für Heatmap vorbereiten: Zeilen = Methoden, Spalten = Rotoren
        if vergleich_aktiv:
            aktive_methoden = [
                "Regelbasiert",
                "k-Nearest Neighbors",
                "PCA-Embedding",
                "Autoencoder",
                "K-Means Clustering",
                "Graph-Embeddings (Node2Vec)",
            ]
            methoden_labels = [
                m.replace("-Nearest Neighbors", "-NN")
                .replace("-Embedding", "")
                .replace(" Clustering", "")
                .replace(" (Node2Vec)", "")
                for m in aktive_methoden
            ]
        else:
            # Hybrid-Methode
            aktive_methoden = ["Autoencoder", "k-NN"]
            methoden_labels = [
                f"Autoencoder (w={gewicht_autoencoder:.2f})",
                f"k-NN (w={gewicht_graph:.2f})",
            ]

        rotor_labels = [item[0] for item in topk_ergebnisse]

        # Matrix erstellen
        heatmap_matrix = []
        for methode_name in aktive_methoden:
            zeile_matrix: list[float] = []
            for item in topk_ergebnisse:
                if len(item) == 4:
                    _, _, _, similarity_pro_methode = item
                    zeile_matrix.append(similarity_pro_methode.get(methode_name, 0.0))
            heatmap_matrix.append(zeile_matrix)

        # Plotly Heatmap erstellen
        fig = go.Figure(
            data=go.Heatmap(
                z=heatmap_matrix,
                x=rotor_labels,
                y=methoden_labels,
                colorscale="RdYlGn",
                zmid=0.5,
                zmin=0.0,
                zmax=1.0,
                text=[[f"{val:.3f}" for val in zeile] for zeile in heatmap_matrix],
                texttemplate="%{text}",
                textfont={"size": 10},
                colorbar=dict(
                    title="Similarity",
                    tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                    ticktext=["0.0", "0.25", "0.5", "0.75", "1.0"],
                ),
                hovertemplate="<b>%{y}</b><br>%{x}<br>Similarity: %{z:.4f}<extra></extra>",
            )
        )

        fig.update_layout(
            title=f"Methoden-Ähnlichkeit zu {query_rotor_id}",
            xaxis_title="Top-K Rotoren",
            yaxis_title="Methoden",
            height=300 + len(aktive_methoden) * 40,
            xaxis={"side": "bottom"},
            yaxis={"side": "left"},
        )

        st.plotly_chart(fig, width="stretch")

    else:
        st.subheader("📊 Kategorie-Ähnlichkeit Heatmap")

        # Daten für Heatmap vorbereiten
        kategorien = sorted(
            [
                kat
                for kat in gewichtung_pro_kategorie.keys()
                if gewichtung_pro_kategorie.get(kat, 0.0) > 0
            ]
        )
        rotor_labels = [item[0] for item in topk_ergebnisse]

        # Matrix erstellen: Zeilen = Kategorien, Spalten = Rotoren
        heatmap_matrix = []
        for kat in kategorien:
            zeile_matrix: list[float] = []
            for item in topk_ergebnisse:
                rotor_id, sim_total, similarity_pro_kategorie = item[0], item[1], item[2]
                zeile_matrix.append(similarity_pro_kategorie.get(kat, 0.0))
            heatmap_matrix.append(zeile_matrix)

        # Kategorie-Labels mit Gewichten
        kategorie_labels = [
            f"{KATEGORIE_LABEL.get(kat, kat)} (w={gewichtung_pro_kategorie.get(kat, 0.0):.2f})"
            for kat in kategorien
        ]

        # Plotly Heatmap erstellen
        fig = go.Figure(
            data=go.Heatmap(
                z=heatmap_matrix,
                x=rotor_labels,
                y=kategorie_labels,
                colorscale="RdYlGn",  # Rot-Gelb-Grün
                zmid=0.5,
                zmin=0.0,
                zmax=1.0,
                text=[[f"{val:.3f}" for val in zeile] for zeile in heatmap_matrix],
                texttemplate="%{text}",
                textfont={"size": 10},
                colorbar=dict(
                    title="Similarity",
                    tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                    ticktext=["0.0", "0.25", "0.5", "0.75", "1.0"],
                ),
                hovertemplate="<b>%{y}</b><br>%{x}<br>Similarity: %{z:.4f}<extra></extra>",
            )
        )

        fig.update_layout(
            title=f"Kategorie-Ähnlichkeit zu {query_rotor_id}",
            xaxis_title="Top-K Rotoren",
            yaxis_title="Kategorien",
            height=300 + len(kategorie_labels) * 40,  # Dynamische Höhe
            xaxis={"side": "bottom"},
            yaxis={"side": "left"},
        )

        st.plotly_chart(fig, width="stretch")

        # Legende: Kategorien-Definitionen
        st.divider()
        st.caption("**Kategorien-Definitionen:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption(
                f"📐 **{KATEGORIE_LABEL[KAT_GEOM_MECH]}:** {KATEGORIE_BESCHREIBUNG[KAT_GEOM_MECH].split('(')[1].rstrip(')')}"
            )
        with col2:
            st.caption(
                f"🔧 **{KATEGORIE_LABEL[KAT_MTRL_PROC]}:** {KATEGORIE_BESCHREIBUNG[KAT_MTRL_PROC].split('(')[1].rstrip(')')}"
            )
        with col3:
            st.caption(
                f"⚡ **{KATEGORIE_LABEL[KAT_REQ_ELEC]}:** {KATEGORIE_BESCHREIBUNG[KAT_REQ_ELEC].split('(')[1].rstrip(')')}"
            )

else:
    st.info("Query-Rotor auswählen, Gewichte setzen und dann **Similarity berechnen** klicken.")

# ===================== Validierungsergebnisse =====================
st.divider()
st.header("📊 Methoden-Validierung")

with st.expander(
    "ℹ️ **Validierung der Ähnlichkeitsmethoden** (ohne Expertenmeinungen)", expanded=False
):
    st.markdown("""
    Diese Analyse vergleicht die drei Ähnlichkeitsmethoden **k-NN**, **Autoencoder** und **Graph-Embeddings** 
    anhand von 5 statistischen Tests:
    
    **Ergebnisse:**
    
    - ✅ k-NN: Range = 54.7%, CV = 11.2% → EXZELLENT (beste Diskriminierung)
    - ✅ Autoencoder: Range = 93.1%, CV = 142.2% → EXZELLENT (findet andere Muster)
    - ⚠️ Graph-Embeddings: Range = 6.1%, CV = 0.8% → SCHWACH (identische Struktur)
    
    **Hybridischer Ansatz (Autoencoder + k-NN):**
    
    - k-NN fokussiert auf direkte Attribut-Ähnlichkeit
    - Autoencoder findet latente Muster (nicht-lineare Beziehungen)
    - Beide Methoden sind komplementär → zusammen decken sie mehr Ähnlichkeitsaspekte ab
    
    **Graph-Embeddings wurde ersetzt:**
    
    Da alle Rotoren die gleiche Struktur haben (identischer Graph), kann Graph-Embeddings keine 
    sinnvollen Unterschiede erkennen (nur 6% Range).
    """)

    # Validierungsbild anzeigen
    from pathlib import Path

    validation_img = Path(__file__).parent.parent.parent / "data" / "similarity_validation.png"
    if validation_img.exists():
        st.image(
            str(validation_img),
            caption="Similarity Validation - Vergleich aller Methoden",
            width="stretch",
        )
    else:
        st.warning("Validierungsbild nicht gefunden. Führe `python validate_similarities.py` aus.")
