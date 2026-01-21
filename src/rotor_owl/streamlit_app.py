from __future__ import annotations

import time
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from SPARQLWrapper import SPARQLWrapper, XML

from rotor_owl.config.konfiguration import FUSEKI_ENDPOINT_STANDARD
from rotor_owl.daten.feature_fetcher import (
    fetch_all_features,
    build_numeric_stats,
    fetch_component_dependencies,
)
from rotor_owl.methoden.pca_aehnlichkeit import (
    build_pca_embeddings,
    berechne_topk_aehnlichkeiten_pca,
)
from rotor_owl.methoden.regelbasierte_aehnlichkeit import (
    berechne_topk_aehnlichkeiten,
    berechne_automatische_gewichte,
    map_komponenten_zu_kategorie_gewichte,
)

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

# Hybrid (modular)
from rotor_owl.methoden.hybrid_aehnlichkeit import berechne_topk_aehnlichkeiten_hybrid

# Graph-Embeddings
from rotor_owl.methoden.graph_embedding_aehnlichkeit import (
    berechne_topk_aehnlichkeiten_graph_embedding,
)
from rdflib import Graph

# Validation
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Rotor-Ähnlichkeitsanalyse", layout="wide")
st.title("🔍 Rotor-Ähnlichkeitsanalyse")


with st.sidebar:
    st.header("Einstellungen")

    endpoint_url = st.text_input("Fuseki SPARQL Endpoint", value=FUSEKI_ENDPOINT_STANDARD)

    # Button zum Laden der Ontologie und Features (einmalig)
    if st.button("📥 Daten laden", help="Lädt Ontologie-Graph und Features von Fuseki (einmalig)"):
        with st.spinner("Lade Ontologie-Graph von Fuseki..."):
            construct_query = """
            CONSTRUCT { ?s ?p ?o }
            WHERE { ?s ?p ?o }
            LIMIT 100000
            """
            sparql = SPARQLWrapper(endpoint_url)
            sparql.setQuery(construct_query)
            sparql.setReturnFormat(XML)
            result_data = sparql.query().convert()
            if isinstance(result_data, Graph):
                st.session_state["ontologie_graph"] = result_data
            else:
                ontologie_graph = Graph()
                if isinstance(result_data, bytes):
                    ontologie_graph.parse(data=result_data, format="xml")
                else:
                    ontologie_graph.parse(data=str(result_data), format="xml")
                st.session_state["ontologie_graph"] = ontologie_graph
            st.session_state["ontologie_geladen"] = True

        with st.spinner("Lade Rotor-Features von Fuseki..."):
            features_by_rotor = fetch_all_features(endpoint_url)
            stats = build_numeric_stats(features_by_rotor)
            st.session_state["features_by_rotor"] = features_by_rotor
            st.session_state["numeric_stats"] = stats
            st.session_state["daten_geladen"] = True

        st.success(f"✅ Ontologie + {len(features_by_rotor)} Rotoren geladen!")

    # Status anzeigen
    if st.session_state.get("daten_geladen", False):
        n_rotors = len(st.session_state.get("features_by_rotor", {}))
        st.caption(f"🟢 Daten geladen ({n_rotors} Rotoren)")
    else:
        st.caption("🔴 Daten nicht geladen")

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
            "Hybrid-Methode",
        ],
        index=6,  # Hybrid-Methode als Standard
    )

    # Latent Dimension für PCA / Autoencoder (32 = bessere Varianz-Erhaltung)
    latent_dim = 32
    if methode == "PCA-Embedding":
        latent_dim = st.slider("PCA Latent Dimension", 2, 64, 32, 1)

    if methode == "Autoencoder":
        latent_dim = st.slider("Autoencoder Latent Dimension", 2, 64, 32, 1)

    # K-Means: Anzahl Cluster (8 = gute Balance für ~50 Rotoren)
    n_clusters = 8
    if methode == "K-Means Clustering":
        n_clusters = st.slider("K-Means Cluster (k)", 2, 30, 8, 1)

    # Graph-Embeddings: Embedding-Dimensionen
    embedding_dim = 32
    num_walks = 1
    walk_length = 10

    # Hybrid-Methode: Methoden-Auswahl und Gewichte
    hybrid_methode_1 = "PCA-Embedding"
    hybrid_methode_2 = "K-Means Clustering"
    hybrid_gewicht_1 = 0.7
    hybrid_gewicht_2 = 0.3

    if methode == "Hybrid-Methode":
        st.caption("**Hybrid-Konfiguration**")

        verfuegbare_methoden = [
            "Regelbasiert",
            "k-Nearest Neighbors",
            "PCA-Embedding",
            "Autoencoder",
            "K-Means Clustering",
        ]

        hybrid_methode_1 = st.selectbox(
            "Methode 1",
            options=verfuegbare_methoden,
            index=2,  # PCA als Standard
        )

        hybrid_gewicht_1 = st.slider(
            f"Gewicht {hybrid_methode_1}",
            0.0,
            1.0,
            0.7,
            0.05,
        )

        hybrid_methode_2 = st.selectbox(
            "Methode 2",
            options=[m for m in verfuegbare_methoden if m != hybrid_methode_1],
            index=3,  # K-Means als Standard (Index nach Filterung)
        )

        hybrid_gewicht_2 = st.slider(
            f"Gewicht {hybrid_methode_2}",
            0.0,
            1.0,
            0.3,
            0.05,
        )

        summe = hybrid_gewicht_1 + hybrid_gewicht_2
        if abs(summe - 1.0) > 0.01:
            st.warning(f"⚠️ Gewichte-Summe: {summe:.2f} (empfohlen: 1.0)")
        else:
            st.success(f"✓ Gewichte-Summe: {summe:.2f}")

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

    auto_gewichte_aktiv = st.checkbox(
        "🤖 Auto-Gewichte aus Dependencies",
        value=False,
        help="Berechne Kategorie-Gewichte automatisch aus Ontologie-Dependencies (hasStrength, hasDependencyPercentage)",
    )

    if auto_gewichte_aktiv:
        st.info("Gewichte werden automatisch aus Dependency-Constraints berechnet")
        gewicht_geom_mech = 0.0
        gewicht_mtrl_proc = 0.0
        gewicht_req_elec = 0.0
    else:
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
        dependencies = fetch_component_dependencies(endpoint_url)
except Exception as fehler:
    st.error(f"Fuseki-Abfrage fehlgeschlagen: {fehler}")
    st.stop()


# Auto-Gewichte berechnen wenn aktiviert
if auto_gewichte_aktiv and dependencies:
    komponenten_gewichte = berechne_automatische_gewichte(dependencies)
    gewichtung_pro_kategorie = map_komponenten_zu_kategorie_gewichte(
        komponenten_gewichte, features_by_rotor
    )

    st.sidebar.success(f"✓ Auto-Gewichte geladen ({len(komponenten_gewichte)} Komponenten)")

    # Zeige Auto-Gewichte in Sidebar
    with st.sidebar.expander("📊 Komponenten-Gewichte"):
        for comp, weight in sorted(komponenten_gewichte.items(), key=lambda x: x[1], reverse=True):
            st.write(f"{comp}: {weight:.3f}")

    with st.sidebar.expander("📊 Kategorie-Gewichte (berechnet)"):
        for kat, weight in sorted(
            gewichtung_pro_kategorie.items(), key=lambda x: x[1], reverse=True
        ):
            st.write(f"{KATEGORIE_LABEL[kat]}: {weight:.3f}")


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

    # Speichere in session_state für Validierungs-Button
    st.session_state["features_by_rotor"] = features_by_rotor
    st.session_state["numeric_stats"] = stats

    # Vergleich aller Methoden (für Analyse)
    if vergleich_aktiv:
        # Setze alle Gewichte gleichmäßig (5 Methoden ohne Graph-Embeddings)
        methoden_gewichte = {
            "Regelbasiert": 1 / 5,
            "k-Nearest Neighbors": 1 / 5,
            "PCA-Embedding": 1 / 5,
            "Autoencoder": 1 / 5,
            "K-Means Clustering": 1 / 5,
        }

        topk_ergebnisse = berechne_topk_aehnlichkeiten_hybrid(
            query_rotor_id=query_rotor_id,
            rotor_ids=rotor_ids,
            features_by_rotor=features_by_rotor,
            stats=stats,
            gewichtung_pro_kategorie=gewichtung_pro_kategorie,
            methoden_gewichte=methoden_gewichte,
            latent_dim=latent_dim,
            n_clusters=n_clusters,
            top_k=top_k,
        )

    elif methode == "Hybrid-Methode":
        # Modulare Hybrid-Methode mit ausgewählten Methoden
        methoden_gewichte = {
            hybrid_methode_1: hybrid_gewicht_1,
            hybrid_methode_2: hybrid_gewicht_2,
        }

        topk_ergebnisse = berechne_topk_aehnlichkeiten_hybrid(
            query_rotor_id=query_rotor_id,
            rotor_ids=rotor_ids,
            features_by_rotor=features_by_rotor,
            stats=stats,
            gewichtung_pro_kategorie=gewichtung_pro_kategorie,
            methoden_gewichte=methoden_gewichte,
            latent_dim=latent_dim,
            n_clusters=n_clusters,
            top_k=top_k,
        )

    elif methode == "Regelbasiert":
        # Regelbasierte Similarity
        topk_ergebnisse = berechne_topk_aehnlichkeiten(
            query_rotor_id=query_rotor_id,
            rotor_ids=rotor_ids,
            features_by_rotor=features_by_rotor,
            stats=stats,
            gewichtung_pro_kategorie=gewichtung_pro_kategorie,
            top_k=top_k,
        )

    elif methode == "k-Nearest Neighbors":
        # kNN Cosine auf Feature-Vektoren
        embeddings = build_knn_embeddings(features_by_rotor, stats)

        topk_ergebnisse = berechne_topk_aehnlichkeiten_knn(
            query_rotor_id=query_rotor_id,
            rotor_ids=rotor_ids,
            embeddings=embeddings,
            gewichtung_pro_kategorie=gewichtung_pro_kategorie,
            top_k=top_k,
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
            top_k=top_k,
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
            top_k=top_k,
        )

    elif methode == "Graph-Embeddings (Node2Vec)":
        # Nutze bereits geladene Ontologie oder lade neu
        if st.session_state.get("ontologie_geladen", False):
            ontologie_graph = st.session_state["ontologie_graph"]
        else:
            with st.spinner("Lade Ontologie-Graph von Fuseki..."):
                construct_query = """
                CONSTRUCT { ?s ?p ?o }
                WHERE { ?s ?p ?o }
                LIMIT 100000
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

                st.session_state["ontologie_graph"] = ontologie_graph
                st.session_state["ontologie_geladen"] = True

        # Graph-Embeddings (Node2Vec) mit gewichteten Kanten
        topk_ergebnisse = berechne_topk_aehnlichkeiten_graph_embedding(
            query_rotor_id=query_rotor_id,
            alle_rotor_ids=rotor_ids,
            ontologie_graph=ontologie_graph,
            kategorie_gewichte=gewichtung_pro_kategorie,
            embedding_dimensions=embedding_dim,
            num_walks=num_walks,
            walk_length=walk_length,
            top_k=top_k,
            dependencies=dependencies if dependencies else None,
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
            top_k=top_k,
        )

    laufzeit = time.perf_counter() - startzeit

    # Methoden-Info anzeigen
    if vergleich_aktiv:
        st.caption(
            f"Vergleich aller Methoden berechnet in {laufzeit:.3f} s | Vergleiche: {len(rotor_ids) - 1}"
        )
    elif methode == "Hybrid-Methode":
        st.caption(
            f"Hybrid ({hybrid_gewicht_1:.0%} {hybrid_methode_1} + {hybrid_gewicht_2:.0%} {hybrid_methode_2}) berechnet in {laufzeit:.3f} s | Vergleiche: {len(rotor_ids) - 1}"
        )
    else:
        st.caption(f"Similarity berechnet in {laufzeit:.3f} s | Vergleiche: {len(rotor_ids) - 1}")

    # Tabelle: Top-k Übersicht
    tabellen_zeilen = []

    if vergleich_aktiv or (
        methode == "Hybrid-Methode" and len(topk_ergebnisse) > 0 and len(topk_ergebnisse[0]) == 4
    ):
        # Hybrid/Vergleich-Modus: Zeige Similarity pro Methode
        for item in topk_ergebnisse:
            if len(item) == 4:  # Hybrid gibt 4 Elemente zurück
                rotor_id, sim_total, _, similarity_pro_methode = item
                zeile: dict[str, str | float] = {"Rotor": rotor_id}

                # Spalten für jede Methode
                if vergleich_aktiv:
                    # Keys müssen mit hybrid_aehnlichkeit.py übereinstimmen
                    methoden_mapping = {
                        "Regelbasiert": "Regelbasiert",
                        "k-NN": "k-Nearest Neighbors",
                        "PCA": "PCA-Embedding",
                        "Autoencoder": "Autoencoder",
                        "K-Means": "K-Means Clustering",
                    }
                    for display_name, key_name in methoden_mapping.items():
                        zeile[display_name] = float(
                            f"{similarity_pro_methode.get(key_name, 0.0):.4f}"
                        )
                else:
                    # Hybrid-Methode: Dynamisch die ausgewählten Methoden anzeigen
                    zeile[hybrid_methode_1] = float(
                        f"{similarity_pro_methode.get(hybrid_methode_1, 0.0):.4f}"
                    )
                    zeile[hybrid_methode_2] = float(
                        f"{similarity_pro_methode.get(hybrid_methode_2, 0.0):.4f}"
                    )

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
        methode == "Hybrid-Methode" and len(topk_ergebnisse) > 0 and len(topk_ergebnisse[0]) == 4
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
            ]
            methoden_labels = [
                m.replace("-Nearest Neighbors", "-NN")
                .replace("-Embedding", "")
                .replace(" Clustering", "")
                for m in aktive_methoden
            ]
        else:
            # Hybrid-Methode: Dynamisch die ausgewählten Methoden anzeigen
            aktive_methoden = [hybrid_methode_1, hybrid_methode_2]
            methoden_labels = [
                f"{hybrid_methode_1} (w={hybrid_gewicht_1:.2f})",
                f"{hybrid_methode_2} (w={hybrid_gewicht_2:.2f})",
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

# Button fuer Live-Validierung mit vollstaendigen Matrizen
if st.button("🔬 Vollständige Matrix-Validierung (alle Rotor-Paare)", type="secondary"):
    if "features_by_rotor" not in st.session_state:
        st.error("Bitte zuerst auf **📥 Daten laden** in der Sidebar klicken")
    else:
        with st.spinner("Berechne vollständige Similarity-Matrizen für alle Methoden..."):
            features_by_rotor = st.session_state["features_by_rotor"]
            numeric_stats = st.session_state["numeric_stats"]
            rotor_ids = sorted(features_by_rotor.keys())
            n_rotors = len(rotor_ids)

            st.info(
                f"📊 Berechne {n_rotors}×{n_rotors} = {n_rotors**2} Similarity-Werte für jede Methode..."
            )

            # Nutze UI-Einstellungen für Validierung
            val_gewichte = gewichtung_pro_kategorie
            val_latent_dim = latent_dim
            val_n_clusters = n_clusters

            # Container für vollständige Matrizen
            similarity_matrices = {}

            progress_bar = st.progress(0)
            status_text = st.empty()

            # 1. REGELBASIERT - Vollständige Matrix
            status_text.text(f"1/6: Regelbasiert (0/{n_rotors} Rotoren)...")
            regelbasiert_matrix = np.ones((n_rotors, n_rotors))
            for i, query_r in enumerate(rotor_ids):
                ergebnisse = berechne_topk_aehnlichkeiten(
                    query_r,
                    rotor_ids,
                    features_by_rotor,
                    numeric_stats,
                    val_gewichte,
                    top_k=len(rotor_ids),
                )
                for rotor_id, sim, _ in ergebnisse:
                    j = rotor_ids.index(rotor_id)
                    regelbasiert_matrix[i, j] = sim
                if i % 10 == 0:
                    status_text.text(f"1/6: Regelbasiert ({i}/{n_rotors} Rotoren)...")
            similarity_matrices["Regelbasiert"] = regelbasiert_matrix
            progress_bar.progress(1 / 6)

            # 2. k-NN - Vollständige Matrix
            status_text.text(f"2/6: k-NN (0/{n_rotors} Rotoren)...")
            knn_emb = build_knn_embeddings(features_by_rotor, numeric_stats)
            knn_matrix = np.ones((n_rotors, n_rotors))
            for i, query_r in enumerate(rotor_ids):
                ergebnisse = berechne_topk_aehnlichkeiten_knn(
                    query_r, rotor_ids, knn_emb, val_gewichte, top_k=len(rotor_ids)
                )
                for item in ergebnisse:
                    rotor_id, sim = item[0], item[1]
                    j = rotor_ids.index(rotor_id)
                    knn_matrix[i, j] = sim
                if i % 10 == 0:
                    status_text.text(f"2/6: k-NN ({i}/{n_rotors} Rotoren)...")
            similarity_matrices["k-NN"] = knn_matrix
            progress_bar.progress(2 / 6)

            # 3. PCA - Vollständige Matrix
            status_text.text(f"3/6: PCA (0/{n_rotors} Rotoren)...")
            pca_emb = build_pca_embeddings(
                features_by_rotor, numeric_stats, latent_dim=val_latent_dim
            )
            pca_matrix = np.ones((n_rotors, n_rotors))
            for i, query_r in enumerate(rotor_ids):
                ergebnisse = berechne_topk_aehnlichkeiten_pca(
                    query_r, rotor_ids, pca_emb, val_gewichte, top_k=len(rotor_ids)
                )
                for item in ergebnisse:
                    rotor_id, sim = item[0], item[1]
                    j = rotor_ids.index(rotor_id)
                    pca_matrix[i, j] = sim
                if i % 10 == 0:
                    status_text.text(f"3/6: PCA ({i}/{n_rotors} Rotoren)...")
            similarity_matrices["PCA"] = pca_matrix
            progress_bar.progress(3 / 6)

            # 4. Autoencoder - Vollständige Matrix
            status_text.text(f"4/6: Autoencoder (0/{n_rotors} Rotoren)...")
            ae_emb = build_autoencoder_embeddings(
                features_by_rotor, numeric_stats, latent_dim=val_latent_dim
            )
            ae_matrix = np.ones((n_rotors, n_rotors))
            for i, query_r in enumerate(rotor_ids):
                ergebnisse = berechne_topk_aehnlichkeiten_autoencoder(
                    query_r, rotor_ids, ae_emb, val_gewichte, top_k=len(rotor_ids)
                )
                for item in ergebnisse:
                    rotor_id, sim = item[0], item[1]
                    j = rotor_ids.index(rotor_id)
                    ae_matrix[i, j] = sim
                if i % 10 == 0:
                    status_text.text(f"4/6: Autoencoder ({i}/{n_rotors} Rotoren)...")
            similarity_matrices["Autoencoder"] = ae_matrix
            progress_bar.progress(4 / 6)

            # 5. K-Means - Vollständige Matrix
            status_text.text(f"5/6: K-Means (0/{n_rotors} Rotoren)...")
            kmeans_matrix = np.ones((n_rotors, n_rotors))
            for i, query_r in enumerate(rotor_ids):
                ergebnisse = berechne_topk_aehnlichkeiten_kmeans(
                    query_r,
                    rotor_ids,
                    features_by_rotor,
                    numeric_stats,
                    val_gewichte,
                    n_clusters=val_n_clusters,
                    top_k=len(rotor_ids),
                )
                for item in ergebnisse:
                    rotor_id, sim = item[0], item[1]
                    j = rotor_ids.index(rotor_id)
                    kmeans_matrix[i, j] = sim
                if i % 10 == 0:
                    status_text.text(f"5/6: K-Means ({i}/{n_rotors} Rotoren)...")
            similarity_matrices["K-Means"] = kmeans_matrix
            progress_bar.progress(5 / 6)

            # 6. Hybrid - verwendet die UI-Einstellungen (hybrid_methode_1 + hybrid_methode_2)
            # Mapping von UI-Namen zu Matrix-Keys
            methoden_matrix_mapping = {
                "Regelbasiert": "Regelbasiert",
                "k-Nearest Neighbors": "k-NN",
                "PCA-Embedding": "PCA",
                "Autoencoder": "Autoencoder",
                "K-Means Clustering": "K-Means",
            }

            matrix_key_1 = methoden_matrix_mapping.get(hybrid_methode_1, "PCA")
            matrix_key_2 = methoden_matrix_mapping.get(hybrid_methode_2, "K-Means")

            status_text.text(
                f"6/6: Hybrid ({hybrid_gewicht_1:.0%} {matrix_key_1} + {hybrid_gewicht_2:.0%} {matrix_key_2})..."
            )

            # Berechne gewichtete Kombination der ausgewählten Matrizen
            matrix_1 = similarity_matrices.get(matrix_key_1, pca_matrix)
            matrix_2 = similarity_matrices.get(matrix_key_2, kmeans_matrix)
            hybrid_matrix = hybrid_gewicht_1 * matrix_1 + hybrid_gewicht_2 * matrix_2

            similarity_matrices["Hybrid"] = hybrid_matrix
            progress_bar.progress(6 / 6)

            status_text.text("Analysiere Matrizen...")

            # Berechne Metriken für jede Methode
            validation_results = {}

            for method_name, sim_matrix in similarity_matrices.items():
                # Extrahiere obere Dreiecks-Matrix (ohne Diagonale) für echte Similarities
                triu_indices = np.triu_indices_from(sim_matrix, k=1)
                sim_values = sim_matrix[triu_indices]

                validation_results[method_name] = {
                    "mean": float(np.mean(sim_values)),
                    "std": float(np.std(sim_values)),
                    "min": float(np.min(sim_values)),
                    "max": float(np.max(sim_values)),
                    "range": float(np.max(sim_values) - np.min(sim_values)),
                    "cv": float(np.std(sim_values) / np.mean(sim_values))
                    if np.mean(sim_values) > 0
                    else 0,
                }

                # Silhouette Score
                try:
                    dist_matrix = np.clip(1 - sim_matrix, 0, 1)
                    np.fill_diagonal(dist_matrix, 0)

                    if n_rotors > 5:
                        clustering = AgglomerativeClustering(
                            n_clusters=5, metric="precomputed", linkage="average"
                        )
                        labels = clustering.fit_predict(dist_matrix)
                        sil_score = silhouette_score(dist_matrix, labels, metric="precomputed")
                        validation_results[method_name]["silhouette"] = float(sil_score)
                    else:
                        validation_results[method_name]["silhouette"] = 0.0
                except Exception:
                    validation_results[method_name]["silhouette"] = 0.0

            progress_bar.empty()
            status_text.empty()

            # Erstelle Visualisierung
            methods = list(validation_results.keys())
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#e377c2"]

            fig, axes = plt.subplots(2, 3, figsize=(16, 10))
            fig.suptitle(
                f"Matrix-Validierung ({n_rotors} Rotoren, {n_rotors*(n_rotors-1)//2} Paare) - 6 Methoden",
                fontsize=14,
                fontweight="bold",
            )

            # Plot 1: Histogramme
            ax = axes[0, 0]
            for i, method in enumerate(methods):
                sim_matrix = similarity_matrices[method]
                triu_indices = np.triu_indices_from(sim_matrix, k=1)
                sim_vals = sim_matrix[triu_indices]
                ax.hist(sim_vals, bins=30, alpha=0.5, label=method, color=colors[i % len(colors)])
            ax.set_xlabel("Similarity")
            ax.set_ylabel("Häufigkeit")
            ax.set_title("Similarity Verteilungen (alle Paare)")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

            # Plot 2: Range
            ax = axes[0, 1]
            ranges = [validation_results[m]["range"] * 100 for m in methods]
            bars = ax.bar(
                methods, ranges, color=colors[: len(methods)], alpha=0.8, edgecolor="black"
            )
            ax.set_ylabel("Range (%)")
            ax.set_title("Range (Höher = Besser)")
            ax.tick_params(axis="x", rotation=45, labelsize=8)
            ax.grid(True, alpha=0.3, axis="y")
            for bar, val in zip(bars, ranges):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold",
                )

            # Plot 3: CV
            ax = axes[0, 2]
            cvs = [validation_results[m]["cv"] * 100 for m in methods]
            bars = ax.bar(methods, cvs, color=colors[: len(methods)], alpha=0.8, edgecolor="black")
            ax.set_ylabel("CV (%)")
            ax.set_title("Variationskoeffizient (Höher = Besser)")
            ax.tick_params(axis="x", rotation=45, labelsize=8)
            ax.grid(True, alpha=0.3, axis="y")
            for bar, val in zip(bars, cvs):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold",
                )

            # Plot 4: Silhouette
            ax = axes[1, 0]
            sils = [validation_results[m]["silhouette"] for m in methods]
            bars = ax.bar(methods, sils, color=colors[: len(methods)], alpha=0.8, edgecolor="black")
            ax.set_ylabel("Silhouette Score")
            ax.set_title("Cluster-Qualität (Höher = Besser)")
            ax.tick_params(axis="x", rotation=45, labelsize=8)
            ax.grid(True, alpha=0.3, axis="y")
            for bar, val in zip(bars, sils):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold",
                )

            # Plot 5: Min/Mean/Max
            ax = axes[1, 1]
            x = np.arange(len(methods))
            width = 0.25
            mins = [validation_results[m]["min"] for m in methods]
            means = [validation_results[m]["mean"] for m in methods]
            maxs = [validation_results[m]["max"] for m in methods]
            ax.bar(x - width, mins, width, label="Min", color="lightcoral", edgecolor="black")
            ax.bar(x, means, width, label="Mean", color="steelblue", edgecolor="black")
            ax.bar(x + width, maxs, width, label="Max", color="lightgreen", edgecolor="black")
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("Similarity")
            ax.set_title("Min / Mean / Max")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3, axis="y")
            ax.set_ylim(0, 1.05)

            # Plot 6: Tabelle
            ax = axes[1, 2]
            ax.axis("off")
            table_data = []
            for m in methods:
                r = validation_results[m]
                table_data.append(
                    [m, f"{r['range']*100:.1f}%", f"{r['cv']*100:.1f}%", f"{r['silhouette']:.3f}"]
                )
            table = ax.table(
                cellText=table_data,
                colLabels=["Methode", "Range", "CV", "Silh."],
                loc="center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.8)
            ax.set_title("Zusammenfassung", fontweight="bold", pad=20)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Beste Methoden
            best_range = max(validation_results.items(), key=lambda x: x[1]["range"])
            best_cv = max(validation_results.items(), key=lambda x: x[1]["cv"])
            best_sil = max(validation_results.items(), key=lambda x: x[1]["silhouette"])

            st.success(f"""
            **Ergebnis der vollständigen Matrix-Validierung:**
            
            📊 **Datenbasis:** {n_rotors} Rotoren, {n_rotors*(n_rotors-1)//2} unique Rotor-Paare
            
            - 🏆 **Beste Trennschärfe (Range):** {best_range[0]} ({best_range[1]['range']*100:.1f}%)
            - 📈 **Höchste Variabilität (CV):** {best_cv[0]} ({best_cv[1]['cv']*100:.1f}%)
            - ⭐ **Beste Cluster-Qualität (Silhouette):** {best_sil[0]} ({best_sil[1]['silhouette']:.3f})
            """)

            # Zeige Details als Dataframe
            df_results = pd.DataFrame(
                [
                    {
                        "Methode": m,
                        "Range (%)": f"{r['range']*100:.1f}",
                        "CV (%)": f"{r['cv']*100:.1f}",
                        "Silhouette": f"{r['silhouette']:.3f}",
                        "Min": f"{r['min']:.3f}",
                        "Mean": f"{r['mean']:.3f}",
                        "Max": f"{r['max']:.3f}",
                    }
                    for m, r in validation_results.items()
                ]
            )

            st.dataframe(df_results, width="stretch", hide_index=True)

with st.expander("📈 Vorberechnete Validierung (aus validate_results.py)", expanded=False):
    from pathlib import Path

    validation_img = Path(__file__).parent.parent.parent / "data" / "similarity_validation.png"
    if validation_img.exists():
        st.image(
            str(validation_img), caption="Similarity Validation - Alle 7 Methoden", width="stretch"
        )
    else:
        st.warning("Validierungsbild nicht gefunden. Fuehre `python validate_results.py` aus.")
