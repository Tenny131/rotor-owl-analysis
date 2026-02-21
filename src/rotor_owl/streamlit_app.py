from __future__ import annotations

import time
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from SPARQLWrapper import SPARQLWrapper, XML

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

from rotor_owl.methoden.vektorbasierte_aehnlichkeit import (
    build_vektor_embeddings,
    berechne_topk_aehnlichkeiten_vektorbasiert,
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

from rdflib import Graph

# Validation
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Rotor-Ähnlichkeitsanalyse", layout="wide")
st.title("🔍 Rotor-Ähnlichkeitsanalyse")


with st.sidebar:
    st.header("Einstellungen")

    fuseki_dataset = st.text_input(
        "Fuseki Dataset Name",
        value="rotors",
        help="Name des Fuseki-Datasets, z. B. 'rotors'. Wird in die Endpoint-URL eingesetzt.",
    )
    fuseki_umgebung = st.radio(
        "Fuseki-Umgebung",
        options=["Localhost", "Docker"],
        index=0,
        help="Localhost: http://localhost:3030. Docker: http://fuseki:3030.",
    )
    endpoint_url = (
        f"http://localhost:3030/{fuseki_dataset}/sparql"
        if fuseki_umgebung == "Localhost"
        else f"http://fuseki:3030/{fuseki_dataset}/sparql"
    )

    # Daten automatisch laden (einmalig)
    if not st.session_state.get("daten_geladen", False):
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
            "Regelbasiert (kein ML)",
            "Vektorbasiert (ML)",
            "PCA-Embedding (ML)",
            "Autoencoder (ML)",
            "K-Means Clustering (ML)",
            "Hybrid-Methode",
        ],
        index=0,  # Regelbasiert als Standard
        help=(
            "**Regelbasiert (kein ML):** Parameterweiser Vergleich mit gewichteter Aggregation. "
            "Keine Dimensionsreduktion, direkte Ähnlichkeit pro Parameterpaar.\n\n"
            "**Vektorbasiert (ML):** Feature-Vektoren pro Kategorie, Cosine-Similarity. "
            "Kodiert numerische und kategorische Features in Vektoren.\n\n"
            "**PCA-Embedding (ML):** Hauptkomponentenanalyse zur Dimensionsreduktion, "
            "dann Cosine-Similarity auf latenten Vektoren.\n\n"
            "**Autoencoder (ML):** Neuronales Netz lernt komprimierte Repräsentation, "
            "Cosine-Similarity im Latent Space.\n\n"
            "**K-Means Clustering (ML):** Spherical K-Means gruppiert Rotoren, "
            "Similarity basiert auf Cluster-Zugehörigkeit + Centroid-Distanz.\n\n"
            "**Hybrid-Methode:** Gewichtete Kombination zweier Methoden."
        ),
    )

    # Latent Dimension für PCA / Autoencoder (32 = bessere Varianz-Erhaltung)
    latent_dim = 48
    if methode == "PCA-Embedding (ML)":
        latent_dim = st.slider("PCA Latent Dimension", 2, 96, 48, 2)

    if methode == "Autoencoder (ML)":
        latent_dim = st.slider("Autoencoder Latent Dimension", 2, 96, 48, 2)

    # K-Means: Anzahl Cluster (8 = gute Balance für ~50 Rotoren)
    n_clusters = 8
    if methode == "K-Means Clustering (ML)":
        n_clusters = st.slider("K-Means Cluster (k)", 2, 30, 8, 1)

    # Hybrid-Methode: Methoden-Auswahl und Gewichte
    hybrid_methode_1 = "PCA-Embedding"
    hybrid_methode_2 = "K-Means Clustering"
    hybrid_gewicht_1 = 0.7
    hybrid_gewicht_2 = 0.3

    if methode == "Hybrid-Methode":
        st.caption("**Hybrid-Konfiguration**")

        verfuegbare_methoden = [
            "Regelbasiert (kein ML)",
            "Vektorbasiert (ML)",
            "PCA-Embedding (ML)",
            "Autoencoder (ML)",
            "K-Means Clustering (ML)",
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


try:
    with st.spinner("Lade Features aus Fuseki..."):
        features_by_rotor = fetch_all_features(endpoint_url)
        dependencies = fetch_component_dependencies(endpoint_url)
except Exception as fehler:
    st.error(f"Fuseki-Abfrage fehlgeschlagen: {fehler}")
    st.stop()

# Sofort in session_state speichern (fuer Matrix-Validierung etc.)
st.session_state["features_by_rotor"] = features_by_rotor
st.session_state["numeric_stats"] = build_numeric_stats(features_by_rotor)
st.session_state["daten_geladen"] = True


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
        # Setze alle Gewichte gleichmäßig (5 Methoden)
        methoden_gewichte = {
            "Regelbasiert (kein ML)": 1 / 5,
            "Vektorbasiert (ML)": 1 / 5,
            "PCA-Embedding (ML)": 1 / 5,
            "Autoencoder (ML)": 1 / 5,
            "K-Means Clustering (ML)": 1 / 5,
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

    elif methode == "Regelbasiert (kein ML)":
        # Regelbasierte Similarity
        topk_ergebnisse = berechne_topk_aehnlichkeiten(
            query_rotor_id=query_rotor_id,
            rotor_ids=rotor_ids,
            features_by_rotor=features_by_rotor,
            stats=stats,
            gewichtung_pro_kategorie=gewichtung_pro_kategorie,
            top_k=top_k,
        )

    elif methode == "Vektorbasiert (ML)":
        # Vektorbasiert: Cosine auf Feature-Vektoren
        embeddings = build_vektor_embeddings(features_by_rotor, stats)

        topk_ergebnisse = berechne_topk_aehnlichkeiten_vektorbasiert(
            query_rotor_id=query_rotor_id,
            rotor_ids=rotor_ids,
            embeddings=embeddings,
            gewichtung_pro_kategorie=gewichtung_pro_kategorie,
            top_k=top_k,
        )

    elif methode == "Autoencoder (ML)":
        # Autoencoder-Embedding + Vektorbasiert/Cosine
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

    elif methode == "K-Means Clustering (ML)":
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

    else:
        # PCA-Embedding + Vektorbasiert/Cosine
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
                        "Regelbasiert": "Regelbasiert (kein ML)",
                        "Vektorbasiert": "Vektorbasiert (ML)",
                        "PCA": "PCA-Embedding (ML)",
                        "Autoencoder": "Autoencoder (ML)",
                        "K-Means": "K-Means Clustering (ML)",
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
                "Regelbasiert (kein ML)",
                "Vektorbasiert (ML)",
                "PCA-Embedding (ML)",
                "Autoencoder (ML)",
                "K-Means Clustering (ML)",
            ]
            methoden_labels = [
                m.replace("-Embedding ", "")
                .replace(" Clustering", "")
                .replace(" (kein ML)", "")
                .replace(" (ML)", "")
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
    with st.spinner("Berechne vollständige Similarity-Matrizen für alle Methoden..."):
        features_by_rotor = st.session_state["features_by_rotor"]
        numeric_stats = st.session_state["numeric_stats"]
        rotor_ids = sorted(features_by_rotor.keys())
        n_rotors = len(rotor_ids)

        datenquelle_label = f"Fuseki ({fuseki_umgebung})"

        st.info(
            f"📊 Berechne {n_rotors}×{n_rotors} = {n_rotors**2} Similarity-Werte für jede Methode "
            f"({datenquelle_label})..."
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

        # 2. Vektorbasiert - Vollständige Matrix
        status_text.text(f"2/6: Vektorbasiert (0/{n_rotors} Rotoren)...")
        vektor_emb = build_vektor_embeddings(features_by_rotor, numeric_stats)
        vektor_matrix = np.ones((n_rotors, n_rotors))
        for i, query_r in enumerate(rotor_ids):
            ergebnisse = berechne_topk_aehnlichkeiten_vektorbasiert(
                query_r, rotor_ids, vektor_emb, val_gewichte, top_k=len(rotor_ids)
            )
            for item in ergebnisse:
                rotor_id, sim = item[0], item[1]
                j = rotor_ids.index(rotor_id)
                vektor_matrix[i, j] = sim
            if i % 10 == 0:
                status_text.text(f"2/6: Vektorbasiert ({i}/{n_rotors} Rotoren)...")
        similarity_matrices["Vektorbasiert"] = vektor_matrix
        progress_bar.progress(2 / 6)

        # 3. PCA - Vollständige Matrix
        status_text.text(f"3/6: PCA (0/{n_rotors} Rotoren)...")
        pca_emb = build_pca_embeddings(features_by_rotor, numeric_stats, latent_dim=val_latent_dim)
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
            "Regelbasiert (kein ML)": "Regelbasiert",
            "Vektorbasiert (ML)": "Vektorbasiert",
            "PCA-Embedding (ML)": "PCA",
            "Autoencoder (ML)": "Autoencoder",
            "K-Means Clustering (ML)": "K-Means",
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
            f"Matrix-Validierung ({n_rotors} Rotoren, {n_rotors*(n_rotors-1)//2} Paare) – {datenquelle_label}",
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
        bars = ax.bar(methods, ranges, color=colors[: len(methods)], alpha=0.8, edgecolor="black")
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
Ergebnis der vollständigen Matrix-Validierung:
            
Datenbasis: {n_rotors} Rotoren, {n_rotors*(n_rotors-1)//2} unique Rotor-Paare ({datenquelle_label})
            
- Beste Trennschärfe (Range): {best_range[0]} ({best_range[1]['range']*100:.1f}%)
- Höchste Variabilität (CV): {best_cv[0]} ({best_cv[1]['cv']*100:.1f}%)
- Beste Cluster-Qualität (Silhouette): {best_sil[0]} ({best_sil[1]['silhouette']:.3f})
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

        # =============================================================
        # Korrelationsanalyse: Spearman + Jaccard + Scatter
        # =============================================================
        st.subheader("🔗 Korrelationsanalyse vs. Regelbasiert")

        referenz_name = "Regelbasiert"
        ml_methoden_namen = [m for m in methods if m != referenz_name]
        methoden_farben_map = {
            "Regelbasiert": "#1f77b4",
            "Vektorbasiert": "#ff7f0e",
            "PCA": "#2ca02c",
            "Autoencoder": "#d62728",
            "K-Means": "#9467bd",
            "Hybrid": "#e377c2",
        }

        # Obere Dreiecks-Vektoren extrahieren
        triu_vektoren = {}
        for m_name, m_matrix in similarity_matrices.items():
            triu_idx = np.triu_indices_from(m_matrix, k=1)
            triu_vektoren[m_name] = m_matrix[triu_idx]

        # Spearman-Korrelation jede Methode vs. Regelbasiert
        spearman_ergebnisse = {}
        for m_name in ml_methoden_namen:
            rho_val, p_val = spearmanr(
                triu_vektoren[referenz_name],
                triu_vektoren[m_name],
            )
            spearman_ergebnisse[m_name] = {"rho": rho_val, "p_wert": p_val}

        # 6x6 paarweise Korrelationsmatrix
        n_meth = len(methods)
        korr_matrix = np.ones((n_meth, n_meth))
        for mi in range(n_meth):
            for mj in range(mi + 1, n_meth):
                rho_val, _ = spearmanr(
                    triu_vektoren[methods[mi]],
                    triu_vektoren[methods[mj]],
                )
                korr_matrix[mi, mj] = rho_val  # type: ignore
                korr_matrix[mj, mi] = rho_val  # type: ignore

        # Jaccard Top-10 Overlap
        jaccard_top_k = 10
        referenz_sim_matrix = similarity_matrices[referenz_name]
        jaccard_ergebnisse_ui = {}
        for m_name, m_matrix in similarity_matrices.items():
            if m_name == referenz_name:
                continue
            jac_summe = 0.0
            for row_i in range(n_rotors):
                ref_row = referenz_sim_matrix[row_i].copy()
                ref_row[row_i] = -1.0
                ref_topk = set(np.argsort(ref_row)[-jaccard_top_k:])
                met_row = m_matrix[row_i].copy()
                met_row[row_i] = -1.0
                met_topk = set(np.argsort(met_row)[-jaccard_top_k:])
                schnitt = len(ref_topk & met_topk)
                verein = len(ref_topk | met_topk)
                jac_summe += schnitt / verein if verein > 0 else 0.0
            jaccard_ergebnisse_ui[m_name] = jac_summe / n_rotors

        # --- Plot: Korrelationsanalyse (2x2) ---
        fig_korr, axes_korr = plt.subplots(2, 2, figsize=(14, 11))
        fig_korr.suptitle(
            f"Korrelationsanalyse – Inhaltliche Validierung vs. {referenz_name}",
            fontsize=14,
            fontweight="bold",
        )

        # Subplot 1: Spearman rho Balken
        ax_k = axes_korr[0, 0]
        rho_vals = [spearman_ergebnisse[m]["rho"] for m in ml_methoden_namen]
        farben_k = [methoden_farben_map.get(m, "gray") for m in ml_methoden_namen]
        bars_k = ax_k.bar(ml_methoden_namen, rho_vals, color=farben_k, alpha=0.8, edgecolor="black")
        ax_k.set_ylabel("Spearman ρ")
        ax_k.set_title(f"Spearman-Korrelation vs. {referenz_name}")
        ax_k.set_ylim(0, 1.05)
        ax_k.tick_params(axis="x", rotation=45, labelsize=8)
        ax_k.grid(True, alpha=0.3, axis="y")
        for bar_k, val_k in zip(bars_k, rho_vals):
            ax_k.text(
                bar_k.get_x() + bar_k.get_width() / 2,
                bar_k.get_height() + 0.02,
                f"{val_k:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

        # Subplot 2: 6x6 Heatmap
        ax_k = axes_korr[0, 1]
        im_k = ax_k.imshow(korr_matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
        ax_k.set_xticks(range(n_meth))
        ax_k.set_yticks(range(n_meth))
        ax_k.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
        ax_k.set_yticklabels(methods, fontsize=8)
        ax_k.set_title("Paarweise Spearman-Korrelation")
        for ci in range(n_meth):
            for cj in range(n_meth):
                txt_c = "white" if korr_matrix[ci, cj] < 0.5 else "black"
                ax_k.text(
                    cj,
                    ci,
                    f"{korr_matrix[ci, cj]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=txt_c,
                    fontweight="bold",
                )
        fig_korr.colorbar(im_k, ax=ax_k, shrink=0.8)

        # Subplot 3: Jaccard Balken
        ax_k = axes_korr[1, 0]
        jac_namen = list(jaccard_ergebnisse_ui.keys())
        jac_werte = [jaccard_ergebnisse_ui[m] for m in jac_namen]
        farben_j = [methoden_farben_map.get(m, "gray") for m in jac_namen]
        bars_j = ax_k.bar(jac_namen, jac_werte, color=farben_j, alpha=0.8, edgecolor="black")
        ax_k.set_ylabel("Jaccard-Index")
        ax_k.set_title(f"Top-{jaccard_top_k} Ranking-Overlap vs. {referenz_name}")
        ax_k.set_ylim(0, 1.05)
        ax_k.tick_params(axis="x", rotation=45, labelsize=8)
        ax_k.grid(True, alpha=0.3, axis="y")
        for bar_j, val_j in zip(bars_j, jac_werte):
            ax_k.text(
                bar_j.get_x() + bar_j.get_width() / 2,
                bar_j.get_height() + 0.02,
                f"{val_j:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

        # Subplot 4: Zusammenfassungs-Tabelle
        ax_k = axes_korr[1, 1]
        ax_k.axis("off")
        tab_rows = []
        for m in ml_methoden_namen:
            rho_m = spearman_ergebnisse[m]["rho"]
            p_m = spearman_ergebnisse[m]["p_wert"]
            jac_m = jaccard_ergebnisse_ui.get(m, 0.0)
            sig_m = "***" if p_m < 0.001 else "**" if p_m < 0.01 else "*" if p_m < 0.05 else "n.s."
            tab_rows.append([m, f"{rho_m:.4f}", sig_m, f"{jac_m:.3f}"])
        tab_k = ax_k.table(
            cellText=tab_rows,
            colLabels=["Methode", "Spearman ρ", "Signif.", f"Jaccard@{jaccard_top_k}"],
            loc="center",
            cellLoc="center",
        )
        tab_k.auto_set_font_size(False)
        tab_k.set_fontsize(9)
        tab_k.scale(1.3, 1.8)
        ax_k.set_title("Zusammenfassung: Inhaltliche Validierung", fontweight="bold", pad=20)

        plt.tight_layout()
        st.pyplot(fig_korr)
        plt.close(fig_korr)

        # Legende: Signifikanzniveaus
        n_paare = n_rotors * (n_rotors - 1) // 2
        st.markdown(
            f"""
**Legende – Signifikanzniveaus (p-Wert):**

| Symbol | Bedeutung | p-Wert |
|:------:|-----------|--------|
| {chr(42)}{chr(42)}{chr(42)} | hochsignifikant | p < 0.001 |
| {chr(42)}{chr(42)} | sehr signifikant | p < 0.01 |
| {chr(42)} | signifikant | p < 0.05 |
| n.s. | nicht signifikant | p ≥ 0.05 |

> **Signifikanz ≠ Stärke:** Die Signifikanz (p-Wert) sagt aus, ob eine Korrelation
> *statistisch zufällig* sein könnte. Die **Stärke** (Spearman ρ) beschreibt dagegen,
> *wie eng* der Zusammenhang ist. Bei großen Stichproben (hier {n_paare} Paare)
> sind selbst schwache Korrelationen signifikant – deshalb ist die Effektstärke
> das relevantere Maß (Cohen, 1988).
>
> **Interpretation nach Hinkle et al. (2003):**
> |ρ| ≥ 0.90 = sehr stark, 0.70–0.89 = stark, 0.50–0.69 = moderat,
> 0.30–0.49 = schwach, < 0.30 = vernachlässigbar.
            """
        )

        # --- Plot: Scatter-Plots (2x3) ---
        st.subheader("📊 Scatter-Plots: ML-Methoden vs. Regelbasiert")

        fig_sc, axes_sc = plt.subplots(2, 3, figsize=(16, 10))
        fig_sc.suptitle(
            "Scatter-Plots: ML-Methoden vs. Regelbasiert (jeder Punkt = 1 Rotor-Paar)",
            fontsize=14,
            fontweight="bold",
        )

        referenz_vektor = triu_vektoren[referenz_name]

        for sc_idx, sc_name in enumerate(ml_methoden_namen):
            sc_row = sc_idx // 3
            sc_col = sc_idx % 3
            ax_sc = axes_sc[sc_row, sc_col]

            methode_vektor = triu_vektoren[sc_name]
            sc_rho = spearman_ergebnisse[sc_name]["rho"]
            sc_farbe = methoden_farben_map.get(sc_name, "gray")

            ax_sc.scatter(
                referenz_vektor,
                methode_vektor,
                alpha=0.08,
                s=3,
                color=sc_farbe,
                edgecolors="none",
                rasterized=True,
            )

            koeff = np.polyfit(referenz_vektor, methode_vektor, 1)
            x_line = np.linspace(0.0, 1.0, 100)
            ax_sc.plot(
                x_line,
                np.polyval(koeff, x_line),
                color="black",
                linewidth=1.5,
                linestyle="--",
                alpha=0.8,
            )
            ax_sc.plot([0, 1], [0, 1], color="gray", linewidth=0.8, linestyle=":", alpha=0.5)

            ax_sc.set_xlabel(referenz_name, fontsize=9)
            ax_sc.set_ylabel(sc_name, fontsize=9)
            ax_sc.set_title(f"{sc_name}  (ρ = {sc_rho:.3f})", fontsize=10, fontweight="bold")
            ax_sc.set_xlim(0, 1.02)
            ax_sc.set_ylim(0, 1.02)
            ax_sc.set_aspect("equal")
            ax_sc.grid(True, alpha=0.3)

        # 6. Subplot: Stärke-Tabelle
        ax_sc = axes_sc[1, 2]
        ax_sc.axis("off")
        sc_rows = []
        for m in ml_methoden_namen:
            r_val = spearman_ergebnisse[m]["rho"]
            if r_val >= 0.7:
                st_label = "stark"
            elif r_val >= 0.4:
                st_label = "moderat"
            else:
                st_label = "schwach"
            sc_rows.append([m, f"{r_val:.4f}", st_label])
        tab_sc = ax_sc.table(
            cellText=sc_rows,
            colLabels=["Methode", "Spearman ρ", "Stärke"],
            loc="center",
            cellLoc="center",
        )
        tab_sc.auto_set_font_size(False)
        tab_sc.set_fontsize(9)
        tab_sc.scale(1.3, 1.8)
        ax_sc.set_title("Korrelationsstärke", fontweight="bold", pad=20)

        plt.tight_layout()
        st.pyplot(fig_sc)
        plt.close(fig_sc)

        # Beste Korrelation + Jaccard
        best_rho_name = max(spearman_ergebnisse, key=lambda x: spearman_ergebnisse[x]["rho"])
        best_jac_name = max(jaccard_ergebnisse_ui, key=lambda x: jaccard_ergebnisse_ui[x])

        # Stärke-Klassifikation nach Hinkle et al. (2003)
        zusammenfassung_zeilen = []
        for m_name in ml_methoden_namen:
            rho_m = spearman_ergebnisse[m_name]["rho"]
            if abs(rho_m) >= 0.90:
                staerke = "sehr stark"
            elif abs(rho_m) >= 0.70:
                staerke = "stark"
            elif abs(rho_m) >= 0.50:
                staerke = "moderat"
            elif abs(rho_m) >= 0.30:
                staerke = "schwach"
            else:
                staerke = "vernachlässigbar"
            zusammenfassung_zeilen.append(f"- **{m_name}:** ρ = {rho_m:.4f} → **{staerke}**")

        st.info(
            f"🔗 **Höchste Korrelation mit Regelbasiert:** {best_rho_name} "
            f"(ρ = {spearman_ergebnisse[best_rho_name]['rho']:.4f})\n\n"
            f"🎯 **Bester Ranking-Overlap:** {best_jac_name} "
            f"(Jaccard@{jaccard_top_k} = {jaccard_ergebnisse_ui[best_jac_name]:.3f})"
        )
        st.markdown(
            "**Korrelationsstärke (Hinkle et al., 2003):**\n\n" + "\n".join(zusammenfassung_zeilen)
        )

with st.expander("Parameterliste (aus Ontologie)", expanded=False):
    from rotor_owl.config.kategorien import map_paramtype_to_kategorie, KATEGORIE_LABEL

    if st.session_state.get("daten_geladen", False):
        _fbr = st.session_state.get("features_by_rotor", {})
        # Sammle alle Parameter mit ptype über alle Rotoren
        _param_infos: dict[str, str] = {}  # param_name -> ptype
        for _rotor_daten in _fbr.values():
            for _pkey, _pdata in _rotor_daten.get("params", {}).items():
                _pname = _pkey[1] if isinstance(_pkey, tuple) else str(_pkey)
                _ptype = _pdata.get("ptype", "UNKNOWN") or "UNKNOWN"
                if _pname not in _param_infos:
                    _param_infos[_pname] = _ptype

        # Kategorien zuordnen
        _kat_params: dict[str, list[str]] = {}
        for _pname, _ptype in sorted(_param_infos.items()):
            _kat = map_paramtype_to_kategorie(_ptype)
            _kat_label = KATEGORIE_LABEL.get(_kat, _kat)
            _kat_params.setdefault(_kat_label, []).append(_pname)

        # Zusammenfassung
        st.markdown(
            f"**{len(_param_infos)} Parameter** in **{len(_kat_params)} Kategorien** erkannt:"
        )
        for _kat_label in sorted(_kat_params.keys()):
            _params = _kat_params[_kat_label]
            st.markdown(f"**{_kat_label}** ({len(_params)} Parameter)")
            _param_df = pd.DataFrame({"Parameter": _params})
            st.dataframe(_param_df, hide_index=True, height=min(35 * len(_params) + 38, 300))
    else:
        st.info("Daten noch nicht geladen. Bitte Fuseki-Verbindung prüfen.")
