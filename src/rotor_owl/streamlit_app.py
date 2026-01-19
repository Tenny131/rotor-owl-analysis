from __future__ import annotations

import time
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

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
        ],
        index=0,
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

    if methode == "Regelbasiert":
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
    st.caption(f"Similarity berechnet in {laufzeit:.3f} s | Vergleiche: {len(rotor_ids) - 1}")

    # Tabelle: Top-k Übersicht mit Kategorie-Similarities
    tabellen_zeilen = []
    for rotor_id, sim_total, similarity_pro_kategorie in topk_ergebnisse:
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

    # Heatmap: Kategorie-Ähnlichkeit Matrix
    st.subheader("📊 Kategorie-Ähnlichkeit Heatmap")

    # Daten für Heatmap vorbereiten
    kategorien = sorted(
        [
            kat
            for kat in gewichtung_pro_kategorie.keys()
            if gewichtung_pro_kategorie.get(kat, 0.0) > 0
        ]
    )
    rotor_labels = [rotor_id for rotor_id, _, _ in topk_ergebnisse]

    # Matrix erstellen: Zeilen = Kategorien, Spalten = Rotoren
    heatmap_matrix = []
    for kat in kategorien:
        zeile_matrix: list[float] = []
        for _, _, similarity_pro_kategorie in topk_ergebnisse:
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
        height=300 + len(kategorien) * 40,  # Dynamische Höhe
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
