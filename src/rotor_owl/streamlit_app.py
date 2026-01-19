from __future__ import annotations

import time
import pandas as pd
import streamlit as st

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
st.set_page_config(page_title="Rotor Similarity UI", layout="wide")
st.title("Rotor Similarity (Fuseki + SPARQL)")


with st.sidebar:
    st.header("Einstellungen")

    endpoint_url = st.text_input("Fuseki SPARQL Endpoint", value=FUSEKI_ENDPOINT_STANDARD)

    st.divider()
    st.subheader("Methode")

    methode = st.selectbox(
        "Similarity-Methode",
        options=[
            "A) Regelbasiert (bestehend)",
            "B) ML kNN (Cosine auf Feature-Vektoren)",
            "C) ML Embedding (PCA Latent Space)",
            "C2) ML Embedding (Autoencoder + kNN)",
            "D) ML Clustering (Custom K-Means nach Ahmed & Day)",
        ],
        index=0,
    )

    # Latent Dimension für C / C2
    latent_dim = 8
    if methode.startswith("C)"):
        latent_dim = st.slider("PCA Latent Dimension", 2, 32, 8, 1)

    if methode.startswith("C2)"):
        latent_dim = st.slider("Autoencoder Latent Dimension", 2, 32, 8, 1)

    # D: Anzahl Cluster
    n_clusters = 5
    if methode.startswith("D)"):
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

    if methode.startswith("A)"):
        # Methode A: regelbasiert
        topk_ergebnisse = berechne_topk_aehnlichkeiten(
            query_rotor_id=query_rotor_id,
            rotor_ids=rotor_ids,
            features_by_rotor=features_by_rotor,
            stats=stats,
            gewichtung_pro_kategorie=gewichtung_pro_kategorie,
            k=top_k,
        )

    elif methode.startswith("B)"):
        # Methode B: kNN Cosine auf "rohen" Feature-Vektoren
        embeddings = build_knn_embeddings(features_by_rotor, stats)

        topk_ergebnisse = berechne_topk_aehnlichkeiten_knn(
            query_rotor_id=query_rotor_id,
            rotor_ids=rotor_ids,
            embeddings=embeddings,
            gewichtung_pro_kategorie=gewichtung_pro_kategorie,
            k=top_k,
        )

    elif methode.startswith("C2)"):
        # Methode C2: Autoencoder-Embedding + kNN/Cosine
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

    elif methode.startswith("D)"):
        # Methode D: Custom K-Means (Ahmed & Day) + Similarity über Cluster/Zentroiden
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
        # Methode C: PCA-Embedding + kNN/Cosine
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

    # Tabelle: Top-k Übersicht
    tabellen_zeilen = [
        {"Rotor": rotor_id, "SIM_total": float(f"{sim_total:.6f}")}
        for rotor_id, sim_total, _ in topk_ergebnisse
    ]
    df_topk = pd.DataFrame(tabellen_zeilen)

    st.subheader(f"Top-{top_k} ähnlich zu {query_rotor_id}")
    st.dataframe(df_topk, width="stretch")

    # Details pro Treffer
    st.subheader("Details pro Ergebnis (Kategorie-Similarity)")
    for rotor_id, sim_total, similarity_pro_kategorie in topk_ergebnisse:
        with st.expander(f"{rotor_id} | SIM_total={sim_total:.4f}", expanded=False):
            detail_zeilen = []
            for kat, similarity in similarity_pro_kategorie.items():
                detail_zeilen.append(
                    {
                        "Kategorie": KATEGORIE_LABEL.get(kat, kat),
                        "SIM": float(f"{similarity:.6f}"),
                        "Gewicht": float(gewichtung_pro_kategorie.get(kat, 0.0)),
                    }
                )
            df_details = pd.DataFrame(detail_zeilen).sort_values(
                by=["Gewicht", "Kategorie"],
                ascending=[False, True],
            )
            st.dataframe(df_details, width="stretch")

else:
    st.info("Query-Rotor auswählen, Gewichte setzen und dann **Similarity berechnen** klicken.")
