from __future__ import annotations

import time
import pandas as pd
import streamlit as st

from rotor_owl.konfiguration import FUSEKI_ENDPOINT_STANDARD
from rotor_owl.feature_fetcher import fetch_all_features
from rotor_owl.numerische_aehnlichkeit import build_numeric_stats
from rotor_owl.gesamt_aehnlichkeit import berechne_topk_aehnlichkeiten


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Rotor Similarity UI", layout="wide")
st.title("Rotor Similarity (Fuseki + SPARQL)")


with st.sidebar:
    st.header("Einstellungen")

    endpoint_url = st.text_input("Fuseki SPARQL Endpoint", value=FUSEKI_ENDPOINT_STANDARD)

    st.divider()
    st.subheader("Top-k")
    top_k = st.slider("k", min_value=1, max_value=20, value=5, step=1)

    st.divider()
    st.subheader("Gewichte (Kategorien)")
    # Sinnvolle Defaults
    gewicht_geom = st.slider("GEOM", 0.0, 5.0, 2.0, 0.1)
    gewicht_mtrl = st.slider("MTRL", 0.0, 5.0, 1.0, 0.1)
    gewicht_struct = st.slider("STRUCT", 0.0, 5.0, 1.0, 0.1)
    gewicht_dyn = st.slider("DYN", 0.0, 5.0, 0.5, 0.1)
    gewicht_elec = st.slider("ELEC", 0.0, 5.0, 0.2, 0.1)
    gewicht_req = st.slider("REQ", 0.0, 5.0, 0.2, 0.1)
    gewicht_mfg = st.slider("MFG", 0.0, 5.0, 0.2, 0.1)
    gewicht_unknown = st.slider("UNKNOWN", 0.0, 5.0, 0.0, 0.1)

    gewichtung_pro_typ = {
        "GEOM": gewicht_geom,
        "MTRL": gewicht_mtrl,
        "STRUCT": gewicht_struct,
        "DYN": gewicht_dyn,
        "ELEC": gewicht_elec,
        "REQ": gewicht_req,
        "MFG": gewicht_mfg,
        "UNKNOWN": gewicht_unknown,
    }

    st.divider()
    daten_neuladen = st.button("Daten neu laden (Fuseki erneut abfragen)")


# Streamlit Cache bewusst löschen
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

    # 1) Stats für numerische Normierung berechnen
    stats = build_numeric_stats(features_by_rotor)

    # 2) Top-k Similarities berechnen
    topk_ergebnisse = berechne_topk_aehnlichkeiten(
        query_rotor_id=query_rotor_id,
        rotor_ids=rotor_ids,
        features_by_rotor=features_by_rotor,
        stats=stats,
        gewichtung_pro_typ=gewichtung_pro_typ,
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
    st.dataframe(df_topk, use_container_width=True)

    # Details pro Treffer
    st.subheader("Details pro Ergebnis (Kategorie-Similarity)")
    for rotor_id, sim_total, similarity_pro_typ in topk_ergebnisse:
        with st.expander(f"{rotor_id} | SIM_total={sim_total:.4f}", expanded=False):
            detail_zeilen = []
            for kategorie, similarity in similarity_pro_typ.items():
                detail_zeilen.append(
                    {
                        "Kategorie": kategorie,
                        "SIM": float(f"{similarity:.6f}"),
                        "Gewicht": float(gewichtung_pro_typ.get(kategorie, 0.0)),
                    }
                )
            df_details = pd.DataFrame(detail_zeilen).sort_values(
                by=["Gewicht", "Kategorie"],
                ascending=[False, True],
            )
            st.dataframe(df_details, use_container_width=True)

else:
    st.info("Query-Rotor auswählen, Gewichte setzen und dann **Similarity berechnen** klicken.")
