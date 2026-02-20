#!/usr/bin/env python3
"""
Vollstaendige Matrix-Validierung aller Similarity-Methoden.

Berechnet echte n x n Similarity-Matrizen fuer alle 6 Methoden
auf Basis der realen WVSC-JSON-Daten (230 Rotoren, 26 335 Paare).
Erzeugt statistische Kennzahlen, Korrelationsanalysen und Visualisierungen.

Verwendung::

    python validate_results.py
    python validate_results.py --latent-dim 16 --n-clusters 8
"""

from __future__ import annotations

import argparse
import datetime
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from rotor_owl.daten.json_parser import fetch_all_features_from_json
from rotor_owl.daten.feature_fetcher import build_numeric_stats
from rotor_owl.methoden.regelbasierte_aehnlichkeit import berechne_topk_aehnlichkeiten
from rotor_owl.methoden.knn_aehnlichkeit import (
    build_knn_embeddings,
    berechne_topk_aehnlichkeiten_knn,
)
from rotor_owl.methoden.pca_aehnlichkeit import (
    build_pca_embeddings,
    berechne_topk_aehnlichkeiten_pca,
)
from rotor_owl.methoden.autoencoder_aehnlichkeit import (
    build_autoencoder_embeddings,
    berechne_topk_aehnlichkeiten_autoencoder,
)
from rotor_owl.methoden.kmeans_aehnlichkeit import berechne_topk_aehnlichkeiten_kmeans
from rotor_owl.config.kategorien import KAT_GEOM_MECH, KAT_MTRL_PROC, KAT_REQ_ELEC

# ===========================================================================
# Konstanten
# ===========================================================================

METHODEN_NAMEN: list[str] = [
    "Regelbasiert",
    "k-NN",
    "PCA",
    "Autoencoder",
    "K-Means",
    "Hybrid",
]

METHODEN_FARBEN: list[str] = [
    "#1f77b4",  # Regelbasiert – blau
    "#ff7f0e",  # k-NN – orange
    "#2ca02c",  # PCA – gruen
    "#d62728",  # Autoencoder – rot
    "#9467bd",  # K-Means – lila
    "#e377c2",  # Hybrid – pink
]

REFERENZ_METHODE: str = "Regelbasiert"

# Standard-Hyperparameter
STANDARD_LATENT_DIM: int = 8
STANDARD_N_CLUSTERS: int = 5
STANDARD_HYBRID_GEWICHT_PCA: float = 0.5
STANDARD_HYBRID_GEWICHT_KMEANS: float = 0.5

# Validierungs-Parameter
SILHOUETTE_N_CLUSTERS: int = 5
TOP_K_JACCARD: int = 10
HISTOGRAMM_BINS: int = 30
SCATTER_ALPHA: float = 0.08
SCATTER_PUNKTGROESSE: int = 3

# Schwellenwerte (aus Evaluationsdesign)
SCHWELLE_RANGE_PROZENT: float = 0.30
SCHWELLE_CV_PROZENT: float = 0.10

# Ausgabe-Verzeichnis
AUSGABE_VERZEICHNIS: Path = Path(__file__).parent / "data"

# Fortschritts-Intervall (alle N Rotoren Status drucken)
FORTSCHRITTS_INTERVALL: int = 50


# ===========================================================================
# Datenladen
# ===========================================================================


def lade_json_daten() -> tuple[dict, dict, list[str]]:
    """
    Lade Rotor-Features aus den realen WVSC-JSON-Dateien.

    :return: Tupel aus (features_by_rotor, numeric_stats, rotor_ids)
    :rtype: tuple[dict, dict, list[str]]
    """
    print("📡 Lade Realdaten aus JSON-Dateien...")
    features_by_rotor = fetch_all_features_from_json()
    numeric_stats = build_numeric_stats(features_by_rotor)
    rotor_ids = sorted(features_by_rotor.keys())

    anzahl_rotoren = len(rotor_ids)
    anzahl_paare = anzahl_rotoren * (anzahl_rotoren - 1) // 2
    print(f"   ✓ {anzahl_rotoren} Rotoren geladen ({anzahl_paare} unique Paare)")

    return features_by_rotor, numeric_stats, rotor_ids


# ===========================================================================
# Matrix-Berechnung
# ===========================================================================


def berechne_alle_matrizen(
    features_by_rotor: dict,
    numeric_stats: dict,
    rotor_ids: list[str],
    latent_dim: int = STANDARD_LATENT_DIM,
    n_clusters: int = STANDARD_N_CLUSTERS,
) -> dict[str, np.ndarray]:
    """
    Berechne vollstaendige n x n Similarity-Matrizen fuer alle 6 Methoden.

    Jeder Rotor dient einmal als Query, sodass echte All-Pairs-Matrizen
    entstehen (nicht nur Single-Query).

    :param features_by_rotor: Feature-Dict aller Rotoren
    :type features_by_rotor: dict
    :param numeric_stats: Min/Max-Statistiken pro Parameter
    :type numeric_stats: dict
    :param rotor_ids: Sortierte Liste der Rotor-IDs
    :type rotor_ids: list[str]
    :param latent_dim: Latent-Dimension fuer PCA und Autoencoder
    :type latent_dim: int
    :param n_clusters: Anzahl Cluster fuer K-Means
    :type n_clusters: int
    :return: Dict mit Methoden-Name -> n x n NumPy-Matrix
    :rtype: dict[str, np.ndarray]
    """
    n = len(rotor_ids)
    rotor_index = {rid: idx for idx, rid in enumerate(rotor_ids)}

    gleichgewichtung = {
        KAT_GEOM_MECH: 1.0,
        KAT_MTRL_PROC: 1.0,
        KAT_REQ_ELEC: 1.0,
    }

    matrizen: dict[str, np.ndarray] = {}

    print(f"\n{'=' * 70}")
    print("BERECHNE SIMILARITY-MATRIZEN")
    print(f"{'=' * 70}")

    # --- 1. Regelbasiert ---
    startzeit = time.time()
    print(f"\n1/6: Regelbasiert ({n} Queries)...", end="", flush=True)

    matrix = np.ones((n, n))
    for i, query_r in enumerate(rotor_ids):
        ergebnisse = berechne_topk_aehnlichkeiten(
            query_r,
            rotor_ids,
            features_by_rotor,
            numeric_stats,
            gleichgewichtung,
            top_k=n,
        )
        for eintrag in ergebnisse:
            j = rotor_index[eintrag[0]]
            matrix[i, j] = eintrag[1]
        if (i + 1) % FORTSCHRITTS_INTERVALL == 0:
            print(f" {i + 1}", end="", flush=True)

    matrizen["Regelbasiert"] = matrix
    print(f" ✓ ({time.time() - startzeit:.1f}s)")

    # --- 2. k-NN ---
    startzeit = time.time()
    print(f"2/6: k-NN ({n} Queries)...", end="", flush=True)

    knn_emb = build_knn_embeddings(features_by_rotor, numeric_stats)
    matrix = np.ones((n, n))
    for i, query_r in enumerate(rotor_ids):
        ergebnisse = berechne_topk_aehnlichkeiten_knn(
            query_r,
            rotor_ids,
            knn_emb,
            gleichgewichtung,
            top_k=n,
        )
        for eintrag in ergebnisse:
            j = rotor_index[eintrag[0]]
            matrix[i, j] = eintrag[1]
        if (i + 1) % FORTSCHRITTS_INTERVALL == 0:
            print(f" {i + 1}", end="", flush=True)

    matrizen["k-NN"] = matrix
    print(f" ✓ ({time.time() - startzeit:.1f}s)")

    # --- 3. PCA ---
    startzeit = time.time()
    print(f"3/6: PCA (latent_dim={latent_dim}, {n} Queries)...", end="", flush=True)

    pca_emb = build_pca_embeddings(
        features_by_rotor,
        numeric_stats,
        latent_dim=latent_dim,
    )
    matrix = np.ones((n, n))
    for i, query_r in enumerate(rotor_ids):
        ergebnisse = berechne_topk_aehnlichkeiten_pca(
            query_r,
            rotor_ids,
            pca_emb,
            gleichgewichtung,
            top_k=n,
        )
        for eintrag in ergebnisse:
            j = rotor_index[eintrag[0]]
            matrix[i, j] = eintrag[1]
        if (i + 1) % FORTSCHRITTS_INTERVALL == 0:
            print(f" {i + 1}", end="", flush=True)

    matrizen["PCA"] = matrix
    print(f" ✓ ({time.time() - startzeit:.1f}s)")

    # --- 4. Autoencoder ---
    startzeit = time.time()
    print(f"4/6: Autoencoder (latent_dim={latent_dim}, {n} Queries)...", end="", flush=True)

    ae_emb = build_autoencoder_embeddings(
        features_by_rotor,
        numeric_stats,
        latent_dim=latent_dim,
    )
    matrix = np.ones((n, n))
    for i, query_r in enumerate(rotor_ids):
        ergebnisse = berechne_topk_aehnlichkeiten_autoencoder(
            query_r,
            rotor_ids,
            ae_emb,
            gleichgewichtung,
            top_k=n,
        )
        for eintrag in ergebnisse:
            j = rotor_index[eintrag[0]]
            matrix[i, j] = eintrag[1]
        if (i + 1) % FORTSCHRITTS_INTERVALL == 0:
            print(f" {i + 1}", end="", flush=True)

    matrizen["Autoencoder"] = matrix
    print(f" ✓ ({time.time() - startzeit:.1f}s)")

    # --- 5. K-Means ---
    startzeit = time.time()
    print(f"5/6: K-Means (n_clusters={n_clusters}, {n} Queries)...", end="", flush=True)

    matrix = np.ones((n, n))
    for i, query_r in enumerate(rotor_ids):
        ergebnisse = berechne_topk_aehnlichkeiten_kmeans(
            query_r,
            rotor_ids,
            features_by_rotor,
            numeric_stats,
            gleichgewichtung,
            n_clusters=n_clusters,
            top_k=n,
        )
        for eintrag in ergebnisse:
            j = rotor_index[eintrag[0]]
            matrix[i, j] = eintrag[1]
        if (i + 1) % FORTSCHRITTS_INTERVALL == 0:
            print(f" {i + 1}", end="", flush=True)

    matrizen["K-Means"] = matrix
    print(f" ✓ ({time.time() - startzeit:.1f}s)")

    # --- 6. Hybrid (PCA + K-Means) ---
    startzeit = time.time()
    print(
        f"6/6: Hybrid ({STANDARD_HYBRID_GEWICHT_PCA:.0%} PCA"
        f" + {STANDARD_HYBRID_GEWICHT_KMEANS:.0%} K-Means)...",
        end="",
        flush=True,
    )

    matrizen["Hybrid"] = (
        STANDARD_HYBRID_GEWICHT_PCA * matrizen["PCA"]
        + STANDARD_HYBRID_GEWICHT_KMEANS * matrizen["K-Means"]
    )
    print(f" ✓ ({time.time() - startzeit:.1f}s)")

    return matrizen


# ===========================================================================
# Hilfs-Funktion: Obere Dreiecksmatrix
# ===========================================================================


def extrahiere_obere_dreiecksmatrix(matrix: np.ndarray) -> np.ndarray:
    """
    Extrahiere die Werte der oberen Dreiecksmatrix (ohne Diagonale).

    :param matrix: Quadratische n x n Matrix
    :type matrix: np.ndarray
    :return: Flaches Array mit n*(n-1)/2 Werten
    :rtype: np.ndarray
    """
    indizes = np.triu_indices_from(matrix, k=1)
    return matrix[indizes]


# ===========================================================================
# Statistische Basis-Metriken
# ===========================================================================


def berechne_basis_metriken(
    similarity_matrizen: dict[str, np.ndarray],
) -> dict[str, dict]:
    """
    Berechne deskriptive Statistiken und Silhouette-Score fuer jede Methode.

    :param similarity_matrizen: Dict mit Methoden-Name -> n x n Matrix
    :type similarity_matrizen: dict[str, np.ndarray]
    :return: Dict mit Methoden-Name -> Metriken-Dict
    :rtype: dict[str, dict]
    """
    print(f"\n{'=' * 70}")
    print("STATISTISCHE KENNZAHLEN")
    print(f"{'=' * 70}")

    ergebnisse: dict[str, dict] = {}

    for name, matrix in similarity_matrizen.items():
        werte = extrahiere_obere_dreiecksmatrix(matrix)

        metriken: dict = {
            "mean": float(np.mean(werte)),
            "std": float(np.std(werte)),
            "min": float(np.min(werte)),
            "max": float(np.max(werte)),
            "range": float(np.max(werte) - np.min(werte)),
            "cv": (float(np.std(werte) / np.mean(werte)) if np.mean(werte) > 0 else 0.0),
        }

        # Silhouette Score via Agglomerative Clustering
        try:
            dist_matrix = np.clip(1.0 - matrix, 0.0, 1.0)
            np.fill_diagonal(dist_matrix, 0.0)

            n = matrix.shape[0]
            if n > SILHOUETTE_N_CLUSTERS:
                clustering = AgglomerativeClustering(
                    n_clusters=SILHOUETTE_N_CLUSTERS,
                    metric="precomputed",
                    linkage="average",
                )
                labels = clustering.fit_predict(dist_matrix)
                metriken["silhouette"] = float(
                    silhouette_score(dist_matrix, labels, metric="precomputed")
                )
            else:
                metriken["silhouette"] = 0.0
        except Exception:
            metriken["silhouette"] = 0.0

        # Schwellenwert-Bewertung
        range_ok = metriken["range"] >= SCHWELLE_RANGE_PROZENT
        cv_ok = metriken["cv"] >= SCHWELLE_CV_PROZENT

        if range_ok and cv_ok:
            metriken["bewertung"] = "BESTANDEN"
        elif range_ok or cv_ok:
            metriken["bewertung"] = "GRENZFALL"
        else:
            metriken["bewertung"] = "DURCHGEFALLEN"

        ergebnisse[name] = metriken

        print(
            f"   {name:<15} Range: {metriken['range'] * 100:5.1f}%  "
            f"CV: {metriken['cv'] * 100:5.1f}%  "
            f"Silh.: {metriken['silhouette']:.3f}  "
            f"→ {metriken['bewertung']}"
        )

    return ergebnisse


# ===========================================================================
# Spearman-Korrelationsanalyse
# ===========================================================================


def berechne_spearman_korrelationen(
    similarity_matrizen: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """
    Berechne paarweise Spearman-Rangkorrelationen zwischen allen Methoden.

    :param similarity_matrizen: Dict mit Methoden-Name -> n x n Matrix
    :type similarity_matrizen: dict[str, np.ndarray]
    :return: Tupel aus (6x6 Korrelationsmatrix, Dict mit vs-Regelbasiert-Ergebnissen)
    :rtype: tuple[pd.DataFrame, dict[str, dict]]
    """
    print(f"\n{'=' * 70}")
    print("SPEARMAN-KORRELATIONSANALYSE")
    print(f"{'=' * 70}")

    methoden = list(similarity_matrizen.keys())
    n_methoden = len(methoden)

    # Obere Dreiecks-Vektoren extrahieren
    vektoren: dict[str, np.ndarray] = {}
    for name, matrix in similarity_matrizen.items():
        vektoren[name] = extrahiere_obere_dreiecksmatrix(matrix)

    # 6x6 paarweise Korrelationsmatrix
    korrelations_werte = np.ones((n_methoden, n_methoden))

    for i in range(n_methoden):
        for j in range(i + 1, n_methoden):
            rho, _ = spearmanr(vektoren[methoden[i]], vektoren[methoden[j]])
            korrelations_werte[i, j] = rho
            korrelations_werte[j, i] = rho

    korrelations_df = pd.DataFrame(
        korrelations_werte,
        index=methoden,
        columns=methoden,
    )

    # Spezifisch: jede ML-Methode vs. Regelbasiert
    vs_regelbasiert: dict[str, dict] = {}

    print(f"\n   Korrelation vs. {REFERENZ_METHODE}:")
    for name in methoden:
        if name == REFERENZ_METHODE:
            continue
        rho, p_wert = spearmanr(vektoren[REFERENZ_METHODE], vektoren[name])
        vs_regelbasiert[name] = {"rho": rho, "p_wert": p_wert}

        if p_wert < 0.001:
            signifikanz = "***"
        elif p_wert < 0.01:
            signifikanz = "**"
        elif p_wert < 0.05:
            signifikanz = "*"
        else:
            signifikanz = "n.s."

        print(f"      {name:<15} ρ = {rho:.4f}  " f"(p = {p_wert:.2e}) {signifikanz}")

    return korrelations_df, vs_regelbasiert


# ===========================================================================
# Jaccard Ranking-Overlap
# ===========================================================================


def berechne_jaccard_overlap(
    similarity_matrizen: dict[str, np.ndarray],
    rotor_ids: list[str],
    top_k: int = TOP_K_JACCARD,
) -> dict[str, float]:
    """
    Berechne den mittleren Jaccard-Index der Top-k Rankings vs. Regelbasiert.

    Fuer jeden Rotor als Query werden die Top-k aehnlichsten Rotoren jeder
    Methode mit den Top-k von Regelbasiert verglichen.

    :param similarity_matrizen: Dict mit Methoden-Name -> n x n Matrix
    :type similarity_matrizen: dict[str, np.ndarray]
    :param rotor_ids: Sortierte Liste der Rotor-IDs
    :type rotor_ids: list[str]
    :param top_k: Anzahl der Top-Treffer fuer den Vergleich
    :type top_k: int
    :return: Dict mit Methoden-Name -> mittlerer Jaccard-Index
    :rtype: dict[str, float]
    """
    print(f"\n{'=' * 70}")
    print(f"RANKING-OVERLAP (Top-{top_k} Jaccard vs. {REFERENZ_METHODE})")
    print(f"{'=' * 70}")

    n = len(rotor_ids)
    referenz_matrix = similarity_matrizen[REFERENZ_METHODE]

    jaccard_ergebnisse: dict[str, float] = {}

    for name, matrix in similarity_matrizen.items():
        if name == REFERENZ_METHODE:
            continue

        jaccard_summe = 0.0

        for i in range(n):
            # Top-k Indizes fuer Regelbasiert (ohne sich selbst)
            referenz_zeile = referenz_matrix[i].copy()
            referenz_zeile[i] = -1.0
            referenz_topk = set(np.argsort(referenz_zeile)[-top_k:])

            # Top-k Indizes fuer aktuelle Methode
            methode_zeile = matrix[i].copy()
            methode_zeile[i] = -1.0
            methode_topk = set(np.argsort(methode_zeile)[-top_k:])

            # Jaccard-Index = |Schnittmenge| / |Vereinigung|
            schnittmenge = len(referenz_topk & methode_topk)
            vereinigung = len(referenz_topk | methode_topk)
            jaccard_summe += schnittmenge / vereinigung if vereinigung > 0 else 0.0

        mittlerer_jaccard = jaccard_summe / n
        jaccard_ergebnisse[name] = mittlerer_jaccard

        print(
            f"   {name:<15} Jaccard@{top_k} = {mittlerer_jaccard:.4f} "
            f"({mittlerer_jaccard * 100:.1f}%)"
        )

    return jaccard_ergebnisse


# ===========================================================================
# Plot 1: Statistische Kennzahlen (6 Subplots)
# ===========================================================================


def erstelle_statistik_plot(
    basis_metriken: dict[str, dict],
    similarity_matrizen: dict[str, np.ndarray],
    anzahl_rotoren: int,
) -> plt.Figure:
    """
    Erstelle die 6-Panel-Uebersicht mit statistischen Kennzahlen.

    :param basis_metriken: Berechnete Metriken pro Methode
    :type basis_metriken: dict[str, dict]
    :param similarity_matrizen: Similarity-Matrizen pro Methode
    :type similarity_matrizen: dict[str, np.ndarray]
    :param anzahl_rotoren: Anzahl der Rotoren
    :type anzahl_rotoren: int
    :return: Matplotlib Figure
    :rtype: plt.Figure
    """
    methoden = list(basis_metriken.keys())
    n_paare = anzahl_rotoren * (anzahl_rotoren - 1) // 2

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f"Matrix-Validierung ({anzahl_rotoren} Rotoren, "
        f"{n_paare} Paare) – Statistische Kennzahlen",
        fontsize=14,
        fontweight="bold",
    )

    # --- Histogramme ---
    ax = axes[0, 0]
    for i, name in enumerate(methoden):
        werte = extrahiere_obere_dreiecksmatrix(similarity_matrizen[name])
        ax.hist(
            werte,
            bins=HISTOGRAMM_BINS,
            alpha=0.5,
            label=name,
            color=METHODEN_FARBEN[i],
        )
    ax.set_xlabel("Similarity")
    ax.set_ylabel("Häufigkeit")
    ax.set_title("Similarity-Verteilungen (alle Paare)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- Range ---
    ax = axes[0, 1]
    ranges = [basis_metriken[m]["range"] * 100 for m in methoden]
    bars = ax.bar(
        methoden,
        ranges,
        color=METHODEN_FARBEN[: len(methoden)],
        alpha=0.8,
        edgecolor="black",
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

    # --- CV ---
    ax = axes[0, 2]
    cvs = [basis_metriken[m]["cv"] * 100 for m in methoden]
    bars = ax.bar(
        methoden,
        cvs,
        color=METHODEN_FARBEN[: len(methoden)],
        alpha=0.8,
        edgecolor="black",
    )
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

    # --- Silhouette ---
    ax = axes[1, 0]
    sils = [basis_metriken[m]["silhouette"] for m in methoden]
    bars = ax.bar(
        methoden,
        sils,
        color=METHODEN_FARBEN[: len(methoden)],
        alpha=0.8,
        edgecolor="black",
    )
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

    # --- Min / Mean / Max ---
    ax = axes[1, 1]
    x = np.arange(len(methoden))
    breite = 0.25
    mins = [basis_metriken[m]["min"] for m in methoden]
    means = [basis_metriken[m]["mean"] for m in methoden]
    maxs = [basis_metriken[m]["max"] for m in methoden]
    ax.bar(x - breite, mins, breite, label="Min", color="lightcoral", edgecolor="black")
    ax.bar(x, means, breite, label="Mean", color="steelblue", edgecolor="black")
    ax.bar(x + breite, maxs, breite, label="Max", color="lightgreen", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(methoden, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Similarity")
    ax.set_title("Min / Mean / Max")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)

    # --- Zusammenfassungs-Tabelle ---
    ax = axes[1, 2]
    ax.axis("off")
    tabellen_daten = []
    for m in methoden:
        r = basis_metriken[m]
        tabellen_daten.append(
            [
                m,
                f"{r['range'] * 100:.1f}%",
                f"{r['cv'] * 100:.1f}%",
                f"{r['silhouette']:.3f}",
            ]
        )
    tabelle = ax.table(
        cellText=tabellen_daten,
        colLabels=["Methode", "Range", "CV", "Silh."],
        loc="center",
        cellLoc="center",
    )
    tabelle.auto_set_font_size(False)
    tabelle.set_fontsize(9)
    tabelle.scale(1.2, 1.8)
    ax.set_title("Zusammenfassung", fontweight="bold", pad=20)

    plt.tight_layout()
    return fig


# ===========================================================================
# Plot 2: Korrelationsanalyse (4 Subplots)
# ===========================================================================


def erstelle_korrelations_plot(
    korrelations_df: pd.DataFrame,
    vs_regelbasiert: dict[str, dict],
    jaccard_ergebnisse: dict[str, float],
) -> plt.Figure:
    """
    Erstelle die 4-Panel-Uebersicht mit Korrelations- und Ranking-Analyse.

    :param korrelations_df: 6x6 Spearman-Korrelationsmatrix
    :type korrelations_df: pd.DataFrame
    :param vs_regelbasiert: Spearman-Ergebnisse je ML-Methode vs. Regelbasiert
    :type vs_regelbasiert: dict[str, dict]
    :param jaccard_ergebnisse: Mittlerer Jaccard-Index je Methode
    :type jaccard_ergebnisse: dict[str, float]
    :return: Matplotlib Figure
    :rtype: plt.Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        f"Korrelationsanalyse – Inhaltliche Validierung vs. {REFERENZ_METHODE}",
        fontsize=14,
        fontweight="bold",
    )

    ml_methoden = list(vs_regelbasiert.keys())

    # --- Spearman rho vs. Regelbasiert (Balken) ---
    ax = axes[0, 0]
    rho_werte = [vs_regelbasiert[m]["rho"] for m in ml_methoden]
    farben_ml = [METHODEN_FARBEN[METHODEN_NAMEN.index(m)] for m in ml_methoden]
    bars = ax.bar(
        ml_methoden,
        rho_werte,
        color=farben_ml,
        alpha=0.8,
        edgecolor="black",
    )
    ax.set_ylabel("Spearman ρ")
    ax.set_title(f"Spearman-Korrelation vs. {REFERENZ_METHODE}")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, rho_werte):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    # --- 6x6 Korrelations-Heatmap ---
    ax = axes[0, 1]
    methoden_labels = korrelations_df.columns.tolist()
    matrix_werte = korrelations_df.values

    im = ax.imshow(matrix_werte, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(methoden_labels)))
    ax.set_yticks(range(len(methoden_labels)))
    ax.set_xticklabels(methoden_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(methoden_labels, fontsize=8)
    ax.set_title("Paarweise Spearman-Korrelation")

    for i in range(len(methoden_labels)):
        for j in range(len(methoden_labels)):
            text_farbe = "white" if matrix_werte[i, j] < 0.5 else "black"
            ax.text(
                j,
                i,
                f"{matrix_werte[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color=text_farbe,
                fontweight="bold",
            )

    fig.colorbar(im, ax=ax, shrink=0.8)

    # --- Jaccard Top-k Overlap ---
    ax = axes[1, 0]
    jaccard_methoden = list(jaccard_ergebnisse.keys())
    jaccard_werte = [jaccard_ergebnisse[m] for m in jaccard_methoden]
    farben_jac = [METHODEN_FARBEN[METHODEN_NAMEN.index(m)] for m in jaccard_methoden]
    bars = ax.bar(
        jaccard_methoden,
        jaccard_werte,
        color=farben_jac,
        alpha=0.8,
        edgecolor="black",
    )
    ax.set_ylabel("Jaccard-Index")
    ax.set_title(f"Top-{TOP_K_JACCARD} Ranking-Overlap vs. {REFERENZ_METHODE}")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, jaccard_werte):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    # --- Zusammenfassungs-Tabelle ---
    ax = axes[1, 1]
    ax.axis("off")
    tabellen_daten = []
    for m in ml_methoden:
        rho = vs_regelbasiert[m]["rho"]
        p = vs_regelbasiert[m]["p_wert"]
        jac = jaccard_ergebnisse.get(m, 0.0)

        if p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = "**"
        elif p < 0.05:
            sig = "*"
        else:
            sig = "n.s."

        tabellen_daten.append([m, f"{rho:.4f}", sig, f"{jac:.3f}"])

    tabelle = ax.table(
        cellText=tabellen_daten,
        colLabels=["Methode", "Spearman ρ", "Signif.", f"Jaccard@{TOP_K_JACCARD}"],
        loc="center",
        cellLoc="center",
    )
    tabelle.auto_set_font_size(False)
    tabelle.set_fontsize(9)
    tabelle.scale(1.3, 1.8)
    ax.set_title("Zusammenfassung: Inhaltliche Validierung", fontweight="bold", pad=20)

    plt.tight_layout()
    return fig


# ===========================================================================
# Plot 3: Scatter-Plots (ML vs. Regelbasiert)
# ===========================================================================


def erstelle_scatter_plot(
    similarity_matrizen: dict[str, np.ndarray],
    vs_regelbasiert: dict[str, dict],
) -> plt.Figure:
    """
    Erstelle 5 Scatter-Plots (jede ML-Methode vs. Regelbasiert).

    Jeder Punkt entspricht einem Rotor-Paar. Eine Regressionslinie
    und der Spearman-rho-Wert werden annotiert.

    :param similarity_matrizen: Similarity-Matrizen pro Methode
    :type similarity_matrizen: dict[str, np.ndarray]
    :param vs_regelbasiert: Spearman-Ergebnisse vs. Regelbasiert
    :type vs_regelbasiert: dict[str, dict]
    :return: Matplotlib Figure
    :rtype: plt.Figure
    """
    ml_methoden = [m for m in METHODEN_NAMEN if m != REFERENZ_METHODE]
    referenz_werte = extrahiere_obere_dreiecksmatrix(similarity_matrizen[REFERENZ_METHODE])

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        "Scatter-Plots: ML-Methoden vs. Regelbasiert " "(jeder Punkt = 1 Rotor-Paar)",
        fontsize=14,
        fontweight="bold",
    )

    for idx, name in enumerate(ml_methoden):
        zeile = idx // 3
        spalte = idx % 3
        ax = axes[zeile, spalte]

        methode_werte = extrahiere_obere_dreiecksmatrix(similarity_matrizen[name])
        rho = vs_regelbasiert[name]["rho"]
        farbe = METHODEN_FARBEN[METHODEN_NAMEN.index(name)]

        ax.scatter(
            referenz_werte,
            methode_werte,
            alpha=SCATTER_ALPHA,
            s=SCATTER_PUNKTGROESSE,
            color=farbe,
            edgecolors="none",
            rasterized=True,
        )

        # Regressionslinie
        koeffizienten = np.polyfit(referenz_werte, methode_werte, 1)
        x_linie = np.linspace(0.0, 1.0, 100)
        y_linie = np.polyval(koeffizienten, x_linie)
        ax.plot(
            x_linie,
            y_linie,
            color="black",
            linewidth=1.5,
            linestyle="--",
            alpha=0.8,
        )

        # Diagonale (perfekte Uebereinstimmung)
        ax.plot(
            [0, 1],
            [0, 1],
            color="gray",
            linewidth=0.8,
            linestyle=":",
            alpha=0.5,
        )

        ax.set_xlabel(REFERENZ_METHODE, fontsize=9)
        ax.set_ylabel(name, fontsize=9)
        ax.set_title(f"{name}  (ρ = {rho:.3f})", fontsize=10, fontweight="bold")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    # 6. Subplot: Zusammenfassungs-Tabelle
    ax = axes[1, 2]
    ax.axis("off")

    zeilen_daten = []
    for m in ml_methoden:
        rho = vs_regelbasiert[m]["rho"]
        if rho >= 0.7:
            staerke = "stark"
        elif rho >= 0.4:
            staerke = "moderat"
        else:
            staerke = "schwach"
        zeilen_daten.append([m, f"{rho:.4f}", staerke])

    tabelle = ax.table(
        cellText=zeilen_daten,
        colLabels=["Methode", "Spearman ρ", "Stärke"],
        loc="center",
        cellLoc="center",
    )
    tabelle.auto_set_font_size(False)
    tabelle.set_fontsize(9)
    tabelle.scale(1.3, 1.8)
    ax.set_title("Korrelationsstärke", fontweight="bold", pad=20)

    plt.tight_layout()
    return fig


# ===========================================================================
# CSV-Export
# ===========================================================================


def exportiere_ergebnisse(
    basis_metriken: dict[str, dict],
    vs_regelbasiert: dict[str, dict],
    jaccard_ergebnisse: dict[str, float],
    ausgabe_verzeichnis: Path,
) -> Path:
    """
    Exportiere alle Validierungsergebnisse als CSV-Datei.

    :param basis_metriken: Statistische Metriken pro Methode
    :type basis_metriken: dict[str, dict]
    :param vs_regelbasiert: Spearman-Ergebnisse vs. Regelbasiert
    :type vs_regelbasiert: dict[str, dict]
    :param jaccard_ergebnisse: Jaccard-Indices pro Methode
    :type jaccard_ergebnisse: dict[str, float]
    :param ausgabe_verzeichnis: Zielverzeichnis fuer die CSV-Datei
    :type ausgabe_verzeichnis: Path
    :return: Pfad zur erzeugten CSV-Datei
    :rtype: Path
    """
    zeitstempel = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_pfad = ausgabe_verzeichnis / f"validierung_ergebnisse_{zeitstempel}.csv"

    zeilen = []
    for name, metriken in basis_metriken.items():
        zeile = {
            "Methode": name,
            "Range_Prozent": round(metriken["range"] * 100, 2),
            "CV_Prozent": round(metriken["cv"] * 100, 2),
            "Silhouette": round(metriken["silhouette"], 4),
            "Min": round(metriken["min"], 4),
            "Mean": round(metriken["mean"], 4),
            "Max": round(metriken["max"], 4),
            "StdDev": round(metriken["std"], 4),
            "Bewertung": metriken["bewertung"],
        }

        if name in vs_regelbasiert:
            zeile["Spearman_rho"] = round(vs_regelbasiert[name]["rho"], 4)
            zeile["Spearman_p"] = vs_regelbasiert[name]["p_wert"]
        else:
            zeile["Spearman_rho"] = 1.0
            zeile["Spearman_p"] = 0.0

        zeile[f"Jaccard_Top{TOP_K_JACCARD}"] = round(
            jaccard_ergebnisse.get(name, 1.0),
            4,
        )

        zeilen.append(zeile)

    df = pd.DataFrame(zeilen)
    df.to_csv(csv_pfad, index=False)

    print(f"\n   ✓ CSV exportiert: {csv_pfad}")
    return csv_pfad


# ===========================================================================
# Zusammenfassung (Konsole)
# ===========================================================================


def drucke_zusammenfassung(
    basis_metriken: dict[str, dict],
    vs_regelbasiert: dict[str, dict],
    jaccard_ergebnisse: dict[str, float],
) -> None:
    """
    Drucke eine tabellarische Zusammenfassung aller Ergebnisse auf die Konsole.

    :param basis_metriken: Statistische Metriken pro Methode
    :type basis_metriken: dict[str, dict]
    :param vs_regelbasiert: Spearman-Ergebnisse vs. Regelbasiert
    :type vs_regelbasiert: dict[str, dict]
    :param jaccard_ergebnisse: Jaccard-Indices pro Methode
    :type jaccard_ergebnisse: dict[str, float]
    """
    print(f"\n{'=' * 70}")
    print("GESAMTZUSAMMENFASSUNG")
    print(f"{'=' * 70}")

    header = (
        f"{'Methode':<15} {'Range':>8} {'CV':>8} {'Silh.':>8} "
        f"{'ρ vs Regel':>12} {'Jaccard':>10} {'Status'}"
    )
    print(f"\n{header}")
    print("-" * 80)

    for name, m in basis_metriken.items():
        rho_str = f"{vs_regelbasiert[name]['rho']:.3f}" if name in vs_regelbasiert else "  1.000"
        jac_str = f"{jaccard_ergebnisse.get(name, 1.0):.3f}"
        print(
            f"{name:<15} {m['range'] * 100:>7.1f}% {m['cv'] * 100:>7.1f}% "
            f"{m['silhouette']:>8.3f} {rho_str:>12} {jac_str:>10} "
            f"{m['bewertung']}"
        )

    # Beste pro Kategorie
    best_range = max(basis_metriken.items(), key=lambda x: x[1]["range"])
    best_cv = max(basis_metriken.items(), key=lambda x: x[1]["cv"])
    best_sil = max(basis_metriken.items(), key=lambda x: x[1]["silhouette"])
    best_rho = max(vs_regelbasiert.items(), key=lambda x: x[1]["rho"])
    best_jac = max(jaccard_ergebnisse.items(), key=lambda x: x[1])

    print(f"\n{'=' * 70}")
    print("BESTE METHODEN")
    print(f"{'=' * 70}")
    print(
        f"\n🏆 Beste Trennschärfe (Range):    "
        f"{best_range[0]} ({best_range[1]['range'] * 100:.1f}%)"
    )
    print(f"📊 Höchste Variabilität (CV):     " f"{best_cv[0]} ({best_cv[1]['cv'] * 100:.1f}%)")
    print(f"⭐ Beste Cluster-Qualität (Silh.): " f"{best_sil[0]} ({best_sil[1]['silhouette']:.3f})")
    print(f"🔗 Höchste Korrelation mit Regel.: " f"{best_rho[0]} (ρ = {best_rho[1]['rho']:.4f})")
    print(
        f"🎯 Bester Ranking-Overlap:         "
        f"{best_jac[0]} (Jaccard@{TOP_K_JACCARD} = {best_jac[1]:.3f})"
    )


# ===========================================================================
# Hauptprogramm
# ===========================================================================


def main() -> None:
    """Hauptprogramm: Matrix-Validierung als CLI-Script."""
    parser = argparse.ArgumentParser(
        description="Vollständige Matrix-Validierung aller Similarity-Methoden",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=STANDARD_LATENT_DIM,
        help=f"Latent-Dimension für PCA/Autoencoder (Standard: {STANDARD_LATENT_DIM})",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=STANDARD_N_CLUSTERS,
        help=f"Anzahl Cluster für K-Means (Standard: {STANDARD_N_CLUSTERS})",
    )
    args = parser.parse_args()

    print(f"{'=' * 70}")
    print("VOLLSTÄNDIGE MATRIX-VALIDIERUNG (Realdaten)")
    print(f"{'=' * 70}")

    gesamtzeit = time.time()

    # 1. Daten laden
    features_by_rotor, numeric_stats, rotor_ids = lade_json_daten()
    anzahl_rotoren = len(rotor_ids)

    # 2. Alle 6 Similarity-Matrizen berechnen
    similarity_matrizen = berechne_alle_matrizen(
        features_by_rotor,
        numeric_stats,
        rotor_ids,
        latent_dim=args.latent_dim,
        n_clusters=args.n_clusters,
    )

    # 3. Statistische Basis-Metriken (Range, CV, Silhouette)
    basis_metriken = berechne_basis_metriken(similarity_matrizen)

    # 4. Spearman-Korrelationen (paarweise + vs. Regelbasiert)
    korrelations_df, vs_regelbasiert = berechne_spearman_korrelationen(
        similarity_matrizen,
    )

    # 5. Jaccard Ranking-Overlap
    jaccard_ergebnisse = berechne_jaccard_overlap(
        similarity_matrizen,
        rotor_ids,
    )

    # 6. Konsolenzusammenfassung
    drucke_zusammenfassung(basis_metriken, vs_regelbasiert, jaccard_ergebnisse)

    # 7. Visualisierungen erstellen und speichern
    ausgabe_dir = AUSGABE_VERZEICHNIS
    ausgabe_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print("ERSTELLE VISUALISIERUNGEN")
    print(f"{'=' * 70}")

    fig_stat = erstelle_statistik_plot(
        basis_metriken,
        similarity_matrizen,
        anzahl_rotoren,
    )
    pfad_stat = ausgabe_dir / "validierung_statistik.png"
    fig_stat.savefig(pfad_stat, dpi=300, bbox_inches="tight")
    print(f"   ✓ {pfad_stat}")
    plt.close(fig_stat)

    fig_korr = erstelle_korrelations_plot(
        korrelations_df,
        vs_regelbasiert,
        jaccard_ergebnisse,
    )
    pfad_korr = ausgabe_dir / "validierung_korrelation.png"
    fig_korr.savefig(pfad_korr, dpi=300, bbox_inches="tight")
    print(f"   ✓ {pfad_korr}")
    plt.close(fig_korr)

    fig_scatter = erstelle_scatter_plot(
        similarity_matrizen,
        vs_regelbasiert,
    )
    pfad_scatter = ausgabe_dir / "validierung_scatter.png"
    fig_scatter.savefig(pfad_scatter, dpi=300, bbox_inches="tight")
    print(f"   ✓ {pfad_scatter}")
    plt.close(fig_scatter)

    # 8. CSV exportieren
    csv_pfad = exportiere_ergebnisse(
        basis_metriken,
        vs_regelbasiert,
        jaccard_ergebnisse,
        ausgabe_dir,
    )

    # Fertig
    dauer = time.time() - gesamtzeit
    print(f"\n{'=' * 70}")
    print(f"✅ FERTIG! (Gesamtzeit: {dauer:.1f}s)")
    print(f"{'=' * 70}")
    print(f"   PNG: {pfad_stat}")
    print(f"   PNG: {pfad_korr}")
    print(f"   PNG: {pfad_scatter}")
    print(f"   CSV: {csv_pfad}")


if __name__ == "__main__":
    main()
