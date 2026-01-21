#!/usr/bin/env python3
"""
Validierungs-Plots für Similarity-Methoden

Berechnet die Similarities mit den GLEICHEN Funktionen wie das Streamlit UI
und erstellt Visualisierungen.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import datetime

# Importiere die Similarity-Funktionen aus dem rotor_owl Package
from rotor_owl.daten.feature_fetcher import fetch_all_features, build_numeric_stats
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
from rotor_owl.methoden.graph_embedding_aehnlichkeit import (
    berechne_topk_aehnlichkeiten_graph_embedding,
)
from rotor_owl.config.kategorien import KAT_GEOM_MECH, KAT_MTRL_PROC, KAT_REQ_ELEC

from rdflib import Graph
from SPARQLWrapper import SPARQLWrapper, XML


def load_data_from_fuseki(endpoint_url="http://localhost:3030/rotors/sparql"):
    """Lade Daten von Fuseki - GENAU wie Streamlit UI"""

    print(f"📡 Lade Daten von Fuseki: {endpoint_url}")

    # Features laden
    features_by_rotor = fetch_all_features(endpoint_url)
    print(f"✓ {len(features_by_rotor)} Rotoren geladen")

    # Stats berechnen
    numeric_stats = build_numeric_stats(features_by_rotor)
    print("✓ Numerische Statistiken berechnet")

    # Ontologie-Graph laden (für Graph-Embeddings)
    print("📊 Lade Ontologie-Graph...")
    ontologie_graph = Graph()
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

    print(f"✓ Graph geladen: {len(ontologie_graph)} Tripel")

    return features_by_rotor, numeric_stats, ontologie_graph


def compute_all_similarities(features_by_rotor, numeric_stats, ontologie_graph):
    """Berechne alle 7 Methoden - GENAU wie im Streamlit UI Live-Validierung"""

    rotor_ids = sorted(features_by_rotor.keys())
    query_rotor = rotor_ids[0]  # Erster Rotor als Query (wie im UI)

    # Gleichgewichtung für Validierung
    val_gewichte = {KAT_GEOM_MECH: 1.0, KAT_MTRL_PROC: 1.0, KAT_REQ_ELEC: 1.0}

    methoden_ergebnisse = {}

    print("\n" + "=" * 80)
    print("BERECHNE ALLE METHODEN (wie Streamlit UI)")
    print("=" * 80)

    # 1. Regelbasiert
    print("\n1/7: Regelbasiert...")
    ergebnisse = berechne_topk_aehnlichkeiten(
        query_rotor, rotor_ids, features_by_rotor, numeric_stats, val_gewichte, k=len(rotor_ids)
    )
    methoden_ergebnisse["Regelbasiert"] = {r[0]: r[1] for r in ergebnisse}
    print(f"   ✓ {len(methoden_ergebnisse['Regelbasiert'])} Similarities berechnet")

    # 2. k-NN
    print("\n2/7: k-NN...")
    knn_emb = build_knn_embeddings(features_by_rotor, numeric_stats)
    ergebnisse = berechne_topk_aehnlichkeiten_knn(
        query_rotor, rotor_ids, knn_emb, val_gewichte, k=len(rotor_ids)
    )
    methoden_ergebnisse["k-NN"] = {r[0]: r[1] for r in ergebnisse}
    print(f"   ✓ {len(methoden_ergebnisse['k-NN'])} Similarities berechnet")

    # 3. PCA
    print("\n3/7: PCA...")
    pca_emb = build_pca_embeddings(features_by_rotor, numeric_stats, latent_dim=8)
    ergebnisse = berechne_topk_aehnlichkeiten_pca(
        query_rotor, rotor_ids, pca_emb, val_gewichte, k=len(rotor_ids)
    )
    methoden_ergebnisse["PCA"] = {r[0]: r[1] for r in ergebnisse}
    print(f"   ✓ {len(methoden_ergebnisse['PCA'])} Similarities berechnet")

    # 4. Autoencoder
    print("\n4/7: Autoencoder...")
    ae_emb = build_autoencoder_embeddings(features_by_rotor, numeric_stats, latent_dim=8)
    ergebnisse = berechne_topk_aehnlichkeiten_autoencoder(
        query_rotor, rotor_ids, ae_emb, val_gewichte, k=len(rotor_ids)
    )
    methoden_ergebnisse["Autoencoder"] = {r[0]: r[1] for r in ergebnisse}
    print(f"   ✓ {len(methoden_ergebnisse['Autoencoder'])} Similarities berechnet")

    # 5. K-Means
    print("\n5/7: K-Means...")
    ergebnisse = berechne_topk_aehnlichkeiten_kmeans(
        query_rotor,
        rotor_ids,
        features_by_rotor,
        numeric_stats,
        val_gewichte,
        n_clusters=5,
        k=len(rotor_ids),
    )
    methoden_ergebnisse["K-Means"] = {r[0]: r[1] for r in ergebnisse}
    print(f"   ✓ {len(methoden_ergebnisse['K-Means'])} Similarities berechnet")

    # 6. Graph-Embeddings
    print("\n6/7: Graph-Embeddings...")
    ergebnisse = berechne_topk_aehnlichkeiten_graph_embedding(
        query_rotor,
        rotor_ids,
        ontologie_graph,
        val_gewichte,
        embedding_dimensions=32,
        num_walks=1,
        walk_length=10,
        k=len(rotor_ids),
    )
    methoden_ergebnisse["Graph"] = {r[0]: r[1] for r in ergebnisse}
    print(f"   ✓ {len(methoden_ergebnisse['Graph'])} Similarities berechnet")

    # 7. Hybrid (K-Means + PCA)
    print("\n7/7: Hybrid (K-Means + PCA)...")
    hybrid_sims = {}
    for rid in rotor_ids:
        if rid != query_rotor:
            kmeans_sim = methoden_ergebnisse["K-Means"].get(rid, 0.5)
            pca_sim = methoden_ergebnisse["PCA"].get(rid, 0.5)
            hybrid_sims[rid] = 0.5 * kmeans_sim + 0.5 * pca_sim
    methoden_ergebnisse["Hybrid"] = hybrid_sims
    print(f"   ✓ {len(methoden_ergebnisse['Hybrid'])} Similarities berechnet")

    return methoden_ergebnisse, query_rotor


def compute_validation_metrics(methoden_ergebnisse):
    """Berechne Validierungs-Metriken für alle Methoden"""

    print("\n" + "=" * 80)
    print("BERECHNE VALIDIERUNGS-METRIKEN")
    print("=" * 80)

    validation_results = {}

    for method_name, sims in methoden_ergebnisse.items():
        print(f"\n📊 {method_name}...")

        sim_values = list(sims.values())

        if len(sim_values) == 0:
            print("   ⚠️  Keine Similarities vorhanden")
            continue

        # Basis-Statistiken
        result = {
            "mean": np.mean(sim_values),
            "std": np.std(sim_values),
            "min": np.min(sim_values),
            "max": np.max(sim_values),
            "range": np.max(sim_values) - np.min(sim_values),
            "cv": np.std(sim_values) / np.mean(sim_values) if np.mean(sim_values) > 0 else 0,
        }

        print(f"   Range: {result['range']*100:.1f}%")
        print(f"   CV: {result['cv']*100:.1f}%")

        # Silhouette Score berechnen
        try:
            n = len(sim_values)
            sim_matrix = np.zeros((n, n))
            rid_list = list(sims.keys())

            for i, ri in enumerate(rid_list):
                for j, rj in enumerate(rid_list):
                    if i == j:
                        sim_matrix[i, j] = 1.0
                    else:
                        # Symmetrische Matrix
                        sim_matrix[i, j] = sims.get(ri, 0.5) if i < j else sims.get(rj, 0.5)

            dist_matrix = np.clip(1 - sim_matrix, 0, 1)
            np.fill_diagonal(dist_matrix, 0)

            if n > 5:
                clustering = AgglomerativeClustering(
                    n_clusters=5, metric="precomputed", linkage="average"
                )
                labels = clustering.fit_predict(dist_matrix)
                sil_score = silhouette_score(dist_matrix, labels, metric="precomputed")
                result["silhouette"] = sil_score
                print(f"   Silhouette: {sil_score:.3f}")
            else:
                result["silhouette"] = 0.0
                print("   Silhouette: N/A (zu wenige Datenpunkte)")
        except Exception as e:
            result["silhouette"] = 0.0
            print(f"   Silhouette: Fehler ({e})")

        validation_results[method_name] = result

    return validation_results


def create_validation_plots(
    validation_results, methoden_ergebnisse, output_png=None, output_pdf=None
):
    """Erstelle umfassende Visualisierung - GENAU wie im Streamlit UI"""

    methods = list(validation_results.keys())
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Methoden-Validierung (Live-Berechnung aus Streamlit UI)", fontsize=14, fontweight="bold"
    )

    # Plot 1: Histogramme der Similarity-Verteilungen
    ax = axes[0, 0]
    for i, (method, sims) in enumerate(methoden_ergebnisse.items()):
        sim_vals = list(sims.values())
        ax.hist(sim_vals, bins=20, alpha=0.5, label=method, color=colors[i % len(colors)])
    ax.set_xlabel("Similarity")
    ax.set_ylabel("Frequency")
    ax.set_title("Similarity Verteilungen")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Plot 2: Range (Trennschärfe)
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
            bar.get_height(),
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    # Plot 3: CV (Variationskoeffizient)
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
            bar.get_height(),
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    # Plot 4: Silhouette Score
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
            bar.get_height(),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
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

    # Plot 6: Zusammenfassungs-Tabelle
    ax = axes[1, 2]
    ax.axis("off")
    table_data = []
    for m in methods:
        r = validation_results[m]
        table_data.append(
            [m, f"{r['range']*100:.1f}%", f"{r['cv']*100:.1f}%", f"{r['silhouette']:.2f}"]
        )
    table = ax.table(
        cellText=table_data,
        colLabels=["Methode", "Range", "CV", "Silhouette"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax.set_title("Zusammenfassung", fontweight="bold", pad=20)

    plt.tight_layout()

    if output_png:
        plt.savefig(output_png, dpi=300, bbox_inches="tight")
        print(f"\n✓ PNG gespeichert: {output_png}")

    if output_pdf:
        plt.savefig(output_pdf, bbox_inches="tight")
        print(f"✓ PDF gespeichert: {output_pdf}")

    plt.show()

    return fig


def export_results_to_csv(validation_results, output_dir):
    """Exportiere Ergebnisse als CSV (für spätere Verwendung)"""

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = output_dir / f"similarity_detailed_{timestamp}.csv"

    rows = []
    for method_name, result in validation_results.items():
        row = {
            "Methode": method_name,
            "Range": result["range"],
            "Range_Percent": result["range"] * 100,
            "Min_Similarity": result["min"],
            "Max_Similarity": result["max"],
            "Mean": result["mean"],
            "StdDev": result["std"],
            "CV": result["cv"],
            "CV_Percent": result["cv"] * 100,
            "Best_Silhouette": result["silhouette"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_file, index=False)

    print(f"✓ CSV exportiert: {csv_file}")
    return csv_file


def print_summary(validation_results):
    """Drucke Zusammenfassung"""

    print("\n" + "=" * 80)
    print("ZUSAMMENFASSUNG")
    print("=" * 80)

    print(f"\n{'Methode':<15} {'Range %':>10} {'CV %':>10} {'Silhouette':>12}")
    print("-" * 80)

    for method, result in validation_results.items():
        print(
            f"{method:<15} {result['range']*100:>10.1f} {result['cv']*100:>10.1f} {result['silhouette']:>12.3f}"
        )

    # Beste Methoden
    best_range = max(validation_results.items(), key=lambda x: x[1]["range"])
    best_cv = max(validation_results.items(), key=lambda x: x[1]["cv"])
    best_sil = max(validation_results.items(), key=lambda x: x[1]["silhouette"])

    print("\n" + "=" * 80)
    print("BESTE METHODEN")
    print("=" * 80)

    print(
        f"\n🏆 Beste Trennschärfe (Range):      {best_range[0]} ({best_range[1]['range']*100:.1f}%)"
    )
    print(f"📊 Höchste Variabilität (CV):      {best_cv[0]} ({best_cv[1]['cv']*100:.1f}%)")
    print(f"⭐ Beste Cluster-Qualität (Silh.): {best_sil[0]} ({best_sil[1]['silhouette']:.3f})")


def main():
    """Hauptprogramm"""

    print("=" * 80)
    print("VALIDIERUNG MIT STREAMLIT UI BERECHNUNGEN")
    print("=" * 80)
    print()

    # Lade Daten von Fuseki
    try:
        features_by_rotor, numeric_stats, ontologie_graph = load_data_from_fuseki()
    except Exception as e:
        print(f"❌ Fehler beim Laden der Daten: {e}")
        print("   Stelle sicher, dass Fuseki läuft!")
        return

    # Berechne alle Methoden
    try:
        methoden_ergebnisse, query_rotor = compute_all_similarities(
            features_by_rotor, numeric_stats, ontologie_graph
        )
    except Exception as e:
        print(f"❌ Fehler bei der Berechnung: {e}")
        import traceback

        traceback.print_exc()
        return

    # Berechne Validierungs-Metriken
    validation_results = compute_validation_metrics(methoden_ergebnisse)

    # Drucke Zusammenfassung
    print_summary(validation_results)

    # Exportiere CSV
    output_dir = Path(__file__).parent / "data" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_file = export_results_to_csv(validation_results, output_dir)

    # Erstelle Visualisierung
    print("\n" + "=" * 80)
    print("ERSTELLE VISUALISIERUNG")
    print("=" * 80)

    data_dir = Path(__file__).parent / "data"
    output_png = data_dir / "similarity_validation.png"
    output_pdf = data_dir / "similarity_validation.pdf"

    create_validation_plots(
        validation_results, methoden_ergebnisse, output_png=output_png, output_pdf=output_pdf
    )

    print("\n" + "=" * 80)
    print("✅ FERTIG!")
    print("=" * 80)
    print(f"Query-Rotor: {query_rotor}")
    print(f"PNG: {output_png}")
    print(f"PDF: {output_pdf}")
    print(f"CSV: {csv_file}")


if __name__ == "__main__":
    main()
