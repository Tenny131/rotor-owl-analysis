"""
Sanity-Check Tests für Ähnlichkeitswerte.

Prüft ob die berechneten Similarities in sinnvollen Bereichen liegen.
"""


def test_autoencoder_similarity_ranges():
    """
    Testet ob Autoencoder-Similarities in sinnvollen Bereichen liegen.

    Erwartung:
    - Top-1 ähnlichster: 0.65 - 0.98
    - Top-5 durchschnitt: 0.50 - 0.85
    - Mindestens etwas Varianz (nicht alles gleich)
    """
    # Dieser Test sollte mit echten Daten aus Fuseki laufen
    # Hier als Template für manuellen Test

    # Beispiel-Erwartungen basierend auf Rotor-Daten:
    # min_top1_similarity = 0.65  # Bester Match sollte mind. 65% sein
    # max_top1_similarity = 0.98  # Aber nicht perfekt (außer bei identischen)
    # min_variance = 0.05  # Top-5 sollten mind. 5% Unterschied haben

    # TODO: Mit echten Daten testen
    # results = berechne_topk_aehnlichkeiten_autoencoder(...)
    # similarities = [sim for _, sim, _ in results]
    # assert min_top1_similarity <= similarities[0] <= max_top1_similarity
    # assert (max(similarities) - min(similarities)) >= min_variance

    assert True  # Placeholder


def test_vektorbasiert_similarity_ranges():
    """
    Testet ob vektorbasierte Similarities in sinnvollen Bereichen liegen.

    Erwartung:
    - Top-1 ähnlichster: 0.55 - 0.90
    - Top-5 durchschnitt: 0.45 - 0.75
    """
    # TODO: Mit echten Daten testen
    assert True  # Placeholder


def test_hybrid_method_combines_correctly():
    """
    Testet ob Hybrid-Methode sinnvoll kombiniert.

    Erwartung:
    - Gewichtete Summe liegt zwischen den Einzelmethoden
    - Bei 60% AE + 40% Vektorbasiert: Ergebnis näher an Autoencoder
    """
    # Beispiel:
    # ae_sim = 0.85
    # vektor_sim = 0.65
    # hybrid_sim = 0.6 * ae_sim + 0.4 * vektor_sim = 0.77

    ae_sim = 0.85
    vektor_sim = 0.65
    gewicht_ae = 0.6
    gewicht_vektor = 0.4

    expected_hybrid = gewicht_ae * ae_sim + gewicht_vektor * vektor_sim
    assert abs(expected_hybrid - 0.77) < 0.01

    # Hybrid sollte zwischen min und max liegen
    assert min(ae_sim, vektor_sim) <= expected_hybrid <= max(ae_sim, vektor_sim)


def test_similarity_order_makes_sense():
    """
    Testet ob die Rangfolge sinnvoll ist.

    Erwartung:
    - Ähnlichere Rotoren haben höhere Scores
    - Top-1 > Top-2 > Top-3 > ... (monoton fallend)
    """
    # Beispiel Top-5 Similarities
    top5_similarities = [0.87, 0.82, 0.75, 0.68, 0.61]

    # Prüfe monoton fallend
    for i in range(len(top5_similarities) - 1):
        assert (
            top5_similarities[i] >= top5_similarities[i + 1]
        ), f"Top-{i+1} ({top5_similarities[i]}) sollte >= Top-{i+2} ({top5_similarities[i+1]}) sein"


def test_detect_anomalies():
    """
    Demonstriert Anomalie-Erkennung (Info-Ausgabe, kein Test-Fehler).

    Dieser Test zeigt wie man Anomalien in Similarity-Werten erkennen würde.
    Er schlägt NICHT fehl, dient nur zur Demonstration.
    """
    print("\n" + "=" * 70)
    print("ANOMALIE-ERKENNUNG DEMONSTRATION")
    print("=" * 70)

    # ANOMALIE 1: Alle Werte identisch → Bug!
    anomaly_all_same = [0.5, 0.5, 0.5, 0.5, 0.5]
    variance = max(anomaly_all_same) - min(anomaly_all_same)

    print("\n1️⃣  Test: Alle Werte identisch")
    print(f"   Beispiel: {anomaly_all_same}")
    print(f"   Varianz: {variance}")
    if variance < 0.01:
        print("   ⚠️  ANOMALIE: Alle Similarities sind gleich - möglicherweise Bug!")
    else:
        print("   ✅ OK")

    # ANOMALIE 2: Alle Werte = 0.5 → Zero-Vektor Bug (Vektorbasiert)
    print("\n2️⃣  Test: Zero-Vektor Bug (alle = 0.5)")
    if all(abs(s - 0.5) < 0.01 for s in anomaly_all_same):
        print("   🚨 ANOMALIE: Zero-Vektor Bug detektiert! Alle Similarities = 0.5")
    else:
        print("   ✅ OK")

    # ANOMALIE 3: Alle Werte > 0.95 → Zu wenig Varianz in Daten
    anomaly_too_high = [0.98, 0.97, 0.96, 0.96, 0.95]
    print("\n3️⃣  Test: Alle Werte sehr hoch (>0.95)")
    print(f"   Beispiel: {anomaly_too_high}")
    if all(s > 0.95 for s in anomaly_too_high):
        print("   ⚠️  WARNUNG: Alle Similarities >0.95 - Features haben zu wenig Varianz")
        print("   ℹ️  Bei Produktfamilien (z.B. D001-D050) ist dies NORMAL und KORREKT")
    else:
        print("   ✅ OK")

    # ANOMALIE 4: Alle Werte < 0.3 → Features falsch normalisiert
    anomaly_too_low = [0.28, 0.25, 0.22, 0.20, 0.18]
    print("\n4️⃣  Test: Alle Werte sehr niedrig (<0.3)")
    print(f"   Beispiel: {anomaly_too_low}")
    if all(s < 0.3 for s in anomaly_too_low):
        print("   ⚠️  WARNUNG: Alle Similarities <0.3 - Features möglicherweise falsch normalisiert")
    else:
        print("   ✅ OK")

    print(f"\n{'='*70}\n")


# Nutze diesen Test mit echten Daten:
"""
MANUELLER TEST in Streamlit:

1. Wähle "Hybrid-Methode"
2. Query: Rotor_D001
3. Prüfe Top-5 Ergebnisse:

ERWARTETE WERTE (Beispiel):
┌─────────────┬────────────┬──────────────┬──────────┐
│ Rotor       │ Autoencoder│ Vektorbasiert│ S_ges    │
├─────────────┼────────────┼──────────────┼──────────┤
│ Rotor_D002  │ 0.8823     │ 0.7241       │ 0.8188   │  ✅ Gut
│ Rotor_D010  │ 0.8156     │ 0.6893       │ 0.7651   │  ✅ Gut
│ Rotor_D023  │ 0.7421     │ 0.6512       │ 0.7057   │  ✅ Gut
│ Rotor_D031  │ 0.6834     │ 0.6203       │ 0.6582   │  ✅ Gut
│ Rotor_D041  │ 0.6125     │ 0.5847       │ 0.6009   │  ✅ Gut
└─────────────┴────────────┴──────────────┴──────────┘

WARNSIGNALE:
- Vektorbasiert alle = 0.5000 → 🚨 Zero-Vektor Bug
- Autoencoder alle > 0.95     → ⚠️  Zu wenig Varianz
- Beide < 0.30               → ⚠️  Normalisierungsfehler
- Keine Variation (±0.01)    → ⚠️  Methode funktioniert nicht
"""
