"""
Diagnose-Test: Feature-Varianz prüfen.

Gibt Warnungen aus für Features mit zu wenig Varianz (nur Info, kein Test-Fehler).
"""

import pytest
from rotor_owl.daten.feature_fetcher import fetch_all_features, build_numeric_stats


def test_feature_variance_diagnostic():
    """
    Prüft ob Features genug Varianz haben und gibt Warnungen aus.

    WICHTIG: Dieser Test schlägt NICHT fehl, er dient nur zur Information!
    Er listet Parameter mit zu wenig Varianz auf.
    """
    print("\n" + "=" * 70)
    print("FEATURE VARIANZ DIAGNOSE (Info-Ausgabe, kein Test-Fehler)")
    print("=" * 70)

    # Versuche Daten zu laden
    try:
        features_by_rotor = fetch_all_features("http://localhost:3030/rotors/sparql")
    except Exception as e:
        print(f"⚠️  Fuseki nicht erreichbar: {e}")
        print("ℹ️  Test wird übersprungen (keine Daten verfügbar)")
        pytest.skip(f"Fuseki nicht erreichbar: {e}")
        return

    if not features_by_rotor:
        print("⚠️  Keine Features gefunden")
        pytest.skip("Keine Features von Fuseki erhalten")
        return

    # Baue Stats
    stats = build_numeric_stats(features_by_rotor)

    # Prüfe numerische Features
    low_variance_numeric = []
    very_low_variance_numeric = []

    for (kategorie, param_name), (mean, std) in stats.items():
        if std == 0:
            very_low_variance_numeric.append(param_name)
        elif std < mean * 0.05:  # Weniger als 5% Varianz
            low_variance_numeric.append(param_name)

    # Prüfe kategorische Features
    categorical_diversity = {}
    for rotor_id, rotor_daten in features_by_rotor.items():
        for parameter_schluessel, parameter_datensatz in rotor_daten["params"].items():
            param_value = parameter_datensatz.get("value")
            if param_value is not None and not isinstance(param_value, (int, float)):
                # Kategorischer Wert
                if parameter_schluessel not in categorical_diversity:
                    categorical_diversity[parameter_schluessel] = set()
                categorical_diversity[parameter_schluessel].add(str(param_value))

    low_diversity_categorical = []
    very_low_diversity_categorical = []

    for parameter_schluessel, unique_values in categorical_diversity.items():
        component, param_name = parameter_schluessel

        if len(unique_values) == 1:
            very_low_diversity_categorical.append(f"{component}/{param_name}")
        elif len(unique_values) <= 2:
            low_diversity_categorical.append(f"{component}/{param_name}")

    # === AUSGABE ===

    if very_low_variance_numeric:
        print("\n⚠️  NUMERISCHE PARAMETER MIT KEINER VARIANZ (alle identisch):")
        print(f"   Anzahl: {len(very_low_variance_numeric)}")
        print(f"   Liste: {very_low_variance_numeric}")

    if low_variance_numeric:
        print("\n⚠️  NUMERISCHE PARAMETER MIT WENIG VARIANZ (CV < 5%):")
        print(f"   Anzahl: {len(low_variance_numeric)}")
        print(f"   Liste: {low_variance_numeric}")

    if very_low_diversity_categorical:
        print("\n⚠️  KATEGORISCHE PARAMETER MIT KEINER VIELFALT (nur 1 Wert):")
        print(f"   Anzahl: {len(very_low_diversity_categorical)}")
        print(f"   Liste: {very_low_diversity_categorical}")

    if low_diversity_categorical:
        print("\n⚠️  KATEGORISCHE PARAMETER MIT WENIG VIELFALT (nur 2 Werte):")
        print(f"   Anzahl: {len(low_diversity_categorical)}")
        print(f"   Liste: {low_diversity_categorical}")

    # Zusammenfassung
    total_low = (
        len(very_low_variance_numeric)
        + len(low_variance_numeric)
        + len(very_low_diversity_categorical)
        + len(low_diversity_categorical)
    )
    total_features = len(stats) + len(categorical_diversity)

    print(f"\n{'='*70}")
    print("ZUSAMMENFASSUNG:")
    print(f"  Gesamt Features: {total_features}")
    print(f"  Features mit zu wenig Varianz: {total_low} ({total_low/total_features*100:.1f}%)")

    if total_low > total_features * 0.5:
        print("\n🚨 WARNUNG: Mehr als 50% der Features haben geringe Varianz!")
        print("   → Hohe Ähnlichkeitswerte (0.95+) sind dadurch erwartbar")
    elif total_low > 0:
        print(f"\nℹ️  INFO: {total_low} Features mit geringer Varianz gefunden")
        print("   → Kann zu erhöhten Ähnlichkeitswerten führen")
    else:
        print("\n✅ OK: Alle Features haben ausreichende Varianz")

    print(f"{'='*70}\n")

    # Test passed IMMER (ist nur Info-Ausgabe)


if __name__ == "__main__":
    # Kann direkt ausgeführt werden
    test_feature_variance_diagnostic()
