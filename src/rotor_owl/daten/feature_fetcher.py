from __future__ import annotations

from collections import defaultdict
import streamlit as st

from rotor_owl.config.konfiguration import IMS_NAMESPACE
from rotor_owl.utils.sparql_helpers import run_sparql
from rotor_owl.utils.aufbereitung import (
    local_name,
    strip_last_suffix,
    normalize_param_name,
    safe_float,
)


@st.cache_data(show_spinner=False, ttl=60)
def fetch_component_dependencies(endpoint_url: str) -> dict[tuple[str, str], dict]:
    """
    Extrahiert Dependency-Relationen zwischen Komponenten aus der Ontologie.

    Erfasst alle Relationen mit Patterns:
    - *_Beeinflusst_* (beeinflusst)
    - *_affectsStrength_* (Festigkeitseinfluss)
    - *_Anfordert_* (Anforderungen)

    Returns:
        Dict mit (source_component, target_component) -> {
            "strength": str,  # "hoch", "mittel", "niedrig"
            "percentage": float  # 0.0 - 1.0
        }
    """
    sparql_query = f"""
PREFIX ims:  <{IMS_NAMESPACE}>
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
SELECT ?property ?strength ?percentage
WHERE {{
  ?property rdf:type <http://www.w3.org/2002/07/owl#ObjectProperty> .
  OPTIONAL {{ ?property ims:hasStrength ?strength . }}
  OPTIONAL {{ ?property ims:hasDependencyPercentage ?percentage . }}
  
  FILTER(
    CONTAINS(STR(?property), "Beeinflusst") ||
    CONTAINS(STR(?property), "affectsStrength") ||
    CONTAINS(STR(?property), "Anfordert")
  )
}}
"""
    query_json = run_sparql(endpoint_url, sparql_query)
    sparql_zeilen = query_json["results"]["bindings"]

    dependencies = {}

    for zeile in sparql_zeilen:
        property_uri = zeile["property"]["value"]
        property_name = local_name(property_uri)

        # Parse verschiedene Patterns:
        # "Blechpaket_Beeinflusst_Welle_1" -> ("Blechpaket", "Welle")
        # "Blechpaket_affectsStrength_Stanzkonzept_1" -> ("Blechpaket", "Stanzkonzept")
        # "Wellenenden_Anfordert_Kundenanforderung_1" -> ("Wellenenden", "Kundenanforderung")

        separator = None
        if "_Beeinflusst_" in property_name:
            separator = "_Beeinflusst_"
        elif "_affectsStrength_" in property_name:
            separator = "_affectsStrength_"
        elif "_Anfordert_" in property_name:
            separator = "_Anfordert_"

        if not separator:
            continue

        parts = property_name.split(separator)
        if len(parts) != 2:
            continue

        source = parts[0].replace(".", "_")
        target = strip_last_suffix(parts[1])  # Entferne _1 am Ende

        strength = zeile.get("strength", {}).get("value", "mittel")
        percentage_str = zeile.get("percentage", {}).get("value", "0.5")
        percentage = safe_float(percentage_str) or 0.5

        dependencies[(source, target)] = {"strength": strength, "percentage": percentage}

    return dependencies


@st.cache_data(show_spinner=False, ttl=60)
def fetch_all_features(endpoint_url: str) -> dict[str, dict]:
    """
    Liest ALLE Rotor-Features aus Fuseki und baut eine kompakte Datenstruktur auf.

    Ergebnisstruktur:
      features_by_rotor[rotor_id]["params"][(basis_component, basis_param)] = {
          "value": float | str | None,
          "unit": str | None,
          "ptype": str | None
      }

    Warum so?
    - So ist die Similarity-Berechnung später extrem schnell
    - Keys sind stabil über Rotor_1 (original) und Rotor_D001 (generiert)
    """
    sparql_query = f"""
PREFIX ims:  <{IMS_NAMESPACE}>
SELECT ?rotor ?component ?param ?value ?unit ?ptype
WHERE {{
  ?rotor ims:composed_of ?component .
  ?component ims:composed_of ?param .

  OPTIONAL {{ ?param ims:hasValue ?value . }}
  OPTIONAL {{ ?param ims:hasUnit ?unit . }}
  OPTIONAL {{ ?param ims:hasType ?ptype . }}

  FILTER(STRSTARTS(STR(?rotor), STR(ims:Rotor_)))
}}
"""
    query_json = run_sparql(endpoint_url, sparql_query)
    sparql_zeilen = query_json["results"]["bindings"]

    features_by_rotor: dict[str, dict] = {}

    for sparql_zeile in sparql_zeilen:
        rotor_uri = sparql_zeile["rotor"]["value"]
        component_uri = sparql_zeile["component"]["value"]
        parameter_uri = sparql_zeile["param"]["value"]

        rotor_id = local_name(rotor_uri)

        # Component: wir entfernen hinten _D001 oder _1, damit alle Designs sauber vergleichbar sind
        component_basis = strip_last_suffix(local_name(component_uri))

        # Parameter: muss beide Namensschemata auf den gleichen "Basisschlüssel" mappen
        parameter_basis = normalize_param_name(local_name(parameter_uri))

        parameter_typ = sparql_zeile.get("ptype", {}).get("value")
        parameter_einheit = sparql_zeile.get("unit", {}).get("value")
        wert_roh = sparql_zeile.get("value", {}).get("value")

        # Zahlenwerte werden zu float, alles andere bleibt String
        wert_num = safe_float(wert_roh)
        wert_final = (
            wert_num if wert_num is not None else (wert_roh if wert_roh is not None else None)
        )

        # Rotor-Eintrag sicherstellen
        if rotor_id not in features_by_rotor:
            features_by_rotor[rotor_id] = {"params": {}}

        parameter_schluessel = (component_basis, parameter_basis)
        features_by_rotor[rotor_id]["params"][parameter_schluessel] = {
            "value": wert_final,
            "unit": parameter_einheit,
            "ptype": parameter_typ,
        }

    return features_by_rotor


def build_numeric_stats(
    features_by_rotor: dict[str, dict],
) -> dict[tuple[str, str], tuple[float, float]]:
    """
    Baut min/max Statistik pro Parameter-Key (component_basis, parameter_basis).

    Diese Funktion wird von allen Similarity-Methoden benötigt, die numerische
    Parameter normieren müssen.

    Args:
        features_by_rotor: Feature-Daten aller Rotoren (von fetch_all_features)

    Returns:
        Dictionary mit min/max Werten pro Parameter-Schlüssel
        stats[(C_WELLE, P_WELLE_TIR)] = (min_wert, max_wert)

    Idee:
        - Für numerische Similarity brauchen wir eine Normierung
        - Dafür ist min/max über alle Rotoren eine einfache, robuste Basis
    """
    numerische_werte_pro_parameter: dict[tuple[str, str], list[float]] = defaultdict(list)

    for _, rotor_daten in features_by_rotor.items():
        for parameter_schluessel, parameter_datensatz in rotor_daten["params"].items():
            parameter_wert = parameter_datensatz.get("value")

            if isinstance(parameter_wert, (int, float)):
                numerische_werte_pro_parameter[parameter_schluessel].append(float(parameter_wert))

    stats: dict[tuple[str, str], tuple[float, float]] = {}

    for parameter_schluessel, werte_liste in numerische_werte_pro_parameter.items():
        if len(werte_liste) >= 2:
            stats[parameter_schluessel] = (min(werte_liste), max(werte_liste))
        elif len(werte_liste) == 1:
            stats[parameter_schluessel] = (werte_liste[0], werte_liste[0])

    return stats
