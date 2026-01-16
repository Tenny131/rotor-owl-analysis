from __future__ import annotations

import streamlit as st

from rotor_owl.konfiguration import IMS_NAMESPACE
from rotor_owl.sparql_helpers import run_sparql
from rotor_owl.aufbereitung import local_name, strip_last_suffix, normalize_param_name, safe_float


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
