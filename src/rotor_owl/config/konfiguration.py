"""
Zentrale Konfiguration für das Projekt.
Hier werden Konstanten gesammelt, damit sie nicht in jedem Modul doppelt stehen.
"""

# Ontologie-Namespace
IMS_NAMESPACE = "http://ontology.innomotics.net/ims#"

# Fuseki SPARQL Endpoint
# FUSEKI_ENDPOINT_STANDARD = "http://fuseki:3030/rotors/sparql"  # Docker
FUSEKI_ENDPOINT_STANDARD = "http://localhost:3030/rotors/sparql"  # Lokal

# Komponenten-Mapping: Component_ID -> deutscher Label
KOMPONENTEN_MAPPING = {
    "C_WELLE": "Welle",
    "C_WELLEENDE": "Wellenende",
    "C_AKTIVTEIL": "Aktivteil",
    "C_LUEFTER": "Lüfter",
    "C_BLECHPAKET": "Blechpaket",
    "C_WUCHTSCHEIBEN": "Wuchtscheiben",
    "C_ROTOR": "Rotor",
    "C_LAGER": "Lager",
    "C_SONSTIGE": "Sonstige",
}

# Komponenten-Keywords für Text-Matching (lowercase!)
KOMPONENTEN_KEYWORDS = {
    "welle": "C_WELLE",
    "wellenende": "C_WELLEENDE",
    "aktivteil": "C_AKTIVTEIL",
    "lüfter": "C_LUEFTER",
    "blechpaket": "C_BLECHPAKET",
    "wuchtscheiben": "C_WUCHTSCHEIBEN",
    "rotor": "C_ROTOR",
    "lager": "C_LAGER",
    "sonstige": "C_SONSTIGE",
}
