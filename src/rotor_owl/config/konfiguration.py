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
    "C_AKTIVTEIL": "Aktivteil",
    "C_LUEFTER": "Lüfter",
    "C_BLECHPAKET": "Blechpaket",
    "C_WUCHTSCHEIBEN": "Wuchtscheiben",
}

# Komponenten-Keywords für Text-Matching (lowercase!)
KOMPONENTEN_KEYWORDS = {
    "welle": "C_WELLE",
    "aktivteil": "C_AKTIVTEIL",
    "lüfter": "C_LUEFTER",
    "blechpaket": "C_BLECHPAKET",
    "wuchtscheiben": "C_WUCHTSCHEIBEN",
}
