from __future__ import annotations

import requests


def run_sparql(endpoint_url: str, sparql_query: str) -> dict:
    """
    Führt eine SPARQL-Query gegen Fuseki aus und gibt JSON im SPARQL-Result-Format zurück.

    Warum extra Funktion?
    - Damit die Netzwerkkommunikation gekapselt ist
    - Später kann man hier Logging, Retry-Mechanismen oder Auth ergänzen
    """
    http_header = {
        "Accept": "application/sparql-results+json",
        "Content-Type": "application/sparql-query",
    }

    antwort = requests.post(
        endpoint_url,
        data=sparql_query.encode("utf-8"),
        headers=http_header,
        timeout=60,
    )
    antwort.raise_for_status()
    return antwort.json()
