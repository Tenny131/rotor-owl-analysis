"""
Hilfsfunktionen f\u00fcr Ontology.py
Extrahiert wiederverwendbare Logik f\u00fcr bessere Wartbarkeit.
"""

from __future__ import annotations

from rotor_owl.config.konfiguration import KOMPONENTEN_KEYWORDS


def komponente_aus_text_erkennen(text: str, instanz_nummer: int) -> tuple[str | None, bool]:
    """
    Erkennt Komponenten-Typ aus Freitext und gibt Component_ID zur\u00fcck.

    Args:
        text: Freitext (z.B. "Welle", "Aktivteil", etc.)
        instanz_nummer: Nummer f\u00fcr Instanz-Suffix

    Returns:
        Tuple aus (component_id, ist_bekannt)
        - component_id: z.B. "C_WELLE_1" oder None
        - ist_bekannt: True wenn Komponente erkannt wurde

    Beispiel:
        komponente_aus_text_erkennen("Welle", 1) -> ("C_WELLE_1", True)
        komponente_aus_text_erkennen("Unbekannt", 1) -> (None, False)
    """
    text_lower = text.lower().strip()

    for keyword, component_id in KOMPONENTEN_KEYWORDS.items():
        if keyword in text_lower:
            return f"{component_id}_{instanz_nummer}", True

    return None, False
