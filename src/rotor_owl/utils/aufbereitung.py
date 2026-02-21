from __future__ import annotations

import re


def local_name(uri: str) -> str:
    """Extrahiert lokalen Namen aus URI (http://.../ims#Rotor_D001 -> Rotor_D001).

    Args:
        uri: Vollständige URI

    Returns:
        Lokaler Name (Fragment nach # oder letztes Segment)
    """
    if "#" in uri:
        return uri.split("#", 1)[1]
    return uri.rsplit("/", 1)[-1]


def strip_last_suffix(name: str) -> str:
    """Entfernt letztes _SUFFIX Segment (C_WELLE_D001 -> C_WELLE).

    Args:
        name: Name mit Suffix

    Returns:
        Name ohne letztes Suffix
    """
    if "_" not in name:
        return name
    return name.rsplit("_", 1)[0]


def normalize_param_name(parameter_name: str) -> str:
    """Normalisiert Parameter-Namen durch Entfernen von Instanz-Suffixen (_D001, _YYYY-MM-DD_1, _1QA1452-8JA60-0HG2-Z).

    Args:
        parameter_name: Parameter-Name mit möglichem Instanz-Suffix

    Returns:
        Normalisierter Parameter-Name
    """
    original_name = parameter_name

    # Explizite bekannte Suffixe entfernen
    parameter_name = re.sub(r"_D\d+$", "", parameter_name)
    parameter_name = re.sub(r"_\d{4}-\d{2}-\d{2}_\d+$", "", parameter_name)

    # Wenn sich der Name geändert hat → fertig
    if parameter_name != original_name:
        return parameter_name

    # Fallback: letztes _SUFFIX prüfen
    if "_" in parameter_name:
        parts = parameter_name.rsplit("_", 1)
        letztes_segment = parts[1]

        # Realdaten-Design-IDs enthalten Hyphens (z.B. 1QA1452-8JA60-0HG2-Z).
        # Normale Parameter-Segmente enthalten nie Hyphens.
        if "-" in letztes_segment:
            return parts[0]

        # Kurze instanz-artige Suffixe (z.B. _1, _D001)
        ist_instanz_suffix = len(letztes_segment) <= 4 and (
            letztes_segment.isupper() or letztes_segment.isdigit()
        )

        if ist_instanz_suffix:
            return parts[0]

    return parameter_name


def safe_float(wert_text: str | None) -> float | None:
    """Konvertiert Text zu float, gibt None zurück wenn nicht parsebar.

    Args:
        wert_text: Text-Wert (unterstützt Komma, Quotes, NaN)

    Returns:
        float oder None bei Fehler/nan/empty
    """
    if wert_text is None:
        return None

    wert_string = str(wert_text).strip().strip('"')
    if wert_string == "" or wert_string.lower() == "nan":
        return None

    try:
        return float(wert_string.replace(",", "."))
    except Exception:
        return None
