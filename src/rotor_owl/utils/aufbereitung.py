from __future__ import annotations

import re


def local_name(uri: str) -> str:
    """
    Extrahiert den lokalen Namen einer URI.
    Beispiel:
      http://.../ims#Rotor_D001  ->  Rotor_D001
    """
    if "#" in uri:
        return uri.split("#", 1)[1]
    return uri.rsplit("/", 1)[-1]


def strip_last_suffix(name: str) -> str:
    """
    Entfernt den letzten '_SUFFIX' Teil.
    Beispiel:
      C_WELLE_D001        -> C_WELLE
      P_WELLE_TIR_D001    -> P_WELLE_TIR
      Rotor_D001          -> Rotor
    """
    if "_" not in name:
        return name
    return name.rsplit("_", 1)[0]


def normalize_param_name(parameter_name: str) -> str:
    """
    Normalisiert Parameter-Namen auf ihre semantische Basis.

    Regeln (in dieser Reihenfolge):
    1) Entferne explizite Instanz-Suffixe wie:
       - _D001, _D002, etc. (generierte Design-IDs)
       - _YYYY-MM-DD_1 (Datums-basierte Instanzen)
    2) Intelligenter Fallback:
       - Entferne letztes Segment NUR wenn es instanz-artig aussieht
       - Instanz-artig = sehr kurz (≤4 Zeichen) UND (nur Großbuchstaben ODER nur Ziffern)

    Beispiele:
      P_WELLE_TIR_D001           -> P_WELLE_TIR
      P_WELLE_TIR_2025-11-30_1   -> P_WELLE_TIR
      P_AKTIV_LAENGE_XYZ         -> P_AKTIV_LAENGE  (XYZ sieht aus wie Instanz)
      P_LUEFTER_DURCHMESSER      -> P_LUEFTER_DURCHMESSER  (bleibt unverändert)
      P_WELLE_1                  -> P_WELLE  (Zahl-Suffix wird entfernt)
    """
    original_name = parameter_name

    # Explizite bekannte Suffixe entfernen
    parameter_name = re.sub(r"_D\d+$", "", parameter_name)
    parameter_name = re.sub(r"_\d{4}-\d{2}-\d{2}_\d+$", "", parameter_name)

    # Wenn sich der Name geändert hat → fertig
    if parameter_name != original_name:
        return parameter_name

    # Fallback: nur "instanz-artige" Suffixe entfernen
    if "_" in parameter_name:
        parts = parameter_name.rsplit("_", 1)
        letztes_segment = parts[1]
        ist_instanz_suffix = len(letztes_segment) <= 4 and (
            letztes_segment.isupper() or letztes_segment.isdigit()
        )

        if ist_instanz_suffix:
            return parts[0]

    return parameter_name


def safe_float(wert_text: str | None) -> float | None:
    """
    Konvertiert Text sicher zu float, wenn möglich.

    Typische Fälle:
    - "458.813" -> 458.813
    - "344.1712e0" -> 344.1712
    - "nan" / "" / None -> None

    Wichtig:
    - Wenn es NICHT als Zahl interpretierbar ist, kommt None zurück
      (dann wird es später als kategorisch / String behandelt)
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
