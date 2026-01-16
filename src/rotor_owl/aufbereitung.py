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

    Achtung:
    - Diese Funktion ist "generisch" und entfernt immer nur das letzte _... Stück.
    """
    if "_" not in name:
        return name
    return name.rsplit("_", 1)[0]


def normalize_param_name(parameter_name: str) -> str:
    """
    Normalisiert Parameter-IDs aus BEIDEN Namensschemata:

    1) Generierte Designs:
        P_WELLE_TIR_D001  -> P_WELLE_TIR

    2) Original Ontologie:
        P_WELLE_TIR_2025-11-30_1  -> P_WELLE_TIR

    3) Fallback:
        Entfernt den letzten '_SUFFIX'

    4) Diese Funktion kann entfernt werden, wenn Rotor_1 aus der Ontologie verschwindet.
    """
    parameter_name = re.sub(r"_D\d+$", "", parameter_name)  # entfernt _D001
    parameter_name = re.sub(
        r"_\d{4}-\d{2}-\d{2}_\d+$", "", parameter_name
    )  # entfernt _YYYY-MM-DD_1
    parameter_name = re.sub(r"_[^_]+$", "", parameter_name)  # fallback: entfernt letztes _...
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
