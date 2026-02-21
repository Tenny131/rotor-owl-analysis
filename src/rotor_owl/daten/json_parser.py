"""
JSON-Parser für reale WVSC-Rotordaten – CSV-gesteuert.

Liest die handgepflegte Datei ``parameter_auswahl.csv``, die festlegt
welche JSON-Pfade extrahiert werden.  Für jeden Rotor wird das JSON
rekursiv geflattened (gleiche Logik wie datenanalyse_realdaten.py) und
anschließend nur die in der CSV definierten Pfade übernommen.

Das Rückgabeformat ist identisch mit dem bisherigen:

    features_by_rotor[rotor_id]["params"][(component, param)] = {
        "value": ...,
        "unit":  ...,
        "ptype": ...,
    }
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Standardpfade
# ---------------------------------------------------------------------------
WVSC_STANDARD_VERZEICHNIS = Path(__file__).resolve().parents[3] / "data" / "reference" / "wvsc"
PARAMETER_CSV = Path(__file__).resolve().parents[3] / "data" / "reference" / "parameter_auswahl.csv"

# ---------------------------------------------------------------------------
# Fachkategorie → ptype-Mapping (wie im bisherigen Parser)
# ---------------------------------------------------------------------------
_PTYPE_MAP: dict[str, str] = {
    "Dynamik": "DYN",
    "Struktur": "STRUCT",
    "Geometrie": "GEOM",
    "Material/Masse": "MTRL",
    "Fertigung": "MFG",
    "Anforderung": "REQ",
    "Elektrisch": "ELEC",
}

# ---------------------------------------------------------------------------
# Skip-Patterns beim Flatten (identisch mit datenanalyse_realdaten.py)
# ---------------------------------------------------------------------------
_SKIP_PATTERNS = [
    "module_info",
    "OrderData",
    "additional_data",
    "tag",
    "machine_template",
    "username",
    "created",
    "last_updated",
    "status_code",
    "versions.",
    "calculation.identifier",
    "calculation.remarks",
    "calculation.user",
    "calculation.time",
    "info.pid",
    "info.tra_suffix",
    "info.mlfb",
    "info.tra",
]


def _sollte_uebersprungen_werden(pfad: str) -> bool:
    for pattern in _SKIP_PATTERNS:
        if pattern in pfad:
            return True
    return False


# ===================================================================
# CSV-Parameterdefinition laden
# ===================================================================


def _lade_parameter_csv(
    csv_pfad: Path | str | None = None,
) -> dict[str, dict]:
    """
    Liest ``parameter_auswahl.csv`` und gibt ein Dict zurück:

        json_pfad → {"component", "parameter", "unit", "ptype", "datentyp"}

    Für *Nicht zugeordnet*-Zeilen (``P_Name == "–"``) wird:

    - ``component`` = ``"C_SONSTIGE"``
    - ``parameter`` = aus dem JSON-Pfad abgeleitet
    """
    pfad = Path(csv_pfad) if csv_pfad else PARAMETER_CSV
    df = pd.read_csv(pfad, sep=",", encoding="utf-8-sig")

    mapping: dict[str, dict] = {}

    for _, row in df.iterrows():
        json_pfad = str(row["Parameter"]).strip()
        p_name = str(row["P_Name"]).strip()
        komponente = str(row["Komponente"]).strip()
        einheit = str(row.get("Einheit_", "–")).strip()
        fachkat = str(row.get("Fachkategorie", "–")).strip()
        datentyp = str(row.get("Datentyp", "numerisch")).strip()

        # Placeholder für nicht-zugeordnete Parameter
        if komponente in ("–", "nan", ""):
            komponente = "C_SONSTIGE"
        if p_name in ("–", "nan", ""):
            p_name = _erzeuge_parameter_id(json_pfad)

        ptype = _PTYPE_MAP.get(fachkat, "MISC")

        mapping[json_pfad] = {
            "component": komponente,
            "parameter": p_name,
            "unit": einheit if einheit not in ("–", "nan") else "",
            "ptype": ptype,
            "datentyp": datentyp,
        }

    return mapping


def _erzeuge_parameter_id(json_pfad: str) -> str:
    """
    Erzeugt eine lesbare Parameter-ID aus einem JSON-Pfad.

    Beispiel::

        RotorWorkflow.output.output.roller_bearings.output.NDE.forces.radial
        → P_NDE_FORCES_RADIAL
    """
    teile = json_pfad.replace(".", "_").split("_")
    skip = {
        "rotorworkflow",
        "edimworkflow",
        "simocalcworkflow",
        "output",
        "input",
        "results",
    }
    relevant = [t for t in teile if t.lower() not in skip and t != ""]
    # Maximal letzte 5 Segmente
    relevant = relevant[-5:] if len(relevant) > 5 else relevant
    name = "_".join(relevant).upper()
    while "__" in name:
        name = name.replace("__", "_")
    name = name.strip("_")
    return f"P_{name}" if name else f"P_{json_pfad.replace('.', '_').upper()}"


# ===================================================================
# JSON-Flatten – gleiche Logik wie datenanalyse_realdaten.py
# ===================================================================


def _flatten_json(obj: dict | list, prefix: str = "") -> dict[str, object]:
    """Flattened ein verschachteltes JSON-Objekt rekursiv."""
    ergebnis: dict[str, object] = {}

    if isinstance(obj, dict):
        for key, val in obj.items():
            neuer_pfad = f"{prefix}.{key}" if prefix else key

            if _sollte_uebersprungen_werden(neuer_pfad):
                continue

            if isinstance(val, dict):
                ergebnis.update(_flatten_json(val, neuer_pfad))
            elif isinstance(val, list):
                _verarbeite_liste(val, neuer_pfad, ergebnis)
            elif val is not None:
                ergebnis[neuer_pfad] = val

    return ergebnis


def _verarbeite_liste(lst: list, pfad: str, ergebnis: dict):
    """Verarbeitet JSON-Listen: Aggregation oder Segmentverarbeitung."""
    if not lst:
        return

    if pfad.endswith("rotor.segments"):
        _verarbeite_segmente(lst, ergebnis)
        return

    if "shaft_safety.output" in pfad or "parallelkey_safety.output" in pfad:
        _verarbeite_sicherheitsliste(lst, pfad, ergebnis)
        return

    if "roller_bearings.output.DE" in pfad or "roller_bearings.output.NDE" in pfad:
        _verarbeite_lagerliste(lst, pfad, ergebnis)
        return

    if "relubrication.masses" in pfad:
        return

    if all(isinstance(x, (int, float)) for x in lst):
        if len(lst) > 1:
            arr = np.array([float(x) for x in lst])
            ergebnis[f"{pfad}._min"] = float(np.min(arr))
            ergebnis[f"{pfad}._max"] = float(np.max(arr))
            ergebnis[f"{pfad}._mean"] = float(np.mean(arr))
            ergebnis[f"{pfad}._count"] = len(arr)
        elif len(lst) == 1:
            ergebnis[pfad] = lst[0]
        return

    if all(isinstance(x, dict) for x in lst):
        if len(lst) <= 3:
            for i, item in enumerate(lst):
                sub = _flatten_json(item, f"{pfad}[{i}]")
                ergebnis.update(sub)
        return


def _verarbeite_segmente(segmente: list[dict], ergebnis: dict):
    """Aggregiert Rotor-Segmente nach Typ."""
    typ_gruppen: dict[str, list[dict]] = defaultdict(list)
    for seg in segmente:
        seg_typ = seg.get("type", "unknown")
        typ_gruppen[seg_typ].append(seg)

    ergebnis["segments._total_count"] = len(segmente)
    ergebnis["segments._type_count"] = len(typ_gruppen)

    for typ, gruppe in typ_gruppen.items():
        basis = f"segments.{typ}"
        ergebnis[f"{basis}._count"] = len(gruppe)

        num_felder = [
            "length",
            "outer_diameter",
            "inner_diameter",
            "mass",
            "coupling_mass",
            "polar_inertia",
            "axial_force",
            "diametral_inertia",
        ]
        for feld in num_felder:
            werte = []
            for seg in gruppe:
                v = seg.get(feld)
                if v is not None:
                    try:
                        werte.append(float(v))
                    except (ValueError, TypeError):
                        pass
            if werte:
                if len(werte) == 1:
                    ergebnis[f"{basis}.{feld}"] = werte[0]
                else:
                    ergebnis[f"{basis}.{feld}._sum"] = sum(werte)
                    ergebnis[f"{basis}.{feld}._max"] = max(werte)
                    ergebnis[f"{basis}.{feld}._min"] = min(werte)

        for feld in ["description", "designation"]:
            werte = [seg.get(feld) for seg in gruppe if seg.get(feld)]
            if werte:
                ergebnis[f"{basis}.{feld}"] = (
                    werte[0] if len(werte) == 1 else "; ".join(str(v) for v in werte[:3])
                )

        pk_segmente = [seg for seg in gruppe if "parallel_key" in seg]
        if pk_segmente:
            pk = pk_segmente[0]["parallel_key"]
            for k, v in pk.items():
                if v is not None:
                    ergebnis[f"{basis}.parallel_key.{k}"] = v

        stiff_segmente = [seg for seg in gruppe if "stiffness" in seg]
        if stiff_segmente:
            stiff = stiff_segmente[0]["stiffness"]
            for k, v in stiff.items():
                if v is not None:
                    ergebnis[f"{basis}.stiffness.{k}"] = v

        shoulder_segmente = [seg for seg in gruppe if "shoulder" in seg]
        if shoulder_segmente:
            sh = shoulder_segmente[0].get("shoulder", {})
            for pos, details in sh.items():
                if isinstance(details, dict):
                    for k, v in details.items():
                        if v is not None:
                            ergebnis[f"{basis}.shoulder.{pos}.{k}"] = v

        func_segmente = [seg for seg in gruppe if "function" in seg]
        if func_segmente:
            alle_funcs = []
            for seg in func_segmente:
                funcs = seg.get("function", [])
                alle_funcs.extend(funcs)
            if alle_funcs:
                ergebnis[f"{basis}._functions"] = "; ".join(sorted(set(alle_funcs)))


def _verarbeite_sicherheitsliste(eintraege: list[dict], pfad: str, ergebnis: dict):
    """Aggregiert Shaft-Safety/Parallelkey Ergebnisse."""
    ergebnis[f"{pfad}._count"] = len(eintraege)

    if "shaft_safety" in pfad:
        fatigue_vals = []
        yield_vals = []
        for e in eintraege:
            safety = e.get("safety", {})
            fs = safety.get("fatigue_strength")
            ys = safety.get("yield_strength")
            if isinstance(fs, (int, float)):
                fatigue_vals.append(float(fs))
            if isinstance(ys, (int, float)):
                yield_vals.append(float(ys))
        if fatigue_vals:
            ergebnis[f"{pfad}.fatigue_strength._min"] = min(fatigue_vals)
            ergebnis[f"{pfad}.fatigue_strength._max"] = max(fatigue_vals)
            ergebnis[f"{pfad}.fatigue_strength._mean"] = float(np.mean(fatigue_vals))
        if yield_vals:
            ergebnis[f"{pfad}.yield_strength._min"] = min(yield_vals)
            ergebnis[f"{pfad}.yield_strength._max"] = max(yield_vals)
            ergebnis[f"{pfad}.yield_strength._mean"] = float(np.mean(yield_vals))

    elif "parallelkey_safety" in pfad:
        for e in eintraege:
            for key in ["transmittable_torque", "shaft_diameter", "position"]:
                v = e.get(key)
                if v is not None:
                    ergebnis[f"{pfad}.{key}"] = v
            for sub in ["key", "shaft"]:
                safety_sub = e.get("safety", {}).get(sub, {})
                for k, v in safety_sub.items():
                    if v is not None:
                        ergebnis[f"{pfad}.safety.{sub}.{k}"] = v


def _verarbeite_lagerliste(eintraege: list[dict], pfad: str, ergebnis: dict):
    """Aggregiert Roller-Bearing DE/NDE Ergebnisse (erstes Lager)."""
    if not eintraege:
        return
    erstes = eintraege[0]

    def _flat_bearing(obj, p):
        if isinstance(obj, dict):
            for k, v in obj.items():
                np_ = f"{p}.{k}"
                if isinstance(v, dict):
                    _flat_bearing(v, np_)
                elif isinstance(v, (int, float)):
                    ergebnis[np_] = v
                elif isinstance(v, str):
                    ergebnis[np_] = v
        elif isinstance(obj, (int, float)):
            ergebnis[p] = obj

    _flat_bearing(erstes, pfad)


def _normalisiere_bearing_properties(rotor_params: dict) -> dict:
    """
    Normalisiert bearing_properties: variable Lagernamen → 'bearing_1', 'bearing_2'.
    """
    normalisiert = {}
    lager_keys: list[str] = []
    bp_prefix = "RotorWorkflow.input.bearing_properties."
    meta_keys = {"axial_preload_forces", "temperature_rise", "grease_slinger", "grease", "type"}

    for pfad, wert in rotor_params.items():
        if not pfad.startswith(bp_prefix):
            normalisiert[pfad] = wert
            continue

        rest = pfad[len(bp_prefix) :]
        teile = rest.split(".", 1)
        top_key = teile[0]

        if top_key in meta_keys:
            normalisiert[pfad] = wert
        else:
            if top_key not in lager_keys:
                lager_keys.append(top_key)
            idx = lager_keys.index(top_key) + 1
            if len(teile) > 1:
                neuer_pfad = f"{bp_prefix}bearing_{idx}.{teile[1]}"
            else:
                neuer_pfad = f"{bp_prefix}bearing_{idx}"
            normalisiert[neuer_pfad] = wert

    return normalisiert


# ===================================================================
# Hilfsfunktion: Wert-Konvertierung
# ===================================================================


def _safe_float(wert) -> float | None:
    if wert is None:
        return None
    try:
        return float(wert)
    except (ValueError, TypeError):
        return None


def _konvertiere_wert(wert, datentyp: str):
    """Konvertiert einen Rohwert je nach Datentyp."""
    if wert is None:
        return None
    if datentyp == "kategorisch":
        return str(wert)
    f = _safe_float(wert)
    if f is not None:
        return f
    return str(wert) if isinstance(wert, str) else wert


# ===================================================================
# Kern: einzelne JSON-Datei parsen und gefilterte Parameter extrahieren
# ===================================================================


def _parse_einzelne_json(
    dateipfad: Path,
    param_mapping: dict[str, dict],
) -> tuple[str, dict] | None:
    """
    Liest eine WVSC-JSON-Datei, flattened sie und extrahiert nur die
    Parameter, die in ``param_mapping`` definiert sind.

    :return: (rotor_id, {"params": {(component, param): {"value", "unit", "ptype"}}})
    """
    try:
        with dateipfad.open("r", encoding="utf-8") as datei:
            rohdaten = json.load(datei)
    except (json.JSONDecodeError, OSError) as fehler:
        logger.warning("JSON konnte nicht gelesen werden: %s – %s", dateipfad.name, fehler)
        return None

    rotor_id = rohdaten.get("machine_id")
    if not rotor_id:
        logger.warning("Kein machine_id in %s", dateipfad.name)
        return None

    # JSON rekursiv flatten (gleiche Logik wie datenanalyse_realdaten.py)
    alle_params = _flatten_json(rohdaten)
    alle_params = _normalisiere_bearing_properties(alle_params)

    # Nur die in der CSV definierten Pfade extrahieren
    parameter: dict[tuple[str, str], dict] = {}

    for json_pfad, meta in param_mapping.items():
        wert = alle_params.get(json_pfad)
        if wert is None:
            continue

        endwert = _konvertiere_wert(wert, meta["datentyp"])
        if endwert is None:
            continue

        schluessel = (meta["component"], meta["parameter"])
        parameter[schluessel] = {
            "value": endwert,
            "unit": meta["unit"],
            "ptype": meta["ptype"],
        }

    return rotor_id, {"params": parameter}


# ===================================================================
# Öffentliche API
# ===================================================================


def fetch_all_features_from_json(
    verzeichnis: Path | str | None = None,
    csv_pfad: Path | str | None = None,
) -> dict[str, dict]:
    """
    Liest alle WVSC-JSON-Dateien und gibt ``features_by_rotor`` zurück.

    Die zu extrahierenden Parameter werden aus ``parameter_auswahl.csv``
    gelesen – keine hardcodierten Mappings.

    :param verzeichnis: Pfad zum JSON-Verzeichnis (Standard: data/reference/wvsc)
    :param csv_pfad: Pfad zur Parameter-CSV (Standard: data/reference/parameter_auswahl.csv)
    :return: features_by_rotor[rotor_id]["params"][(component, param)] = {"value", "unit", "ptype"}
    """
    if verzeichnis is None:
        json_verzeichnis = WVSC_STANDARD_VERZEICHNIS
    else:
        json_verzeichnis = Path(verzeichnis)

    if not json_verzeichnis.is_dir():
        logger.error("WVSC-Verzeichnis existiert nicht: %s", json_verzeichnis)
        return {}

    # Parameter-Mapping aus CSV laden
    param_mapping = _lade_parameter_csv(csv_pfad)
    logger.info("%d Parameter aus CSV geladen", len(param_mapping))

    json_dateien = sorted(json_verzeichnis.glob("*.json"))
    if not json_dateien:
        logger.warning("Keine JSON-Dateien in %s gefunden", json_verzeichnis)
        return {}

    features_by_rotor: dict[str, dict] = {}
    fehlerhafte = 0

    for dateipfad in json_dateien:
        ergebnis = _parse_einzelne_json(dateipfad, param_mapping)
        if ergebnis is None:
            fehlerhafte += 1
            continue
        rotor_id, rotor_daten = ergebnis
        features_by_rotor[rotor_id] = rotor_daten

    logger.info(
        "%d Rotoren geladen (%d fehlerhaft), %d Parameter pro Rotor (max)",
        len(features_by_rotor),
        fehlerhafte,
        len(param_mapping),
    )

    return features_by_rotor
