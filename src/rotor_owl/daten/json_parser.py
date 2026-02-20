"""
JSON-Parser für reale WVSC-Rotordaten.

Liest die JSON-Dateien aus dem WVSC-Verzeichnis und konvertiert sie
in das gleiche Format wie fetch_all_features() aus feature_fetcher.py,
sodass alle nachgelagerten Algorithmen nahtlos funktionieren.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Standardpfad zum Verzeichnis mit den realen JSON-Dateien
# ---------------------------------------------------------------------------
WVSC_STANDARD_VERZEICHNIS = Path(__file__).resolve().parents[3] / "data" / "real_data" / "wvsc"


# ---------------------------------------------------------------------------
# Datenklasse für Parameter-Mapping: JSON-Pfad → Feature-Schlüssel
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ParameterMapping:
    """
    Beschreibt die Zuordnung eines JSON-Felds zu einem Feature-Schlüssel.

    :param component: Komponenten-ID (z.B. "C_WELLE")
    :type component: str
    :param parameter: Parameter-ID (z.B. "P_WELLE_LAENGE")
    :type parameter: str
    :param unit: Physikalische Einheit
    :type unit: str
    :param ptype: Parametertyp für Kategorie-Zuordnung (GEOM, STRUCT, DYN, MTRL, MFG, REQ, ELEC)
    :type ptype: str
    """

    component: str
    parameter: str
    unit: str
    ptype: str


# ---------------------------------------------------------------------------
# Zentrale Mapping-Tabelle: alle ausgewählten Parameter
# ---------------------------------------------------------------------------

# Welle (Shaft)
MAPPING_WELLE_LAENGE = ParameterMapping("C_WELLE", "P_WELLE_LAENGE", "mm", "GEOM")
MAPPING_WELLE_MASSE = ParameterMapping("C_WELLE", "P_WELLE_MASSE", "kg", "MTRL")
MAPPING_WELLE_MATERIAL = ParameterMapping("C_WELLE", "P_WELLE_MATERIAL", "–", "MTRL")
MAPPING_WELLE_DREHZAHL = ParameterMapping("C_WELLE", "P_WELLE_DREHZAHLBEREICH", "1/min", "DYN")
MAPPING_WELLE_TORSIONSSTEIFIGKEIT = ParameterMapping(
    "C_WELLE", "P_WELLE_TORSIONSSTEIFIGKEIT", "Nm/rad", "STRUCT"
)
MAPPING_WELLE_KUPPLUNGSMASSE = ParameterMapping("C_WELLE", "P_WELLE_KUPPLUNGSMASSE", "kg", "MTRL")
MAPPING_WELLE_VOLUMEN = ParameterMapping("C_WELLE", "P_WELLE_VOLUMEN", "m³", "GEOM")
MAPPING_WELLE_ZERSPANUNGSRATE = ParameterMapping("C_WELLE", "P_WELLE_ZERSPANUNGSRATE", "%", "MFG")
MAPPING_WELLE_TRAEGHEITSMOMENT = ParameterMapping(
    "C_WELLE", "P_WELLE_TRAEGHEITSMOMENT", "kg·m²", "DYN"
)
MAPPING_WELLE_MIN_SICHERHEIT = ParameterMapping("C_WELLE", "P_WELLE_MIN_SICHERHEIT", "–", "STRUCT")
MAPPING_WELLE_PK_DREHMOMENT = ParameterMapping("C_WELLE", "P_WELLE_PK_DREHMOMENT", "Nm", "STRUCT")

# Aktivteil (Laminated Core)
MAPPING_AKTIV_LAENGE = ParameterMapping("C_AKTIVTEIL", "P_AKTIV_LAENGE", "mm", "GEOM")
MAPPING_AKTIV_D_AUSSEN = ParameterMapping("C_AKTIVTEIL", "P_AKTIV_D_AUSSEN", "mm", "GEOM")
MAPPING_AKTIV_D_INNEN = ParameterMapping("C_AKTIVTEIL", "P_AKTIV_D_INNEN", "mm", "GEOM")
MAPPING_AKTIV_MASSE = ParameterMapping("C_AKTIVTEIL", "P_AKTIV_MASSE", "kg", "MTRL")
MAPPING_AKTIV_MAGNETFEDER = ParameterMapping("C_AKTIVTEIL", "P_AKTIV_MAGNETFEDER", "N/mm", "STRUCT")
MAPPING_AKTIV_AXIALKRAFT = ParameterMapping("C_AKTIVTEIL", "P_AKTIV_AXIALKRAFT_MAG", "N", "STRUCT")

# Lüfter (Fan)
MAPPING_LUEFTER_MASSE = ParameterMapping("C_LUEFTER", "P_LUEFTER_GEWICHT", "kg", "MTRL")
MAPPING_LUEFTER_D = ParameterMapping("C_LUEFTER", "P_LUEFTER_D", "mm", "GEOM")
MAPPING_LUEFTER_J = ParameterMapping("C_LUEFTER", "P_LUEFTER_J", "kg·mm²", "DYN")
MAPPING_LUEFTER_ANZAHL = ParameterMapping("C_LUEFTER", "P_LUEFTER_ANZAHL", "–", "GEOM")

# Rotor (Gesamtsystem)
MAPPING_ROTOR_GESAMTMASSE = ParameterMapping("C_ROTOR", "P_ROTOR_GESAMTMASSE", "kg", "MTRL")
MAPPING_ROTOR_NENNMOMENT = ParameterMapping("C_ROTOR", "P_ROTOR_NENNMOMENT", "Nm", "DYN")
MAPPING_ROTOR_AXIALLAST = ParameterMapping("C_ROTOR", "P_ROTOR_AXIALLAST", "N", "STRUCT")
MAPPING_ROTOR_MAX_TORSION = ParameterMapping("C_ROTOR", "P_ROTOR_MAX_TORSION", "Nm", "DYN")
MAPPING_ROTOR_POLZAHL = ParameterMapping("C_ROTOR", "P_ROTOR_POLZAHL", "–", "ELEC")
MAPPING_ROTOR_LAGERTYP = ParameterMapping("C_ROTOR", "P_ROTOR_LAGERTYP", "–", "REQ")
MAPPING_ROTOR_BAUFORM = ParameterMapping("C_ROTOR", "P_ROTOR_BAUFORM", "–", "REQ")
MAPPING_ROTOR_C_MASS = ParameterMapping("C_ROTOR", "P_ROTOR_C_MASS", "mm", "GEOM")

# Rotordynamik (Output)
MAPPING_ROTOR_EIGENFREQ_1 = ParameterMapping("C_ROTOR", "P_ROTOR_EIGENFREQ_1", "Hz", "DYN")
MAPPING_ROTOR_EIGENFREQ_2 = ParameterMapping("C_ROTOR", "P_ROTOR_EIGENFREQ_2", "Hz", "DYN")
MAPPING_ROTOR_MAX_BIEGUNG = ParameterMapping("C_ROTOR", "P_ROTOR_MAX_BIEGUNG", "mm", "STRUCT")
MAPPING_ROTOR_BIEGUNG_KERN = ParameterMapping("C_ROTOR", "P_ROTOR_BIEGUNG_KERN", "mm", "STRUCT")
MAPPING_ROTOR_LAGERABSTAND = ParameterMapping("C_ROTOR", "P_ROTOR_LAGERABSTAND", "mm", "GEOM")
MAPPING_ROTOR_KERNMITTE = ParameterMapping("C_ROTOR", "P_ROTOR_KERNMITTE", "mm", "GEOM")

# Lager (Bearing)
MAPPING_LAGER_BEZEICHNUNG = ParameterMapping("C_LAGER", "P_LAGER_BEZEICHNUNG", "–", "GEOM")
MAPPING_LAGER_BAUREIHE = ParameterMapping("C_LAGER", "P_LAGER_BAUREIHE", "–", "GEOM")
MAPPING_LAGER_TYP_DETAIL = ParameterMapping("C_LAGER", "P_LAGER_TYP_DETAIL", "–", "REQ")
MAPPING_LAGER_INNER_D = ParameterMapping("C_LAGER", "P_LAGER_INNER_D", "mm", "GEOM")
MAPPING_LAGER_OUTER_D = ParameterMapping("C_LAGER", "P_LAGER_OUTER_D", "mm", "GEOM")
MAPPING_LAGER_DYN_TRAGZAHL = ParameterMapping("C_LAGER", "P_LAGER_DYN_TRAGZAHL", "N", "STRUCT")
MAPPING_LAGER_STAT_TRAGZAHL = ParameterMapping("C_LAGER", "P_LAGER_STAT_TRAGZAHL", "N", "STRUCT")
MAPPING_LAGER_LEBENSDAUER = ParameterMapping("C_LAGER", "P_LAGER_LEBENSDAUER", "h", "REQ")

# MLFB-Baureihe
MAPPING_ROTOR_MLFB = ParameterMapping("C_ROTOR", "P_ROTOR_MLFB_BAUREIHE", "–", "REQ")


# ---------------------------------------------------------------------------
# Hilfsfunktionen für sicheren Zugriff auf verschachtelte JSON-Strukturen
# ---------------------------------------------------------------------------


def _sicherer_zugriff(daten: dict, *schluessel, fallback=None):
    """
    Navigiert sicher durch verschachtelte Dicts.

    :param daten: Verschachteltes Dictionary
    :type daten: dict
    :param schluessel: Schlüssel-Kette zum Zielwert
    :type schluessel: str
    :param fallback: Rückgabewert wenn Pfad nicht existiert
    :return: Gefundener Wert oder fallback
    """
    aktuell = daten
    for key in schluessel:
        if not isinstance(aktuell, dict):
            return fallback
        aktuell = aktuell.get(key, fallback)
        if aktuell is fallback:
            return fallback
    return aktuell


def _safe_float(wert) -> float | None:
    """
    Konvertiert einen Wert sicher zu float.

    :param wert: Beliebiger Wert
    :return: float oder None bei Fehlschlag
    :rtype: float | None
    """
    if wert is None:
        return None
    try:
        return float(wert)
    except (ValueError, TypeError):
        return None


def _parameter_eintrag(wert, mapping: ParameterMapping, erzwinge_string: bool = False) -> dict:
    """
    Erzeugt einen einzelnen Parameter-Eintrag im Feature-Fetcher-Format.

    :param wert: Parameterwert (numerisch oder kategorisch)
    :param mapping: Zugehöriges ParameterMapping
    :type mapping: ParameterMapping
    :param erzwinge_string: Wenn True, wird der Wert immer als String behandelt (für kategorische Parameter)
    :type erzwinge_string: bool
    :return: Dict mit value, unit, ptype
    :rtype: dict
    """
    if erzwinge_string:
        endwert = str(wert) if wert is not None else None
    elif isinstance(wert, (int, float)):
        endwert = float(wert)
    elif isinstance(wert, str):
        numerisch = _safe_float(wert)
        endwert = numerisch if numerisch is not None else wert
    else:
        endwert = None

    return {
        "value": endwert,
        "unit": mapping.unit,
        "ptype": mapping.ptype,
    }


# ---------------------------------------------------------------------------
# Segment-Aggregation: mehrere Segmente eines Typs zusammenfassen
# ---------------------------------------------------------------------------


def _segmente_nach_typ(segmente: list[dict]) -> dict[str, list[dict]]:
    """
    Gruppiert Rotor-Segmente nach ihrem Typ-Feld.

    :param segmente: Liste aller Segment-Dicts aus dem JSON
    :type segmente: list[dict]
    :return: Dict von Typ → Liste der Segmente
    :rtype: dict[str, list[dict]]
    """
    gruppiert: dict[str, list[dict]] = {}
    for segment in segmente:
        segment_typ = segment.get("type", "unknown")
        if segment_typ not in gruppiert:
            gruppiert[segment_typ] = []
        gruppiert[segment_typ].append(segment)
    return gruppiert


def _erstes_lager_aus_properties(bearing_properties: dict) -> dict | None:
    """
    Extrahiert das erste benannte Lager aus bearing_properties.

    :param bearing_properties: bearing_properties-Dict aus dem JSON
    :type bearing_properties: dict
    :return: Lager-Dict mit designation, type etc. oder None
    :rtype: dict | None
    """
    for _, wert in bearing_properties.items():
        if isinstance(wert, dict) and "designation" in wert:
            return wert
    return None


# ---------------------------------------------------------------------------
# Kernfunktion: einzelnes JSON → Parameter-Dict
# ---------------------------------------------------------------------------


def _parse_einzelne_json(dateipfad: Path) -> tuple[str, dict] | None:
    """
    Liest eine einzelne WVSC-JSON-Datei und extrahiert alle Parameter.

    :param dateipfad: Pfad zur JSON-Datei
    :type dateipfad: Path
    :return: Tuple (rotor_id, {"params": {...}}) oder None bei Fehler
    :rtype: tuple[str, dict] | None
    """
    try:
        with dateipfad.open("r", encoding="utf-8") as datei:
            rohdaten = json.load(datei)
    except (json.JSONDecodeError, OSError) as fehler:
        logger.warning("JSON-Datei konnte nicht gelesen werden: %s – %s", dateipfad.name, fehler)
        return None

    rotor_id = rohdaten.get("machine_id")
    if not rotor_id:
        logger.warning("Kein machine_id in %s", dateipfad.name)
        return None

    rotor_workflow = rohdaten.get("RotorWorkflow", {})
    eingabe = rotor_workflow.get("input", {})
    ausgabe = rotor_workflow.get("output", {})

    parameter: dict[tuple[str, str], dict] = {}

    # ------------------------------------------------------------------
    # INPUT-Parameter
    # ------------------------------------------------------------------

    betriebsdaten = eingabe.get("operational_data", {})
    last = eingabe.get("load", {})
    materialien = eingabe.get("materials", {})
    bearing_properties = eingabe.get("bearing_properties", {})
    segmente = eingabe.get("rotor", {}).get("segments", [])
    segmente_gruppiert = _segmente_nach_typ(segmente)

    # Welle – Betriebsdaten
    parameter[(MAPPING_WELLE_DREHZAHL.component, MAPPING_WELLE_DREHZAHL.parameter)] = (
        _parameter_eintrag(betriebsdaten.get("operational_speed"), MAPPING_WELLE_DREHZAHL)
    )
    parameter[(MAPPING_WELLE_MATERIAL.component, MAPPING_WELLE_MATERIAL.parameter)] = (
        _parameter_eintrag(
            _sicherer_zugriff(materialien, "shaft_material", "name"),
            MAPPING_WELLE_MATERIAL,
        )
    )

    # Rotor – Lastdaten
    parameter[(MAPPING_ROTOR_NENNMOMENT.component, MAPPING_ROTOR_NENNMOMENT.parameter)] = (
        _parameter_eintrag(last.get("nominal_torque"), MAPPING_ROTOR_NENNMOMENT)
    )
    parameter[(MAPPING_ROTOR_MAX_TORSION.component, MAPPING_ROTOR_MAX_TORSION.parameter)] = (
        _parameter_eintrag(
            _sicherer_zugriff(last, "torsion", "maximum"),
            MAPPING_ROTOR_MAX_TORSION,
        )
    )
    parameter[(MAPPING_ROTOR_AXIALLAST.component, MAPPING_ROTOR_AXIALLAST.parameter)] = (
        _parameter_eintrag(last.get("axial_load"), MAPPING_ROTOR_AXIALLAST)
    )

    # Rotor – kategorische Eingabedaten
    parameter[(MAPPING_ROTOR_POLZAHL.component, MAPPING_ROTOR_POLZAHL.parameter)] = (
        _parameter_eintrag(eingabe.get("pole_number"), MAPPING_ROTOR_POLZAHL)
    )
    parameter[(MAPPING_ROTOR_LAGERTYP.component, MAPPING_ROTOR_LAGERTYP.parameter)] = (
        _parameter_eintrag(eingabe.get("bearing_type"), MAPPING_ROTOR_LAGERTYP)
    )
    parameter[(MAPPING_ROTOR_BAUFORM.component, MAPPING_ROTOR_BAUFORM.parameter)] = (
        _parameter_eintrag(betriebsdaten.get("construction_type"), MAPPING_ROTOR_BAUFORM)
    )
    parameter[(MAPPING_ROTOR_C_MASS.component, MAPPING_ROTOR_C_MASS.parameter)] = (
        _parameter_eintrag(eingabe.get("c_dimension"), MAPPING_ROTOR_C_MASS)
    )

    # MLFB-Baureihe (erste 7 Zeichen der machine_id)
    mlfb_baureihe = rotor_id[:7] if len(rotor_id) >= 7 else rotor_id
    parameter[(MAPPING_ROTOR_MLFB.component, MAPPING_ROTOR_MLFB.parameter)] = _parameter_eintrag(
        mlfb_baureihe, MAPPING_ROTOR_MLFB
    )

    # ------------------------------------------------------------------
    # Segment-basierte Parameter
    # ------------------------------------------------------------------

    # Aktivteil (Laminated Core) – genau 1 pro Rotor
    kernpakete = segmente_gruppiert.get("laminated_core", [])
    if kernpakete:
        kern = kernpakete[0]
        parameter[(MAPPING_AKTIV_LAENGE.component, MAPPING_AKTIV_LAENGE.parameter)] = (
            _parameter_eintrag(kern.get("length"), MAPPING_AKTIV_LAENGE)
        )
        parameter[(MAPPING_AKTIV_D_AUSSEN.component, MAPPING_AKTIV_D_AUSSEN.parameter)] = (
            _parameter_eintrag(kern.get("outer_diameter"), MAPPING_AKTIV_D_AUSSEN)
        )
        parameter[(MAPPING_AKTIV_D_INNEN.component, MAPPING_AKTIV_D_INNEN.parameter)] = (
            _parameter_eintrag(kern.get("inner_diameter"), MAPPING_AKTIV_D_INNEN)
        )
        parameter[(MAPPING_AKTIV_MASSE.component, MAPPING_AKTIV_MASSE.parameter)] = (
            _parameter_eintrag(kern.get("mass"), MAPPING_AKTIV_MASSE)
        )
        parameter[(MAPPING_AKTIV_MAGNETFEDER.component, MAPPING_AKTIV_MAGNETFEDER.parameter)] = (
            _parameter_eintrag(kern.get("magnetic_spring"), MAPPING_AKTIV_MAGNETFEDER)
        )
        parameter[(MAPPING_AKTIV_AXIALKRAFT.component, MAPPING_AKTIV_AXIALKRAFT.parameter)] = (
            _parameter_eintrag(kern.get("axial_magnetic_force"), MAPPING_AKTIV_AXIALKRAFT)
        )

    # Lüfter (Fan) – Aggregation über alle Fan-Segmente
    luefter_segmente = segmente_gruppiert.get("fan", [])
    anzahl_luefter = len(luefter_segmente)
    parameter[(MAPPING_LUEFTER_ANZAHL.component, MAPPING_LUEFTER_ANZAHL.parameter)] = (
        _parameter_eintrag(anzahl_luefter, MAPPING_LUEFTER_ANZAHL)
    )
    if luefter_segmente:
        summe_masse = sum(_safe_float(s.get("mass")) or 0.0 for s in luefter_segmente)
        max_durchmesser = max(
            (_safe_float(s.get("outer_diameter")) or 0.0 for s in luefter_segmente),
            default=0.0,
        )
        summe_traegheit = sum(_safe_float(s.get("polar_inertia")) or 0.0 for s in luefter_segmente)

        parameter[(MAPPING_LUEFTER_MASSE.component, MAPPING_LUEFTER_MASSE.parameter)] = (
            _parameter_eintrag(summe_masse, MAPPING_LUEFTER_MASSE)
        )
        parameter[(MAPPING_LUEFTER_D.component, MAPPING_LUEFTER_D.parameter)] = _parameter_eintrag(
            max_durchmesser, MAPPING_LUEFTER_D
        )
        parameter[(MAPPING_LUEFTER_J.component, MAPPING_LUEFTER_J.parameter)] = _parameter_eintrag(
            summe_traegheit, MAPPING_LUEFTER_J
        )

    # Wellenende (Shaft End) – erstes Wellenende für Kupplungsmasse + Passfeder
    wellenenden = segmente_gruppiert.get("shaft_end", [])
    if wellenenden:
        wellenende = wellenenden[0]
        parameter[
            (MAPPING_WELLE_KUPPLUNGSMASSE.component, MAPPING_WELLE_KUPPLUNGSMASSE.parameter)
        ] = _parameter_eintrag(wellenende.get("coupling_mass"), MAPPING_WELLE_KUPPLUNGSMASSE)

    # ------------------------------------------------------------------
    # Lager-Parameter aus bearing_properties
    # ------------------------------------------------------------------
    erstes_lager = _erstes_lager_aus_properties(bearing_properties)
    if erstes_lager:
        parameter[(MAPPING_LAGER_BEZEICHNUNG.component, MAPPING_LAGER_BEZEICHNUNG.parameter)] = (
            _parameter_eintrag(
                erstes_lager.get("designation"), MAPPING_LAGER_BEZEICHNUNG, erzwinge_string=True
            )
        )
        parameter[(MAPPING_LAGER_BAUREIHE.component, MAPPING_LAGER_BAUREIHE.parameter)] = (
            _parameter_eintrag(
                erstes_lager.get("bearing_series"), MAPPING_LAGER_BAUREIHE, erzwinge_string=True
            )
        )
        parameter[(MAPPING_LAGER_TYP_DETAIL.component, MAPPING_LAGER_TYP_DETAIL.parameter)] = (
            _parameter_eintrag(erstes_lager.get("type"), MAPPING_LAGER_TYP_DETAIL)
        )
        parameter[(MAPPING_LAGER_INNER_D.component, MAPPING_LAGER_INNER_D.parameter)] = (
            _parameter_eintrag(erstes_lager.get("inner_diameter"), MAPPING_LAGER_INNER_D)
        )
        parameter[(MAPPING_LAGER_OUTER_D.component, MAPPING_LAGER_OUTER_D.parameter)] = (
            _parameter_eintrag(erstes_lager.get("outer_diameter"), MAPPING_LAGER_OUTER_D)
        )
        parameter[(MAPPING_LAGER_DYN_TRAGZAHL.component, MAPPING_LAGER_DYN_TRAGZAHL.parameter)] = (
            _parameter_eintrag(erstes_lager.get("basic_dynamic_load"), MAPPING_LAGER_DYN_TRAGZAHL)
        )
        parameter[
            (MAPPING_LAGER_STAT_TRAGZAHL.component, MAPPING_LAGER_STAT_TRAGZAHL.parameter)
        ] = _parameter_eintrag(erstes_lager.get("basic_static_load"), MAPPING_LAGER_STAT_TRAGZAHL)

    # ------------------------------------------------------------------
    # OUTPUT-Parameter (berechnete Ergebnisse)
    # ------------------------------------------------------------------
    ausgabe_ergebnisse = _sicherer_zugriff(ausgabe, "output") or {}

    # Shaft-Form-Ergebnisse
    wellenform = _sicherer_zugriff(ausgabe_ergebnisse, "form", "output", "shaft") or {}

    parameter[(MAPPING_WELLE_LAENGE.component, MAPPING_WELLE_LAENGE.parameter)] = (
        _parameter_eintrag(wellenform.get("length"), MAPPING_WELLE_LAENGE)
    )
    parameter[(MAPPING_WELLE_MASSE.component, MAPPING_WELLE_MASSE.parameter)] = _parameter_eintrag(
        wellenform.get("mass"), MAPPING_WELLE_MASSE
    )
    parameter[
        (MAPPING_WELLE_TORSIONSSTEIFIGKEIT.component, MAPPING_WELLE_TORSIONSSTEIFIGKEIT.parameter)
    ] = _parameter_eintrag(wellenform.get("torsional_stiffness"), MAPPING_WELLE_TORSIONSSTEIFIGKEIT)
    parameter[(MAPPING_WELLE_VOLUMEN.component, MAPPING_WELLE_VOLUMEN.parameter)] = (
        _parameter_eintrag(wellenform.get("volume"), MAPPING_WELLE_VOLUMEN)
    )
    parameter[
        (MAPPING_WELLE_ZERSPANUNGSRATE.component, MAPPING_WELLE_ZERSPANUNGSRATE.parameter)
    ] = _parameter_eintrag(wellenform.get("metal_removal_rate"), MAPPING_WELLE_ZERSPANUNGSRATE)
    parameter[
        (MAPPING_WELLE_TRAEGHEITSMOMENT.component, MAPPING_WELLE_TRAEGHEITSMOMENT.parameter)
    ] = _parameter_eintrag(wellenform.get("mass_moment_of_inertia"), MAPPING_WELLE_TRAEGHEITSMOMENT)
    parameter[(MAPPING_ROTOR_LAGERABSTAND.component, MAPPING_ROTOR_LAGERABSTAND.parameter)] = (
        _parameter_eintrag(
            _sicherer_zugriff(wellenform, "bearing_positions", "distance"),
            MAPPING_ROTOR_LAGERABSTAND,
        )
    )
    parameter[(MAPPING_ROTOR_KERNMITTE.component, MAPPING_ROTOR_KERNMITTE.parameter)] = (
        _parameter_eintrag(wellenform.get("core_center"), MAPPING_ROTOR_KERNMITTE)
    )

    # Shaft Safety – minimaler Sicherheitsfaktor
    sicherheitswerte = (
        _sicherer_zugriff(ausgabe_ergebnisse, "shaft", "output", "shaft_safety", "output") or []
    )
    alle_sicherheiten = [
        eintrag.get("safety", {}).get("fatigue_strength")
        for eintrag in sicherheitswerte
        if isinstance(eintrag.get("safety", {}).get("fatigue_strength"), (int, float))
    ]
    if alle_sicherheiten:
        parameter[
            (MAPPING_WELLE_MIN_SICHERHEIT.component, MAPPING_WELLE_MIN_SICHERHEIT.parameter)
        ] = _parameter_eintrag(min(alle_sicherheiten), MAPPING_WELLE_MIN_SICHERHEIT)

    # Parallelkey-Drehmoment
    pk_eintraege = (
        _sicherer_zugriff(ausgabe_ergebnisse, "shaft", "output", "parallelkey_safety", "output")
        or []
    )
    if pk_eintraege:
        pk_drehmoment = pk_eintraege[0].get("transmittable_torque")
        if pk_drehmoment is not None:
            parameter[
                (MAPPING_WELLE_PK_DREHMOMENT.component, MAPPING_WELLE_PK_DREHMOMENT.parameter)
            ] = _parameter_eintrag(pk_drehmoment, MAPPING_WELLE_PK_DREHMOMENT)

    # Rotordynamik
    rotordynamik = _sicherer_zugriff(ausgabe_ergebnisse, "rotordynamics", "output") or {}

    rotor_masse = _sicherer_zugriff(rotordynamik, "rotor_properties", "mass")
    if rotor_masse is not None:
        parameter[(MAPPING_ROTOR_GESAMTMASSE.component, MAPPING_ROTOR_GESAMTMASSE.parameter)] = (
            _parameter_eintrag(rotor_masse, MAPPING_ROTOR_GESAMTMASSE)
        )

    statische_loesung = rotordynamik.get("static_solution", {})
    max_biegung = statische_loesung.get("maximum_bending")
    if max_biegung is not None:
        parameter[(MAPPING_ROTOR_MAX_BIEGUNG.component, MAPPING_ROTOR_MAX_BIEGUNG.parameter)] = (
            _parameter_eintrag(abs(max_biegung), MAPPING_ROTOR_MAX_BIEGUNG)
        )
    mittlere_biegung = statische_loesung.get("average_bending_laminated_core")
    if mittlere_biegung is not None:
        parameter[(MAPPING_ROTOR_BIEGUNG_KERN.component, MAPPING_ROTOR_BIEGUNG_KERN.parameter)] = (
            _parameter_eintrag(abs(mittlere_biegung), MAPPING_ROTOR_BIEGUNG_KERN)
        )

    eigenfrequenzen = _sicherer_zugriff(rotordynamik, "modal_solution", "eigenfrequencies") or {}
    if isinstance(eigenfrequenzen, dict):
        mode_1 = eigenfrequenzen.get("mode_1")
        mode_2 = eigenfrequenzen.get("mode_2")
        if mode_1 is not None:
            parameter[
                (MAPPING_ROTOR_EIGENFREQ_1.component, MAPPING_ROTOR_EIGENFREQ_1.parameter)
            ] = _parameter_eintrag(mode_1, MAPPING_ROTOR_EIGENFREQ_1)
        if mode_2 is not None:
            parameter[
                (MAPPING_ROTOR_EIGENFREQ_2.component, MAPPING_ROTOR_EIGENFREQ_2.parameter)
            ] = _parameter_eintrag(mode_2, MAPPING_ROTOR_EIGENFREQ_2)

    # Lager-Lebensdauer (nur bei Wälzlagern vorhanden)
    lager_lebensdauer = _sicherer_zugriff(
        ausgabe_ergebnisse, "roller_bearings", "output", "total", "rating_life_in_hours"
    )
    if lager_lebensdauer is not None:
        parameter[(MAPPING_LAGER_LEBENSDAUER.component, MAPPING_LAGER_LEBENSDAUER.parameter)] = (
            _parameter_eintrag(lager_lebensdauer, MAPPING_LAGER_LEBENSDAUER)
        )

    return rotor_id, {"params": parameter}


# ---------------------------------------------------------------------------
# Öffentliche API: alle JSON-Dateien laden
# ---------------------------------------------------------------------------


def fetch_all_features_from_json(
    verzeichnis: Path | str | None = None,
) -> dict[str, dict]:
    """
    Liest alle WVSC-JSON-Dateien und gibt features_by_rotor im Feature-Fetcher-Format zurück.

    :param verzeichnis: Pfad zum Verzeichnis mit JSON-Dateien (optional, Standard: data/real_data/wvsc)
    :type verzeichnis: Path | str | None
    :return: features_by_rotor[rotor_id]["params"][(component, param)] = {"value", "unit", "ptype"}
    :rtype: dict[str, dict]
    """
    if verzeichnis is None:
        json_verzeichnis = WVSC_STANDARD_VERZEICHNIS
    else:
        json_verzeichnis = Path(verzeichnis)

    if not json_verzeichnis.is_dir():
        logger.error("WVSC-Verzeichnis existiert nicht: %s", json_verzeichnis)
        return {}

    json_dateien = sorted(json_verzeichnis.glob("*.json"))
    if not json_dateien:
        logger.warning("Keine JSON-Dateien in %s gefunden", json_verzeichnis)
        return {}

    features_by_rotor: dict[str, dict] = {}
    fehlerhafte_dateien = 0

    for dateipfad in json_dateien:
        ergebnis = _parse_einzelne_json(dateipfad)
        if ergebnis is None:
            fehlerhafte_dateien += 1
            continue
        rotor_id, rotor_daten = ergebnis
        features_by_rotor[rotor_id] = rotor_daten

    logger.info(
        "%d Rotoren aus JSON geladen (%d Dateien fehlerhaft)",
        len(features_by_rotor),
        fehlerhafte_dateien,
    )

    return features_by_rotor
