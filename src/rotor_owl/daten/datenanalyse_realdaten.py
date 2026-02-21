"""
Datenanalyse und Bereinigung der realen WVSC-Rotordaten.

Dieses Skript kombiniert zwei Schritte:

**Schritt 1 – Analyse (vormals datenanalyse_realdaten.py):**
  Analysiert rekursiv ALLE Felder aus den JSON-Dateien, nicht nur die
  44 vom json_parser ausgewaehlten. Umfasst auch SimocalcWorkflow,
  sleeve_bearing_calculation, Segment-Details, Material-Eigenschaften etc.
  Berechnet statistische Kennzahlen und Eignungs-Scores.

**Schritt 2 – Bereinigung (vormals bereinige_parameter.py):**
  Ordnet JSON-Pfade den bekannten Parametern zu, identifiziert fachlich
  relevante NEUE Parameter, entfernt Duplikate, Konstanten und
  Identifikatoren. Gibt eine bereinigte CSV mit fachlicher Kategorisierung
  aus.

Eingabe:  data/reference/wvsc/*.json
Ausgabe:  data/real_data/analyse/parameter_analyse_alle.csv
          data/real_data/analyse/parameter_bereinigt.csv
          data/real_data/analyse/*.png  (3 Visualisierungen)

Ausfuehrung:
  python -m rotor_owl.daten.datenanalyse_realdaten
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import get_cmap  # noqa: E402

# ---------------------------------------------------------------------------
# Pfade (relativ zum Projektverzeichnis)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = _SCRIPT_DIR / ".." / ".." / ".." / "data"
WVSC_VERZEICHNIS = (_DATA_DIR / "reference" / "wvsc").resolve()
AUSGABE_VERZEICHNIS = (_DATA_DIR / "real_data" / "analyse").resolve()
AUSGABE_VERZEICHNIS.mkdir(parents=True, exist_ok=True)

# Pfade, die uebersprungen werden (Meta-Daten, Versions-Infos, leere Felder)
SKIP_PATTERNS = [
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


# ═══════════════════════════════════════════════════════════════════════════
# TEIL 1: JSON-Parameter extrahieren und analysieren
# ═══════════════════════════════════════════════════════════════════════════


def _sollte_uebersprungen_werden(pfad: str) -> bool:
    """Prueft ob ein JSON-Pfad uebersprungen werden soll."""
    for pattern in SKIP_PATTERNS:
        if pattern in pfad:
            return True
    return False


def _flatten_json(obj: dict | list, prefix: str = "") -> dict[str, object]:
    """
    Flattened ein verschachteltes JSON-Objekt rekursiv.
    Gibt dict {pfad: wert} fuer alle Blattknoten zurueck.
    """
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
    Normalisiert bearing_properties: variable Lagernamen -> 'bearing_1', 'bearing_2'.
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


def lade_alle_parameter() -> tuple[pd.DataFrame, int]:
    """Laedt alle JSON-Dateien und extrahiert rekursiv alle Parameter."""
    print("[1/8] Lade und parse alle JSON-Dateien ...")

    dateien = sorted(WVSC_VERZEICHNIS.glob("*.json"))
    print(f"      {len(dateien)} JSON-Dateien gefunden.\n")

    alle_zeilen = []
    n_rotoren = 0

    for datei in dateien:
        with open(datei, "r", encoding="utf-8") as f:
            d = json.load(f)

        rotor_id = d.get("machine_id", datei.stem)
        n_rotoren += 1

        params = _flatten_json(d)
        params = _normalisiere_bearing_properties(params)

        for pfad, wert in params.items():
            alle_zeilen.append(
                {
                    "rotor_id": rotor_id,
                    "parameter": pfad,
                    "value": wert,
                }
            )

    df = pd.DataFrame(alle_zeilen)
    print(f"      {n_rotoren} Rotoren geladen.")
    print(f"      {df['parameter'].nunique()} eindeutige Parameter gefunden.")
    print(f"      {len(df)} Datenpunkte insgesamt.\n")

    return df, n_rotoren


# ---------------------------------------------------------------------------
# Numerisch vs. kategorisch klassifizieren
# ---------------------------------------------------------------------------
def klassifiziere_parameter(df: pd.DataFrame) -> dict[str, str]:
    """Bestimmt pro Parameter ob numerisch oder kategorisch."""
    klassifikation = {}
    for param in df["parameter"].unique():
        werte = pd.Series(df.loc[df["parameter"] == param, "value"]).dropna()
        numerisch_count = 0
        for w in werte:
            if isinstance(w, (int, float)):
                numerisch_count += 1
            elif isinstance(w, str):
                try:
                    float(w)
                    numerisch_count += 1
                except (ValueError, TypeError):
                    pass
        anteil = numerisch_count / len(werte) if len(werte) > 0 else 0
        klassifikation[param] = "numerisch" if anteil > 0.8 else "kategorisch"
    return klassifikation


# ---------------------------------------------------------------------------
# Entropie
# ---------------------------------------------------------------------------
def berechne_entropie(werte: list) -> float:
    """Shannon-Entropie (log2)."""
    if not werte:
        return 0.0
    counts = Counter(werte)
    n = len(werte)
    return -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)


# ---------------------------------------------------------------------------
# Bereich aus Pfad ableiten
# ---------------------------------------------------------------------------
def _bereich_aus_pfad(pfad: str) -> str:
    """Leitet den logischen Bereich aus dem JSON-Pfad ab."""
    p = pfad.lower()
    if p.startswith("simocalcworkflow"):
        if ".input." in p:
            return "Simocalc (Input)"
        return "Simocalc (Output)"
    if p.startswith("segments"):
        return "Segmente"
    if "sleeve_bearing" in p:
        return "Gleitlager"
    if "roller_bearings" in p:
        return "Waelzlager (Output)"
    if "bearing_properties" in p:
        return "Lager (Input)"
    if "rotordynamics" in p:
        return "Rotordynamik"
    if "shaft_safety" in p or "parallelkey" in p:
        return "Wellensicherheit"
    if "form.output.shaft" in p:
        return "Wellenform"
    if "operational_data" in p:
        return "Betriebsdaten"
    if "load." in p:
        return "Lastdaten"
    if "material" in p:
        return "Materialien"
    if "surface" in p:
        return "Oberflaeche"
    if p.startswith("rotorworkflow.input"):
        return "Rotor (Input)"
    if p.startswith("rotorworkflow.output"):
        return "Rotor (Output)"
    if p.startswith("machine_id"):
        return "Meta"
    return "Sonstige"


# ---------------------------------------------------------------------------
# Statistische Analyse
# ---------------------------------------------------------------------------
def analysiere_parameter(df: pd.DataFrame, n_rotoren: int) -> pd.DataFrame:
    """Berechnet umfassende Statistiken pro Parameter."""
    klassifikation = klassifiziere_parameter(df)
    ergebnisse = []

    for param in sorted(df["parameter"].unique()):
        subset = df.loc[df["parameter"] == param]
        werte_raw = subset["value"].dropna().tolist()
        datentyp = klassifikation[param]
        n_vorhanden = len(werte_raw)
        abdeckung_pct = round(100 * n_vorhanden / n_rotoren, 1)
        bereich = _bereich_aus_pfad(param)

        eintrag = {
            "Parameter": param,
            "Bereich": bereich,
            "Datentyp": datentyp,
            "Anzahl": n_vorhanden,
            "Abdeckung (%)": abdeckung_pct,
        }

        if datentyp == "numerisch":
            numerische_werte = []
            for w in werte_raw:
                try:
                    numerische_werte.append(float(w))
                except (ValueError, TypeError):
                    pass
            arr = np.array(numerische_werte) if numerische_werte else np.array([])

            if len(arr) > 0:
                mean_val = float(np.mean(arr))
                std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
                cv = (std_val / abs(mean_val) * 100) if mean_val != 0 else 0.0
                q1, median_val, q3 = np.percentile(arr, [25, 50, 75])
                iqr = q3 - q1
                n_unique = len(set(numerische_werte))

                if len(arr) > 2 and std_val > 0:
                    schiefe = float(
                        (len(arr) / ((len(arr) - 1) * (len(arr) - 2)))
                        * np.sum(((arr - mean_val) / std_val) ** 3)
                    )
                else:
                    schiefe = 0.0

                eintrag.update(
                    {
                        "Min": round(float(np.min(arr)), 2),
                        "Max": round(float(np.max(arr)), 2),
                        "Mittelwert": round(mean_val, 2),
                        "Median": round(float(median_val), 2),
                        "Std.Abw.": round(std_val, 2),
                        "CV (%)": round(cv, 1),
                        "IQR": round(float(iqr), 2),
                        "Schiefe": round(schiefe, 2),
                        "Unique": n_unique,
                        "Haeufigster Wert": "–",
                        "Entropie (bit)": "–",
                    }
                )
            else:
                eintrag.update(
                    {
                        k: "–"
                        for k in [
                            "Min",
                            "Max",
                            "Mittelwert",
                            "Median",
                            "Std.Abw.",
                            "CV (%)",
                            "IQR",
                            "Schiefe",
                            "Unique",
                            "Haeufigster Wert",
                            "Entropie (bit)",
                        ]
                    }
                )
        else:
            str_werte = [str(w) for w in werte_raw]
            counts = Counter(str_werte)
            n_unique = len(counts)
            haeufigster = counts.most_common(1)[0] if counts else ("–", 0)
            entropie = berechne_entropie(str_werte)
            max_entropie = math.log2(n_unique) if n_unique > 1 else 0.0

            eintrag.update(
                {
                    "Min": "–",
                    "Max": "–",
                    "Mittelwert": "–",
                    "Median": "–",
                    "Std.Abw.": "–",
                    "CV (%)": "–",
                    "IQR": "–",
                    "Schiefe": "–",
                    "Unique": n_unique,
                    "Haeufigster Wert": f"{str(haeufigster[0])[:40]} ({haeufigster[1]}x)",
                    "Entropie (bit)": f"{entropie:.2f} / {max_entropie:.2f}",
                }
            )

        ergebnisse.append(eintrag)

    return pd.DataFrame(ergebnisse)


# ---------------------------------------------------------------------------
# Eignung bewerten
# ---------------------------------------------------------------------------
def bewerte_eignung(df_stats: pd.DataFrame, n_rotoren: int) -> pd.DataFrame:
    """
    Bewertet jeden Parameter 0-100 hinsichtlich Eignung fuer
    Aehnlichkeitsanalysen.

    Kriterien (gewichtet):
    - Unique       (50 %): Anzahl verschiedener Werte (log-skaliert)
    - Variabilitaet (35 %): CV (numerisch) bzw. normierte Entropie (kategorisch)
    - Verteilung   (15 %): Symmetrie (geringe Schiefe = besser)
    """
    scores = []
    for _, row in df_stats.iterrows():
        # Variabilitaet (35 %)
        if row["Datentyp"] == "numerisch" and row["CV (%)"] != "–":
            s_var = min(float(row["CV (%)"]), 100.0)
        elif row["Datentyp"] == "kategorisch" and row["Entropie (bit)"] != "–":
            teile = str(row["Entropie (bit)"]).split(" / ")
            if len(teile) == 2:
                try:
                    ent, max_ent = float(teile[0]), float(teile[1])
                    s_var = (ent / max_ent * 100) if max_ent > 0 else 0.0
                except ValueError:
                    s_var = 0.0
            else:
                s_var = 0.0
        else:
            s_var = 0.0

        # Unique (50 %)
        if row["Unique"] != "–":
            n_u = int(row["Unique"])
            s_unique = (
                min(math.log2(n_u) / math.log2(max(n_rotoren, 2)) * 100, 100.0) if n_u > 1 else 0.0
            )
        else:
            s_unique = 0.0

        # Verteilung (15 %)
        if row["Datentyp"] == "numerisch" and row["Schiefe"] != "–":
            s_vert = max(0.0, 100.0 - abs(float(row["Schiefe"])) * 33.33)
        else:
            s_vert = 50.0

        gesamt = 0.50 * s_unique + 0.35 * s_var + 0.15 * s_vert
        scores.append(round(gesamt, 1))

    df_stats = df_stats.copy()
    df_stats["Eignung"] = scores
    return df_stats.sort_values("Eignung", ascending=False)


# ---------------------------------------------------------------------------
# Visualisierungen
# ---------------------------------------------------------------------------
def erstelle_visualisierungen(df_stats: pd.DataFrame, n_rotoren: int):
    """Erzeugt alle Analysegrafiken."""
    print("[5/8] Erstelle Visualisierungen ...\n")

    plt.rcParams.update(
        {
            "font.size": 8,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.3,
        }
    )

    _plot_bereich_uebersicht(df_stats)
    _plot_abdeckung_verteilung(df_stats, n_rotoren)
    _plot_datentyp_verteilung(df_stats)


def _plot_bereich_uebersicht(df_stats: pd.DataFrame):
    """Parameter-Anzahl und Durchschnitt Eignung pro Bereich."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    counts = df_stats["Bereich"].value_counts().sort_values()
    axes[0].barh(counts.index, counts.values, color="#3498db", edgecolor="white")
    axes[0].set_xlabel("Anzahl Parameter")
    axes[0].set_title("Parameter pro Bereich", fontweight="bold")
    for i, (_, val) in enumerate(zip(counts.index, counts.values)):
        axes[0].text(val + 0.5, i, str(val), va="center", fontsize=9)

    mean_scores = df_stats.groupby("Bereich")["Eignung"].mean().sort_values()
    farben = get_cmap("RdYlGn")(mcolors.Normalize(0, 100)(np.asarray(mean_scores.values)))
    axes[1].barh(mean_scores.index, mean_scores.values, color=farben, edgecolor="white")
    axes[1].set_xlabel("Durchschnittlicher Eignungs-Score")
    axes[1].set_title("Durchschnittliche Eignung pro Bereich", fontweight="bold")
    axes[1].set_xlim(0, 100)
    for i, (_, val) in enumerate(zip(mean_scores.index, mean_scores.values)):
        axes[1].text(val + 1, i, f"{val:.0f}", va="center", fontsize=9)

    plt.tight_layout()
    pfad = AUSGABE_VERZEICHNIS / "03_bereich_uebersicht.png"
    fig.savefig(pfad)
    plt.close(fig)
    print(f"      -> {pfad.name}")


def _plot_abdeckung_verteilung(df_stats: pd.DataFrame, n_rotoren: int):
    """Histogramm: Verteilung der Abdeckung."""
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100.1]
    ax.hist(df_stats["Abdeckung (%)"], bins=bins, color="#2ecc71", edgecolor="white", rwidth=0.9)
    ax.set_xlabel("Abdeckung (%)")
    ax.set_ylabel("Anzahl Parameter")
    ax.set_title(
        f"Verteilung der Parameterabdeckung ({len(df_stats)} Parameter, {n_rotoren} Rotoren)",
        fontsize=11,
        fontweight="bold",
    )

    voll = (df_stats["Abdeckung (%)"] == 100).sum()
    ax.annotate(
        f"{voll} Parameter mit\n100 % Abdeckung",
        xy=(95, voll * 0.3),
        fontsize=9,
        ha="center",
        color="#27ae60",
        fontweight="bold",
    )

    plt.tight_layout()
    pfad = AUSGABE_VERZEICHNIS / "04_abdeckung_histogramm.png"
    fig.savefig(pfad)
    plt.close(fig)
    print(f"      -> {pfad.name}")


def _plot_datentyp_verteilung(df_stats: pd.DataFrame):
    """Pie-Chart: Datentyp + Abdeckungsstufen."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    dt_counts = df_stats["Datentyp"].value_counts()
    axes[0].pie(
        dt_counts.values,
        labels=dt_counts.index,
        autopct="%1.0f%%",
        colors=["#3498db", "#e74c3c"],
        textprops={"fontsize": 10},
    )
    axes[0].set_title("Numerisch vs. Kategorisch", fontweight="bold")

    def abdeckung_stufe(pct):
        if pct == 100:
            return "100 % (vollstaendig)"
        elif pct >= 70:
            return "70-99 %"
        elif pct >= 30:
            return "30-69 %"
        else:
            return "< 30 %"

    stufen = df_stats["Abdeckung (%)"].apply(abdeckung_stufe).value_counts()
    farben_map = {
        "100 % (vollstaendig)": "#2ecc71",
        "70-99 %": "#f39c12",
        "30-69 %": "#e67e22",
        "< 30 %": "#e74c3c",
    }
    c = [farben_map.get(s, "#95a5a6") for s in stufen.index]
    axes[1].pie(
        stufen.values,
        labels=stufen.index,
        autopct="%1.0f%%",
        colors=c,
        textprops={"fontsize": 9},
    )
    axes[1].set_title("Abdeckungsstufen", fontweight="bold")

    plt.tight_layout()
    pfad = AUSGABE_VERZEICHNIS / "05_datentyp_abdeckung.png"
    fig.savefig(pfad)
    plt.close(fig)
    print(f"      -> {pfad.name}")


# ---------------------------------------------------------------------------
# Konsolenausgabe
# ---------------------------------------------------------------------------
def drucke_tabellen(df_stats: pd.DataFrame, n_rotoren: int):
    """Druckt die Ergebnisse tabellarisch."""
    print("[4/8] Ergebnisse:\n")

    trenn = "=" * 130
    print(trenn)
    print("  DATENANALYSE DER REALEN WVSC-ROTORDATEN (ALLE PARAMETER)")
    print(f"  {n_rotoren} Rotoren, {len(df_stats)} Parameter")
    print(trenn)

    # Zusammenfassung nach Bereich
    print("\n--- Uebersicht nach Bereich ---\n")
    zusammenfassung = (
        df_stats.groupby("Bereich")
        .agg(
            Anzahl_Params=("Parameter", "count"),
            Avg_Abdeckung=("Abdeckung (%)", "mean"),
            Avg_Eignung=("Eignung", "mean"),
            Numerisch=("Datentyp", lambda x: (x == "numerisch").sum()),
            Kategorisch=("Datentyp", lambda x: (x == "kategorisch").sum()),
        )
        .round(1)
        .sort_values("Avg_Eignung", ascending=False)
    )
    print(zusammenfassung.to_string())

    # Top 30
    print("\n--- Top 30 geeignetste Parameter ---\n")
    top_cols = [
        "Parameter",
        "Bereich",
        "Datentyp",
        "Abdeckung (%)",
        "CV (%)",
        "Unique",
        "Eignung",
    ]
    top30 = df_stats.head(30)[top_cols].copy()
    top30["Parameter"] = top30["Parameter"].apply(lambda p: p if len(p) <= 60 else "..." + p[-57:])
    print(top30.to_string(index=False))

    # Bottom 15
    print("\n--- Bottom 15 (am wenigsten geeignet) ---\n")
    bottom15 = df_stats.tail(15)[top_cols].copy()
    bottom15["Parameter"] = bottom15["Parameter"].apply(
        lambda p: p if len(p) <= 60 else "..." + p[-57:]
    )
    print(bottom15.to_string(index=False))

    # Numerische Highlights
    num_df = df_stats[df_stats["Datentyp"] == "numerisch"].copy()
    num_df = num_df[num_df["CV (%)"] != "–"].copy()
    if not num_df.empty:
        num_df["CV_float"] = num_df["CV (%)"].astype(float)
        print("\n--- Numerische Parameter mit hoechster Varianz (Top 20 nach CV) ---\n")
        top_cv = num_df.nlargest(20, "CV_float")[
            [
                "Parameter",
                "Bereich",
                "Abdeckung (%)",
                "Min",
                "Max",
                "CV (%)",
                "Unique",
                "Eignung",
            ]
        ].copy()
        top_cv["Parameter"] = top_cv["Parameter"].apply(
            lambda p: p if len(p) <= 55 else "..." + p[-52:]
        )
        print(top_cv.to_string(index=False))

    # Kategorische
    kat_df = df_stats[df_stats["Datentyp"] == "kategorisch"].copy()
    if not kat_df.empty:
        print(f"\n--- Kategorische Parameter ({len(kat_df)}) ---\n")
        kat_cols = [
            "Parameter",
            "Bereich",
            "Abdeckung (%)",
            "Unique",
            "Haeufigster Wert",
            "Entropie (bit)",
            "Eignung",
        ]
        kat_out = (
            kat_df.rename(columns={"Haeufigster Wert": "Haeufigster Wert"})[kat_cols]
            .sort_values("Eignung", ascending=False)
            .copy()
        )
        kat_out["Parameter"] = kat_out["Parameter"].apply(
            lambda p: p if len(p) <= 50 else "..." + p[-47:]
        )
        print(kat_out.to_string(index=False))

    print(f"\n{trenn}")
    print("  Legende Eignungs-Score:")
    print("    80-100: Sehr gut geeignet")
    print("    60-79:  Gut geeignet")
    print("    40-59:  Bedingt geeignet")
    print("    < 40:   Wenig geeignet")
    print(trenn)


# ---------------------------------------------------------------------------
# CSV-Export (Analyse)
# ---------------------------------------------------------------------------
def _fmt_komma(val):
    """Dezimalpunkt -> Dezimalkomma fuer numerische Werte."""
    if isinstance(val, float):
        return str(val).replace(".", ",")
    return val


def exportiere_analyse_csv(df_stats: pd.DataFrame):
    """Exportiert die Analyse-Ergebnisse als CSV (Dezimalkomma)."""
    pfad = AUSGABE_VERZEICHNIS / "parameter_analyse_alle.csv"
    df_export = df_stats.copy()

    num_cols = [
        "Abdeckung (%)",
        "Min",
        "Max",
        "Mittelwert",
        "Median",
        "Std.Abw.",
        "CV (%)",
        "IQR",
        "Schiefe",
        "Eignung",
    ]
    for col in num_cols:
        if col in df_export.columns:
            df_export[col] = df_export[col].apply(_fmt_komma)

    if "Entropie (bit)" in df_export.columns:
        df_export["Entropie (bit)"] = df_export["Entropie (bit)"].apply(
            lambda x: str(x).replace(".", ",") if str(x) != "–" else x
        )

    df_export.to_csv(pfad, index=False, sep=";", encoding="utf-8-sig")
    print(f"      -> CSV: {pfad.name}")


# ═══════════════════════════════════════════════════════════════════════════
# TEIL 2: Bereinigung und fachliche Kategorisierung
# ═══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# MAPPING: JSON-Pfad -> bekannter P_-Parametername
# ---------------------------------------------------------------------------
BEKANNTE_PARAMETER: dict[str, dict] = {
    # --- Welle (C_WELLE) ---
    "RotorWorkflow.output.output.form.output.shaft.mass_moment_of_inertia": {
        "P_Name": "P_WELLE_TRAEGHEITSMOMENT",
        "Komponente": "C_WELLE",
        "Einheit": "kg*m2",
        "Kategorie": "Dynamik",
    },
    "RotorWorkflow.output.output.form.output.shaft.torsional_stiffness": {
        "P_Name": "P_WELLE_TORSIONSSTEIFIGKEIT",
        "Komponente": "C_WELLE",
        "Einheit": "Nm/rad",
        "Kategorie": "Struktur",
    },
    "RotorWorkflow.output.output.form.output.shaft.mass": {
        "P_Name": "P_WELLE_MASSE",
        "Komponente": "C_WELLE",
        "Einheit": "kg",
        "Kategorie": "Material/Masse",
    },
    "RotorWorkflow.output.output.form.output.shaft.volume": {
        "P_Name": "P_WELLE_VOLUMEN",
        "Komponente": "C_WELLE",
        "Einheit": "m3",
        "Kategorie": "Geometrie",
    },
    "RotorWorkflow.output.output.form.output.shaft.metal_removal_rate": {
        "P_Name": "P_WELLE_ZERSPANUNGSRATE",
        "Komponente": "C_WELLE",
        "Einheit": "%",
        "Kategorie": "Fertigung",
    },
    "RotorWorkflow.output.output.form.output.shaft.length": {
        "P_Name": "P_WELLE_LAENGE",
        "Komponente": "C_WELLE",
        "Einheit": "mm",
        "Kategorie": "Geometrie",
    },
    "RotorWorkflow.output.output.form.output.shaft.core_center": {
        "P_Name": "P_ROTOR_KERNMITTE",
        "Komponente": "C_ROTOR",
        "Einheit": "mm",
        "Kategorie": "Geometrie",
    },
    "RotorWorkflow.output.output.form.output.shaft.bearing_positions.distance": {
        "P_Name": "P_ROTOR_LAGERABSTAND",
        "Komponente": "C_ROTOR",
        "Einheit": "mm",
        "Kategorie": "Geometrie",
    },
    "RotorWorkflow.output.output.shaft.output.parallelkey_safety.output.transmittable_torque": {
        "P_Name": "P_WELLE_PK_DREHMOMENT",
        "Komponente": "C_WELLE",
        "Einheit": "Nm",
        "Kategorie": "Struktur",
    },
    "RotorWorkflow.output.output.shaft.output.shaft_safety.output.fatigue_strength._min": {
        "P_Name": "P_WELLE_MIN_SICHERHEIT",
        "Komponente": "C_WELLE",
        "Einheit": "–",
        "Kategorie": "Struktur",
    },
    "segments.shaft_end.coupling_mass": {
        "P_Name": "P_WELLE_KUPPLUNGSMASSE",
        "Komponente": "C_WELLE",
        "Einheit": "kg",
        "Kategorie": "Material/Masse",
    },
    "RotorWorkflow.input.materials.shaft_material.material_name": {
        "P_Name": "P_WELLE_MATERIAL",
        "Komponente": "C_WELLE",
        "Einheit": "–",
        "Kategorie": "Material/Masse",
    },
    "RotorWorkflow.input.materials.shaft_material.name": {
        "P_Name": "P_WELLE_MATERIAL (Duplikat)",
        "Komponente": "C_WELLE",
        "Einheit": "–",
        "Kategorie": "Material/Masse",
        "_duplikat_von": "RotorWorkflow.input.materials.shaft_material.material_name",
    },
    # --- Rotor (C_ROTOR) ---
    "RotorWorkflow.output.output.rotordynamics.output.rotor_properties.mass": {
        "P_Name": "P_ROTOR_GESAMTMASSE",
        "Komponente": "C_ROTOR",
        "Einheit": "kg",
        "Kategorie": "Material/Masse",
    },
    "RotorWorkflow.input.load.nominal_torque": {
        "P_Name": "P_ROTOR_NENNMOMENT",
        "Komponente": "C_ROTOR",
        "Einheit": "Nm",
        "Kategorie": "Dynamik",
    },
    "RotorWorkflow.input.load.torsion.maximum": {
        "P_Name": "P_ROTOR_MAX_TORSION",
        "Komponente": "C_ROTOR",
        "Einheit": "Nm",
        "Kategorie": "Dynamik",
    },
    "RotorWorkflow.input.operational_data.construction_type": {
        "P_Name": "P_ROTOR_BAUFORM",
        "Komponente": "C_ROTOR",
        "Einheit": "–",
        "Kategorie": "Anforderung",
    },
    "RotorWorkflow.input.bearing_type": {
        "P_Name": "P_ROTOR_LAGERTYP",
        "Komponente": "C_ROTOR",
        "Einheit": "–",
        "Kategorie": "Anforderung",
    },
    "RotorWorkflow.input.pole_number": {
        "P_Name": "P_ROTOR_POLZAHL",
        "Komponente": "C_ROTOR",
        "Einheit": "–",
        "Kategorie": "Elektrisch",
    },
    "RotorWorkflow.input.c_dimension": {
        "P_Name": "P_ROTOR_C_MASS",
        "Komponente": "C_ROTOR",
        "Einheit": "mm",
        "Kategorie": "Geometrie",
    },
    "RotorWorkflow.input.operational_data.operational_speed": {
        "P_Name": "P_WELLE_DREHZAHLBEREICH",
        "Komponente": "C_WELLE",
        "Einheit": "1/min",
        "Kategorie": "Dynamik",
    },
    "RotorWorkflow.output.output.rotordynamics.output.modal_solution.eigenfrequencies.mode_1": {
        "P_Name": "P_ROTOR_EIGENFREQ_1",
        "Komponente": "C_ROTOR",
        "Einheit": "Hz",
        "Kategorie": "Dynamik",
    },
    "RotorWorkflow.output.output.rotordynamics.output.modal_solution.eigenfrequencies.mode_2": {
        "P_Name": "P_ROTOR_EIGENFREQ_2",
        "Komponente": "C_ROTOR",
        "Einheit": "Hz",
        "Kategorie": "Dynamik",
    },
    "RotorWorkflow.output.output.rotordynamics.output.static_solution.maximum_bending": {
        "P_Name": "P_ROTOR_MAX_BIEGUNG",
        "Komponente": "C_ROTOR",
        "Einheit": "mm",
        "Kategorie": "Struktur",
    },
    "RotorWorkflow.output.output.rotordynamics.output.static_solution.bending_at_core_center": {
        "P_Name": "P_ROTOR_BIEGUNG_KERN",
        "Komponente": "C_ROTOR",
        "Einheit": "mm",
        "Kategorie": "Struktur",
    },
    # --- Lager (C_LAGER) ---
    "RotorWorkflow.output.output.roller_bearings.output.total.rating_life_in_hours": {
        "P_Name": "P_LAGER_LEBENSDAUER",
        "Komponente": "C_LAGER",
        "Einheit": "h",
        "Kategorie": "Anforderung",
    },
    "RotorWorkflow.input.bearing_properties.bearing_1.static_load_rating": {
        "P_Name": "P_LAGER_STAT_TRAGZAHL",
        "Komponente": "C_LAGER",
        "Einheit": "N",
        "Kategorie": "Struktur",
    },
    "RotorWorkflow.input.bearing_properties.bearing_1.dynamic_load_rating": {
        "P_Name": "P_LAGER_DYN_TRAGZAHL",
        "Komponente": "C_LAGER",
        "Einheit": "N",
        "Kategorie": "Struktur",
    },
    "RotorWorkflow.input.bearing_properties.bearing_1.inner_diameter": {
        "P_Name": "P_LAGER_INNER_D",
        "Komponente": "C_LAGER",
        "Einheit": "mm",
        "Kategorie": "Geometrie",
    },
    "RotorWorkflow.input.bearing_properties.bearing_1.outer_diameter": {
        "P_Name": "P_LAGER_OUTER_D",
        "Komponente": "C_LAGER",
        "Einheit": "mm",
        "Kategorie": "Geometrie",
    },
    "RotorWorkflow.input.bearing_properties.bearing_1.type": {
        "P_Name": "P_LAGER_TYP_DETAIL",
        "Komponente": "C_LAGER",
        "Einheit": "–",
        "Kategorie": "Anforderung",
    },
    "RotorWorkflow.output.output.roller_bearings.output.NDE.designation": {
        "P_Name": "P_LAGER_BEZEICHNUNG",
        "Komponente": "C_LAGER",
        "Einheit": "–",
        "Kategorie": "Geometrie",
    },
    # --- Aktivteil (C_AKTIVTEIL) ---
    "segments.laminated_core.mass": {
        "P_Name": "P_AKTIV_MASSE",
        "Komponente": "C_AKTIVTEIL",
        "Einheit": "kg",
        "Kategorie": "Material/Masse",
    },
    "segments.laminated_core.length": {
        "P_Name": "P_AKTIV_LAENGE",
        "Komponente": "C_AKTIVTEIL",
        "Einheit": "mm",
        "Kategorie": "Geometrie",
    },
    "segments.laminated_core.outer_diameter": {
        "P_Name": "P_AKTIV_D_AUSSEN",
        "Komponente": "C_AKTIVTEIL",
        "Einheit": "mm",
        "Kategorie": "Geometrie",
    },
    "segments.laminated_core.inner_diameter": {
        "P_Name": "P_AKTIV_D_INNEN",
        "Komponente": "C_AKTIVTEIL",
        "Einheit": "mm",
        "Kategorie": "Geometrie",
    },
    # --- Luefter (C_LUEFTER) ---
    "segments.fan._count": {
        "P_Name": "P_LUEFTER_ANZAHL",
        "Komponente": "C_LUEFTER",
        "Einheit": "–",
        "Kategorie": "Geometrie",
    },
}

# ---------------------------------------------------------------------------
# NEUE fachlich relevante Parameter
# ---------------------------------------------------------------------------
NEUE_RELEVANTE: dict[str, dict] = {
    # --- Elektrisch / Leistung ---
    "EdimWorkflow.input.rated.power": {
        "P_Name": "P_NENNLEISTUNG",
        "Komponente": "C_ROTOR",
        "Einheit": "W",
        "Kategorie": "Elektrisch",
        "Begruendung": "Nennleistung – fundamentaler Auslegungsparameter jeder E-Maschine",
    },
    "EdimWorkflow.input.rated.frequency": {
        "P_Name": "P_NENNFREQUENZ",
        "Komponente": "C_ROTOR",
        "Einheit": "Hz",
        "Kategorie": "Elektrisch",
        "Begruendung": "Nennfrequenz – bestimmt Drehzahl zusammen mit Polzahl",
    },
    "EdimWorkflow.input.nameplate.rated_current": {
        "P_Name": "P_NENNSTROM",
        "Komponente": "C_ROTOR",
        "Einheit": "A",
        "Kategorie": "Elektrisch",
        "Begruendung": "Nennstrom – dimensioniert Wicklung und thermische Auslegung",
    },
    "EdimWorkflow.input.nameplate.rated_speed": {
        "P_Name": "P_NENNDREHZAHL",
        "Komponente": "C_ROTOR",
        "Einheit": "1/min",
        "Kategorie": "Elektrisch",
        "Begruendung": "Nenndrehzahl – bestimmt Betriebspunkt und Rotordynamik",
    },
    "EdimWorkflow.input.nameplate.rated_efficiency": {
        "P_Name": "P_NENNWIRKUNGSGRAD",
        "Komponente": "C_ROTOR",
        "Einheit": "–",
        "Kategorie": "Elektrisch",
        "Begruendung": "Wirkungsgrad – zentrale Leistungskennzahl (IE-Klasse)",
    },
    "EdimWorkflow.input.nameplate.rated_power_factor": {
        "P_Name": "P_LEISTUNGSFAKTOR",
        "Komponente": "C_ROTOR",
        "Einheit": "–",
        "Kategorie": "Elektrisch",
        "Begruendung": "cos(phi) – bestimmt Blindleistungsbedarf",
    },
    "EdimWorkflow.input.nameplate.breakdown_torque_ratio": {
        "P_Name": "P_KIPPMOMENT_VERHAELTNIS",
        "Komponente": "C_ROTOR",
        "Einheit": "–",
        "Kategorie": "Dynamik",
        "Begruendung": "Kippmoment/Nennmoment – Ueberlastfaehigkeit",
    },
    "EdimWorkflow.input.nameplate.locked_rotor_current_ratio": {
        "P_Name": "P_ANLAUFSTROM_VERHAELTNIS",
        "Komponente": "C_ROTOR",
        "Einheit": "–",
        "Kategorie": "Elektrisch",
        "Begruendung": "Anlaufstrom/Nennstrom – Netzbelastung beim Start",
    },
    "EdimWorkflow.input.nameplate.locked_rotor_torque_ratio": {
        "P_Name": "P_ANLAUFMOMENT_VERHAELTNIS",
        "Komponente": "C_ROTOR",
        "Einheit": "–",
        "Kategorie": "Dynamik",
        "Begruendung": "Anlaufmoment/Nennmoment – Startfaehigkeit",
    },
    # --- Stator-Geometrie ---
    "EdimWorkflow.input.stator.core.outer_diameter": {
        "P_Name": "P_STATOR_D_AUSSEN",
        "Komponente": "C_AKTIVTEIL",
        "Einheit": "mm",
        "Kategorie": "Geometrie",
        "Begruendung": "Stator-Aussendurchmesser – definiert Baugroesse",
    },
    "EdimWorkflow.input.stator.core.inner_diameter": {
        "P_Name": "P_STATOR_D_INNEN",
        "Komponente": "C_AKTIVTEIL",
        "Einheit": "mm",
        "Kategorie": "Geometrie",
        "Begruendung": "Stator-Innendurchmesser – definiert Luftspalt",
    },
    "EdimWorkflow.input.stator.core.iron_length": {
        "P_Name": "P_STATOR_EISENLAENGE",
        "Komponente": "C_AKTIVTEIL",
        "Einheit": "mm",
        "Kategorie": "Geometrie",
        "Begruendung": "Stator-Eisenlaenge – bestimmt elektromagnetisch aktives Volumen",
    },
    "EdimWorkflow.input.stator.core.number_of_slots": {
        "P_Name": "P_STATOR_NUTZAHL",
        "Komponente": "C_AKTIVTEIL",
        "Einheit": "–",
        "Kategorie": "Geometrie",
        "Begruendung": "Stator-Nutzahl – bestimmt Wicklungsauslegung und Oberwellen",
    },
    "EdimWorkflow.input.rotor.core.number_of_slots": {
        "P_Name": "P_ROTOR_NUTZAHL",
        "Komponente": "C_AKTIVTEIL",
        "Einheit": "–",
        "Kategorie": "Geometrie",
        "Begruendung": "Rotor-Nutzahl – Kaefiglaeufer-Stabzahl",
    },
    "EdimWorkflow.input.compound.air_gap_height": {
        "P_Name": "P_LUFTSPALT",
        "Komponente": "C_AKTIVTEIL",
        "Einheit": "mm",
        "Kategorie": "Geometrie",
        "Begruendung": "Luftspalt – kritischer Parameter fuer Magnetkreis und Rotordynamik",
    },
    # --- Wicklung ---
    "EdimWorkflow.input.stator.winding.number_of_poles": {
        "P_Name": "P_STATOR_POLZAHL",
        "Komponente": "C_AKTIVTEIL",
        "Einheit": "–",
        "Kategorie": "Elektrisch",
        "Begruendung": "Polzahl Stator – bestimmt Synchrondrehzahl",
    },
    "EdimWorkflow.input.stator.winding.coil_pitch": {
        "P_Name": "P_SPULENSCHRITT",
        "Komponente": "C_AKTIVTEIL",
        "Einheit": "–",
        "Kategorie": "Elektrisch",
        "Begruendung": "Spulenschritt – beeinflusst Wicklungsfaktor",
    },
    "EdimWorkflow.input.stator.winding.bare_wire_height": {
        "P_Name": "P_DRAHTHOEHE",
        "Komponente": "C_AKTIVTEIL",
        "Einheit": "mm",
        "Kategorie": "Geometrie",
        "Begruendung": "Drahthoehe – bestimmt Kupferquerschnitt",
    },
    "EdimWorkflow.input.stator.winding.bare_wire_width": {
        "P_Name": "P_DRAHTBREITE",
        "Komponente": "C_AKTIVTEIL",
        "Einheit": "mm",
        "Kategorie": "Geometrie",
        "Begruendung": "Drahtbreite – bestimmt Nutfuellfaktor",
    },
    # --- Rotor-Kaefig (Kurzschlusslaeufer) ---
    "EdimWorkflow.input.rotor.winding.bar.width": {
        "P_Name": "P_ROTORSTAB_BREITE",
        "Komponente": "C_AKTIVTEIL",
        "Einheit": "mm",
        "Kategorie": "Geometrie",
        "Begruendung": "Rotorstabreite – Querschnitt bestimmt Rotorwiderstand",
    },
    "EdimWorkflow.input.rotor.winding.bar.height": {
        "P_Name": "P_ROTORSTAB_HOEHE",
        "Komponente": "C_AKTIVTEIL",
        "Einheit": "mm",
        "Kategorie": "Geometrie",
        "Begruendung": "Rotor-Stabhoehe – beeinflusst Anlaufverhalten",
    },
    "EdimWorkflow.input.rotor.winding.bar.length": {
        "P_Name": "P_ROTORSTAB_LAENGE",
        "Komponente": "C_AKTIVTEIL",
        "Einheit": "mm",
        "Kategorie": "Geometrie",
        "Begruendung": "Rotor-Stablaenge – bestimmt Widerstand und Ueberstand",
    },
    "EdimWorkflow.input.rotor.winding.end_ring.width": {
        "P_Name": "P_KURZSCHLUSSRING_BREITE",
        "Komponente": "C_AKTIVTEIL",
        "Einheit": "mm",
        "Kategorie": "Geometrie",
        "Begruendung": "Kurzschlussring-Breite – bestimmt Ringwiderstand",
    },
    "EdimWorkflow.input.rotor.winding.end_ring.height": {
        "P_Name": "P_KURZSCHLUSSRING_HOEHE",
        "Komponente": "C_AKTIVTEIL",
        "Einheit": "mm",
        "Kategorie": "Geometrie",
        "Begruendung": "Kurzschlussring-Hoehe – bestimmt Ringquerschnitt",
    },
    "EdimWorkflow.input.rotor.winding.end_ring.outer_diameter": {
        "P_Name": "P_KURZSCHLUSSRING_D_AUSSEN",
        "Komponente": "C_AKTIVTEIL",
        "Einheit": "mm",
        "Kategorie": "Geometrie",
        "Begruendung": "Kurzschlussring-Aussendurchmesser – Massentraegheit am Rotorende",
    },
    # --- Lagerkraefte (berechnete Ergebnisse) ---
    "RotorWorkflow.output.output.roller_bearings.output.DE.forces.radial": {
        "P_Name": "P_LAGER_DE_RADIALKRAFT",
        "Komponente": "C_LAGER",
        "Einheit": "N",
        "Kategorie": "Struktur",
        "Begruendung": "Radialkraft am DE-Lager – zentral fuer Lagerdimensionierung",
    },
    "RotorWorkflow.output.output.roller_bearings.output.NDE.forces.radial": {
        "P_Name": "P_LAGER_NDE_RADIALKRAFT",
        "Komponente": "C_LAGER",
        "Einheit": "N",
        "Kategorie": "Struktur",
        "Begruendung": "Radialkraft am NDE-Lager – bestimmt Lagerbelastung",
    },
    "RotorWorkflow.output.output.roller_bearings.output.NDE.forces.axial": {
        "P_Name": "P_LAGER_NDE_AXIALKRAFT",
        "Komponente": "C_LAGER",
        "Einheit": "N",
        "Kategorie": "Struktur",
        "Begruendung": "Axialkraft am NDE-Lager – bestimmt Lagertyp-Eignung",
    },
    "RotorWorkflow.output.output.roller_bearings.output.total.friction.power_loss": {
        "P_Name": "P_LAGER_REIBUNGSVERLUST",
        "Komponente": "C_LAGER",
        "Einheit": "W",
        "Kategorie": "Dynamik",
        "Begruendung": "Gesamte Lagerreibungsverluste – Waermeentwicklung und Wirkungsgrad",
    },
    "RotorWorkflow.output.output.roller_bearings.output.DE.results.bearing_rating_life.basic_rating_life_in_hours": {
        "P_Name": "P_LAGER_DE_LEBENSDAUER",
        "Komponente": "C_LAGER",
        "Einheit": "h",
        "Kategorie": "Anforderung",
        "Begruendung": "Nominelle Lagerlebensdauer DE – Dimensionierungskriterium",
    },
    "RotorWorkflow.output.output.roller_bearings.output.NDE.results.bearing_rating_life.basic_rating_life_in_hours": {
        "P_Name": "P_LAGER_NDE_LEBENSDAUER",
        "Komponente": "C_LAGER",
        "Einheit": "h",
        "Kategorie": "Anforderung",
        "Begruendung": "Nominelle Lagerlebensdauer NDE – Dimensionierungskriterium",
    },
    "RotorWorkflow.output.output.roller_bearings.output.DE.results.static_safety_factor.static_load_safety": {
        "P_Name": "P_LAGER_DE_STAT_SICHERHEIT",
        "Komponente": "C_LAGER",
        "Einheit": "–",
        "Kategorie": "Struktur",
        "Begruendung": "Statische Tragfaehigkeitssicherheit DE – Schutz vor Stillstandsschaeden",
    },
    # --- Wellensicherheit (erweitert) ---
    "RotorWorkflow.output.output.shaft.output.shaft_safety.output.fatigue_strength._mean": {
        "P_Name": "P_WELLE_DAUERFESTIGKEIT_MITTEL",
        "Komponente": "C_WELLE",
        "Einheit": "–",
        "Kategorie": "Struktur",
        "Begruendung": "Mittlere Dauerfestigkeitssicherheit – Gesamtbild der Wellenauslegung",
    },
    "RotorWorkflow.output.output.shaft.output.shaft_safety.output.fatigue_strength._max": {
        "P_Name": "P_WELLE_DAUERFESTIGKEIT_MAX",
        "Komponente": "C_WELLE",
        "Einheit": "–",
        "Kategorie": "Struktur",
        "Begruendung": "Maximale Dauerfestigkeitssicherheit – Reserven an unkritischen Stellen",
    },
    "RotorWorkflow.output.output.shaft.output.shaft_safety.output.yield_strength._min": {
        "P_Name": "P_WELLE_STRECKGRENZE_MIN",
        "Komponente": "C_WELLE",
        "Einheit": "–",
        "Kategorie": "Struktur",
        "Begruendung": "Minimale Streckgrenzensicherheit – plastische Verformung vermeiden",
    },
    "RotorWorkflow.output.output.shaft.output.shaft_safety.output.yield_strength._mean": {
        "P_Name": "P_WELLE_STRECKGRENZE_MITTEL",
        "Komponente": "C_WELLE",
        "Einheit": "–",
        "Kategorie": "Struktur",
        "Begruendung": "Mittlere Streckgrenzensicherheit – Gesamtbild statische Sicherheit",
    },
    "RotorWorkflow.output.output.shaft.output.parallelkey_safety.output.safety.key.maximum_load": {
        "P_Name": "P_PK_SICHERHEIT_FEDER_MAX",
        "Komponente": "C_WELLE",
        "Einheit": "–",
        "Kategorie": "Struktur",
        "Begruendung": "Max. Passfedersicherheit (Feder) – Dimensionierung Welle-Nabe-Verbindung",
    },
    "RotorWorkflow.output.output.shaft.output.parallelkey_safety.output.safety.shaft.maximum_load": {
        "P_Name": "P_PK_SICHERHEIT_WELLE_MAX",
        "Komponente": "C_WELLE",
        "Einheit": "–",
        "Kategorie": "Struktur",
        "Begruendung": "Max. Passfedersicherheit (Welle) – Flaechenpressung Wellennut",
    },
    # --- Rotordynamik (erweitert) ---
    "RotorWorkflow.output.output.rotordynamics.output.modal_solution.eigenfrequencies.mode_3": {
        "P_Name": "P_ROTOR_EIGENFREQ_3",
        "Komponente": "C_ROTOR",
        "Einheit": "Hz",
        "Kategorie": "Dynamik",
        "Begruendung": "3. Eigenfrequenz – Campbell-Diagramm, hoehere Resonanzstellen",
    },
    "RotorWorkflow.output.output.rotordynamics.output.modal_solution.modal_damping.mode_1": {
        "P_Name": "P_ROTOR_DAEMPFUNG_1",
        "Komponente": "C_ROTOR",
        "Einheit": "%",
        "Kategorie": "Dynamik",
        "Begruendung": "Modale Daempfung Mode 1 – Stabilitaetskriterium (>0 = stabil)",
    },
    "RotorWorkflow.output.output.rotordynamics.output.modal_solution.modal_damping.mode_2": {
        "P_Name": "P_ROTOR_DAEMPFUNG_2",
        "Komponente": "C_ROTOR",
        "Einheit": "%",
        "Kategorie": "Dynamik",
        "Begruendung": "Modale Daempfung Mode 2 – Stabilitaetskriterium 2. Biegemode",
    },
    # --- Gleitlager ---
    "RotorWorkflow.output.sleeve_bearing_calculation.DE.c12._max": {
        "P_Name": "P_GLEITLAGER_DE_C12_MAX",
        "Komponente": "C_LAGER",
        "Einheit": "N/um",
        "Kategorie": "Dynamik",
        "Begruendung": "Kreuzsteifigkeit DE-Gleitlager – Stabilitaets-relevanter Koeffizient",
    },
    "RotorWorkflow.output.sleeve_bearing_calculation.NDE.c12._max": {
        "P_Name": "P_GLEITLAGER_NDE_C12_MAX",
        "Komponente": "C_LAGER",
        "Einheit": "N/um",
        "Kategorie": "Dynamik",
        "Begruendung": "Kreuzsteifigkeit NDE-Gleitlager – kann Instabilitaet (Oil Whirl) verursachen",
    },
    # --- Betriebsdaten ---
    "RotorWorkflow.input.operational_data.ambient_temperature": {
        "P_Name": "P_UMGEBUNGSTEMPERATUR",
        "Komponente": "C_ROTOR",
        "Einheit": "degC",
        "Kategorie": "Anforderung",
        "Begruendung": "Umgebungstemperatur – Einfluss auf Lagerung, Wicklung, Kuehlung",
    },
    "RotorWorkflow.input.load.bending.maximum": {
        "P_Name": "P_BIEGEBELASTUNG_MAX",
        "Komponente": "C_ROTOR",
        "Einheit": "Nm",
        "Kategorie": "Struktur",
        "Begruendung": "Maximale Biegebelastung – Wellendimensionierung",
    },
    "RotorWorkflow.input.load.torsion.mean": {
        "P_Name": "P_TORSION_MITTEL",
        "Komponente": "C_ROTOR",
        "Einheit": "Nm",
        "Kategorie": "Struktur",
        "Begruendung": "Mittlere Torsionsbelastung – Dauerfestigkeit",
    },
    "RotorWorkflow.input.load.torsion.amplitude": {
        "P_Name": "P_TORSION_AMPLITUDE",
        "Komponente": "C_ROTOR",
        "Einheit": "Nm",
        "Kategorie": "Dynamik",
        "Begruendung": "Torsionsamplitude – Wechselfestigkeit der Welle",
    },
    # --- Lagerpositionen ---
    "RotorWorkflow.output.output.form.output.shaft.bearing_positions.DE": {
        "P_Name": "P_LAGERPOSITION_DE",
        "Komponente": "C_WELLE",
        "Einheit": "mm",
        "Kategorie": "Geometrie",
        "Begruendung": "Axiale Position des DE-Lagers – Beeinflusst Biegemoment",
    },
    "RotorWorkflow.output.output.form.output.shaft.bearing_positions.NDE": {
        "P_Name": "P_LAGERPOSITION_NDE",
        "Komponente": "C_WELLE",
        "Einheit": "mm",
        "Kategorie": "Geometrie",
        "Begruendung": "Axiale Position des NDE-Lagers – Beeinflusst Biegemoment",
    },
    # --- Unterbau-Steifigkeit ---
    "segments.substructure.stiffness.horizontal": {
        "P_Name": "P_UNTERBAU_STEIFIGKEIT_H",
        "Komponente": "C_ROTOR",
        "Einheit": "N/m",
        "Kategorie": "Struktur",
        "Begruendung": "Horizontale Unterbausteifigkeit – kritisch fuer Rotordynamik",
    },
    "segments.substructure.stiffness.vertical": {
        "P_Name": "P_UNTERBAU_STEIFIGKEIT_V",
        "Komponente": "C_ROTOR",
        "Einheit": "N/m",
        "Kategorie": "Struktur",
        "Begruendung": "Vertikale Unterbausteifigkeit – kritisch fuer Rotordynamik",
    },
    # --- Segmente (aggregiert) ---
    "segments._total_count": {
        "P_Name": "P_SEGMENT_ANZAHL",
        "Komponente": "C_ROTOR",
        "Einheit": "–",
        "Kategorie": "Geometrie",
        "Begruendung": "Gesamtanzahl Segmente – Komplexitaet des Rotoraufbaus",
    },
    "segments.shaft.length._sum": {
        "P_Name": "P_WELLE_SEGMENTLAENGE_SUMME",
        "Komponente": "C_WELLE",
        "Einheit": "mm",
        "Kategorie": "Geometrie",
        "Begruendung": "Summe der Wellensegmentlaengen – Gesamte Wellenlaenge aus Segmenten",
    },
    # --- Rotor-Nutgeometrie ---
    "EdimWorkflow.input.rotor.core.slot.B1": {
        "P_Name": "P_ROTOR_NUT_B1",
        "Komponente": "C_AKTIVTEIL",
        "Einheit": "mm",
        "Kategorie": "Geometrie",
        "Begruendung": "Rotor-Nutoeffnung B1 – Einfluss auf Oberwellen, Geraeusch, Anlauf",
    },
    "EdimWorkflow.input.rotor.core.skewing_angle": {
        "P_Name": "P_ROTOR_SCHRAEGUNG",
        "Komponente": "C_AKTIVTEIL",
        "Einheit": "deg",
        "Kategorie": "Geometrie",
        "Begruendung": "Rotorschraegung – reduziert Nutoberwellen, beeinflusst Anlauf",
    },
    # --- Magnetkreis-Ergebnisse ---
    "EdimWorkflow.output.output.forces.radial_magnetic_spring_constant": {
        "P_Name": "P_MAGNETFEDER",
        "Komponente": "C_AKTIVTEIL",
        "Einheit": "N/mm",
        "Kategorie": "Struktur",
        "Begruendung": "Radiale Magnetfederkonstante – = P_AKTIV_MAGNETFEDER aus den bekannten 44",
    },
    # --- Windage/Friction ---
    "EdimWorkflow.input.rotor.windage_and_friction_losses": {
        "P_Name": "P_VENTILATIONSVERLUSTE",
        "Komponente": "C_ROTOR",
        "Einheit": "W",
        "Kategorie": "Dynamik",
        "Begruendung": "Ventilations-/Reibungsverluste – Kuehlung und Wirkungsgrad",
    },
    # --- Passungsgeometrie ---
    "segments.shaft_end.parallel_key.width": {
        "P_Name": "P_PK_BREITE",
        "Komponente": "C_WELLE",
        "Einheit": "mm",
        "Kategorie": "Geometrie",
        "Begruendung": "Passfederbreite – normiert nach DIN 6885",
    },
    "segments.shaft_end.parallel_key.height": {
        "P_Name": "P_PK_HOEHE",
        "Komponente": "C_WELLE",
        "Einheit": "mm",
        "Kategorie": "Geometrie",
        "Begruendung": "Passfederhoehe – bestimmt Flaechenpressung",
    },
    "segments.shaft_end.parallel_key.length": {
        "P_Name": "P_PK_LAENGE",
        "Komponente": "C_WELLE",
        "Einheit": "mm",
        "Kategorie": "Geometrie",
        "Begruendung": "Passfederlaenge – bestimmt tragende Laenge",
    },
    "segments.shaft_end.outer_diameter": {
        "P_Name": "P_WELLENENDE_D",
        "Komponente": "C_WELLE",
        "Einheit": "mm",
        "Kategorie": "Geometrie",
        "Begruendung": "Wellenenddurchmesser – Abtriebsseite",
    },
    "segments.shaft_end.length": {
        "P_Name": "P_WELLENENDE_LAENGE",
        "Komponente": "C_WELLE",
        "Einheit": "mm",
        "Kategorie": "Geometrie",
        "Begruendung": "Wellenende-Laenge – normiert nach IEC/DIN",
    },
}

# ---------------------------------------------------------------------------
# Identifikatoren / Teilenummern -> entfernen
# ---------------------------------------------------------------------------
IDENTIFIKATOREN = {
    "machine_id",
    "MachineConfig.mlfb",
    "MachineConfig.shaft",
    "MachineConfig.bearing",
    "MachineConfig.enclosure",
    "MachineConfig.endShieldDE",
    "MachineConfig.endShieldNDE",
    "MachineConfig.bearingInsertDE",
    "MachineConfig.bearingInsertNDE",
    "MachineConfig.pressurePlateDE",
    "MachineConfig.pressurePlateNDE",
    "MachineConfig.frame",
    "MachineConfig.cDimension",
    "MachineConfig.shaftEnd",
    "MachineConfig.balancingDiscDE",
    "MachineConfig.balancingDiscNDE",
    "MachineConfig.baffleHolderDE",
    "MachineConfig.baffleHolderNDE",
    "MachineConfig.cBaffleDE",
    "MachineConfig.cBaffleNDE",
    "MachineConfig.cylinderBaffleDE",
    "MachineConfig.cylinderBaffleNDE",
    "MachineConfig.sBaffleDE",
    "MachineConfig.sBaffleNDE",
    "MachineConfig.axialFanInnerDE",
    "MachineConfig.axialFanInnerNDE",
    "MachineConfig.axialFanOuter",
    "MachineConfig.radialInnerFans",
    "MachineConfig.radialOuterFans",
    "MachineConfig.airInletHousing",
    "MachineConfig.airInletSilencer",
    "MachineConfig.terminalBox",
    "MachineConfig.catalogueDimensionDrawing",
    "MachineConfig.position",
    "MachineConfig.ndeLength",
}


# ---------------------------------------------------------------------------
# Bereinigungslogik
# ---------------------------------------------------------------------------
def bereinige_parameter(df_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Bereinigt die Analyse-Ergebnisse:
    1. Entfernt SimocalcWorkflow-Duplikate
    2. Entfernt Konstanten (Unique = 1)
    3. Entfernt Identifikatoren
    4. Weist fachliche Relevanz zu
    """
    n_gesamt = len(df_stats)
    print(f"\n[7/8] Bereinigung ({n_gesamt} Parameter) ...\n")

    # 1. Simocalc-Duplikate
    edim_suffixe = set()
    for p in df_stats["Parameter"]:
        if p.startswith("EdimWorkflow."):
            suffix = p[len("EdimWorkflow.") :]
            edim_suffixe.add(suffix)

    simocalc_duplikate = set()
    for p in df_stats["Parameter"]:
        if p.startswith("SimocalcWorkflow."):
            suffix = p[len("SimocalcWorkflow.") :]
            if suffix in edim_suffixe:
                simocalc_duplikate.add(p)

    df = df_stats[~df_stats["Parameter"].isin(simocalc_duplikate)].copy()
    print(
        f"      {len(simocalc_duplikate)} SimocalcWorkflow-Duplikate entfernt "
        f"(identische Pfade in EdimWorkflow vorhanden)."
    )
    print(f"      Verbleibend: {len(df)} Parameter.\n")

    # 2. Konstanten entfernen (Unique = 1)
    unique_col = df["Unique"]
    konstanten_mask = unique_col.apply(lambda x: str(x).strip() in ("1", "1.0"))
    n_konstanten = konstanten_mask.sum()
    df = df[~konstanten_mask].copy()
    print(f"      {n_konstanten} Konstanten entfernt (Unique = 1).")
    print(f"      Verbleibend: {len(df)} Parameter.\n")

    # 3. Identifikatoren entfernen
    ident_mask = df["Parameter"].isin(IDENTIFIKATOREN)
    n_ident = ident_mask.sum()
    df = df[~ident_mask].copy()
    print(f"      {n_ident} Identifikatoren/Teilenummern entfernt.")
    print(f"      Verbleibend: {len(df)} Parameter.\n")

    # 4. Fachliche Relevanz zuweisen
    bekannt_set = set(BEKANNTE_PARAMETER.keys())
    neu_set = set(NEUE_RELEVANTE.keys())

    relevanz_liste = []
    p_name_liste = []
    komponente_liste = []
    einheit_liste = []
    kategorie_liste = []
    begruendung_liste = []

    for _, row in df.iterrows():
        param = row["Parameter"]

        if param in bekannt_set:
            info = BEKANNTE_PARAMETER[param]
            if "_duplikat_von" in info:
                relevanz_liste.append("Entfernt (internes Duplikat)")
                p_name_liste.append(info["P_Name"])
                komponente_liste.append(info["Komponente"])
                einheit_liste.append(info["Einheit"])
                kategorie_liste.append(info["Kategorie"])
                begruendung_liste.append(f"Duplikat von {info['_duplikat_von']}")
            else:
                relevanz_liste.append("Bekannt (44)")
                p_name_liste.append(info["P_Name"])
                komponente_liste.append(info["Komponente"])
                einheit_liste.append(info["Einheit"])
                kategorie_liste.append(info["Kategorie"])
                begruendung_liste.append("Aus der kuratierten Parameterliste")
        elif param in neu_set:
            info = NEUE_RELEVANTE[param]
            relevanz_liste.append("Neu (fachlich relevant)")
            p_name_liste.append(info["P_Name"])
            komponente_liste.append(info["Komponente"])
            einheit_liste.append(info["Einheit"])
            kategorie_liste.append(info["Kategorie"])
            begruendung_liste.append(info["Begruendung"])
        else:
            relevanz_liste.append("Nicht zugeordnet")
            p_name_liste.append("–")
            komponente_liste.append("–")
            einheit_liste.append("–")
            kategorie_liste.append("–")
            begruendung_liste.append("–")

    df = df.copy()
    df.insert(0, "Fachliche_Relevanz", relevanz_liste)
    df.insert(1, "P_Name", p_name_liste)
    df.insert(2, "Komponente", komponente_liste)
    df.insert(3, "Einheit_", einheit_liste)
    df.insert(4, "Fachkategorie", kategorie_liste)
    df["Begruendung"] = begruendung_liste

    # Interne Duplikate entfernen
    df = df[df["Fachliche_Relevanz"] != "Entfernt (internes Duplikat)"].copy()

    # Statistik ausgeben
    n_bekannt = (df["Fachliche_Relevanz"] == "Bekannt (44)").sum()
    n_neu = (df["Fachliche_Relevanz"] == "Neu (fachlich relevant)").sum()
    n_rest = (df["Fachliche_Relevanz"] == "Nicht zugeordnet").sum()

    print("      Fachliche Kategorisierung:")
    print(f"        {n_bekannt:>4} Parameter den bekannten 44 zugeordnet")
    print(f"        {n_neu:>4} NEUE fachlich relevante Parameter identifiziert")
    print(f"        {n_rest:>4} nicht zugeordnet (verbleibend)")
    print(f"        {'─' * 30}")
    print(f"        {len(df):>4} Parameter in der bereinigten CSV\n")

    # Neue Parameter auflisten
    print("      --- Neu identifizierte fachlich relevante Parameter ---\n")
    neu_df = df[df["Fachliche_Relevanz"] == "Neu (fachlich relevant)"][
        [
            "P_Name",
            "Fachkategorie",
            "Einheit_",
            "Abdeckung (%)",
            "Unique",
            "Eignung",
            "Begruendung",
        ]
    ].sort_values("Eignung", ascending=False)
    for _, row in neu_df.iterrows():
        print(
            f"        {row['P_Name']:<40s} [{row['Fachkategorie']:<10s}] "
            f"Eignung={row['Eignung']:>5}  {row['Einheit_']:<6s}  "
            f"Abdeckung={row['Abdeckung (%)']:>5}%  Unique={row['Unique']}"
        )
        print(f"          -> {row['Begruendung']}")

    # Zusammenfassung
    print("\n      --- Zusammenfassung Bereinigung ---\n")
    print(f"        Ausgangsbasis:                {n_gesamt:>5} Parameter")
    print(f"        - SimocalcWorkflow-Duplikate: {len(simocalc_duplikate):>5}")
    print(f"        - Konstanten (Unique=1):      {n_konstanten:>5}")
    print(f"        - Identifikatoren/Teilenr.:   {n_ident:>5}")
    print(f"        - Interne Duplikate:          {1:>5}")
    print(f"        {'=' * 45}")
    print(f"        Bereinigte Parameterliste:    {len(df):>5} Parameter\n")

    # Sortieren
    sort_order = {"Bekannt (44)": 0, "Neu (fachlich relevant)": 1, "Nicht zugeordnet": 2}
    df["_sort"] = df["Fachliche_Relevanz"].map(sort_order)
    df = df.sort_values(["_sort", "Eignung"], ascending=[True, False]).drop(columns=["_sort"])

    return df


def exportiere_bereinigte_csv(df: pd.DataFrame):
    """Exportiert die bereinigte CSV (Dezimalkomma)."""
    pfad = AUSGABE_VERZEICHNIS / "parameter_bereinigt.csv"
    df_export = df.copy()

    num_cols = [
        "Abdeckung (%)",
        "Min",
        "Max",
        "Mittelwert",
        "Median",
        "Std.Abw.",
        "CV (%)",
        "IQR",
        "Schiefe",
        "Eignung",
    ]
    for col in num_cols:
        if col in df_export.columns:
            df_export[col] = df_export[col].apply(_fmt_komma)

    if "Entropie (bit)" in df_export.columns:
        df_export["Entropie (bit)"] = df_export["Entropie (bit)"].apply(
            lambda x: str(x).replace(".", ",") if str(x) != "–" else x
        )

    df_export.to_csv(pfad, index=False, sep=";", encoding="utf-8-sig")
    print(f"      -> CSV: {pfad.name}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print()
    print("+" + "=" * 62 + "+")
    print("|   Datenanalyse und Bereinigung der WVSC-Realdaten           |")
    print("+" + "=" * 62 + "+")
    print()
    print(f"  Eingabe: {WVSC_VERZEICHNIS}")
    print(f"  Ausgabe: {AUSGABE_VERZEICHNIS}")
    print()

    # --- Teil 1: Analyse ---
    df, n_rotoren = lade_alle_parameter()

    print("[2/8] Berechne Statistiken ...\n")
    df_stats = analysiere_parameter(df, n_rotoren)

    print("[3/8] Bewerte Eignung ...\n")
    df_stats = bewerte_eignung(df_stats, n_rotoren)

    drucke_tabellen(df_stats, n_rotoren)
    erstelle_visualisierungen(df_stats, n_rotoren)

    print("\n[6/8] Exportiere Analyse-CSV ...")
    exportiere_analyse_csv(df_stats)

    # --- Teil 2: Bereinigung ---
    df_bereinigt = bereinige_parameter(df_stats)

    print("[8/8] Exportiere bereinigte CSV ...")
    exportiere_bereinigte_csv(df_bereinigt)

    print(f"\n{'=' * 62}")
    print("  Analyse und Bereinigung abgeschlossen.")
    print(f"{'=' * 62}\n")


if __name__ == "__main__":
    main()
