"""
Datenanalyse der realen WVSC-Rotordaten – ALLE Parameter.

Analysiert rekursiv ALLE Felder aus den JSON-Dateien, nicht nur die
44 vom json_parser ausgewählten. Umfasst auch SimocalcWorkflow,
sleeve_bearing_calculation, Segment-Details, Material-Eigenschaften etc.

Ausgabe: Konsolentabellen + Visualisierungen als PNG + CSV
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
WVSC_VERZEICHNIS = Path(__file__).resolve().parent / "data" / "real_data" / "wvsc"
AUSGABE_VERZEICHNIS = Path(__file__).resolve().parent / "data" / "real_data" / "analyse"
AUSGABE_VERZEICHNIS.mkdir(parents=True, exist_ok=True)

# Pfade, die übersprungen werden (Meta-Daten, Versions-Infos, leere Felder)
SKIP_PATTERNS = [
    "module_info",  # Software-Versionierung, nicht rotorrelevant
    "OrderData",  # Projektdaten (IDs, Angebote)
    "additional_data",  # Interne TRA-Referenzen
    "tag",  # Interne Tags
    "machine_template",
    "username",
    "created",
    "last_updated",
    "status_code",  # Workflow-Statuscodes
    "versions.",  # Simocalc Output-Versionen (alle None)
    "calculation.identifier",
    "calculation.remarks",
    "calculation.user",
    "calculation.time",
    "info.pid",
    "info.tra_suffix",
    "info.mlfb",  # Simocalc MLFB (Duplikat)
    "info.tra",
]


def _sollte_uebersprungen_werden(pfad: str) -> bool:
    """Prüft ob ein JSON-Pfad übersprungen werden soll."""
    for pattern in SKIP_PATTERNS:
        if pattern in pfad:
            return True
    return False


# ---------------------------------------------------------------------------
# 1. JSON-Dateien laden und rekursiv alle Parameter extrahieren
# ---------------------------------------------------------------------------
def _flatten_json(obj: dict | list, prefix: str = "") -> dict[str, object]:
    """
    Flattened ein verschachteltes JSON-Objekt rekursiv.
    Gibt dict {pfad: wert} für alle Blattknoten zurück.
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

    # Rotor-Segmente: nach Typ aggregieren
    if pfad.endswith("rotor.segments"):
        _verarbeite_segmente(lst, ergebnis)
        return

    # Shaft-Safety und Parallelkey: aggregierte Werte
    if "shaft_safety.output" in pfad or "parallelkey_safety.output" in pfad:
        _verarbeite_sicherheitsliste(lst, pfad, ergebnis)
        return

    # Roller-Bearing DE/NDE Listen
    if "roller_bearings.output.DE" in pfad or "roller_bearings.output.NDE" in pfad:
        _verarbeite_lagerliste(lst, pfad, ergebnis)
        return

    # Relubrication masses (skip)
    if "relubrication.masses" in pfad:
        return

    # Numerische Listen → Aggregation (Sleeve-Bearing, Speeds, etc.)
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

    # Listen von Dicts (z.B. characteristics)
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

        # Numerische Felder aggregieren
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

        # Kategorische Felder
        for feld in ["description", "designation"]:
            werte = [seg.get(feld) for seg in gruppe if seg.get(feld)]
            if werte:
                ergebnis[f"{basis}.{feld}"] = (
                    werte[0] if len(werte) == 1 else "; ".join(str(v) for v in werte[:3])
                )

        # Parallel key
        pk_segmente = [seg for seg in gruppe if "parallel_key" in seg]
        if pk_segmente:
            pk = pk_segmente[0]["parallel_key"]
            for k, v in pk.items():
                if v is not None:
                    ergebnis[f"{basis}.parallel_key.{k}"] = v

        # Stiffness (z.B. bei substructure)
        stiff_segmente = [seg for seg in gruppe if "stiffness" in seg]
        if stiff_segmente:
            stiff = stiff_segmente[0]["stiffness"]
            for k, v in stiff.items():
                if v is not None:
                    ergebnis[f"{basis}.stiffness.{k}"] = v

        # Shoulder
        shoulder_segmente = [seg for seg in gruppe if "shoulder" in seg]
        if shoulder_segmente:
            sh = shoulder_segmente[0].get("shoulder", {})
            for pos, details in sh.items():
                if isinstance(details, dict):
                    for k, v in details.items():
                        if v is not None:
                            ergebnis[f"{basis}.shoulder.{pos}.{k}"] = v

        # Functions
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


def lade_alle_parameter() -> tuple[pd.DataFrame, int]:
    """Lädt alle JSON-Dateien und extrahiert rekursiv alle Parameter."""
    print("[1/6] Lade und parse alle JSON-Dateien ...")

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
# 2. Numerisch vs. kategorisch klassifizieren
# ---------------------------------------------------------------------------
def klassifiziere_parameter(df: pd.DataFrame) -> dict[str, str]:
    """Bestimmt pro Parameter ob numerisch oder kategorisch."""
    klassifikation = {}
    for param in df["parameter"].unique():
        werte = df.loc[df["parameter"] == param, "value"].dropna()
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
# 3. Entropie
# ---------------------------------------------------------------------------
def berechne_entropie(werte: list) -> float:
    """Shannon-Entropie (log2)."""
    if not werte:
        return 0.0
    counts = Counter(werte)
    n = len(werte)
    return -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)


# ---------------------------------------------------------------------------
# 4. Bereich aus Pfad ableiten
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
        return "Wälzlager (Output)"
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
        return "Oberfläche"
    if p.startswith("rotorworkflow.input"):
        return "Rotor (Input)"
    if p.startswith("rotorworkflow.output"):
        return "Rotor (Output)"
    if p.startswith("machine_id"):
        return "Meta"
    return "Sonstige"


# ---------------------------------------------------------------------------
# 5. Statistische Analyse
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
                        "Häufigster Wert": "–",
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
                            "Häufigster Wert",
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
                    "Häufigster Wert": f"{str(haeufigster[0])[:40]} ({haeufigster[1]}x)",
                    "Entropie (bit)": f"{entropie:.2f} / {max_entropie:.2f}",
                }
            )

        ergebnisse.append(eintrag)

    return pd.DataFrame(ergebnisse)


# ---------------------------------------------------------------------------
# 6. Eignung bewerten
# ---------------------------------------------------------------------------
def bewerte_eignung(df_stats: pd.DataFrame, n_rotoren: int) -> pd.DataFrame:
    """
    Bewertet jeden Parameter 0–100 hinsichtlich Eignung für
    Ähnlichkeitsanalysen.

    Die Abdeckung (Coverage) wird bewusst NICHT in die Eignung einbezogen,
    da sie eine Eigenschaft der Datenverfügbarkeit ist, nicht der
    Unterscheidungskraft eines Parameters. Ein Parameter, der nur bei 65
    Rotoren vorkommt, aber dort 65 verschiedene Werte hat, ist wertvoller
    als einer, der bei allen 230 vorkommt, aber immer gleich ist.

    Kriterien (gewichtet):
    - Unique       (50 %): Entscheidendes Kriterium. Die Anzahl
      verschiedener Werte bestimmt direkt, wie fein ein Parameter
      Rotoren voneinander unterscheiden kann. Ein Parameter mit nur
      1 Unique-Wert kann keine Ähnlichkeit messen – egal wie gut
      die anderen Metriken sind. Log-skaliert, damit der Sprung
      von 1→10 stärker zählt als von 100→110.
    - Variabilität (35 %): Zweitwichtigstes Kriterium. Selbst bei
      vielen Unique-Werten kann die Streuung gering sein (z. B.
      99 % der Werte bei 0, ein Ausreißer bei 1000). CV (numerisch)
      bzw. normierte Entropie (kategorisch) messen, wie gleichmäßig
      die Werte verteilt sind – also die tatsächliche Trennschärfe.
    - Verteilung   (15 %): Qualitätskriterium. Stark schiefe
      Verteilungen verzerren Distanzmaße (Euklidisch, Manhattan)
      und können Ähnlichkeitsberechnungen dominieren. Symmetrische
      Verteilungen liefern robustere Ergebnisse.
    """
    scores = []
    for _, row in df_stats.iterrows():
        # --- Variabilität (35 %) ---
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

        # --- Unique (50 %) ---
        if row["Unique"] != "–":
            n_u = int(row["Unique"])
            s_unique = (
                min(math.log2(n_u) / math.log2(max(n_rotoren, 2)) * 100, 100.0) if n_u > 1 else 0.0
            )
        else:
            s_unique = 0.0

        # --- Verteilung (15 %) ---
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
# 7. Visualisierungen
# ---------------------------------------------------------------------------
def erstelle_visualisierungen(df_stats: pd.DataFrame, n_rotoren: int):
    """Erzeugt alle Analysegrafiken."""
    print("[5/6] Erstelle Visualisierungen ...\n")

    plt.rcParams.update(
        {
            "font.size": 8,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.3,
        }
    )

    _plot_abdeckung_nach_bereich(df_stats, n_rotoren)
    _plot_eignungs_top(df_stats)
    _plot_bereich_uebersicht(df_stats)
    _plot_abdeckung_verteilung(df_stats, n_rotoren)
    _plot_datentyp_verteilung(df_stats)
    _plot_cv_verteilung(df_stats)


def _plot_abdeckung_nach_bereich(df_stats: pd.DataFrame, n_rotoren: int):
    """Boxplot: Abdeckung nach Bereich."""
    fig, ax = plt.subplots(figsize=(12, 7))

    bereiche = df_stats.groupby("Bereich")["Abdeckung (%)"].apply(list).to_dict()
    bereiche_sorted = sorted(bereiche.keys(), key=lambda b: np.median(bereiche[b]), reverse=True)

    daten = [bereiche[b] for b in bereiche_sorted]
    labels = [f"{b}\n(n={len(bereiche[b])})" for b in bereiche_sorted]

    bp = ax.boxplot(daten, vert=False, patch_artist=True, widths=0.6)
    farben = plt.cm.Set3(np.linspace(0, 1, len(bereiche_sorted)))
    for patch, farbe in zip(bp["boxes"], farben):
        patch.set_facecolor(farbe)
        patch.set_alpha(0.8)

    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Abdeckung (%)")
    ax.set_title(
        f"Parameterabdeckung nach Bereich ({n_rotoren} Rotoren)", fontsize=12, fontweight="bold"
    )
    ax.axvline(100, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    pfad = AUSGABE_VERZEICHNIS / "01_abdeckung_bereiche.png"
    fig.savefig(pfad)
    plt.close(fig)
    print(f"      -> {pfad.name}")


def _plot_eignungs_top(df_stats: pd.DataFrame):
    """Top-50 und Bottom-20 Parameter nach Eignungs-Score."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 14))

    # Top 50
    top = df_stats.head(50).sort_values("Eignung")
    cmap = plt.cm.RdYlGn
    norm = plt.Normalize(0, 100)
    farben = [cmap(norm(s)) for s in top["Eignung"]]

    labels_top = []
    for _, r in top.iterrows():
        p = r["Parameter"]
        short = p.split(".")[-1][:30]
        labels_top.append(f"{short} [{r['Bereich'][:8]}]")

    axes[0].barh(range(len(top)), top["Eignung"].values, color=farben, edgecolor="white")
    axes[0].set_yticks(range(len(top)))
    axes[0].set_yticklabels(labels_top, fontsize=6)
    axes[0].set_xlabel("Eignungs-Score")
    axes[0].set_title("Top 50 Parameter", fontsize=11, fontweight="bold")
    axes[0].set_xlim(0, 105)

    for i, score in enumerate(top["Eignung"].values):
        axes[0].text(score + 0.5, i, f"{score:.0f}", va="center", fontsize=5)

    # Bottom 20
    bottom = df_stats.tail(20).sort_values("Eignung")
    farben_b = [cmap(norm(s)) for s in bottom["Eignung"]]

    labels_bot = []
    for _, r in bottom.iterrows():
        p = r["Parameter"]
        short = p.split(".")[-1][:30]
        labels_bot.append(f"{short} [{r['Bereich'][:8]}]")

    axes[1].barh(range(len(bottom)), bottom["Eignung"].values, color=farben_b, edgecolor="white")
    axes[1].set_yticks(range(len(bottom)))
    axes[1].set_yticklabels(labels_bot, fontsize=7)
    axes[1].set_xlabel("Eignungs-Score")
    axes[1].set_title("Bottom 20 Parameter", fontsize=11, fontweight="bold")
    axes[1].set_xlim(0, 105)

    for i, score in enumerate(bottom["Eignung"].values):
        axes[1].text(score + 0.5, i, f"{score:.0f}", va="center", fontsize=6)

    plt.tight_layout()
    pfad = AUSGABE_VERZEICHNIS / "02_eignungs_top_bottom.png"
    fig.savefig(pfad)
    plt.close(fig)
    print(f"      -> {pfad.name}")


def _plot_bereich_uebersicht(df_stats: pd.DataFrame):
    """Parameter-Anzahl und Ø Eignung pro Bereich."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    counts = df_stats["Bereich"].value_counts().sort_values()
    axes[0].barh(counts.index, counts.values, color="#3498db", edgecolor="white")
    axes[0].set_xlabel("Anzahl Parameter")
    axes[0].set_title("Parameter pro Bereich", fontweight="bold")
    for i, (_, val) in enumerate(zip(counts.index, counts.values)):
        axes[0].text(val + 0.5, i, str(val), va="center", fontsize=9)

    mean_scores = df_stats.groupby("Bereich")["Eignung"].mean().sort_values()
    farben = plt.cm.RdYlGn(plt.Normalize(0, 100)(mean_scores.values))
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
        stufen.values, labels=stufen.index, autopct="%1.0f%%", colors=c, textprops={"fontsize": 9}
    )
    axes[1].set_title("Abdeckungsstufen", fontweight="bold")

    plt.tight_layout()
    pfad = AUSGABE_VERZEICHNIS / "05_datentyp_abdeckung.png"
    fig.savefig(pfad)
    plt.close(fig)
    print(f"      -> {pfad.name}")


def _plot_cv_verteilung(df_stats: pd.DataFrame):
    """Histogramm: CV-Verteilung der numerischen Parameter."""
    num_df = df_stats[(df_stats["Datentyp"] == "numerisch") & (df_stats["CV (%)"] != "–")].copy()
    if num_df.empty:
        return
    num_df["CV_float"] = num_df["CV (%)"].astype(float)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        num_df["CV_float"].clip(upper=200), bins=40, color="#9b59b6", edgecolor="white", rwidth=0.9
    )
    ax.set_xlabel("Variationskoeffizient CV (%)")
    ax.set_ylabel("Anzahl Parameter")
    ax.set_title(
        f"Verteilung des Variationskoeffizienten ({len(num_df)} numerische Parameter)",
        fontsize=11,
        fontweight="bold",
    )
    ax.axvline(50, color="red", linestyle="--", alpha=0.5, label="CV = 50 %")
    ax.legend()

    plt.tight_layout()
    pfad = AUSGABE_VERZEICHNIS / "06_cv_verteilung.png"
    fig.savefig(pfad)
    plt.close(fig)
    print(f"      -> {pfad.name}")


# ---------------------------------------------------------------------------
# 8. Konsolenausgabe
# ---------------------------------------------------------------------------
def drucke_tabellen(df_stats: pd.DataFrame, n_rotoren: int):
    """Druckt die Ergebnisse tabellarisch."""
    print("[4/6] Ergebnisse:\n")

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
    top_cols = ["Parameter", "Bereich", "Datentyp", "Abdeckung (%)", "CV (%)", "Unique", "Eignung"]
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
            ["Parameter", "Bereich", "Abdeckung (%)", "Min", "Max", "CV (%)", "Unique", "Eignung"]
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
            kat_df.rename(columns={"Häufigster Wert": "Haeufigster Wert"})[kat_cols]
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
# 9. CSV-Export
# ---------------------------------------------------------------------------
def _fmt_komma(val):
    """Dezimalpunkt → Dezimalkomma für numerische Werte."""
    if isinstance(val, float):
        return str(val).replace(".", ",")
    return val


def exportiere_csv(df_stats: pd.DataFrame):
    """Exportiert die Ergebnisse als CSV (Dezimalkomma für Google Docs)."""
    pfad = AUSGABE_VERZEICHNIS / "parameter_analyse_alle.csv"
    df_export = df_stats.copy()

    # Spalten mit gemischten Typen (float / "–"): manuell konvertieren
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

    # Entropie (String "3.45 / 4.81"): Punkt → Komma
    if "Entropie (bit)" in df_export.columns:
        df_export["Entropie (bit)"] = df_export["Entropie (bit)"].apply(
            lambda x: str(x).replace(".", ",") if str(x) != "–" else x
        )

    df_export.to_csv(pfad, index=False, sep=";", encoding="utf-8-sig")
    print(f"\n      -> CSV: {pfad.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print()
    print("+" + "=" * 58 + "+")
    print("|   Datenanalyse ALLER Parameter aus WVSC-Realdaten        |")
    print("+" + "=" * 58 + "+")
    print()

    df, n_rotoren = lade_alle_parameter()

    print("[2/6] Berechne Statistiken ...\n")
    df_stats = analysiere_parameter(df, n_rotoren)

    print("[3/6] Bewerte Eignung ...\n")
    df_stats = bewerte_eignung(df_stats, n_rotoren)

    drucke_tabellen(df_stats, n_rotoren)
    erstelle_visualisierungen(df_stats, n_rotoren)

    print("\n[6/6] Exportiere Ergebnisse ...")
    exportiere_csv(df_stats)

    print("\n>> Analyse abgeschlossen.\n")


if __name__ == "__main__":
    main()
