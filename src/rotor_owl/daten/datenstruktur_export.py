"""
Exportiert die reale Parameterstruktur als LaTeX-Tabelle.

Erzeugt eine LaTeX-Datei mit allen 44 Parametern: Name, Typ,
Einheit, Komponente, Kategorie und einem Beispielwert aus dem
ersten geladenen Rotor. Für den Einsatz im Evaluationskapitel
der Bachelorarbeit.

Ausgabe: latex/bilder/data/parameterstruktur.tex
"""

from __future__ import annotations

import sys
from pathlib import Path

# Projekt-Root zum Import-Pfad hinzufügen
PROJEKT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJEKT_ROOT / "src"))

from rotor_owl.daten.json_parser import (  # noqa: E402
    fetch_all_features_from_json,
    MAPPING_WELLE_LAENGE,
    MAPPING_WELLE_MASSE,
    MAPPING_WELLE_MATERIAL,
    MAPPING_WELLE_DREHZAHL,
    MAPPING_WELLE_TORSIONSSTEIFIGKEIT,
    MAPPING_WELLE_KUPPLUNGSMASSE,
    MAPPING_WELLE_VOLUMEN,
    MAPPING_WELLE_ZERSPANUNGSRATE,
    MAPPING_WELLE_TRAEGHEITSMOMENT,
    MAPPING_WELLE_MIN_SICHERHEIT,
    MAPPING_WELLE_PK_DREHMOMENT,
    MAPPING_AKTIV_LAENGE,
    MAPPING_AKTIV_D_AUSSEN,
    MAPPING_AKTIV_D_INNEN,
    MAPPING_AKTIV_MASSE,
    MAPPING_AKTIV_MAGNETFEDER,
    MAPPING_AKTIV_AXIALKRAFT,
    MAPPING_LUEFTER_MASSE,
    MAPPING_LUEFTER_D,
    MAPPING_LUEFTER_J,
    MAPPING_LUEFTER_ANZAHL,
    MAPPING_ROTOR_GESAMTMASSE,
    MAPPING_ROTOR_NENNMOMENT,
    MAPPING_ROTOR_AXIALLAST,
    MAPPING_ROTOR_MAX_TORSION,
    MAPPING_ROTOR_POLZAHL,
    MAPPING_ROTOR_LAGERTYP,
    MAPPING_ROTOR_BAUFORM,
    MAPPING_ROTOR_C_MASS,
    MAPPING_ROTOR_EIGENFREQ_1,
    MAPPING_ROTOR_EIGENFREQ_2,
    MAPPING_ROTOR_MAX_BIEGUNG,
    MAPPING_ROTOR_BIEGUNG_KERN,
    MAPPING_ROTOR_LAGERABSTAND,
    MAPPING_ROTOR_KERNMITTE,
    MAPPING_LAGER_BEZEICHNUNG,
    MAPPING_LAGER_BAUREIHE,
    MAPPING_LAGER_TYP_DETAIL,
    MAPPING_LAGER_INNER_D,
    MAPPING_LAGER_OUTER_D,
    MAPPING_LAGER_DYN_TRAGZAHL,
    MAPPING_LAGER_STAT_TRAGZAHL,
    MAPPING_LAGER_LEBENSDAUER,
    MAPPING_ROTOR_MLFB,
)
from rotor_owl.config.kategorien import PTYPE_TO_KATEGORIE_3, KATEGORIE_LABEL  # noqa: E402

# Geordnete Liste aller Mappings (Reihenfolge = Tabellenreihenfolge)
ALLE_MAPPINGS = [
    # Welle
    MAPPING_WELLE_LAENGE,
    MAPPING_WELLE_MASSE,
    MAPPING_WELLE_MATERIAL,
    MAPPING_WELLE_DREHZAHL,
    MAPPING_WELLE_TORSIONSSTEIFIGKEIT,
    MAPPING_WELLE_KUPPLUNGSMASSE,
    MAPPING_WELLE_VOLUMEN,
    MAPPING_WELLE_ZERSPANUNGSRATE,
    MAPPING_WELLE_TRAEGHEITSMOMENT,
    MAPPING_WELLE_MIN_SICHERHEIT,
    MAPPING_WELLE_PK_DREHMOMENT,
    # Aktivteil
    MAPPING_AKTIV_LAENGE,
    MAPPING_AKTIV_D_AUSSEN,
    MAPPING_AKTIV_D_INNEN,
    MAPPING_AKTIV_MASSE,
    MAPPING_AKTIV_MAGNETFEDER,
    MAPPING_AKTIV_AXIALKRAFT,
    # Lüfter
    MAPPING_LUEFTER_MASSE,
    MAPPING_LUEFTER_D,
    MAPPING_LUEFTER_J,
    MAPPING_LUEFTER_ANZAHL,
    # Rotor (Gesamtsystem)
    MAPPING_ROTOR_GESAMTMASSE,
    MAPPING_ROTOR_NENNMOMENT,
    MAPPING_ROTOR_AXIALLAST,
    MAPPING_ROTOR_MAX_TORSION,
    MAPPING_ROTOR_POLZAHL,
    MAPPING_ROTOR_LAGERTYP,
    MAPPING_ROTOR_BAUFORM,
    MAPPING_ROTOR_C_MASS,
    MAPPING_ROTOR_MLFB,
    # Rotordynamik
    MAPPING_ROTOR_EIGENFREQ_1,
    MAPPING_ROTOR_EIGENFREQ_2,
    MAPPING_ROTOR_MAX_BIEGUNG,
    MAPPING_ROTOR_BIEGUNG_KERN,
    MAPPING_ROTOR_LAGERABSTAND,
    MAPPING_ROTOR_KERNMITTE,
    # Lager
    MAPPING_LAGER_BEZEICHNUNG,
    MAPPING_LAGER_BAUREIHE,
    MAPPING_LAGER_TYP_DETAIL,
    MAPPING_LAGER_INNER_D,
    MAPPING_LAGER_OUTER_D,
    MAPPING_LAGER_DYN_TRAGZAHL,
    MAPPING_LAGER_STAT_TRAGZAHL,
    MAPPING_LAGER_LEBENSDAUER,
]

# Lesbarer Komponenten-Name
KOMPONENTEN_LABEL = {
    "C_WELLE": "Welle",
    "C_AKTIVTEIL": "Aktivteil",
    "C_LUEFTER": "Lüfter",
    "C_ROTOR": "Rotor",
    "C_LAGER": "Lager",
}

# Lesbarer Parameter-Name (Kurzform für Tabelle)
PARAMETER_LABEL = {
    "P_WELLE_LAENGE": "Wellenlänge",
    "P_WELLE_MASSE": "Wellenmasse",
    "P_WELLE_MATERIAL": "Wellenmaterial",
    "P_WELLE_DREHZAHLBEREICH": "Betriebsdrehzahl",
    "P_WELLE_TORSIONSSTEIFIGKEIT": "Torsionssteifigkeit",
    "P_WELLE_KUPPLUNGSMASSE": "Kupplungsmasse",
    "P_WELLE_VOLUMEN": "Wellenvolumen",
    "P_WELLE_ZERSPANUNGSRATE": "Zerspanungsrate",
    "P_WELLE_TRAEGHEITSMOMENT": "Trägheitsmoment",
    "P_WELLE_MIN_SICHERHEIT": "Min. Sicherheitsfaktor",
    "P_WELLE_PK_DREHMOMENT": "PK-Drehmoment",
    "P_AKTIV_LAENGE": "Kernlänge",
    "P_AKTIV_D_AUSSEN": "Kern-Außendurchmesser",
    "P_AKTIV_D_INNEN": "Kern-Innendurchmesser",
    "P_AKTIV_MASSE": "Kernmasse",
    "P_AKTIV_MAGNETFEDER": "Magnetfeder",
    "P_AKTIV_AXIALKRAFT_MAG": "Axiale Magnetkraft",
    "P_LUEFTER_GEWICHT": "Lüftermasse",
    "P_LUEFTER_D": "Lüfterdurchmesser",
    "P_LUEFTER_J": "Lüfterträgheit",
    "P_LUEFTER_ANZAHL": "Lüfteranzahl",
    "P_ROTOR_GESAMTMASSE": "Rotorgesamtmasse",
    "P_ROTOR_NENNMOMENT": "Nennmoment",
    "P_ROTOR_AXIALLAST": "Axiallast",
    "P_ROTOR_MAX_TORSION": "Max. Torsion",
    "P_ROTOR_POLZAHL": "Polzahl",
    "P_ROTOR_LAGERTYP": "Lagertyp",
    "P_ROTOR_BAUFORM": "Bauform",
    "P_ROTOR_C_MASS": "C-Maß",
    "P_ROTOR_MLFB_BAUREIHE": "MLFB-Baureihe",
    "P_ROTOR_EIGENFREQ_1": "1. Eigenfrequenz",
    "P_ROTOR_EIGENFREQ_2": "2. Eigenfrequenz",
    "P_ROTOR_MAX_BIEGUNG": "Max. Biegung",
    "P_ROTOR_BIEGUNG_KERN": "Biegung am Kern",
    "P_ROTOR_LAGERABSTAND": "Lagerabstand",
    "P_ROTOR_KERNMITTE": "Kernmitte",
    "P_LAGER_BEZEICHNUNG": "Lagerbezeichnung",
    "P_LAGER_BAUREIHE": "Lagerbaureihe",
    "P_LAGER_TYP_DETAIL": "Lagertyp (Detail)",
    "P_LAGER_INNER_D": "Lager-Innendurchm.",
    "P_LAGER_OUTER_D": "Lager-Außendurchm.",
    "P_LAGER_DYN_TRAGZAHL": "Dyn. Tragzahl",
    "P_LAGER_STAT_TRAGZAHL": "Stat. Tragzahl",
    "P_LAGER_LEBENSDAUER": "Lagerlebensdauer",
}


def _latex_escape(text: str) -> str:
    """Escaped Sonderzeichen für LaTeX."""
    ersetzungen = {
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    for original, ersatz in ersetzungen.items():
        text = text.replace(original, ersatz)
    return text


def _ist_numerisch(wert) -> bool:
    """Prüft ob ein Wert numerisch ist."""
    return isinstance(wert, (int, float))


def _formatiere_wert(wert) -> str:
    """Formatiert einen Parameterwert für die LaTeX-Tabelle."""
    if wert is None:
        return "---"
    if isinstance(wert, float):
        if abs(wert) >= 10000:
            return f"{wert:.1f}"
        if abs(wert) >= 1:
            return f"{wert:.2f}"
        if abs(wert) >= 0.001:
            return f"{wert:.4f}"
        return f"{wert:.2e}"
    return _latex_escape(str(wert))


def erzeuge_latex_tabelle() -> str:
    """
    Erzeugt eine vollständige LaTeX-Tabelle der Parameterstruktur.

    :return: LaTeX-Code als String
    :rtype: str
    """
    features_by_rotor = fetch_all_features_from_json()
    if not features_by_rotor:
        return "% FEHLER: Keine JSON-Daten gefunden"

    # Ersten Rotor als Beispiel verwenden
    beispiel_rotor_id = sorted(features_by_rotor.keys())[0]
    beispiel_params = features_by_rotor[beispiel_rotor_id]["params"]

    # Zähler für numerisch/kategorisch
    anzahl_numerisch = 0
    anzahl_kategorisch = 0

    zeilen = []
    aktuelle_komponente = None

    for mapping in ALLE_MAPPINGS:
        schluessel = (mapping.component, mapping.parameter)
        eintrag = beispiel_params.get(schluessel, {})
        wert = eintrag.get("value")
        einheit = mapping.unit
        ptype = mapping.ptype
        kategorie = PTYPE_TO_KATEGORIE_3.get(ptype, "K_REQ_ELEC")
        kategorie_kurz = KATEGORIE_LABEL.get(kategorie, kategorie)

        datentyp = "num." if _ist_numerisch(wert) else "kat."
        if _ist_numerisch(wert):
            anzahl_numerisch += 1
        else:
            anzahl_kategorisch += 1

        komponente_label = KOMPONENTEN_LABEL.get(mapping.component, mapping.component)
        param_label = PARAMETER_LABEL.get(mapping.parameter, mapping.parameter)
        wert_str = _formatiere_wert(wert)

        # Komponentengruppen-Trenner
        if mapping.component != aktuelle_komponente:
            if aktuelle_komponente is not None:
                zeilen.append("\\midrule")
            aktuelle_komponente = mapping.component

        zeilen.append(
            f"    {komponente_label} & {param_label} & {datentyp} & "
            f"{_latex_escape(einheit)} & {kategorie_kurz} & {wert_str} \\\\"
        )

    tabellen_zeilen = "\n".join(zeilen)

    latex = f"""% Automatisch generiert von datenstruktur_export.py
% Beispielrotor: {_latex_escape(beispiel_rotor_id)}
% {anzahl_numerisch} numerische + {anzahl_kategorisch} kategorische = {len(ALLE_MAPPINGS)} Parameter

\\begin{{table}}[htbp]
\\centering
\\caption{{Parameterstruktur der realen Rotordaten (Beispiel: {_latex_escape(beispiel_rotor_id[:12])}\\dots)}}
\\label{{tab:parameterstruktur-real}}
\\scriptsize
\\begin{{tabular}}{{llclll}}
\\toprule
\\textbf{{Komp.}} & \\textbf{{Parameter}} & \\textbf{{Typ}} & \\textbf{{Einheit}} & \\textbf{{Kat.}} & \\textbf{{Beispielwert}} \\\\
\\midrule
{tabellen_zeilen}
\\bottomrule
\\end{{tabular}}
\\vspace{{4pt}}
\\parbox{{0.95\\textwidth}}{{\\footnotesize
\\textbf{{Legende:}} Typ: num.~= numerisch, kat.~= kategorisch.
Kategorien: Geometrie~= Geometrie \\& Mechanik (GEOM, STRUCT, DYN),
Material~= Material \\& Prozess (MTRL, MFG),
Anforderungen~= Anforderungen \\& Elektrik (REQ, ELEC).
Insgesamt {anzahl_numerisch} numerische und {anzahl_kategorisch} kategorische Parameter
aus {len(features_by_rotor)} Rotoren.}}
\\end{{table}}"""

    return latex


def main():
    """Hauptfunktion: LaTeX-Tabelle generieren und speichern."""
    ausgabepfad = PROJEKT_ROOT / "latex" / "bilder" / "data" / "parameterstruktur.tex"
    ausgabepfad.parent.mkdir(parents=True, exist_ok=True)

    latex_code = erzeuge_latex_tabelle()

    with open(ausgabepfad, "w", encoding="utf-8") as datei:
        datei.write(latex_code)

    print(f"LaTeX-Tabelle geschrieben: {ausgabepfad}")
    print(f"Anzahl Parameter: {len(ALLE_MAPPINGS)}")


if __name__ == "__main__":
    main()
