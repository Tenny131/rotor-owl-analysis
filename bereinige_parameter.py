"""
Bereinigung und fachliche Kategorisierung der 2205 JSON-Parameter.

Dieses Skript:
1. Ordnet JSON-Pfade den 44 bekannten Parametern zu (wo möglich)
2. Identifiziert fachlich relevante NEUE Parameter (Ingenieurwissen)
3. Entfernt Duplikate (SimocalcWorkflow ↔ EdimWorkflow, gleiche Werte)
4. Entfernt Konstanten (Unique = 1)
5. Entfernt reine Identifikatoren / Teilenummern
6. Gibt eine bereinigte CSV mit Spalte „Fachliche_Relevanz" aus

Ausgabe: data/real_data/analyse/parameter_bereinigt.csv
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Pfade
# ---------------------------------------------------------------------------
ANALYSE_DIR = Path(__file__).resolve().parent / "data" / "real_data" / "analyse"
EINGABE_CSV = ANALYSE_DIR / "parameter_analyse_alle.csv"
AUSGABE_CSV = ANALYSE_DIR / "parameter_bereinigt.csv"

# ---------------------------------------------------------------------------
# 1. MAPPING: JSON-Pfad → bekannter P_-Parametername
#    Basierend auf dem Abgleich der Werte in parameter_analyse.csv
# ---------------------------------------------------------------------------
BEKANNTE_PARAMETER: dict[str, dict] = {
    # --- Welle (C_WELLE) ---
    "RotorWorkflow.output.output.form.output.shaft.mass_moment_of_inertia": {
        "P_Name": "P_WELLE_TRAEGHEITSMOMENT",
        "Komponente": "C_WELLE",
        "Einheit": "kg·m²",
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
        "Einheit": "m³",
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
    # --- Lüfter (C_LUEFTER) ---
    "segments.fan._count": {
        "P_Name": "P_LUEFTER_ANZAHL",
        "Komponente": "C_LUEFTER",
        "Einheit": "–",
        "Kategorie": "Geometrie",
    },
}

# ---------------------------------------------------------------------------
# 2. NEUE fachlich relevante Parameter
#    (nicht in den 44, aber ingenieurstechnisch wichtig)
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
        "Begruendung": "Kippmoment/Nennmoment – Überlastfähigkeit",
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
        "Begruendung": "Anlaufmoment/Nennmoment – Startfähigkeit",
    },
    # --- Stator-Geometrie ---
    "EdimWorkflow.input.stator.core.outer_diameter": {
        "P_Name": "P_STATOR_D_AUSSEN",
        "Komponente": "C_AKTIVTEIL",
        "Einheit": "mm",
        "Kategorie": "Geometrie",
        "Begruendung": "Stator-Außendurchmesser – definiert Baugröße",
    },
    "EdimWorkflow.input.stator.core.inner_diameter": {
        "P_Name": "P_STATOR_D_INNEN",
        "Komponente": "C_AKTIVTEIL",
        "Einheit": "mm",
        "Kategorie": "Geometrie",
        "Begruendung": "Stator-Innendurchmesser – definiert Luftspalt zusammen mit Rotor-Außendurchmesser",
    },
    "EdimWorkflow.input.stator.core.iron_length": {
        "P_Name": "P_STATOR_EISENLAENGE",
        "Komponente": "C_AKTIVTEIL",
        "Einheit": "mm",
        "Kategorie": "Geometrie",
        "Begruendung": "Stator-Eisenlänge – bestimmt elektromagnetisch aktives Volumen",
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
        "Begruendung": "Rotor-Nutzahl – Käfigläufer-Stabzahl, bestimmt Geräusch/Oberwellen",
    },
    "EdimWorkflow.input.compound.air_gap_height": {
        "P_Name": "P_LUFTSPALT",
        "Komponente": "C_AKTIVTEIL",
        "Einheit": "mm",
        "Kategorie": "Geometrie",
        "Begruendung": "Luftspalt – kritischer Parameter für Magnetkreis und Rotordynamik",
    },
    # --- Wicklung ---
    "EdimWorkflow.input.stator.winding.number_of_poles": {
        "P_Name": "P_STATOR_POLZAHL",
        "Komponente": "C_AKTIVTEIL",
        "Einheit": "–",
        "Kategorie": "Elektrisch",
        "Begruendung": "Polzahl Stator (≈ Rotor-Polzahl) – bestimmt Synchrondrehzahl",
    },
    "EdimWorkflow.input.stator.winding.coil_pitch": {
        "P_Name": "P_SPULENSCHRITT",
        "Komponente": "C_AKTIVTEIL",
        "Einheit": "–",
        "Kategorie": "Elektrisch",
        "Begruendung": "Spulenschritt – beeinflusst Wicklungsfaktor und Oberwellenunterdrückung",
    },
    "EdimWorkflow.input.stator.winding.bare_wire_height": {
        "P_Name": "P_DRAHTHOEHE",
        "Komponente": "C_AKTIVTEIL",
        "Einheit": "mm",
        "Kategorie": "Geometrie",
        "Begruendung": "Drahthöhe – bestimmt Kupferquerschnitt und Strombelastbarkeit",
    },
    "EdimWorkflow.input.stator.winding.bare_wire_width": {
        "P_Name": "P_DRAHTBREITE",
        "Komponente": "C_AKTIVTEIL",
        "Einheit": "mm",
        "Kategorie": "Geometrie",
        "Begruendung": "Drahtbreite – bestimmt Kupferquerschnitt und Nutfüllfaktor",
    },
    # --- Rotor-Käfig (Kurzschlussläufer) ---
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
        "Begruendung": "Rotor-Stabhöhe – beeinflusst Anlaufverhalten (Stromverdrängung)",
    },
    "EdimWorkflow.input.rotor.winding.bar.length": {
        "P_Name": "P_ROTORSTAB_LAENGE",
        "Komponente": "C_AKTIVTEIL",
        "Einheit": "mm",
        "Kategorie": "Geometrie",
        "Begruendung": "Rotor-Stablänge – bestimmt Widerstand und Überstand",
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
        "Begruendung": "Kurzschlussring-Höhe – bestimmt Ringquerschnitt",
    },
    "EdimWorkflow.input.rotor.winding.end_ring.outer_diameter": {
        "P_Name": "P_KURZSCHLUSSRING_D_AUSSEN",
        "Komponente": "C_AKTIVTEIL",
        "Einheit": "mm",
        "Kategorie": "Geometrie",
        "Begruendung": "Kurzschlussring-Außendurchmesser – Massenträgheit am Rotorende",
    },
    # --- Lagerkräfte (berechnete Ergebnisse) ---
    "RotorWorkflow.output.output.roller_bearings.output.DE.forces.radial": {
        "P_Name": "P_LAGER_DE_RADIALKRAFT",
        "Komponente": "C_LAGER",
        "Einheit": "N",
        "Kategorie": "Struktur",
        "Begruendung": "Radialkraft am DE-Lager – zentral für Lagerdimensionierung",
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
        "Begruendung": "Gesamte Lagerreibungsverluste – Wärmeentwicklung und Wirkungsgrad",
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
        "Begruendung": "Statische Tragfähigkeitssicherheit DE – Schutz vor Stillstandsschäden",
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
        "Begruendung": "Max. Passfedersicherheit (Welle) – Flächenpressung Wellennut",
    },
    # --- Rotordynamik (erweitert) ---
    "RotorWorkflow.output.output.rotordynamics.output.modal_solution.eigenfrequencies.mode_3": {
        "P_Name": "P_ROTOR_EIGENFREQ_3",
        "Komponente": "C_ROTOR",
        "Einheit": "Hz",
        "Kategorie": "Dynamik",
        "Begruendung": "3. Eigenfrequenz – Campbell-Diagramm, höhere Resonanzstellen",
    },
    "RotorWorkflow.output.output.rotordynamics.output.modal_solution.modal_damping.mode_1": {
        "P_Name": "P_ROTOR_DAEMPFUNG_1",
        "Komponente": "C_ROTOR",
        "Einheit": "%",
        "Kategorie": "Dynamik",
        "Begruendung": "Modale Dämpfung Mode 1 – Stabilitätskriterium (>0 = stabil)",
    },
    "RotorWorkflow.output.output.rotordynamics.output.modal_solution.modal_damping.mode_2": {
        "P_Name": "P_ROTOR_DAEMPFUNG_2",
        "Komponente": "C_ROTOR",
        "Einheit": "%",
        "Kategorie": "Dynamik",
        "Begruendung": "Modale Dämpfung Mode 2 – Stabilitätskriterium 2. Biegemode",
    },
    # --- Gleitlager (wenn vorhanden) ---
    "RotorWorkflow.output.sleeve_bearing_calculation.DE.c12._max": {
        "P_Name": "P_GLEITLAGER_DE_C12_MAX",
        "Komponente": "C_LAGER",
        "Einheit": "N/µm",
        "Kategorie": "Dynamik",
        "Begruendung": "Kreuzsteifigkeit DE-Gleitlager – Stabilitäts-relevanter Koeffizient",
    },
    "RotorWorkflow.output.sleeve_bearing_calculation.NDE.c12._max": {
        "P_Name": "P_GLEITLAGER_NDE_C12_MAX",
        "Komponente": "C_LAGER",
        "Einheit": "N/µm",
        "Kategorie": "Dynamik",
        "Begruendung": "Kreuzsteifigkeit NDE-Gleitlager – kann Instabilität (Oil Whirl) verursachen",
    },
    # --- Betriebsdaten ---
    "RotorWorkflow.input.operational_data.ambient_temperature": {
        "P_Name": "P_UMGEBUNGSTEMPERATUR",
        "Komponente": "C_ROTOR",
        "Einheit": "°C",
        "Kategorie": "Anforderung",
        "Begruendung": "Umgebungstemperatur – Einfluss auf Lagerung, Wicklung, Kühlung",
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
        "Begruendung": "Horizontale Unterbausteifigkeit – kritisch für Rotordynamik-Berechnung",
    },
    "segments.substructure.stiffness.vertical": {
        "P_Name": "P_UNTERBAU_STEIFIGKEIT_V",
        "Komponente": "C_ROTOR",
        "Einheit": "N/m",
        "Kategorie": "Struktur",
        "Begruendung": "Vertikale Unterbausteifigkeit – kritisch für Rotordynamik-Berechnung",
    },
    # --- Segmente (aggregiert) ---
    "segments._total_count": {
        "P_Name": "P_SEGMENT_ANZAHL",
        "Komponente": "C_ROTOR",
        "Einheit": "–",
        "Kategorie": "Geometrie",
        "Begruendung": "Gesamtanzahl Segmente – Komplexität des Rotoraufbaus",
    },
    "segments.shaft.length._sum": {
        "P_Name": "P_WELLE_SEGMENTLAENGE_SUMME",
        "Komponente": "C_WELLE",
        "Einheit": "mm",
        "Kategorie": "Geometrie",
        "Begruendung": "Summe der Wellensegmentlängen – Gesamte Wellenlänge aus Segmenten",
    },
    # --- Rotor-Nutgeometrie ---
    "EdimWorkflow.input.rotor.core.slot.B1": {
        "P_Name": "P_ROTOR_NUT_B1",
        "Komponente": "C_AKTIVTEIL",
        "Einheit": "mm",
        "Kategorie": "Geometrie",
        "Begruendung": "Rotor-Nutöffnung B1 – Einfluss auf Oberwellen, Geräusch, Anlauf",
    },
    "EdimWorkflow.input.rotor.core.skewing_angle": {
        "P_Name": "P_ROTOR_SCHRAEGUNG",
        "Komponente": "C_AKTIVTEIL",
        "Einheit": "°",
        "Kategorie": "Geometrie",
        "Begruendung": "Rotorschrägung – reduziert Nutoberwellen, beeinflusst Anlauf",
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
        "Begruendung": "Ventilations-/Reibungsverluste – Kühlung und Wirkungsgrad",
    },
    # --- Passungsgeometrie ---
    "segments.shaft_end.parallel_key.width": {
        "P_Name": "P_PK_BREITE",
        "Komponente": "C_WELLE",
        "Einheit": "mm",
        "Kategorie": "Geometrie",
        "Begruendung": "Passfederbreite – normiert nach DIN 6885, definiert Drehmomentübertragung",
    },
    "segments.shaft_end.parallel_key.height": {
        "P_Name": "P_PK_HOEHE",
        "Komponente": "C_WELLE",
        "Einheit": "mm",
        "Kategorie": "Geometrie",
        "Begruendung": "Passfederhöhe – bestimmt Flächenpressung",
    },
    "segments.shaft_end.parallel_key.length": {
        "P_Name": "P_PK_LAENGE",
        "Komponente": "C_WELLE",
        "Einheit": "mm",
        "Kategorie": "Geometrie",
        "Begruendung": "Passfederlänge – bestimmt tragende Länge",
    },
    "segments.shaft_end.outer_diameter": {
        "P_Name": "P_WELLENENDE_D",
        "Komponente": "C_WELLE",
        "Einheit": "mm",
        "Kategorie": "Geometrie",
        "Begruendung": "Wellenenddurchmesser – Abtriebsseite, Kupplung/Riemenscheibe",
    },
    "segments.shaft_end.length": {
        "P_Name": "P_WELLENENDE_LAENGE",
        "Komponente": "C_WELLE",
        "Einheit": "mm",
        "Kategorie": "Geometrie",
        "Begruendung": "Wellenende-Länge – normiert nach IEC/DIN, Kupplungsanbindung",
    },
}

# ---------------------------------------------------------------------------
# 3. Identifikatoren / Teilenummern → entfernen
# ---------------------------------------------------------------------------
IDENTIFIKATOREN = {
    "machine_id",
    "MachineConfig.mlfb",
    "MachineConfig.shaft",  # Regex-Muster für Wellennummer
    "MachineConfig.bearing",  # Regex-Muster für Lagernummer
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
# Hauptlogik
# ---------------------------------------------------------------------------
def main():
    print("\n" + "=" * 70)
    print("  Parameterbereinigung und fachliche Kategorisierung")
    print("=" * 70 + "\n")

    # --- 1. Lade alle Parameter ---
    df = pd.read_csv(EINGABE_CSV, sep=";", decimal=",", encoding="utf-8-sig")
    n_gesamt = len(df)
    print(f"[1/5] {n_gesamt} Parameter geladen.\n")

    # --- 2. Simocalc-Duplikate identifizieren und entfernen ---
    #   Alle SimocalcWorkflow-Parameter, die auch als EdimWorkflow existieren
    edim_suffixe = set()
    for p in df["Parameter"]:
        if p.startswith("EdimWorkflow."):
            suffix = p[len("EdimWorkflow.") :]
            edim_suffixe.add(suffix)

    simocalc_duplikate = set()
    for p in df["Parameter"]:
        if p.startswith("SimocalcWorkflow."):
            suffix = p[len("SimocalcWorkflow.") :]
            if suffix in edim_suffixe:
                simocalc_duplikate.add(p)

    df = df[~df["Parameter"].isin(simocalc_duplikate)].copy()
    print(
        f"[2/5] {len(simocalc_duplikate)} SimocalcWorkflow-Duplikate entfernt "
        f"(identische Pfade in EdimWorkflow vorhanden)."
    )
    print(f"      Verbleibend: {len(df)} Parameter.\n")

    # --- 3. Konstanten entfernen (Unique = 1) ---
    unique_col = df["Unique"]
    # Unique-Spalte kann "–" enthalten
    konstanten_mask = unique_col.apply(lambda x: str(x).strip() in ("1", "1.0"))
    n_konstanten = konstanten_mask.sum()
    df = df[~konstanten_mask].copy()
    print(f"[3/5] {n_konstanten} Konstanten entfernt (Unique = 1, alle Rotoren gleich).")
    print(f"      Verbleibend: {len(df)} Parameter.\n")

    # --- 4. Identifikatoren entfernen ---
    ident_mask = df["Parameter"].isin(IDENTIFIKATOREN)
    n_ident = ident_mask.sum()
    df = df[~ident_mask].copy()
    print(f"[4/5] {n_ident} Identifikatoren/Teilenummern entfernt.")
    print(f"      Verbleibend: {len(df)} Parameter.\n")

    # --- 5. Fachliche Relevanz zuweisen ---
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

    # --- Statistik ---
    n_bekannt = (df["Fachliche_Relevanz"] == "Bekannt (44)").sum()
    n_neu = (df["Fachliche_Relevanz"] == "Neu (fachlich relevant)").sum()
    n_rest = (df["Fachliche_Relevanz"] == "Nicht zugeordnet").sum()

    print("[5/5] Fachliche Kategorisierung:")
    print(f"      ✓ {n_bekannt:>4} Parameter den bekannten 44 zugeordnet")
    print(f"      ✓ {n_neu:>4} NEUE fachlich relevante Parameter identifiziert")
    print(f"      ○ {n_rest:>4} nicht zugeordnet (verbleibend)")
    print("      ─────────────")
    print(f"        {len(df):>4} Parameter in der bereinigten CSV\n")

    # --- Zusammenfassung der neuen Parameter ---
    print("--- Neu identifizierte fachlich relevante Parameter ---\n")
    neu_df = df[df["Fachliche_Relevanz"] == "Neu (fachlich relevant)"][
        ["P_Name", "Fachkategorie", "Einheit_", "Abdeckung (%)", "Unique", "Eignung", "Begruendung"]
    ].sort_values("Eignung", ascending=False)
    for _, row in neu_df.iterrows():
        print(
            f"  {row['P_Name']:<40s} [{row['Fachkategorie']:<10s}] "
            f"Eignung={row['Eignung']:>5}  {row['Einheit_']:<6s}  "
            f"Abdeckung={row['Abdeckung (%)']:>5}%  Unique={row['Unique']}"
        )
        print(f"    → {row['Begruendung']}")

    # --- Übersicht der Entfernungen ---
    print("\n--- Zusammenfassung Bereinigung ---\n")
    print(f"  Ausgangsbasis:                {n_gesamt:>5} Parameter")
    print(f"  − SimocalcWorkflow-Duplikate: {len(simocalc_duplikate):>5}")
    print(f"  − Konstanten (Unique=1):      {n_konstanten:>5}")
    print(f"  − Identifikatoren/Teilenr.:   {n_ident:>5}")
    print(f"  − Interne Duplikate:          {1:>5}")  # material_name/name
    print("  ═══════════════════════════════════════")
    print(f"  Bereinigte Parameterliste:    {len(df):>5} Parameter\n")

    # --- CSV Export ---
    # Sortieren: Bekannt → Neu → Rest, jeweils nach Eignung
    sort_order = {"Bekannt (44)": 0, "Neu (fachlich relevant)": 1, "Nicht zugeordnet": 2}
    df["_sort"] = df["Fachliche_Relevanz"].map(sort_order)
    df = df.sort_values(["_sort", "Eignung"], ascending=[True, False]).drop(columns=["_sort"])

    # Dezimalkomma: Alle float-Werte Punkt → Komma
    def _fmt_komma(val):
        if isinstance(val, float):
            return str(val).replace(".", ",")
        return val

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

    df_export.to_csv(AUSGABE_CSV, index=False, sep=";", encoding="utf-8-sig")
    print(f"  → CSV exportiert: {AUSGABE_CSV.name}")
    print(f"\n{'=' * 70}")
    print("  Bereinigung abgeschlossen.")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
