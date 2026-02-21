from __future__ import annotations


# 3 solide, logisch sinnvolle Oberkategorien
#
# K1: "Geometrie & Mechanik"  -> alles was Form, Maße, Dynamik, Struktur beschreibt
# K2: "Material & Prozess"    -> Materialwahl + Fertigungs-/Prozessentscheidungen + Kosten
# K3: "Anforderung & Elektrik"-> Kundenanforderungen + elektrische Vorgaben (+ unknown)
#
# Warum so?
# - K1 ist "engineering-nah" und oft numerisch dominiert
# - K2 ist oft enum/text dominiert (Material, Normen, Verfahren)
# - K3 ist meist funktional/kontextuell (REQ) und technische Vorgaben (ELEC)

KAT_GEOM_MECH = "K_GEOM_MECH"
KAT_MTRL_PROC = "K_MTRL_PROC"
KAT_REQ_ELEC = "K_REQ_ELEC"

KATEGORIEN_3 = [KAT_GEOM_MECH, KAT_MTRL_PROC, KAT_REQ_ELEC]

KATEGORIE_LABEL = {
    KAT_GEOM_MECH: "Geometrie",
    KAT_MTRL_PROC: "Material",
    KAT_REQ_ELEC: "Anforderungen",
}

# Detaillierte Beschreibung für Legende
KATEGORIE_BESCHREIBUNG = {
    KAT_GEOM_MECH: "Geometrie & Mechanik (GEOM, STRUCT, DYN)",
    KAT_MTRL_PROC: "Material & Prozess (MTRL, MFG, COST)",
    KAT_REQ_ELEC: "Anforderungen & Elektrik (REQ, ELEC, UNKNOWN)",
}

# ParamType_ID -> 3er Kategorie
PTYPE_TO_KATEGORIE_3 = {
    # Engineering/Mechanik
    "GEOM": KAT_GEOM_MECH,
    "STRUCT": KAT_GEOM_MECH,
    "DYN": KAT_GEOM_MECH,
    # Material/Prozess/Kosten
    "MTRL": KAT_MTRL_PROC,
    "MFG": KAT_MTRL_PROC,
    "COST": KAT_MTRL_PROC,
    # Anforderungen / Elektrik / Unknown / Sonstige
    "REQ": KAT_REQ_ELEC,
    "ELEC": KAT_REQ_ELEC,
    "UNKNOWN": KAT_REQ_ELEC,
    "MISC": KAT_REQ_ELEC,
}


def map_paramtype_to_kategorie(ptype: str | None) -> str:
    if not ptype:
        return KAT_REQ_ELEC
    return PTYPE_TO_KATEGORIE_3.get(str(ptype).strip(), KAT_REQ_ELEC)
