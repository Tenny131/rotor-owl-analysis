from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ParameterSpec:
    parameter_id: str
    component_id: str
    paramtype_id: str
    datatype: str
    unit: str
    enum_domain: str  # string vom csv, z.B. "{val1, val2, val3}"


def _read_parameters_csv(path: Path) -> list[ParameterSpec]:
    """
    Liest Parameter-Spezifikationen aus CSV-Datei ein.

    :param path: Path zur CSV-Datei
    :type path: Path
    :return: Liste von ParameterSpec Objekten
    :rtype: list[ParameterSpec]
    """
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        specs: list[ParameterSpec] = []
        for row in r:
            specs.append(
                ParameterSpec(
                    parameter_id=row["Parameter_ID"].strip(),
                    component_id=row["Component_ID"].strip(),
                    paramtype_id=row["ParamType_ID"].strip(),
                    datatype=row["DataType"].strip(),
                    unit=row["Unit"].strip(),
                    enum_domain=(row.get("EnumDomain") or "").strip(),
                )
            )
    return specs


def _parse_enum_domain(s: str) -> list[str]:
    """
    Parst den Enum-Domain-String aus der CSV in eine Liste von erlaubten Werten.

    :param s: Enum-Domain-String, z.B. "{val1, val2, val3}"
    :type s: str
    :return: Liste von erlaubten Werten
    :rtype: list[str]
    """
    s = s.strip()
    if not s:
        return []
    s = s.strip("{}").strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _sample_numeric(
    rng: random.Random, unit: str, name_hint: str, variance_factor: float = 1.0
) -> float:
    """
    Samples einen numerischen Wert basierend auf der Einheit und einem Namens-Hinweis.

    :type rng: random.Random
    :type unit: str
    :type name_hint: str
    :type variance_factor: float - Faktor zur Erhöhung der Varianz (1.0 = Standard, >1.0 = mehr Varianz)
    :rtype: float
    """
    u = unit.strip()
    if u == "mm":
        lo, hi = 10.0, 500.0
    elif u in ("µm", "um"):
        lo, hi = 0.1, 10.0
    elif u == "1/min":
        lo, hi = 300.0, 12000.0
    elif u == "kg":
        lo, hi = 0.1, 500.0
    elif "kg" in u and "mm" in u:
        lo, hi = 1_000.0, 5_000_000.0
    elif u == "%":
        lo, hi = 0.0, 100.0
    elif u.lower() == "grad":
        lo, hi = 0.0, 45.0
    elif "m³/h" in u or "m3/h" in u:
        lo, hi = 0.0, 5000.0
    else:
        lo, hi = 0.0, 100.0

    nh = name_hint.lower()
    if "rauheit" in nh:
        lo, hi = 0.2, 3.2
    if "tir" in nh or "rundlauf" in nh:
        lo, hi = 0.001, 0.2

    # Varianz erhöhen durch Vergrößerung des Bereichs
    if variance_factor != 1.0:
        center = (lo + hi) / 2
        range_half = (hi - lo) / 2
        range_half *= variance_factor
        lo = max(0.0 if u != "grad" else lo, center - range_half)
        hi = center + range_half

    return round(rng.uniform(lo, hi), 4)


def generate_instances(
    parameters_csv: Path,
    out_dir: Path,
    n: int,
    seed: int,
    missing_rate: float = 0.05,
    variance_factor: float = 1.0,
) -> Path:
    """Generiert eine CSV-Datei mit synthetischen Instanz-Daten.

    Args:
        parameters_csv: Path zur Parameter-Spezifikations-CSV
        out_dir: Ausgabe-Verzeichnis
        n: Anzahl zu generierender Designs
        seed: Random Seed für Reproduzierbarkeit
        missing_rate: Anteil fehlender Werte (0.0-1.0)
        variance_factor: Faktor zur Erhöhung der Varianz (1.0 = Standard, 2.0 = doppelte Range)

    Returns:
        Path zur generierten generated.csv
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = _read_parameters_csv(parameters_csv)

    out_path = out_dir / "generated.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "Design_ID",
                "Component_ID",
                "Parameter_ID",
                "ParamType_ID",
                "DataType",
                "Unit",
                "Value",
                "IsMissing",
            ]
        )

        for i in range(1, n + 1):
            design_id = f"D{i:03d}"

            rng_d = random.Random(
                seed + i
            )  # pro Design eigener RNG (seed-abhängig, aber verschieden)

            for spec in specs:
                is_missing = 1 if rng_d.random() < missing_rate else 0
                value: str = ""

                if not is_missing:
                    dt = spec.datatype.lower()
                    if dt == "numeric":
                        value = str(
                            _sample_numeric(rng_d, spec.unit, spec.parameter_id, variance_factor)
                        )
                    elif dt == "boolean":
                        value = rng_d.choice(["ja", "nein"])
                    elif dt == "enum":
                        choices = _parse_enum_domain(spec.enum_domain)
                        value = rng_d.choice(choices) if choices else "UNSPECIFIED"
                    elif dt == "text":
                        value = rng_d.choice(["k6/H7", "h6/H7", "m6/H7", "g6/H7"])
                    else:
                        value = "UNSPECIFIED"

                w.writerow(
                    [
                        design_id,
                        spec.component_id,
                        spec.parameter_id,
                        spec.paramtype_id,
                        spec.datatype,
                        spec.unit,
                        value,
                        is_missing,
                    ]
                )

    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generiert synthetische Rotor-Datensätze mit konfigurierbarer Varianz"
    )
    parser.add_argument(
        "--n", type=int, default=50, help="Anzahl zu generierender Rotoren (default: 50)"
    )
    parser.add_argument(
        "--v",
        type=float,
        default=1.0,
        help="Varianz-Faktor (1.0=Standard, 2.0=doppelt, 3.0=dreifach, default: 1.0)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="Random Seed für Reproduzierbarkeit (default: 1)"
    )
    parser.add_argument(
        "--missing", type=float, default=0.0, help="Anteil fehlender Werte 0.0-1.0 (default: 0.0)"
    )

    args = parser.parse_args()

    # Standard-Pfade
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent

    # Suche nach parameters.csv in data/ und data/reference/
    parameters_csv = project_root / "data" / "parameters.csv"
    if not parameters_csv.exists():
        parameters_csv = project_root / "data" / "reference" / "parameters.csv"

    output_dir = project_root / "data" / "generated"

    if not parameters_csv.exists():
        print("❌ Fehler: parameters.csv nicht gefunden!")
        print("   Gesucht in:")
        print(f"   - {project_root / 'data' / 'parameters.csv'}")
        print(f"   - {project_root / 'data' / 'reference' / 'parameters.csv'}")
        print("\n   Bitte erstelle zuerst eine parameters.csv mit den Spalten:")
        print("   Parameter_ID, Component_ID, ParamType_ID, DataType, Unit, EnumDomain")
        exit(1)

    print(f"Generiere {args.n} Rotoren...")
    print(f"  Varianz-Faktor: {args.v}x")
    print(f"  Missing Rate: {args.missing*100:.1f}%")
    print(f"  Seed: {args.seed}")

    result = generate_instances(
        parameters_csv=parameters_csv,
        out_dir=output_dir,
        n=args.n,
        seed=args.seed,
        missing_rate=args.missing,
        variance_factor=args.v,
    )

    print(f"✓ Generiert: {result}")
    print(f"  {args.n} Rotoren mit {args.v}x Varianz")
