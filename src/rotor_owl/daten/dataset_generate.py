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


def _sample_numeric(rng: random.Random, unit: str, name_hint: str) -> float:
    """
    Samples einen numerischen Wert basierend auf der Einheit und einem Namens-Hinweis.

    :type rng: random.Random
    :type unit: str
    :type name_hint: str
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

    return round(rng.uniform(lo, hi), 4)


def generate_instances(
    parameters_csv: Path,
    out_dir: Path,
    n: int,
    seed: int,
    missing_rate: float = 0.05,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = _read_parameters_csv(parameters_csv)

    """Generiert eine CSV-Datei mit synthetischen Instanz-Daten."""

    out_path = out_dir / "instances.csv"
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
                        value = str(_sample_numeric(rng_d, spec.unit, spec.parameter_id))
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
