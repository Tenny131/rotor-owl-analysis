from __future__ import annotations

import csv
from pathlib import Path


def list_design_ids(instances_csv: Path) -> list[str]:
    ids: set[str] = set()
    with instances_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            ids.add(row["Design_ID"].strip())
    return sorted(ids)


def load_feature_weights(
    instances_csv: Path,
    design_id: str,
    paramtype_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    # Lade Feature-Gewichte für ein gegebenes Design aus der CSV-Datei
    feats: dict[str, float] = {}
    with instances_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["Design_ID"].strip() != design_id:
                continue
            if row.get("IsMissing", "0").strip() == "1":
                continue

            pid = row["Parameter_ID"].strip()
            ptype = row["ParamType_ID"].strip()

            w = 1.0
            if paramtype_weights is not None:
                w = float(paramtype_weights.get(ptype, 1.0))

            feats[pid] = w

    return feats
