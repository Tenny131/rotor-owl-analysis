from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from rotor_owl.prep import list_design_ids, load_feature_weights
import csv


@dataclass(frozen=True)
class _NumData:
    # design_id -> (parameter_id -> value)
    values: dict[str, dict[str, float]]
    # parameter_id -> (min, max)
    ranges: dict[str, tuple[float, float]]


def _load_numeric_values_and_ranges(instances_csv: Path) -> _NumData:
    values: dict[str, dict[str, float]] = {}
    mins: dict[str, float] = {}
    maxs: dict[str, float] = {}

    with instances_csv.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if (row.get("DataType") or "").strip().lower() != "numeric":
                continue
            if (row.get("IsMissing") or "").strip() == "1":
                continue

            design_id = (row.get("Design_ID") or "").strip()
            param_id = (row.get("Parameter_ID") or "").strip()
            v_raw = (row.get("Value") or "").strip()
            if not design_id or not param_id or not v_raw:
                continue

            try:
                v = float(v_raw)
            except ValueError:
                continue

            values.setdefault(design_id, {})[param_id] = v

            if param_id not in mins:
                mins[param_id] = v
                maxs[param_id] = v
            else:
                if v < mins[param_id]:
                    mins[param_id] = v
                if v > maxs[param_id]:
                    maxs[param_id] = v

    ranges = {pid: (mins[pid], maxs[pid]) for pid in mins.keys()}
    return _NumData(values=values, ranges=ranges)


def _numeric_sim_from_range(x: float, y: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 1.0  # alle Werte gleich -> immer "gleich"
    return max(0.0, 1.0 - abs(x - y) / (hi - lo))


def numeric_similarity(
    instances_csv: Path,
    query_design: str,
    other_design: str,
) -> float:
    """
    Mittlere Similarity über gemeinsame numerische Parameter (Value-Spalte!),
    normiert über globale (min,max)-Range pro Parameter_ID.
    """
    data = _load_numeric_values_and_ranges(instances_csv)

    q_map = data.values.get(query_design, {})
    o_map = data.values.get(other_design, {})
    common = q_map.keys() & o_map.keys()

    if not common:
        return 0.0

    sims: list[float] = []
    for pid in common:
        lo, hi = data.ranges.get(pid, (0.0, 0.0))
        sims.append(_numeric_sim_from_range(q_map[pid], o_map[pid], lo, hi))

    return sum(sims) / len(sims)


def top_k_numeric_similarity(
    instances_csv: Path,
    query_design: str,
    k: int = 5,
) -> list[tuple[str, float]]:
    data = _load_numeric_values_and_ranges(instances_csv)
    all_ids = sorted(data.values.keys())

    if query_design not in all_ids:
        raise ValueError(f"Design_ID nicht gefunden: {query_design}")

    results: list[tuple[str, float]] = []
    for other in all_ids:
        if other == query_design:
            continue
        s = numeric_similarity(instances_csv, query_design, other)
        results.append((other, s))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:k]


def _weighted_jaccard(a: dict[str, float], b: dict[str, float]) -> float:
    # Weighted Jaccard Similarity zwischen zwei Feature-Gewicht-Dictionaries
    keys = set(a) | set(b)
    if not keys:
        return 1.0

    num = 0.0
    den = 0.0
    for k in keys:
        wa = a.get(k, 0.0)
        wb = b.get(k, 0.0)
        num += min(wa, wb)
        den += max(wa, wb)

    return num / den


def top_k_jaccard(
    instances_csv: Path,
    query_design: str,
    k: int = 5,
    paramtype_weights: dict[str, float] | None = None,
) -> list[tuple[str, float]]:
    # Finde die Top-k Designs ähnlich zum query_design basierend auf gewichteter Jaccard-Ähnlichkeit
    all_ids = list_design_ids(instances_csv)
    if query_design not in all_ids:
        raise ValueError(f"Design_ID nicht gefunden: {query_design}")

    q = load_feature_weights(instances_csv, query_design, paramtype_weights=paramtype_weights)

    results: list[tuple[str, float]] = []
    for other in all_ids:
        if other == query_design:
            continue
        o = load_feature_weights(instances_csv, other, paramtype_weights=paramtype_weights)
        results.append((other, _weighted_jaccard(q, o)))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:k]
