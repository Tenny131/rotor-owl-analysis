from __future__ import annotations

from pathlib import Path

from rotor_owl.prep import list_design_ids, load_feature_weights


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
