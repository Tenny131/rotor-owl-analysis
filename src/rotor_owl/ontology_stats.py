from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class OntologyStats:
    classes_total: int
    obj_props_total: int
    data_props_total: int
    individuals_total: int
    iri_prefix_counts: list[tuple[str, int]]


def _iter_iris(entities: Iterable[object]) -> Iterable[str]:
    for e in entities:
        iri = getattr(e, "iri", None)
        if isinstance(iri, str) and iri:
            yield iri


# Domain Ontologie Prefix extraktion
def _prefix(iri: str) -> str:
    if "#" in iri:
        return iri.rsplit("#", 1)[0] + "#"
    if "/" in iri:
        return iri.rsplit("/", 1)[0] + "/"
    return iri


# Hauptfunktion zur Berechnung der Ontologie-Statistiken
def compute_stats(ontology, top_prefixes: int = 10) -> OntologyStats:
    classes = list(ontology.classes())
    obj_props = list(ontology.object_properties())
    data_props = list(ontology.data_properties())
    individuals = list(ontology.individuals())

    prefixes = Counter(
        _prefix(i) for i in _iter_iris([*classes, *obj_props, *data_props, *individuals])
    )
    top = prefixes.most_common(top_prefixes)

    return OntologyStats(
        classes_total=len(classes),
        obj_props_total=len(obj_props),
        data_props_total=len(data_props),
        individuals_total=len(individuals),
        iri_prefix_counts=top,
    )
