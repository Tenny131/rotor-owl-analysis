from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable
from owlready2 import Thing, get_ontology
from pathlib import Path


# OWL Datei laden
def load_owl(path: str | Path):
    "Owl Ontologie laden und als ontologie Objekt zurückgeben"

    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Die angegebene Datei wurde nicht gefunden: {path}")
    ontology = get_ontology(str(path)).load()

    return ontology


# INSPECTION DER ONTOLOGIE
@dataclass
class OntologyInventory:
    classes: Counter[str]
    object_props_defined: list[str]
    data_props_defined: list[str]
    annotation_props_defined: list[str]


def inspect_ontology(path: str) -> OntologyInventory:
    ont = get_ontology(path).load()

    classes = Counter()
    for c in ont.classes():
        classes[str(c)] += 1

    obj_props = sorted(str(p) for p in ont.object_properties())
    data_props = sorted(str(p) for p in ont.data_properties())
    ann_props = sorted(str(p) for p in ont.annotation_properties())

    return OntologyInventory(
        classes=classes,
        object_props_defined=obj_props,
        data_props_defined=data_props,
        annotation_props_defined=ann_props,
    )


# Statistik der Ontologie
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


def _prefix(iri: str) -> str:
    if "#" in iri:
        return iri.rsplit("#", 1)[0] + "#"
    if "/" in iri:
        return iri.rsplit("/", 1)[0] + "/"
    return iri


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


# Feature-Extraktion
@dataclass(frozen=True)
class FeatureRecord:
    feature_iri: str
    feature_name: str
    feature_class_iri: str | None
    comment: str | None
    value: str | None
    unit: str | None
    ftype: str | None


def _safe_first(x: Any) -> str | None:
    # Erstes Element aus Liste/Tupel holen oder None
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return str(x[0]) if x else None
    return str(x)


def _get_prop(ns: Any, name: str):
    # Property aus Namespace holen (oder None)
    return getattr(ns, name, None)


def extract_features(ontology, assembly_iri: str | None = None) -> list[FeatureRecord]:
    # Extrahiert Parameter-Features (hasValue/hasUnit/hasType) aus der Ontologie.

    ns = ontology.get_namespace("http://ontology.innomotics.net/ims#")

    composed_of = _get_prop(ns, "composed_of")
    has_value = _get_prop(ns, "hasValue")
    has_unit = _get_prop(ns, "hasUnit")
    has_type = _get_prop(ns, "hasType")

    if has_value is None and has_unit is None and has_type is None:
        raise RuntimeError("Keine Feature-Eigenschaften (hasValue/hasUnit/hasType) gefunden.")

    # Kandidaten bestimmen: Wenn assembly_iri gesetzt ist, gehen wir 2 Ebenen tief:
    # Rotor -> Komponenten -> Parameter
    candidates: list[Any] = []

    if assembly_iri and composed_of is not None:
        assembly = ontology.world[assembly_iri]
        if assembly is None:
            raise FileNotFoundError(f"Assembly mit IRI '{assembly_iri}' nicht gefunden.")

        level1 = list(getattr(assembly, "composed_of", []))  # Komponenten
        level2: list[Any] = []
        for comp in level1:
            level2.extend(list(getattr(comp, "composed_of", [])))  # Parameter unter Komponente

        candidates = level2
    else:
        # Fallback: alle Individuals durchsuchen
        candidates = list(ontology.individuals())

    # Duplikate entfernen, Reihenfolge beibehalten
    candidates = list(dict.fromkeys(candidates))

    records: list[FeatureRecord] = []
    for ind in candidates:
        iri = getattr(ind, "iri", None)
        if not isinstance(iri, str):
            continue

        val = _safe_first(getattr(ind, "hasValue", None))
        unit = _safe_first(getattr(ind, "hasUnit", None))
        ftype = _safe_first(getattr(ind, "hasType", None))

        # Nur echte Parameter aufnehmen
        if val is None and unit is None and ftype is None:
            continue

        comment = _safe_first(getattr(ind, "comment", None)) or _safe_first(
            getattr(ind, "rdfs_comment", None)
        )
        name = getattr(ind, "name", None) or iri.rsplit("#", 1)[-1]

        cls = None
        try:
            types = list(ind.is_a)
            if types:
                cls_iri = getattr(types[0], "iri", None)
                cls = cls_iri if isinstance(cls_iri, str) else None
        except Exception:
            cls = None

        records.append(
            FeatureRecord(
                feature_iri=iri,
                feature_name=str(name),
                feature_class_iri=cls,
                comment=str(comment) if comment else None,
                value=val,
                unit=unit,
                ftype=ftype,
            )
        )

    return records


# Dependency Graph Extraktion
@dataclass
class DepGraph:
    edges: dict[str, set[str]]  # src_iri -> set(dst_iri)
    rel_counts: Counter[str]  # relation -> count


def extract_dependency_graph(path: str, include_prefix: str = "ims.") -> DepGraph:
    ont = get_ontology(path).load()

    edges: dict[str, set[str]] = defaultdict(set)
    rel_counts: Counter[str] = Counter()

    obj_props = [p for p in ont.object_properties() if str(p).startswith(include_prefix)]

    for prop in obj_props:
        prop_name = str(prop)

        # Owlready2 kann hier einen Generator liefern:
        # Iteriere direkt über (subject, object)-Paare.
        for s, t in prop.get_relations():
            if not isinstance(s, Thing) or not isinstance(t, Thing):
                continue
            edges[s.iri].add(t.iri)
            rel_counts[prop_name] += 1

    return DepGraph(edges=dict(edges), rel_counts=rel_counts)
