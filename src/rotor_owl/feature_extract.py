from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


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
    # Sicheres Extrahieren des ersten Elements aus Liste/Tupel oder None
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return str(x[0]) if x else None
    return str(x)


def _get_prop(ns: Any, name: str):
    # Sicheres Extrahieren einer Property aus einem Namespace
    return getattr(ns, name, None)


def extract_features(ontology, assembly_iri: str | None = None) -> list[FeatureRecord]:
    # Extrahiert Feature-Informationen aus einer IMS-Ontologie.

    ns = ontology.get_namespace("http://ontology.innomotics.net/ims#")

    composed_of = _get_prop(ns, "composed_of")
    has_value = _get_prop(ns, "hasValue")
    has_unit = _get_prop(ns, "hasUnit")
    has_type = _get_prop(ns, "hasType")

    if has_value is None and has_unit is None and has_type is None:
        # Fallback: falls Names im Namespace anders heißen
        raise RuntimeError("Keine Feature-Eigenschaften in der Ontologie gefunden.")

    # Kandidaten für Features bestimmen
    candidates: Iterable[Any]
    if assembly_iri and composed_of is not None:
        assembly = ontology.world[assembly_iri]
        if assembly is None:
            raise FileNotFoundError(f"Assembly mit IRI '{assembly_iri}' nicht gefunden.")
        candidates = list(getattr(assembly, "composed_of", []))
    else:
        # alle Individuen die mindestens eines der Felder haben
        candidates = list(ontology.individuals())
        candidates = list(dict.fromkeys(candidates))
    if assembly_iri and composed_of is not None:
        assembly = ontology.world[assembly_iri]
        ...
        candidates = list(getattr(assembly, "composed_of", []))
        candidates = list(dict.fromkeys(candidates))  # Duplikate entfernen

    records: list[FeatureRecord] = []
    for ind in candidates:
        iri = getattr(ind, "iri", None)
        if not isinstance(iri, str):
            continue

        # Datenwerte lesen (falls vorhanden)
        val = _safe_first(getattr(ind, "hasValue", None))
        unit = _safe_first(getattr(ind, "hasUnit", None))
        ftype = _safe_first(getattr(ind, "hasType", None))

        # Nur echte "Parameter-Features" aufnehmen
        if val is None and unit is None and ftype is None:
            continue

        # Kommentar/Label
        comment = _safe_first(getattr(ind, "comment", None)) or _safe_first(
            getattr(ind, "rdfs_comment", None)
        )
        name = getattr(ind, "name", None) or iri.rsplit("#", 1)[-1]

        # Klasse (Typ des Individuums)
        cls = None
        try:
            types = list(ind.is_a)  # owlready2 types
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
