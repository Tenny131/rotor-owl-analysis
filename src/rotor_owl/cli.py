from __future__ import annotations

import argparse
import csv
from pathlib import Path
from rotor_owl.dataset_generate import generate_instances
from rotor_owl.similarity import top_k_jaccard, top_k_numeric_similarity
from rotor_owl.ontology_stats import (
    load_owl,
    inspect_ontology,
    compute_stats,
    extract_features,
    extract_dependency_graph,
)


# CLI SETUP
def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def _parse_weights(s: str) -> dict[str, float]:
    s = (s or "").strip()
    if not s:
        return {}

    out: dict[str, float] = {}
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            raise ValueError(f"Invalid weight token: {p} (expected KEY=VALUE)")
        k, v = p.split("=", 1)
        out[k.strip()] = float(v.strip())
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rotor_owl", description="Rotor OWL tools")
    sub = parser.add_subparsers(dest="command", required=True)

    p_load = sub.add_parser("load", help="Load an OWL file and show basic info")
    p_load.add_argument("path", type=Path, help="Path to .owl file")
    p_load.add_argument("--list-classes", action="store_true", help="Print classes")
    p_load.add_argument("--limit", type=int, default=20, help="Max classes to print")
    p_load.set_defaults(func=_cmd_load)
    p_stats = sub.add_parser("stats", help="Show ontology statistics")
    p_stats.add_argument("path", type=Path, help="Path to .owl file")
    p_stats.add_argument("--top-prefixes", type=int, default=10, help="Top IRI prefixes to show")
    p_stats.set_defaults(func=lambda a: _cmd_stats(a))
    p_feat = sub.add_parser(
        "features", help="Extract IMS parameter-features (hasValue/hasUnit/hasType)"
    )
    p_feat.add_argument("path", type=Path, help="Path to .owl file")
    p_feat.add_argument(
        "--assembly-iri",
        type=str,
        default=None,
        help="Limit to one assembly individual by full IRI",
    )
    p_feat.add_argument("--limit", type=int, default=20, help="Max features to print")
    p_feat.set_defaults(func=_cmd_features)
    p_feat.add_argument("--out", type=Path, default=None, help="Write extracted features to CSV")
    p_gen = sub.add_parser("generate", help="Generate synthetic instance dataset (CSV)")
    p_gen.add_argument("--n", type=int, default=50, help="Number of designs")
    p_gen.add_argument("--seed", type=int, default=42, help="Random seed")
    p_gen.add_argument("--out", type=Path, default=Path("data/generated"), help="Output directory")
    p_gen.add_argument(
        "--parameters-csv",
        type=Path,
        default=Path("data/reference/parameters.csv"),
        help="Reference parameters CSV",
    )
    p_gen.add_argument("--missing-rate", type=float, default=0.05, help="Missing value probability")
    p_gen.set_defaults(func=_cmd_generate)
    p_sim = sub.add_parser("similarity", help="TOP-k Jaccard similarity for a design")
    p_sim.add_argument("instances", type=Path, help="instances.csv")
    p_sim.add_argument("design", help="Query Design_ID (e.g. D001)")
    p_sim.add_argument("--k", type=int, default=5, help="Number of similar designs")
    p_sim.set_defaults(func=_cmd_similarity)
    p_sim.add_argument(
        "--weights",
        default="",
        help='ParamType weights, e.g. "GEOM=1,REQ=0.3,DYN=1.2" (default: all 1.0)',
    )
    p_insp = sub.add_parser("inspect", help="Inspect OWL: most used relations & data properties")
    p_insp.add_argument("owl", help="Path to OWL file (e.g. example.owl)")
    p_insp.add_argument("--top", type=int, default=30, help="How many items to show")
    p_insp.set_defaults(func=_cmd_inspect)
    p_deps = sub.add_parser(
        "deps", help="Extract dependency relations (object properties) as a graph"
    )
    p_deps.add_argument("owl", help="Path to OWL file")
    p_deps.add_argument(
        "--prefix",
        default="ims.",
        help='Only include properties starting with this prefix (default: "ims.")',
    )
    p_deps.add_argument("--top", type=int, default=30, help="How many relations to show")
    p_deps.set_defaults(func=_cmd_deps)
    p_simnum = sub.add_parser(
        "similarity-numeric",
        help="TOP-k Similarity basierend nur auf numerischen Parameterwerten",
    )
    p_simnum.add_argument("instances", type=Path, help="instances.csv")
    p_simnum.add_argument("design", help="Query Design_ID (z. B. D001)")
    p_simnum.add_argument("--k", type=int, default=5, help="Anzahl ähnlicher Designs")
    p_simnum.set_defaults(func=_cmd_similarity_numeric)

    return parser


# CLI COMMANDS
def _cmd_load(args: argparse.Namespace) -> int:
    ont = load_owl(args.path)
    classes = list(ont.classes())
    print(f"Loaded ontology: {ont.base_iri or ont.name}")
    print(f"Number of classes: {len(classes)}")
    if args.list_classes:
        for c in classes[: args.limit]:
            print(c)
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    ont = load_owl(args.path)
    s = compute_stats(ont, top_prefixes=args.top_prefixes)

    print(f"Loaded ontology: {ont.base_iri or ont.name}")
    print(f"Classes:        {s.classes_total}")
    print(f"Object props:   {s.obj_props_total}")
    print(f"Data props:     {s.data_props_total}")
    print(f"Individuals:    {s.individuals_total}")
    print("\nTop IRI prefixes:")
    for pref, cnt in s.iri_prefix_counts:
        print(f"  {cnt:6d}  {pref}")
    return 0


def _cmd_features(args: argparse.Namespace) -> int:
    ont = load_owl(args.path)
    recs = extract_features(ont, assembly_iri=args.assembly_iri)

    print(f"Features found: {len(recs)}")
    for r in recs[: args.limit]:
        print(f"- {r.feature_name} | value={r.value} {r.unit} | type={r.ftype}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "feature_name",
                    "value",
                    "unit",
                    "type",
                    "feature_iri",
                    "feature_class_iri",
                    "comment",
                ]
            )
            for r in recs:
                w.writerow(
                    [
                        r.feature_name,
                        r.value,
                        r.unit,
                        r.ftype,
                        r.feature_iri,
                        r.feature_class_iri,
                        r.comment,
                    ]
                )
        print(f"Wrote features to {args.out}")
    return 0


def _cmd_generate(args):
    out = generate_instances(
        parameters_csv=args.parameters_csv,
        out_dir=args.out,
        n=args.n,
        seed=args.seed,
        missing_rate=args.missing_rate,
    )
    print(f"Wrote dataset: {out}")
    return 0


def _cmd_similarity(args):
    weights = _parse_weights(args.weights)
    results = top_k_jaccard(
        instances_csv=args.instances,
        query_design=args.design,
        k=args.k,
        paramtype_weights=(weights if weights else None),
    )
    if weights:
        print(f"Using ParamType weights: {weights}")
    print(f"TOP {len(results)} Similar Designs for {args.design}")
    for rank, (other, score) in enumerate(results, start=1):
        print(f"{rank:>2}. {other}  similarity={score:.4f}")

    return 0


def _cmd_inspect(args):
    inv = inspect_ontology(args.owl)

    print(f"Loaded ontology: {args.owl}")
    print(f"Classes: {len(inv.classes)}")
    print(f"Defined object properties: {len(inv.object_props_defined)}")
    print(f"Defined data properties:   {len(inv.data_props_defined)}")
    print(f"Defined annotation props:  {len(inv.annotation_props_defined)}")

    print("\nObject properties (defined):")
    for name in inv.object_props_defined[: args.top]:
        print(f"- {name}")

    print("\nData properties (defined):")
    for name in inv.data_props_defined[: args.top]:
        print(f"- {name}")

    print("\nAnnotation properties (defined):")
    for name in inv.annotation_props_defined[: args.top]:
        print(f"- {name}")

    return 0


def _cmd_deps(args):
    g = extract_dependency_graph(args.owl, include_prefix=args.prefix)

    print(f"Loaded ontology: {args.owl}")
    print(f"Dependency graph nodes: {len(g.edges)}")
    print(f"Dependency graph edges: {sum(len(v) for v in g.edges.values())}")

    print("\nTop relations (by asserted edges):")
    for rel, n in g.rel_counts.most_common(args.top):
        print(f"{n:>6}  {rel}")

    return 0


def _cmd_similarity_numeric(args):
    results = top_k_numeric_similarity(
        instances_csv=args.instances,
        query_design=args.design,
        k=args.k,
    )

    print(f"TOP {len(results)} numerisch ähnliche Designs für {args.design}")
    for rank, (other, score) in enumerate(results, start=1):
        print(f"{rank:>2}. {other}  similarity={score:.4f}")

    return 0
