from __future__ import annotations

import argparse
from pathlib import Path

from rotor_owl.owl_loader import load_owl


def _cmd_load(args: argparse.Namespace) -> int:
    ont = load_owl(args.path)
    classes = list(ont.classes())
    print(f"Loaded ontology: {ont.base_iri or ont.name}")
    print(f"Number of classes: {len(classes)}")
    if args.list_classes:
        for c in classes[: args.limit]:
            print(c)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rotor_owl", description="Rotor OWL tools")
    sub = parser.add_subparsers(dest="command", required=True)

    p_load = sub.add_parser("load", help="Load an OWL file and show basic info")
    p_load.add_argument("path", type=Path, help="Path to .owl file")
    p_load.add_argument("--list-classes", action="store_true", help="Print classes")
    p_load.add_argument("--limit", type=int, default=20, help="Max classes to print")
    p_load.set_defaults(func=_cmd_load)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
