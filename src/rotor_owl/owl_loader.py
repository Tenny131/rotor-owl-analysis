from pathlib import Path
from owlready2 import get_ontology


def load_owl(path: str | Path):
    "Owl Ontologie laden und als ontologie Objekt zurückgeben"

    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Die angegebene Datei wurde nicht gefunden: {path}")
    ontology = get_ontology(str(path)).load()

    return ontology
