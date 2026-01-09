from pathlib import Path

from rotor_owl.owl_loader import load_owl


def test_load_owl_loads_local_example():
    owl_path = Path("example.owl")
    ont = load_owl(owl_path)
    assert ont is not None
