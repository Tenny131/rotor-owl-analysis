from __future__ import annotations

from rotor_owl.cli import main


def test_cli_stats_outputs_counts(tmp_path, capsys, mini_owl):
    owl_path = tmp_path / "mini.owl"
    owl_path.write_text(mini_owl, encoding="utf-8")

    rc = main(["stats", str(owl_path), "--top-prefixes", "3"])
    assert rc == 0

    out = capsys.readouterr().out
    assert "Classes:" in out
    assert "Object props:" in out
    assert "Data props:" in out
    assert "Individuals:" in out
    assert "http://ontology.innomotics.net/ims#" in out
