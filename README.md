# rotor-owl-analysis

Das Projekt stellt eine Kommandozeilenanwendung (CLI) bereit, um
- strukturelle Kennzahlen einer Ontologie zu analysieren
- parameterisierte Features aus konkreten Baugruppen zu extrahieren
- Ergebnisse reproduzierbar weiterzuverarbeiten (z. B. CSV-Export)

---

## Voraussetzungen
- Python ≥ 3.12
- Git
- Windows PowerShell (oder vergleichbares Terminal)

---

## Installation (Entwicklungsmodus)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

### Ontologie-Statistik anzeigen

```powershell
rotor-owl stats example.owl --top-prefixes 5
```

Ausgabe u. a.:

* Anzahl Klassen
* Anzahl Objekt-Properties
* Anzahl Daten-Properties
* Anzahl Individuen
* wichtigste IRI-Namensräume

---

### Features einer Baugruppe extrahieren

```powershell
rotor-owl features example.owl \
    --assembly-iri "http://ontology.innomotics.net/ims#C_WELLE_1" \
    --limit 20
```

Extrahiert parameterisierte Features (z. B. Geometrie, Material, Toleranzen),
die über Relationen wie `ims:composed_of` an eine konkrete Baugruppe gebunden sind.

---

### Feature-Export als CSV

```powershell
rotor-owl features example.owl \
    --assembly-iri "http://ontology.innomotics.net/ims#C_WELLE_1" \
    --out features_welle.csv
```

CSV-Spalten:

* feature_name
* value
* unit
* type
* feature_iri
* feature_class_iri
* comment

---

## Entwicklung & Qualitätssicherung

### Code-Qualität prüfen

```powershell
ruff check .
```

### Tests ausführen

```powershell
pytest
```

Automatisierte Tests und Linting werden zusätzlich über **GitHub Actions (CI)**
bei jedem Push ausgeführt.

---
