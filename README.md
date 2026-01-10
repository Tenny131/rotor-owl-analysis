# rotor-owl-analysis

Dieses Projekt stellt eine **Kommandozeilenanwendung (CLI)** zur Verfügung für

* Analyse von OWL-Ontologien (Struktur & Statistik)
* Extraktion parameterisierter Features aus Baugruppen
* Generierung synthetischer Instanzdatensätze (CSV)
* Ähnlichkeitsanalyse von Designs mittels **Top-k Jaccard Similarity**

  * optional **gewichtete** Similarity nach ParamType
* reproduzierbare Auswertung (Seed-basiert, CI-fähig)

Ziel ist es, **Similarity-Ansätze für rotierende Maschinenkomponenten**
(z. B. Rotoren, Wellen, Aktivteile) systematisch zu untersuchen.

---

## Voraussetzungen

* Python **≥ 3.12**
* Git
* Windows PowerShell (oder vergleichbares Terminal)

---

## Installation (Entwicklungsmodus)

```powershell
git clone https://github.com/<user>/rotor-owl-analysis.git
cd rotor-owl-analysis

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

Der `-e`-Modus stellt sicher, dass Änderungen am Code sofort wirksam sind.

---

## CLI-Überblick

```powershell
rotor-owl --help
```

Verfügbare Subcommands (Auszug):

* `stats` – Ontologie-Statistiken
* `features` – Feature-Extraktion aus OWL
* `generate` – Instanz-Datensätze erzeugen
* `similarity` – Top-k Jaccard-Similarity

---

## Ontologie analysieren

### Ontologie-Statistik anzeigen

```powershell
rotor-owl stats example.owl --top-prefixes 5
```

Ausgabe u. a.:

* Anzahl Klassen
* Anzahl Object Properties
* Anzahl Data Properties
* Anzahl Individuen
* wichtigste IRI-Namensräume

---

## Features aus einer Baugruppe extrahieren

```powershell
rotor-owl features example.owl \
  --assembly-iri "http://ontology.innomotics.net/ims#C_WELLE_1" \
  --limit 20
```

Extrahiert parameterisierte Features (z. B. Geometrie, Material, Anforderungen),
die über Relationen wie `ims:composed_of` an eine konkrete Baugruppe gebunden sind.

### Feature-Export als CSV

```powershell
rotor-owl features example.owl \
  --assembly-iri "http://ontology.innomotics.net/ims#C_WELLE_1" \
  --out features_welle.csv
```

CSV-Spalten:

* `feature_name`
* `value`
* `unit`
* `type`
* `feature_iri`
* `feature_class_iri`
* `comment`

---

## Instanzdatensatz generieren (synthetisch)

Zur Entwicklung und zum Testen werden **reproduzierbare CSV-Datensätze**
aus der Feature-Struktur erzeugt.

```powershell
rotor-owl generate --n 10 --seed 1
```

Ergebnis:

```text
data/generated/instances.csv
```

**Interpretation:**

* Jede `Design_ID` (z. B. `D001`) entspricht **einer Instanz / einem Design**
* Jede Zeile ist ein **Parameter-Feature** dieser Instanz
* `IsMissing=1` kennzeichnet bewusst fehlende Werte
* Seed garantiert reproduzierbare Datensätze

---

## Similarity-Analyse (Top-k Jaccard)

### Ungewichtete Similarity

```powershell
rotor-owl similarity data/generated/instances.csv D001 --k 5
```

Ausgabe (Beispiel):

```text
Query: D001
TOP-5 similar designs (Jaccard):

D002  similarity=0.8537
D003  similarity=0.8124
D004  similarity=0.7912
...
```

---

### Gewichtete Similarity (ParamType-Gewichte)

```powershell
rotor-owl similarity data/generated/instances.csv D001 \
  --k 5 \
  --weights GEOM=1.0,REQ=0.3,DYN=1.2
```

* Gewichtung erfolgt **auf Feature-Ebene**
* Standard: alle ParamTypes Gewicht = 1.0
* Semantische Erweiterungen (Relationen, Abhängigkeiten) sind vorbereitet

---

## Entwicklung & Qualitätssicherung

### Code-Qualität prüfen

```powershell
ruff check .
```

### Tests ausführen

```powershell
pytest -v
```

Alle Similarity-Tests prüfen ausschließlich **Top-k-Ergebnisse**
(keine redundanten Paarvergleiche).

---

## Continuous Integration (CI)

* Automatische Ausführung von

  * `ruff`
  * `pytest`
* bei jedem Push & Pull Request über **GitHub Actions**
