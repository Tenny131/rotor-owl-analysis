# rotor-owl-analysis

[**English Version**](README_EN.md) | **Deutsche Version**

**Ähnlichkeitsanalyse für rotierende Maschinenkomponenten (Rotoren)**

Dieses Projekt ermöglicht:
* Ontologie-basierte Feature-Extraktion aus Rotor-Designs
* Generierung synthetischer Datensätze (CSV)
* **Multi-Methoden Similarity-Analyse**:
  * klassisch (gewichtete Parameter-Similarity)
  * ML-basiert (Vektorbasiert, PCA, Autoencoder, K-Means)
* **Interaktive Web-UI** (Streamlit) zur Visualisierung

Das System nutzt OWL-Ontologien zur semantischen Modellierung von Rotor-Parametern und Apache Jena Fuseki als SPARQL-Endpoint.

**Features:**
* Query-Rotor auswählen
* Similarity-Methode wählen (Regelbasiert, Vektorbasiert, PCA, Autoencoder, K-Means, Hybrid)
* Kategorie-Gewichte anpassen
* Top-k ähnliche Rotoren finden
* Detaillierte Parameter-Vergleiche ansehen
---

## 📋 Voraussetzungen

* [**Docker**](https://www.docker.com/products/docker-desktop/) installiert
  * Für Windows: WSL2 Backend empfohlen
* Alternativ funktioniert die manuelle lokale Installation mit:
  * [Python](https://www.python.org/downloads/release/python-31212/) **3.12**
  * [Apache Jena Fuseki](https://jena.apache.org/download/) **5.6.0**

### ⚠️ Manuelle Dateien (nicht im Repository)

Folgende Dateien sind aus Vertraulichkeitsgründen im `.gitignore` und müssen **manuell** in den `data/reference/` Ordner gelegt werden:

| Datei | Zielordner | Beschreibung |
|-------|-----------|-------------|
| `AE_Ontology_Entwurf_IN_Feedback.xlsx` | `data/reference/` | Excel mit Komponenten, Parametern und Abhängigkeiten |
| `Ontology_Base.owl` | `data/reference/` | Basis-Ontologie (OWL) |
| `parameters.csv` | `data/reference/` | Parameterdefinitionen für synthetische Daten |
| `parameter_auswahl.csv` | `data/reference/` | Parameter-Auswahl für Realdaten (44 Parameter) |
| `*.json` (WVSC-Rotordaten) | `data/reference/wvsc/` | Reale Rotor-JSON-Dateien |

---

## 🚀 Schnellstart mit Docker

### 1. Repository klonen

```powershell
# In einem Ordner deiner Wahl cmd/PowerShell als Admin öffnen und Repository klonen
git clone https://github.com/Tenny131/rotor-owl-analysis.git
cd rotor-owl-analysis
```

### 2. Docker-Container starten

```powershell
# Docker Desktop öffnen und sicherstellen, dass Docker läuft
# Services starten (Fuseki + Streamlit App)
docker-compose up -d

# Logs ansehen (optional)
docker-compose logs -f
```

**Services:**
* **Fuseki**: http://localhost:3030
* **Streamlit App**: http://localhost:8501

### 3. Ontologie in Fuseki laden

Docker erstellt das Dataset `rotors` automatisch, aber die Ontologie muss manuell hochgeladen werden:

1. Öffne http://localhost:3030
2. Login: `admin` / `admin`
3. Wähle Dataset `rotors`
4. "upload files" → `data/ontologien/rotor_ontologie.owl` hochladen
5. Hochladen klicken

### 4. Streamlit App nutzen

Öffne http://localhost:8501 im Browser.

 In der Streamlit-Sidebar die Fuseki-Umgebung "Docker" auswahlen, um die Verbindung zum SPARQL-Endpoint herzustellen.





## 🔧 Lokale Entwicklung (ohne Docker)

### 1. Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

### 2. Synthetische Datensätze generieren (Optional)

```powershell
# CLI-Interface
python -m rotor_owl.daten.dataset_generate --n 100 --v 2.0 --seed 42

# Parameter:
# --n: Anzahl Rotor-Varianten (Standard: 50)
# --v: Varianz-Faktor (Standard: 1.0, Bereich: 1.0-3.0)
#      1.0 = Standard-Streuung
#      2.0 = doppelte Parameter-Range
#      3.0 = dreifache Parameter-Range
# --seed: Reproduzierbarkeit (Optional)
# --missing: Prozent fehlende Werte (Standard: 0.0)
```

Erzeugt `data/generated/generated.csv` mit synthetischen Rotor-Parametern.

### 3. Ontologie erstellen

**Generierte Daten (synthetisch):**
```powershell
python -m rotor_owl.daten.ontology_generierte_daten
```
Erstellt `data/ontologien/rotor_ontologie.owl` aus `data/generated/generated.csv`.

**Realdaten (WVSC-JSON):**
```powershell
python -m rotor_owl.daten.ontology_realdaten
```
Erstellt `data/ontologien/rotor_ontologie_realdaten.owl` aus den JSON-Dateien in `data/reference/wvsc/`.


### 4. Fuseki manuell starten

```powershell
fuseki-server --loc=tdb2 --update /rotors
```

### 5. Ontologie in Fuseki laden
1. Öffne http://localhost:3030
2. Login: `admin` / `admin`
3. Gehe zu "manage datasets"
4. Erstelle Dataset `rotors`
5. "upload files" → `data/ontologien/rotor_ontologie.owl` hochladen
* Persistent speichern auswählen und Hochladen klicken

### 6. Streamlit lokal

```powershell
streamlit run src/rotor_owl/streamlit_app.py
```

Der Fuseki-Endpoint (Localhost/Docker) und der Dataset-Name werden direkt in der Streamlit-Sidebar konfiguriert.

---