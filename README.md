# rotor-owl-analysis

[**English Version**](README_EN.md) | **Deutsche Version**

**Ähnlichkeitsanalyse für rotierende Maschinenkomponenten (Rotoren)**

Dieses Projekt ermöglicht:
* Ontologie-basierte Feature-Extraktion aus Rotor-Designs
* Generierung synthetischer Datensätze (CSV)
* **Multi-Methoden Similarity-Analyse**:
  * klassisch (gewichtete Parameter-Similarity)
  * ML-basiert (kNN, PCA, Autoencoder, K-Means)
* **Interaktive Web-UI** (Streamlit) zur Visualisierung

Das System nutzt OWL-Ontologien zur semantischen Modellierung von Rotor-Parametern und Apache Jena Fuseki als SPARQL-Endpoint.

**Features:**
* Query-Rotor auswählen
* Similarity-Methode wählen (A-D)
* Kategorie-Gewichte anpassen
* Top-k ähnliche Rotoren finden
* Detaillierte Parameter-Vergleiche ansehen
---

## 📋 Voraussetzungen

* [**Docker**](https://www.docker.com/products/docker-desktop/) installiert
  * Für Windows: WSL2 Backend empfohlen
* Alternativ:
  * [Python](https://www.python.org/downloads/release/python-31212/) **3.12**
  * [Apache Jena Fuseki](https://jena.apache.org/download/) **5.6.0**

---

## 🚀 Schnellstart mit Docker

### 1. Repository klonen

```powershell
git clone https://github.com/Tenny131/rotor-owl-analysis.git
cd rotor-owl-analysis
```

### 2. Docker-Container starten

```powershell
# Services starten (Fuseki + Streamlit App)
docker-compose up -d

# Logs ansehen
docker-compose logs -f
```

**Services:**
* **Fuseki**: http://localhost:3030
* **Streamlit App**: http://localhost:8501

### 3. Ontologie in Fuseki laden

* Docker lädt die Ontologie automatisch hoch.

### 4. Streamlit App nutzen

Öffne http://localhost:8501 im Browser.


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
python src/rotor_owl/daten/dataset_generate.py --n 100 --v 2.0 --seed 42

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

```powershell
# Virtual Environment aktivieren
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python.exe -m pip install --upgrade pip
pip install -e .

# Ontologie erstellen
python src/rotor_owl/Ontology.py
```

Dies erstellt `data/rotor_ontologie.owl`.


### 4. Fuseki manuell starten

```powershell
fuseki-server --loc=tdb2 --update /rotors
```

### 5. Ontologie in Fuseki laden
1. Öffne http://localhost:3030
2. Login: `admin` / `admin`
3. Gehe zu "manage datasets"
4. Erstelle Dataset `rotors`
5. "upload files" → `data/rotor_ontologie.owl` hochladen
* Persistent speichern auswählen und Hochladen klicken

### 6. Streamlit lokal

```powershell
streamlit run src/rotor_owl/streamlit_app.py
```

Endpoint konfigurieren in `src/rotor_owl/konfiguration.py`:

```python
FUSEKI_ENDPOINT_STANDARD = "http://localhost:3030/rotors/sparql"
```
test
---