# rotor-owl-analysis

**English Version** | [**Deutsche Version**](README.md)

**Similarity Analysis for Rotating Machine Components (Rotors)**

This project enables:
* Ontology-based feature extraction from rotor designs
* Generation of synthetic datasets (CSV)
* **Multi-method similarity analysis**:
  * classical (weighted parameter similarity)
  * ML-based (kNN, PCA, Autoencoder, K-Means)
* **Interactive web UI** (Streamlit) for visualization

The system uses OWL ontologies for semantic modeling of rotor parameters and Apache Jena Fuseki as SPARQL endpoint.

**Features:**
* Select query rotor
* Choose similarity method (A-D)
* Adjust category weights
* Find top-k similar rotors
* View detailed parameter comparisons
---

## 📋 Requirements

* [**Docker**](https://www.docker.com/products/docker-desktop/) installed
  * For Windows: WSL2 backend recommended
* Alternatively:
  * [Python](https://www.python.org/downloads/release/python-31212/) **3.12**
  * [Apache Jena Fuseki](https://jena.apache.org/download/) **5.6.0**

---

## 🚀 Quick Start with Docker

### 1. Clone repository

```powershell
git clone https://github.com/Tenny131/rotor-owl-analysis.git
cd rotor-owl-analysis
```

### 2. Start Docker containers

```powershell
# Start services (Fuseki + Streamlit App)
docker-compose up -d

# View logs
docker-compose logs -f
```

**Services:**
* **Fuseki**: http://localhost:3030
* **Streamlit App**: http://localhost:8501

### 3. Load ontology into Fuseki

* Docker uploads the ontology automatically.

### 4. Use Streamlit app

Open http://localhost:8501 in browser.


## 🔧 Local Development (without Docker)

### 1. Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

### 2. Generate synthetic datasets (Optional)

```powershell
# CLI interface
python src/rotor_owl/daten/dataset_generate.py --n 100 --v 2.0 --seed 42

# Parameters:
# --n: Number of rotor variants (default: 50)
# --v: Variance factor (default: 1.0, range: 1.0-3.0)
#      1.0 = standard deviation
#      2.0 = double parameter range
#      3.0 = triple parameter range
# --seed: Reproducibility (optional)
# --missing: Percentage of missing values (default: 0.0)
```

Creates `data/generated/generated.csv` with synthetic rotor parameters.

### 3. Create ontology

```powershell
# Activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .

# Create ontology
python src/rotor_owl/Ontology.py
```

This creates `data/rotor_ontologie.owl`.


### 4. Start Fuseki manually

```powershell
fuseki-server --loc=tdb2 --update /rotors
```

### 5. Load ontology into Fuseki
1. Open http://localhost:3030
2. Login: `admin` / `admin`
3. Go to "manage datasets"
4. Create dataset `rotors`
5. "upload files" → upload `data/rotor_ontologie.owl`
* Select persistent storage and click upload

### 6. Streamlit locally

```powershell
streamlit run src/rotor_owl/streamlit_app.py
```

Configure endpoint in `src/rotor_owl/konfiguration.py`:

```python
FUSEKI_ENDPOINT_STANDARD = "http://localhost:3030/rotors/sparql"
```

---

## 🔬 Validation of Similarity Methods

### Problem

Graph-Embeddings show only 4.5% range with identical rotor structure.
All rotors have identical RDF structure (only parameter values differ).

### Solution

Validation without expert opinions using 5 statistical tests:

1. Physical plausibility (correlation with power/geometry)
2. Silhouette score (cluster quality)
3. Extreme cases (identical vs. maximum differences)
4. Spread analysis (range, coefficient of variation)
5. Bootstrap stability (ranking consistency)

### Run validation

```powershell
python validate_similarities.py
```

Creates:
* `data/similarity_validation.png` (visualization)
* `data/similarity_validation.pdf` (vector format)
* `temp/similarity_*.csv` (raw data with timestamp)

### Results

| Method | Range | CV | Kendall-Tau | Rating |
|---------|-------|----|----|-------|
| k-NN | 54.7% | 11.2% | 1.0 | Excellent |
| Autoencoder | 93.1% | 142.2% | 1.0 | Excellent |
| Graph-Embeddings | 6.1% | 0.8% | 0.99 | Poor |

Interpretation:
* k-NN and Autoencoder are complementary (different feature spaces)
* Graph-Embeddings unsuitable with identical structure
* Hybrid method uses 50% Autoencoder + 50% k-NN (Graph replaced)

Validation results are displayed in Streamlit UI at the bottom (expandable).

---
