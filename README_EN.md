# rotor-owl-analysis

**English Version** | [**Deutsche Version**](README.md)

**Similarity Analysis for Rotating Machine Components (Rotors)**

This project enables:
* Ontology-based feature extraction from rotor designs
* Generation of synthetic datasets (CSV)
* **Multi-method similarity analysis**:
  * classical (weighted parameter similarity)
  * ML-based (Vector-based, PCA, Autoencoder, K-Means)
* **Interactive web UI** (Streamlit) for visualization

The system uses OWL ontologies for semantic modeling of rotor parameters and Apache Jena Fuseki as SPARQL endpoint.

**Features:**
* Select query rotor
* Choose similarity method (Rule-based, Vector-based, PCA, Autoencoder, K-Means, Hybrid)
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

### ⚠️ Manual Files (not in repository)

The following files are excluded via `.gitignore` for confidentiality reasons and must be **manually** placed in the `data/reference/` folder:

| File | Target folder | Description |
|------|--------------|-------------|
| `AE_Ontology_Entwurf_IN_Feedback.xlsx` | `data/reference/` | Excel with components, parameters, and dependencies |
| `Ontology_Base.owl` | `data/reference/` | Base ontology (OWL) |
| `parameters.csv` | `data/reference/` | Parameter definitions for synthetic data |
| `parameter_auswahl.csv` | `data/reference/` | Parameter selection for real data (44 parameters) |
| `*.json` (WVSC rotor data) | `data/reference/wvsc/` | Real rotor JSON files |

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

Docker creates the `rotors` dataset automatically, but the ontology must be uploaded manually:

1. Open http://localhost:3030
2. Login: `admin` / `admin`
3. Select dataset `rotors`
4. "upload files" → upload `data/ontologien/rotor_ontologie.owl`
5. Select persistent storage and click upload

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
python -m rotor_owl.daten.dataset_generate --n 100 --v 2.0 --seed 42

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

**Generated data (synthetic):**
```powershell
python -m rotor_owl.daten.ontology_generierte_daten
```
Creates `data/ontologien/rotor_ontologie.owl` from `data/generated/generated.csv`.

**Real data (WVSC-JSON):**
```powershell
python -m rotor_owl.daten.ontology_realdaten
```
Creates `data/ontologien/rotor_ontologie_realdaten.owl` from JSON files in `data/reference/wvsc/`.


### 4. Start Fuseki manually

```powershell
fuseki-server --loc=tdb2 --update /rotors
```

### 5. Load ontology into Fuseki
1. Open http://localhost:3030
2. Login: `admin` / `admin`
3. Go to "manage datasets"
4. Create dataset `rotors`
5. "upload files" → upload `data/ontologien/rotor_ontologie.owl`
* Select persistent storage and click upload

### 6. Streamlit locally

```powershell
streamlit run src/rotor_owl/streamlit_app.py
```

The Fuseki endpoint (Localhost/Docker) and dataset name are configured directly in the Streamlit sidebar.

---

## 🔬 Similarity Methods

| Method | Type | Description |
|--------|------|-------------|
| Rule-based | no ML | Parameter-wise comparison with weighted aggregation |
| Vector-based | ML | Feature vectors per category, cosine similarity |
| PCA-Embedding | ML | Dimensionality reduction via PCA, cosine similarity |
| Autoencoder | ML | Neural network for latent representations |
| K-Means Clustering | ML | Cluster assignment, centroid distance similarity |
| Hybrid | ML | Weighted combination of two selected methods |

---
