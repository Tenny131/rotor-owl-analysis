# 🏗️ Projekt-Architektur: Rotor-OWL-Analysis

## 📊 System-Übersicht (High-Level)

```mermaid
flowchart TB
    subgraph Input["📥 DATENQUELLEN"]
        OWL[("OWL Ontologie<br/>rotor_ontologie.owl")]
        CSV[("Parameter CSV<br/>parameters.csv")]
        GEN[("Generated Data<br/>generated.csv")]
    end

    subgraph Fuseki["🗄️ APACHE JENA FUSEKI"]
        SPARQL["SPARQL Endpoint<br/>:3030/rotors"]
    end

    subgraph Backend["⚙️ PYTHON BACKEND"]
        FETCH["Feature Fetcher<br/>SPARQL → Python Dict"]
        DEP["Dependency Extractor<br/>Constraints → Weights"]
        METH["7 Similarity Methods<br/>Regelbasiert, k-NN, etc."]
    end

    subgraph Frontend["🖥️ STREAMLIT UI"]
        UI["Interactive Dashboard<br/>Query + Results"]
    end

    subgraph Output["📤 ERGEBNISSE"]
        VIS["Visualisierungen<br/>Plotly Charts"]
        CSV_OUT["CSV Exports<br/>Similarity Matrizen"]
    end

    OWL --> Fuseki
    CSV --> GEN
    GEN --> Fuseki
    Fuseki --> FETCH
    FETCH --> DEP
    FETCH --> METH
    DEP --> METH
    METH --> UI
    UI --> VIS
    UI --> CSV_OUT

    style Input fill:#e1f5ff
    style Fuseki fill:#fff4e1
    style Backend fill:#e8f5e9
    style Frontend fill:#f3e5f5
    style Output fill:#fce4ec
```

---

## 🔄 Datenfluss-Diagramm (Detailed)

```mermaid
flowchart LR
    subgraph Stage1["1️⃣ DATEN LADEN"]
        A1["fetch_all_features()"]
        A2["fetch_component_dependencies()"]
    end

    subgraph Stage2["2️⃣ PREPROCESSING"]
        B1["normalize_param_name()"]
        B2["safe_float()"]
        B3["build_numeric_stats()"]
    end

    subgraph Stage3["3️⃣ GEWICHTUNG"]
        C1{"Auto-Gewichte?"}
        C2["berechne_automatische_gewichte()"]
        C3["map_komponenten_zu_kategorie_gewichte()"]
        C4["Manuelle Slider"]
    end

    subgraph Stage4["4️⃣ SIMILARITY"]
        D1["Regelbasiert<br/>Parameter-weise"]
        D2["k-NN<br/>Feature Vektoren"]
        D3["Autoencoder<br/>Latent Space"]
        D4["Graph-Embeddings<br/>Node2Vec"]
        D5["PCA<br/>Dimensionsreduktion"]
        D6["KMeans<br/>Clustering"]
        D7["Hybrid<br/>Kombiniert"]
    end

    subgraph Stage5["5️⃣ AUSGABE"]
        E1["Top-k Ranking"]
        E2["Similarity Matrix"]
        E3["Visualisierung"]
    end

    A1 --> B1
    A1 --> B2
    A2 --> C1
    B1 --> B3
    B2 --> B3
    B3 --> C1

    C1 -->|JA| C2
    C2 --> C3
    C1 -->|NEIN| C4
    C3 --> Stage4
    C4 --> Stage4

    Stage4 --> E1
    Stage4 --> E2
    E1 --> E3
    E2 --> E3

    style Stage1 fill:#bbdefb
    style Stage2 fill:#c5e1a5
    style Stage3 fill:#fff9c4
    style Stage4 fill:#ffccbc
    style Stage5 fill:#f8bbd0
```

---

## 📁 Verzeichnisstruktur mit Funktionen

```mermaid
graph TD
    ROOT["📦 rotor-owl-analysis"]
    
    ROOT --> DATA["📂 data/"]
    ROOT --> SRC["📂 src/rotor_owl/"]
    ROOT --> TESTS["📂 tests/"]
    ROOT --> CONFIG["📄 Config Files"]

    DATA --> OWL["rotor_ontologie.owl<br/>💾 OWL/RDF Ontologie"]
    DATA --> REF["reference/<br/>📋 Parameter-Templates"]
    DATA --> GEN_DIR["generated/<br/>🎲 Synthetische Daten"]

    SRC --> CONFIG_PY["config/<br/>⚙️ Kategorien, Konfiguration"]
    SRC --> DATEN["daten/<br/>📊 Feature Extraction"]
    SRC --> METHODEN["methoden/<br/>🧮 Similarity Algorithms"]
    SRC --> UTILS["utils/<br/>🔧 Helper Functions"]
    SRC --> APP["streamlit_app.py<br/>🖥️ UI Entry Point"]

    DATEN --> FETCH["feature_fetcher.py<br/>SPARQL → Dict"]
    DATEN --> ONTO["Ontology.py<br/>OWL Generation"]
    DATEN --> DATASET["dataset_generate.py<br/>Synthetic Data"]

    METHODEN --> REGEL["regelbasierte_aehnlichkeit.py<br/>✓ Numerisch + Kategorisch"]
    METHODEN --> KNN_M["knn_aehnlichkeit.py<br/>✓ Feature Vectors"]
    METHODEN --> AUTO["autoencoder_aehnlichkeit.py<br/>✓ Neural Network"]
    METHODEN --> GRAPH["graph_embedding_aehnlichkeit.py<br/>✓ Node2Vec"]
    METHODEN --> PCA_M["pca_aehnlichkeit.py<br/>✓ PCA Embeddings"]
    METHODEN --> KMEANS["kmeans_aehnlichkeit.py<br/>✓ Clustering"]
    METHODEN --> HYB["hybrid_aehnlichkeit.py<br/>✓ Multi-Method"]

    UTILS --> AUFB["aufbereitung.py<br/>normalize, safe_float"]
    UTILS --> MATH["math_utils.py<br/>cosine_similarity"]
    UTILS --> SPARQL_U["sparql_helpers.py<br/>run_sparql()"]

    TESTS --> T1["test_automatic_weights.py<br/>✓ 6 Tests"]
    TESTS --> T2["test_graph_embeddings.py<br/>✓ 5 Tests"]
    TESTS --> T3["test_*.py<br/>✓ 54 Tests Total"]

    CONFIG --> DOCKER["🐳 Dockerfile + compose"]
    CONFIG --> PYPROJ["📦 pyproject.toml"]
    CONFIG --> REQ["📋 requirements.txt"]

    style ROOT fill:#1976d2,color:#fff
    style DATA fill:#4caf50,color:#fff
    style SRC fill:#ff9800,color:#fff
    style TESTS fill:#9c27b0,color:#fff
    style METHODEN fill:#f44336,color:#fff
```

---

## 🛠️ Tech-Stack

### **Backend**
```mermaid
graph LR
    PY["🐍 Python 3.12"]
    
    PY --> ML["Machine Learning"]
    PY --> OWL_LIB["Ontology"]
    PY --> WEB["Web Framework"]
    PY --> DATA_LIB["Data Processing"]

    ML --> SK["scikit-learn<br/>k-NN, PCA, KMeans"]
    ML --> GS["gensim<br/>Word2Vec/Node2Vec"]
    ML --> NP["numpy<br/>Numerik"]

    OWL_LIB --> RDF["rdflib<br/>OWL/RDF Parsing"]
    OWL_LIB --> SPARQL["SPARQLWrapper<br/>SPARQL Queries"]

    WEB --> ST["Streamlit<br/>Interactive UI"]
    WEB --> PL["Plotly<br/>Visualisierungen"]

    DATA_LIB --> PD["pandas<br/>DataFrames"]
    DATA_LIB --> NX["networkx<br/>Graph Operations"]

    style PY fill:#3776ab,color:#fff
    style ML fill:#ff6f00,color:#fff
    style OWL_LIB fill:#4caf50,color:#fff
    style WEB fill:#9c27b0,color:#fff
```

### **Begründungen (Stichpunkte)**

| Technologie | Zweck | Begründung |
|-------------|-------|------------|
| **Python 3.12** | Hauptsprache | • Starke ML/Data-Science Libraries<br/>• OWL/RDF Support (rdflib)<br/>• Rapid Prototyping |
| **Apache Jena Fuseki** | Triple Store | • SPARQL-Endpoint<br/>• OWL-Reasoning<br/>• Skalierbar |
| **Streamlit** | UI Framework | • Schnelle Entwicklung<br/>• Python-nativ<br/>• Interaktive Widgets |
| **scikit-learn** | ML Algorithms | • k-NN, PCA, KMeans<br/>• Standard-Library<br/>• Gut dokumentiert |
| **gensim** | Graph Embeddings | • Node2Vec Implementation<br/>• Word2Vec (Skip-Gram)<br/>• Effizient |
| **rdflib** | OWL/RDF | • OWL-Ontologie Parsing<br/>• SPARQL Integration<br/>• Python-nativ |
| **networkx** | Graph Operations | • RDF → NetworkX Konvertierung<br/>• Random Walk Generation<br/>• Visualisierung |
| **Plotly** | Visualisierung | • Interaktive Charts<br/>• Export (PNG, PDF)<br/>• Professionell |

---

## 🔀 Similarity-Methoden Entscheidungsbaum

```mermaid
flowchart TD
    START{"Welche Methode?"}
    
    START -->|"Transparent & Erklärbar"| REGEL["📊 Regelbasiert<br/><b>Parameter-weise</b>"]
    START -->|"Feature-basiert"| FEAT{"Dimensionalität?"}
    START -->|"Graph-Struktur"| GRAPH_M["🕸️ Graph-Embeddings<br/><b>Node2Vec</b>"]
    START -->|"Kombination"| HYB_M["🔀 Hybrid<br/><b>Multi-Method</b>"]

    FEAT -->|"Hoch-dimensional"| DIM{"Reduktion?"}
    FEAT -->|"Low-dimensional"| KNN_M["📐 k-NN<br/><b>Cosine Similarity</b>"]

    DIM -->|"Linear"| PCA_DIA["🔍 PCA<br/><b>Variance Maximierung</b>"]
    DIM -->|"Non-linear"| AUTO_DIA["🧠 Autoencoder<br/><b>Latent Space</b>"]
    DIM -->|"Clustering"| KMEANS_DIA["🎯 KMeans<br/><b>Cluster Distance</b>"]

    REGEL -.->|"Range: 54.7%"| RES1["✅ Gut interpretierbar"]
    KNN_M -.->|"Range: 54.7%"| RES2["✅ Beste Balance"]
    AUTO_DIA -.->|"Range: 93.1%"| RES3["✅ Höchste Diskriminierung"]
    GRAPH_M -.->|"Range: 6.1%"| RES4["⚠️ Schwach (identische Struktur)"]
    PCA_DIA -.->|"Range: variabel"| RES5["✓ Effizient"]
    KMEANS_DIA -.->|"Range: variabel"| RES6["✓ Gruppen-basiert"]
    HYB_M -.->|"Kombiniert beste"| RES7["✅ Optimal für Produktion"]

    style START fill:#1976d2,color:#fff
    style REGEL fill:#4caf50,color:#fff
    style KNN_M fill:#4caf50,color:#fff
    style AUTO_DIA fill:#4caf50,color:#fff
    style GRAPH_M fill:#ff9800,color:#fff
    style HYB_M fill:#9c27b0,color:#fff
```

---

## 📈 Validation-Pipeline

```mermaid
flowchart TB
    subgraph VAL["🔬 VALIDATION PROCESS"]
        V1["1. Lade 50 Rotoren"]
        V2["2. Berechne Similarity Matrix<br/>50x50 = 2,500 Vergleiche"]
        V3["3. Statistiken<br/>Range, CV, Kendall-Tau"]
        V4{"Kriterien erfüllt?"}
        V5["✅ BESTANDEN"]
        V6["⚠️ GRENZFALL"]
        V7["❌ DURCHGEFALLEN"]
    end

    V1 --> V2
    V2 --> V3
    V3 --> V4
    
    V4 -->|"Range ≥ 5%<br/>CV ≥ 3%"| V5
    V4 -->|"1-2 Kriterien"| V6
    V4 -->|"0 Kriterien"| V7

    V5 -.->|"k-NN: 54.7%"| OK1["Produktionsreif"]
    V5 -.->|"Autoencoder: 93.1%"| OK2["Produktionsreif"]
    V6 -.->|"Graph: 6.1%"| WARN["Eingeschränkt nutzbar"]

    style VAL fill:#e8eaf6
    style V5 fill:#4caf50,color:#fff
    style V6 fill:#ff9800,color:#fff
    style V7 fill:#f44336,color:#fff
```

---

## 🎯 Verwendungszwecke

| Use Case | Empfohlene Methode | Begründung |
|----------|-------------------|------------|
| **🔍 Ähnliche Rotoren finden** | k-NN oder Autoencoder | Hohe Diskriminierung (54-93%) |
| **📊 Explainable AI** | Regelbasiert | Parameter-weise Similarity transparent |
| **🏭 Produktion (Best Quality)** | Hybrid (Autoencoder + k-NN) | Kombiniert Pattern + Attribute |
| **⚡ Schnelle Suche** | k-NN | Einfache Cosine Similarity |
| **🧪 Explorative Analyse** | Alle Methoden parallel | Vergleich der Ergebnisse |
| **🤖 Automatische Gewichtung** | Auto-Gewichte + Regelbasiert | Dependencies aus Ontologie |

---

## 🚀 Deployment-Architektur

```mermaid
graph TB
    subgraph Docker["🐳 DOCKER CONTAINER"]
        FUSE["Apache Jena Fuseki<br/>Port 3030"]
        STREAM["Streamlit App<br/>Port 8501"]
    end

    subgraph Host["💻 HOST SYSTEM"]
        BROWSER["🌐 Web Browser<br/>localhost:8501"]
        DATA_VOL["📁 Volume Mount<br/>./data:/data"]
    end

    BROWSER --> STREAM
    STREAM --> FUSE
    DATA_VOL --> FUSE

    style Docker fill:#0db7ed,color:#fff
    style FUSE fill:#ff6f00,color:#fff
    style STREAM fill:#ff4785,color:#fff
```

**Vorteile Docker-Setup:**
- ✅ Reproduzierbar (gleiche Umgebung überall)
- ✅ Isoliert (keine Konflikte mit Host-System)
- ✅ Einfaches Deployment (`docker-compose up`)
- ✅ Volume Mounts für Live-Daten-Updates

---

## 📊 Performance-Charakteristik

```mermaid
graph LR
    subgraph Metrics["📈 VALIDATION METRIKEN"]
        M1["Range<br/>Diskriminierung"]
        M2["CV<br/>Variabilität"]
        M3["Kendall-Tau<br/>Korrelation"]
    end

    subgraph Results["🏆 ERGEBNISSE"]
        R1["k-NN: 54.7%<br/>CV: 11.2%"]
        R2["Autoencoder: 93.1%<br/>CV: 142%"]
        R3["Graph: 6.1%<br/>CV: 0.82%"]
    end

    M1 --> R1
    M1 --> R2
    M1 --> R3
    M2 --> R1
    M2 --> R2
    M2 --> R3

    R1 -.->|"Empfehlung"| BEST1["✅ Produktion"]
    R2 -.->|"Empfehlung"| BEST2["✅ Beste Diskriminierung"]
    R3 -.->|"Warnung"| WARN2["⚠️ Nur für identische Struktur"]

    style Results fill:#e8f5e9
    style BEST1 fill:#4caf50,color:#fff
    style BEST2 fill:#4caf50,color:#fff
    style WARN2 fill:#ff9800,color:#fff
```

---

## 🔧 Entwicklungs-Workflow

```mermaid
flowchart LR
    subgraph Dev["👨‍💻 ENTWICKLUNG"]
        CODE["Code schreiben"]
        TEST["pytest tests/"]
        LINT["ruff check"]
    end

    subgraph Val["🔬 VALIDIERUNG"]
        RUN["validate_similarities.py"]
        CHECK["Metriken prüfen"]
    end

    subgraph Deploy["🚀 DEPLOYMENT"]
        DOCKER_B["docker build"]
        DOCKER_R["docker-compose up"]
    end

    CODE --> TEST
    TEST --> LINT
    LINT --> RUN
    RUN --> CHECK
    CHECK --> DOCKER_B
    DOCKER_B --> DOCKER_R

    style Dev fill:#bbdefb
    style Val fill:#c5e1a5
    style Deploy fill:#ffccbc
```

**Qualitätssicherung:**
- ✅ **54 Unit-Tests** (pytest)
- ✅ **Ruff Linting** (pre-commit hooks)
- ✅ **Type Hints** (Python 3.12+)
- ✅ **Docstrings** (standardisiert)
- ✅ **Validation Framework** (automatisiert)

---

## 📝 Zusammenfassung

**Kernkomponenten:**
1. 🗄️ **Apache Jena Fuseki** → SPARQL Triple Store
2. 🐍 **Python Backend** → 7 Similarity-Methoden
3. 🖥️ **Streamlit UI** → Interaktive Suche
4. 🔬 **Validation Framework** → Automatische Qualitätsprüfung

**Best Practices implementiert:**
- ✅ Dependency-basierte Auto-Gewichtung
- ✅ Comprehensive Testing (54 Tests)
- ✅ Docker-basiertes Deployment
- ✅ Standardisierte Dokumentation
- ✅ Production-ready Code

**Empfohlene Methode für Produktion:**
🏆 **Hybrid (Autoencoder + k-NN)** → Beste Balance zwischen Genauigkeit und Performance
