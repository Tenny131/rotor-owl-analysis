# Rotor-OWL – befehle

## Installation & Setup

```bash
.\.venv\Scripts\activate

pip install -e .
```

## Datensatz-Generierung (CSV)

```bash
rotor-owl generate --n 50 --missing-rate 0.05 --seed 1 --out data/generated
```

---

## Ontologie Laden und inspizieren

```bash
rotor-owl load example.owl --list-classes --limit 20

rotor-owl inspect example.owl --top 10

rotor-owl stats example.owl

rotor-owl features example.owl --assembly-iri http://ontology.innomotics.net/ims#Rotor_1 --limit 10

rotor-owl deps example.owl --top 10
```


## Numerische Aehnlichkeit

```bash
rotor-owl similarity-numeric data/generated/instances.csv D001 --k 5

python src/rotor_owl/topk_similarity_fuseki.py --query Rotor_D001 --k 5 --weights "GEOM=2,MTRL=1,STRUCT=1,DYN=0.5,REQ=0.2,MFG=0.2,ELEC=0.2,UNKNOWN=0"

streamlit run src/rotor_owl/app_similarity_ui.py
```

---
