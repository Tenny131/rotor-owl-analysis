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
```

---
