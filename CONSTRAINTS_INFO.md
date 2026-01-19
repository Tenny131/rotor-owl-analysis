# Constraints für Auto-Gewichtsberechnung

## Verwendete OWL-Konstrukte

### 1. Object Properties (Relationen)

**Pattern:** `*_Beeinflusst_*`

Extrahiert via SPARQL:
```sparql
SELECT ?property ?strength ?percentage
WHERE {
  ?property rdf:type owl:ObjectProperty .
  OPTIONAL { ?property ims:hasStrength ?strength . }
  OPTIONAL { ?property ims:hasDependencyPercentage ?percentage . }
  
  FILTER(CONTAINS(STR(?property), "Beeinflusst"))
}
```

**Beispiele aus der Ontologie:**
- `Blechpaket_Beeinflusst_Welle_1`
- `Aktivteil_Beeinflusst_Welle_1`
- `Wuchtscheiben_Beeinflusst_Welle_1`
- `Welle.Geometrie_Beeinflusst_Blechpaket.Geometrie_1`
- `Welle.Wellenabsatz_Beeinflusst_Innenlüfter.Position_1`
- `Aktivteil_Beeinflusst_Maschinenparameter_1`

### 2. Data Properties (Attribute)

Jede `*_Beeinflusst_*` Relation hat:

**`ims:hasStrength`**
- Typ: String
- Werte: `"hoch"` | `"mittel"` | `"niedrig"`
- Bedeutung: Qualitative Stärke der Abhängigkeit

**`ims:hasDependencyPercentage`**
- Typ: Float
- Wertebereich: 0.0 - 1.0
- Bedeutung: Quantitative Stärke der Abhängigkeit
- Beispiele: 0.8 (80%), 0.6 (60%), 0.3 (30%)

## Verwendung im System

### Auto-Gewichte AN (Checkbox aktiv)

**1. Kategorie-Gewichte:**
```
Dependencies laden
  → Komponenten-Gewichte berechnen (via DependencyPercentage)
  → Auf Kategorien mappen (GEOM_MECH, MTRL_PROC, REQ_ELEC)
  → Für ALLE Methoden verwenden
```

**2. Graph-Embeddings:**
```
Dependencies laden
  → Kantengewichte aus DependencyPercentage
  → Gewichtete Random Walks
```

### Auto-Gewichte AUS (Checkbox inaktiv)

**1. Kategorie-Gewichte:**
```
MANUELL via Slider gesetzt
→ Dependencies werden NICHT verwendet
```

**2. Graph-Embeddings:**
```
Dependencies werden TROTZDEM geladen und verwendet
→ Kantengewichte aus DependencyPercentage
→ Unabhängig von Kategorie-Gewichten
```

## Technische Details

### Komponenten-Gewichte Berechnung

```python
für jede Dependency (source, target, percentage):
    component_importance[target] += percentage      # Ziel wird beeinflusst
    component_importance[source] += percentage * 0.5  # Quelle beeinflusst auch
    
normalisiere auf Summe = 1.0
```

### Kategorie-Mapping

```python
für jede Komponente:
    zähle Parameter-Typen (GEOM, MTRL, ELEC, DYN)
    verteile Komponenten-Gewicht proportional auf Kategorien
    
summiere über alle Komponenten
normalisiere auf Summe = 1.0
```

### Graph-Embedding Kantengewichte

```python
für jede RDF-Triple (subject, predicate, object):
    extrahiere Komponenten-Namen aus URIs
    suche Dependency (source_component, target_component)
    
    wenn gefunden:
        kantengewicht = dependency["percentage"]
    sonst:
        kantengewicht = 1.0  # Default
```

## Beispiel

**Dependencies:**
```
Blechpaket → Welle: strength=hoch, percentage=0.8
Aktivteil → Welle: strength=mittel, percentage=0.6
Wuchtscheiben → Welle: strength=niedrig, percentage=0.3
```

**Komponenten-Gewichte:**
```
Welle: 0.667 (wird von 3 Komponenten beeinflusst)
Blechpaket: 0.157
Aktivteil: 0.118
Wuchtscheiben: 0.059
```

**Kategorie-Gewichte (bei 50% GEOM, 50% MTRL pro Komponente):**
```
GEOM_MECH: 0.471
MTRL_PROC: 0.412
REQ_ELEC: 0.118
```

## Einschränkungen

**Graph-Embeddings:**
- Dependencies helfen NICHT bei identischer Rotor-Struktur
- Alle Rotoren haben gleiche RDF-Struktur
- Gewichte ändern nur Random-Walk-Wahrscheinlichkeiten
- Resultierende Embeddings bleiben sehr ähnlich
- Range bleibt bei ~6% (vs. k-NN 55%, Autoencoder 93%)

**Empfehlung:**
- Auto-Gewichte für regelbasierte Methoden nützlich
- Für Graph-Embeddings bringt es keinen Vorteil
