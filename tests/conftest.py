import pytest


@pytest.fixture()
def mini_owl() -> str:
    return """<?xml version="1.0"?>
<rdf:RDF xmlns:ims="http://ontology.innomotics.net/ims#"
     xml:base="http://ontology.innomotics.net/ims"
     xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
     xmlns:owl="http://www.w3.org/2002/07/owl#"
     xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
     xmlns:xsd="http://www.w3.org/2001/XMLSchema#">

  <owl:Ontology rdf:about="http://ontology.innomotics.net/ims"/>

  <owl:Class rdf:about="http://ontology.innomotics.net/ims#Welle"/>

  <owl:ObjectProperty rdf:about="http://ontology.innomotics.net/ims#composed_of"/>
  <owl:DatatypeProperty rdf:about="http://ontology.innomotics.net/ims#hasValue"/>
  <owl:DatatypeProperty rdf:about="http://ontology.innomotics.net/ims#hasUnit"/>
  <owl:DatatypeProperty rdf:about="http://ontology.innomotics.net/ims#hasType"/>

  <owl:NamedIndividual rdf:about="http://ontology.innomotics.net/ims#C_WELLE_1">
    <rdf:type rdf:resource="http://ontology.innomotics.net/ims#Welle"/>
    <ims:composed_of rdf:resource="http://ontology.innomotics.net/ims#P_TEST_1"/>
  </owl:NamedIndividual>

  <owl:NamedIndividual rdf:about="http://ontology.innomotics.net/ims#P_TEST_1">
    <ims:hasValue rdf:datatype="http://www.w3.org/2001/XMLSchema#string">120</ims:hasValue>
    <ims:hasUnit rdf:datatype="http://www.w3.org/2001/XMLSchema#string">mm</ims:hasUnit>
    <ims:hasType rdf:datatype="http://www.w3.org/2001/XMLSchema#string">GEOM</ims:hasType>
  </owl:NamedIndividual>

</rdf:RDF>
"""
