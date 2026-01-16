# ======================import pakages =========================
import pandas as pd
import datetime

# RDFLib pakages
from rdflib import Graph, Literal, URIRef, Namespace
from rdflib.namespace import RDF, RDFS, XSD, OWL


# ======================create ontology =========================


def component_class_mapper(data, graph):
    """
    Maps components from the data to ontology classes.
    Args:
        data (pd.DataFrame): DataFrame containing component information.
        graph (rdflib.Graph): RDFLib graph to which the classes will be added.

    """
    # define Artefact as subclass of Object
    print("\n[component_class_mapper] --- Defining Artefact as subclass of Object...")

    for i in range(data.shape[0]):
        str(data.loc[i, "Component_ID"])
        label_ = str(data.loc[i, "Label"])
        description_ = str(data.loc[i, "Description"]) if "Description" in data.columns else None

        Parent = IMS["Artefact"]
        fragment = label_.strip().replace(" ", "_")

        child = IMS[fragment]
        graph.add((child, RDFS.subClassOf, Parent))
        graph.add((child, RDFS.label, Literal(label_)))
        if description_ and description_.lower() != "nan":
            graph.add((child, RDFS.comment, Literal(description_)))

    print("\n[component_class_mapper] --- Component classes mapped successfully.")


def parameter_class_mapper(data, graph):
    """Maps parameters from the data to ontology classes.
    Args:
    data (pd.DataFrame): DataFrame containing parameter information.
    graph (rdflib.Graph): RDFLib graph to which the classes will be added.
    """

    # define Parameter as subclass of Object
    print("\n[parameter_class_mapper] --- Defining Parameter as subclass of Object...")
    parent = IMS["Object"]
    child = IMS["Parameter"]
    graph.add((child, RDFS.subClassOf, parent))
    graph.add((child, RDFS.label, Literal("Parameter")))
    # create subclasses of Parameter

    for i in range(data.shape[0]):
        parameter_ = str(data.loc[i, "Parameter_ID"])
        label_ = str(data.loc[i, "Name"])
        description_ = str(data.loc[i, "Description"]) if "Description" in data.columns else None

        Parent = IMS["Parameter"]
        fragment = parameter_.strip().replace(" ", "_")
        child = IMS[fragment]
        graph.add((child, RDFS.subClassOf, Parent))
        graph.add((child, RDFS.label, Literal(label_)))
        if description_ and description_.lower() != "nan":
            graph.add((child, RDFS.comment, Literal(description_)))

    print("\n[parameter_class_mapper] --- Parameter classes mapped successfully.")


def data_mapper(data, g, num):
    """
    Maps the components and parameters from the data to the RDF graph.
    Args:
        data (dict): A dictionary containing the data from the Excel sheets.
        g (Graph): An RDFLib Graph object to which the data will be added.
        num (int): A numeric identifier for the mapping process.

    """

    print("\n[data_mapper] --- Mapping data instances to ontology...")
    # get the unique component IDs
    liste = data["Component_ID"].tolist()
    liste = list(set(liste))

    # dictionary to map component IDs to class names
    dict_components = {
        "C_WELLE": "Welle",
        "C_AKTIVTEIL": "Aktivteil",
        "C_LUEFTER": "Lüfter",
        "C_BLECHPAKET": "Blechpaket",
        "C_WUCHTSCHEIBEN": "Wuchtscheiben",
    }

    # == create a Rotor instance
    rotor_instance = URIRef("http://ontology.innomotics.net/ims#Rotor_" + str(num))
    PhysicalObject = URIRef("http://ontology.innomotics.net/ims#PhysicalObject")
    g.add((rotor_instance, RDF.type, PhysicalObject))

    # add component instances to the graph
    for component in liste:
        class_name = dict_components[component]
        component = component + "_" + str(num)
        component_ = IMS[component]
        g.add((component_, RDF.type, IMS[class_name]))
        g.add((rotor_instance, IMS.composed_of, component_))

    # define properties
    composed_of = URIRef("http://ontology.innomotics.net/ims#composed_of")
    hasvalue = URIRef("http://ontology.innomotics.net/ims#hasValue")
    hasunit = URIRef("http://ontology.innomotics.net/ims#hasUnit")
    hastype = URIRef("http://ontology.innomotics.net/ims#hasType")
    #
    g.add((hasvalue, RDF.type, OWL.DatatypeProperty))
    g.add((hasvalue, RDFS.domain, IMS["Parameter"]))
    g.add((composed_of, RDF.type, OWL.ObjectProperty))
    g.add((composed_of, RDFS.domain, IMS["Artefact"]))
    g.add((composed_of, RDFS.range, IMS["Parameter"]))

    g.add((hasunit, RDF.type, OWL.DatatypeProperty))
    g.add((hasunit, RDFS.domain, IMS["Parameter"]))
    # g.add((hasunit, RDFS.range, IMS['Unit']))

    g.add((hastype, RDF.type, OWL.DatatypeProperty))
    g.add((hastype, RDFS.domain, IMS["Parameter"]))
    # g.add((hastype, RDFS.range, IMS['Type']))

    # add parameter instances to the graph
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    for i in range(data.shape[0]):
        param_id = data.loc[i, "Parameter_ID"]
        param_type = data.loc[i, "ParamType_ID"]
        Parent = data.loc[i, "Component_ID"]
        data.loc[i, "Name"]
        Definition = data.loc[i, "Definition"]
        value = data.loc[i, "Value"]
        unit = data.loc[i, "Unit"]
        fragment = param_id.strip().replace(" ", "_").replace(")", "_").replace("(", "_")

        # Create URI for the parameter and map its properties
        fragment = fragment + "_" + date + "_" + str(num)
        param_id_uri = URIRef("http://ontology.innomotics.net/ims#" + fragment)
        # print(f"\n[data_mapper] --- Mapping parameter instance: {fragment}")
        g.add((param_id_uri, RDF.type, IMS[param_id]))
        g.add((param_id_uri, RDFS.label, Literal(fragment, datatype=XSD.string)))
        g.add((param_id_uri, RDFS.comment, Literal(Definition, datatype=XSD.string)))
        g.add((param_id_uri, hasvalue, Literal(value, datatype=XSD.string)))
        g.add((param_id_uri, hasunit, Literal(unit, datatype=XSD.string)))
        g.add((param_id_uri, hastype, Literal(param_type, datatype=XSD.string)))

        # parent composed of parameter
        Parent = Parent + "_" + str(num)
        g.add((IMS[Parent], composed_of, param_id_uri))

    print("\n[data_mapper] --- Data instances mapped successfully.")


def abhaengigkeit_mapper(data, g, num):
    """
    Maps Abhängigkeiten from the data to the RDF graph.
    Args:
        data (pd.DataFrame): DataFrame containing Abhängigkeiten information.
        g (Graph): An RDFLib Graph object to which the Abhängigkeiten will be added.
        num (int): A numeric identifier for the mapping process.
    """

    print("\n[abhaengigkeit_mapper] --- Starting mapping of Abhängigkeiten...")

    for i in range(data.shape[0]):
        Quelle = data.loc[i, "Quelle"]
        relation = data.loc[i, "Relation"]
        Ziel = data.loc[i, "Ziel"]
        data.loc[i, "Typ"]
        Staerke = data.loc[i, "Staerke"]
        Abhängigkeit = data.loc[i, "Abhängigkeit_%"]
        Notiz = data.loc[i, "Notiz"]

        Quelle = Quelle.split("(")[0].strip()
        Quelle = "" if Quelle.endswith(".") else Quelle
        Quelle_ = Quelle
        Ziel = Ziel.split("(")[0].strip()
        Ziel = Ziel.split(";")[0].strip()
        Ziel = "" if Ziel.endswith(".") else Ziel
        Ziel_ = Ziel

        # template.
        """
        instance_Quelle  abhängig_von instance_Ziel[Staerke, Abhängigkeit, Notiz]
        
        """
        ziel_0 = None
        quelle_0 = None
        if "welle" in Quelle.lower():
            Quelle = "C_WELLE_" + str(num)
        elif "aktivteil" in Quelle.lower():
            Quelle = "C_AKTIVTEIL_" + str(num)
        elif "lüfter" in Quelle.lower():
            Quelle = "C_LUEFTER_" + str(num)
        elif "blechpaket" in Quelle.lower():
            Quelle = "C_BLECHPAKET_" + str(num)
        elif "wuchtscheiben" in Quelle.lower():
            Quelle = "C_WUCHTSCHEIBEN_" + str(num)
        else:
            print(f"\n[abhaengigkeit_mapper] --- Warning: Unknown Quelle component '{Quelle}'")
            quelle_0 = URIRef("http://ontology.innomotics.net/ims#" + str(Quelle) + "_" + str(num))
            g.add((quelle_0, RDF.type, IMS.Feature))
            continue

        if "welle" in Ziel.lower():
            Ziel = "C_WELLE_" + str(num)
        elif "aktivteil" in Ziel.lower():
            Ziel = "C_AKTIVTEIL_" + str(num)
        elif "lüfter" in Ziel.lower():
            Ziel = "C_LUEFTER_" + str(num)
        elif "blechpaket" in Ziel.lower():
            Ziel = "C_BLECHPAKET_" + str(num)
        elif "wuchtscheiben" in Ziel.lower():
            Ziel = "C_WUCHTSCHEIBEN_" + str(num)
        else:
            print(f"\n[abhaengigkeit_mapper] --- Warning: Unknown Ziel component '{Ziel}'")
            ziel_0 = URIRef("http://ontology.innomotics.net/ims#" + str(Ziel) + "_" + str(num))
            g.add((ziel_0, RDF.type, IMS.Feature))

        # Uri erstellen
        relation_fragment = relation.strip().replace(" ", "_").replace("(", "_").replace(")", "_")
        relation_fragment = Quelle_ + "_" + relation_fragment + "_" + Ziel_ + "_" + str(num)
        relation_uri = URIRef("http://ontology.innomotics.net/ims#" + relation_fragment)

        g.add((relation_uri, RDF.type, OWL.ObjectProperty))
        g.add((relation_uri, RDFS.label, Literal(relation_fragment, datatype=XSD.string)))
        # Abhängigkeit als Tripel hinzufügen
        if ziel_0 is not None:
            if quelle_0 is not None:
                g.add((quelle_0, relation_uri, ziel_0))
            else:
                g.add((IMS[Quelle], relation_uri, ziel_0))
        else:
            if quelle_0 is not None:
                g.add((quelle_0, relation_uri, IMS[Ziel]))
            else:
                g.add((IMS[Quelle], relation_uri, IMS[Ziel]))

        # Zusätzliche Informationen als Annotationen hinzufügen
        g.add((relation_uri, IMS.hasStrength, Literal(Staerke, datatype=XSD.string)))
        g.add(
            (relation_uri, IMS.hasDependencyPercentage, Literal(Abhängigkeit, datatype=XSD.string))
        )
        g.add((relation_uri, RDFS.comment, Literal(Notiz, datatype=XSD.string)))

    print("\n[abhaengigkeit_mapper] --- Mapping Abhängigkeiten...")


if __name__ == "__main__":
    # choose a sheet (e.g. first sheet) and run
    print("\n[MAIN] --- Starting ontology creation process...")
    # ======================load data =========================
    print("\n[MAIN] --- Loading data from Excel file...")
    file = "../Data/AE_Ontology_Entwurf.xlsx"
    sheet_names = pd.ExcelFile(file).sheet_names  # Get all sheet names
    data = pd.read_excel(file, sheet_name=None)  # Load all sheets
    # print(sheet_names)

    # =======================load filled data =========================
    file_filled = "../Data/AE_Ontology_Entwurf_filled.xlsx"
    file_filled_feedback = "../Data/AE_Ontology_Entwurf_IN_Feedback.xlsx"
    sheet_names_filled = pd.ExcelFile(file_filled).sheet_names  # Get all sheet names
    data_filled = pd.read_excel(file_filled, sheet_name=None)  # Load all sheets
    # print(sheet_names_filled)
    # ======================print sheet names =========================
    print("\n[MAIN] --- Data loaded successfully.")

    # ====================== Load basis ontology =========================
    print("\n[MAIN] --- Loading base ontology...")

    file_ontology = "../Data/Ontology_Base.owl"
    g = Graph()
    g.parse(file_ontology, format="xml")

    # ======================build namespaces =========================
    # build namespaces
    IMS = Namespace("http://ontology.innomotics.net/ims#")
    rdf = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
    rdfs = Namespace("http://www.w3.org/2000/01/rdf-schema#")
    xsd = Namespace("http://www.w3.org/2001/XMLSchema#")
    owl = Namespace("http://www.w3.org/2002/07/owl#")

    # bind namespaces
    g.bind("ims", IMS)
    g.bind("rdf", rdf)
    g.bind("rdfs", rdfs)
    g.bind("xsd", xsd)
    g.bind("owl", owl)

    print("\n[MAIN] --- Base ontology loaded successfully.")

    num = 1  # Initialize numeric identifier for data mapping
    # Increment this value if needed for multiple mappings( for example in a loop for more Rotor types)
    # ============ component sheet =========================
    # TODO:sheet = sheet_names[0] # "Components"
    sheet = str(sheet_names[0])  # "Components"
    df_components = data[sheet]

    print("\n[MAIN] --- Mapping components class to ontology classes...")
    # run the class-mapper function
    component_class_mapper(df_components, g)
    print("\n[MAIN] --- Component classes mapped successfully.")

    # ============ parameter sheet =========================
    print("\n[MAIN] --- Mapping parameter classes to ontology classes...")
    sheet_param = "Parameters"
    df_parameters = data[sheet_param]
    # run the Parameter-mapper function
    parameter_class_mapper(df_parameters, g)
    print("\n[MAIN] --- Parameter classes mapped successfully.")

    # ============ map instances =========================
    print("\n[MAIN] --- Mapping data instances to ontology...")
    data_param = data_filled["Parameters"]
    # run the data-mapper function
    data_mapper(data_param, g, num)
    print("\n[MAIN] --- Mapping data instances to ontology...")

    # ====== Map Abhangigkeit =========================
    print("\n[MAIN] --- Mapping Abhängigkeiten to ontology...")

    sheet_param = "Rotor_abhaengigkeiten_umfrage"
    df_relation = pd.read_excel(file_filled_feedback, sheet_name=sheet_param, header=1)
    abhaengigkeit_mapper(df_relation, g, num)

    print("\n[MAIN] --- Abhängigkeiten mapped successfully.")

    # ======================save ontology =========================
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    print("\n[MAIN] --- Saving ontology to file...")
    output_file = "../Data/Ontology_Components_" + date + ".owl"
    g.serialize(destination=output_file, format="xml")

    print(f"\n[MAIN] --- Ontology saved successfully to {output_file}.")
