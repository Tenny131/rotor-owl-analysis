# ======================import pakages =========================
import pandas as pd
import os

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


def data_mapper(data, g, num=None):
    """
    Maps the components and parameters from the data to the RDF graph.
    Creates separate individuals for each Design_ID with data property assertions.

    Args:
        data (pd.DataFrame): DataFrame containing parameter data with Design_ID column.
        g (Graph): An RDFLib Graph object to which the data will be added.
        num (int, optional): Legacy parameter, not used when Design_ID is present.
    """

    print("\n[data_mapper] --- Mapping data instances to ontology...")

    # Dictionary to map component IDs to class names
    dict_components = {
        "C_WELLE": "Welle",
        "C_WELLEENDE": "Welleende",
        "C_AKTIVTEIL": "Aktivteil",
        "C_LUEFTER": "Lüfter",
        "C_BLECHPAKET": "Blechpaket",
        "C_WUCHTSCHEIBEN": "Wuchtscheiben",
    }

    # Define properties
    composed_of = URIRef("http://ontology.innomotics.net/ims#composed_of")
    hasvalue = URIRef("http://ontology.innomotics.net/ims#hasValue")
    hasunit = URIRef("http://ontology.innomotics.net/ims#hasUnit")
    hastype = URIRef("http://ontology.innomotics.net/ims#hasType")

    # Define property types (only once)
    g.add((hasvalue, RDF.type, OWL.DatatypeProperty))
    g.add((hasvalue, RDFS.domain, IMS["Parameter"]))

    g.add((hasunit, RDF.type, OWL.DatatypeProperty))
    g.add((hasunit, RDFS.domain, IMS["Parameter"]))

    g.add((hastype, RDF.type, OWL.DatatypeProperty))
    g.add((hastype, RDFS.domain, IMS["Parameter"]))

    g.add((composed_of, RDF.type, OWL.ObjectProperty))
    g.add((composed_of, RDFS.domain, IMS["Artefact"]))
    g.add((composed_of, RDFS.range, IMS["Parameter"]))

    # Get unique Design IDs
    design_ids = data["Design_ID"].unique() if "Design_ID" in data.columns else [str(num)]

    for design_id in design_ids:
        # Filter data for this design
        if "Design_ID" in data.columns:
            design_data = data[data["Design_ID"] == design_id]
        else:
            design_data = data

        print(f"  Processing design: {design_id}")

        # Create Rotor instance for this design
        rotor_instance = URIRef(f"http://ontology.innomotics.net/ims#Rotor_{design_id}")
        PhysicalObject = URIRef("http://ontology.innomotics.net/ims#PhysicalObject")
        g.add((rotor_instance, RDF.type, PhysicalObject))
        g.add((rotor_instance, RDFS.label, Literal(f"Rotor_{design_id}", datatype=XSD.string)))

        # Get unique components for this design
        components = design_data["Component_ID"].unique()

        # Create component instances
        for component in components:
            if component in dict_components:
                class_name = dict_components[component]
                component_instance = f"{component}_{design_id}"
                component_uri = IMS[component_instance]
                g.add((component_uri, RDF.type, IMS[class_name]))
                g.add((component_uri, RDFS.label, Literal(component_instance, datatype=XSD.string)))
                g.add((rotor_instance, composed_of, component_uri))

        # Create parameter instances with data property assertions
        for idx, row in design_data.iterrows():
            param_id = row["Parameter_ID"]
            param_type = row["ParamType_ID"]
            component_id = row["Component_ID"]
            value = row["Value"]
            unit = row["Unit"]

            # Create unique parameter instance name
            param_fragment = param_id.strip().replace(" ", "_").replace("(", "_").replace(")", "_")
            param_instance_name = f"{param_fragment}_{design_id}"
            param_instance_uri = URIRef(f"http://ontology.innomotics.net/ims#{param_instance_name}")

            # Create parameter individual as instance of its parameter class
            g.add((param_instance_uri, RDF.type, IMS[param_id]))
            g.add(
                (param_instance_uri, RDFS.label, Literal(param_instance_name, datatype=XSD.string))
            )

            # Add data property assertions
            g.add((param_instance_uri, hasvalue, Literal(str(value), datatype=XSD.string)))
            g.add((param_instance_uri, hasunit, Literal(str(unit), datatype=XSD.string)))
            g.add((param_instance_uri, hastype, Literal(param_type, datatype=XSD.string)))

            # Link parameter to its parent component
            component_instance_name = f"{component_id}_{design_id}"
            g.add((IMS[component_instance_name], composed_of, param_instance_uri))

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
    # Bestimme Basis-Pfad relativ zu diesem Skript (jetzt in src/rotor_owl/daten/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "..", "..", "data")
    reference_dir = os.path.join(data_dir, "reference")

    file = os.path.join(reference_dir, "AE_Ontology_Entwurf_IN_Feedback.xlsx")
    sheet_names = pd.ExcelFile(file).sheet_names  # Get all sheet names
    data = pd.read_excel(file, sheet_name=None)  # Load all sheets
    # print(sheet_names)

    # =======================load filled data from CSV =========================
    print("\n[MAIN] --- Loading generated parameter data from CSV...")
    csv_file = os.path.join(data_dir, "generated", "generated.csv")
    df_generated = pd.read_csv(csv_file)
    # Konvertiere CSV-Format zu Excel-Sheet-Format für data_mapper
    # CSV: Design_ID,Component_ID,Parameter_ID,ParamType_ID,DataType,Unit,Value,IsMissing
    # Behalte alle relevanten Spalten inklusive Design_ID
    df_parameters_filled = df_generated[
        ["Design_ID", "Parameter_ID", "Component_ID", "ParamType_ID", "Value", "Unit"]
    ].copy()
    df_parameters_filled["Name"] = df_parameters_filled["Parameter_ID"]  # Name = Parameter_ID
    df_parameters_filled["Definition"] = ""  # Platzhalter für Definition
    # ======================print sheet names =========================
    print("\n[MAIN] --- Data loaded successfully.")

    # ====================== Load basis ontology =========================
    print("\n[MAIN] --- Loading base ontology...")

    file_ontology = os.path.join(reference_dir, "Ontology_Base.owl")
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
    sheet = sheet_names[0]  # "Components"
    df_components = data[sheet]  # type: ignore

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
    # Verwende die aus CSV konvertierten Parameter-Daten
    data_mapper(df_parameters_filled, g, num)
    print("\n[MAIN] --- Data instances mapped successfully.")

    # ====== Map Abhangigkeit =========================
    print("\n[MAIN] --- Mapping Abhängigkeiten to ontology...")

    file_feedback = os.path.join(reference_dir, "AE_Ontology_Entwurf_IN_Feedback.xlsx")
    sheet_param = "Rotor_abhaengigkeiten_umfrage"
    df_relation = pd.read_excel(file_feedback, sheet_name=sheet_param, header=1)
    abhaengigkeit_mapper(df_relation, g, num)

    print("\n[MAIN] --- Abhängigkeiten mapped successfully.")

    # ======================save ontology =========================
    print("\n[MAIN] --- Saving ontology to file...")
    output_dir = os.path.join(data_dir, "ontologien")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "rotor_ontologie_generiert.owl")
    g.serialize(destination=output_file, format="xml")

    print(f"\n[MAIN] --- Ontology saved successfully to {output_file}.")
