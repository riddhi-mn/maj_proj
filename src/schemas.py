"""Schema reference and query helpers for Neo4j graph."""
from typing import Dict, List


# Neo4j Node Labels
NODE_LABELS = {
    "Plant": "Plant",
    "LocalName": "LocalName",
    "BiologicalName": "BiologicalName",
    "PlantPart": "PlantPart",
    "Region": "Region",
    "SoilType": "SoilType",
    "Preparation": "Preparation",
    "ChemicalCompound": "ChemicalCompound",
    "ConstituentClass": "ConstituentClass",
    "CuringProperty": "CuringProperty",
    "Symptom": "Symptom",
    "Illness": "Illness",
    "BodySystem": "BodySystem",
    "BodyParts": "BodyParts"
}

# Relationship Types
RELATIONSHIPS = {
    "HAS_LOCAL_NAME": "HAS_LOCAL_NAME",
    "HAS_BIOLOGICAL_NAME": "HAS_BIOLOGICAL_NAME",
    "HAS_USEFUL_PART": "HAS_USEFUL_PART",
    "NATIVE_TO": "NATIVE_TO",
    "THRIVES_IN": "THRIVES_IN",
    "PREPARED_AS": "PREPARED_AS",
    "CONTAINS_COMPOUND": "CONTAINS_COMPOUND",
    "HAS_HEALING_PROPERTY": "HAS_HEALING_PROPERTY",
    "TREATS_ILLNESS": "TREATS_ILLNESS",
    "BELONGS_TO_CLASS": "BELONGS_TO_CLASS",
    "EXHIBITS_PROPERTY": "EXHIBITS_PROPERTY",
    "HELPS_WITH": "HELPS_WITH",
    "ALLEVIATES_SYMPTOM": "ALLEVIATES_SYMPTOM",
    "HAS_SYMPTOM": "HAS_SYMPTOM",
    "AFFECTS_BODY_SYSTEM": "AFFECTS_BODY_SYSTEM",
    "AFFECTS_SYSTEM": "AFFECTS_SYSTEM",
    "COMPRISES_PART": "COMPRISES_PART"
}


def build_plant_query(plant_name: str = None, illness_name: str = None, symptom_name: str = None) -> str:
    """Build Cypher query to find plants based on illness or symptom."""
    if illness_name:
        return """MATCH (p:Plant)-[:TREATS_ILLNESS]->(i:Illness {name: $illness_name})
OPTIONAL MATCH (i)-[:HAS_SYMPTOM]->(s:Symptom)
OPTIONAL MATCH (p)-[:PREPARED_AS]->(prep:Preparation)
OPTIONAL MATCH (p)-[:HAS_HEALING_PROPERTY]->(prop:CuringProperty)
OPTIONAL MATCH (p)-[:CONTAINS_COMPOUND]->(chem:ChemicalCompound)
OPTIONAL MATCH (prop)-[:ALLEVIATES_SYMPTOM]->(s2:Symptom)
RETURN p.name as plant_name, 
       i.name as illness_name,
       collect(DISTINCT s.name) as symptoms,
       collect(DISTINCT prep.name) as preparations,
       collect(DISTINCT prop.name) as properties,
       collect(DISTINCT chem.name) as compounds
LIMIT 10"""
    elif symptom_name:
        return """MATCH (i:Illness)-[:HAS_SYMPTOM]->(s:Symptom {name: $symptom_name})
MATCH (p:Plant)-[:TREATS_ILLNESS]->(i)
OPTIONAL MATCH (p)-[:HAS_HEALING_PROPERTY]->(prop:CuringProperty)
OPTIONAL MATCH (prop)-[:ALLEVIATES_SYMPTOM]->(s)
OPTIONAL MATCH (p)-[:PREPARED_AS]->(prep:Preparation)
OPTIONAL MATCH (p)-[:CONTAINS_COMPOUND]->(chem:ChemicalCompound)
RETURN p.name as plant_name,
       i.name as illness_name,
       s.name as symptom_name,
       collect(DISTINCT prop.name) as properties,
       collect(DISTINCT prep.name) as preparations,
       collect(DISTINCT chem.name) as compounds
LIMIT 10"""
    elif plant_name:
        return """MATCH (p:Plant {name: $plant_name})
OPTIONAL MATCH (p)-[:TREATS_ILLNESS]->(i:Illness)
OPTIONAL MATCH (i)-[:HAS_SYMPTOM]->(s:Symptom)
OPTIONAL MATCH (p)-[:PREPARED_AS]->(prep:Preparation)
OPTIONAL MATCH (p)-[:HAS_HEALING_PROPERTY]->(prop:CuringProperty)
OPTIONAL MATCH (p)-[:CONTAINS_COMPOUND]->(chem:ChemicalCompound)
OPTIONAL MATCH (chem)-[:BELONGS_TO_CLASS]->(cls:ConstituentClass)
OPTIONAL MATCH (p)-[:HAS_LOCAL_NAME]->(ln:LocalName)
OPTIONAL MATCH (p)-[:HAS_BIOLOGICAL_NAME]->(bn:BiologicalName)
OPTIONAL MATCH (p)-[:HAS_USEFUL_PART]->(part:PlantPart)
OPTIONAL MATCH (p)-[:NATIVE_TO]->(reg:Region)
OPTIONAL MATCH (p)-[:THRIVES_IN]->(soil:SoilType)
RETURN p.name as plant_name,
       collect(DISTINCT i.name) as illnesses,
       collect(DISTINCT s.name) as symptoms,
       collect(DISTINCT prep.name) as preparations,
       collect(DISTINCT prop.name) as properties,
       collect(DISTINCT chem.name) as compounds,
       collect(DISTINCT cls.name) as classes,
       collect(DISTINCT ln.name) as local_names,
       collect(DISTINCT bn.name) as biological_names,
       collect(DISTINCT part.name) as plant_parts,
       collect(DISTINCT reg.name) as regions,
       collect(DISTINCT soil.name) as soil_types
LIMIT 1"""
    else:
        return """MATCH (p:Plant)
OPTIONAL MATCH (p)-[:TREATS_ILLNESS]->(i:Illness)
OPTIONAL MATCH (i)-[:HAS_SYMPTOM]->(s:Symptom)
RETURN p.name as plant_name,
       collect(DISTINCT i.name) as illnesses,
       collect(DISTINCT s.name) as symptoms
LIMIT 10"""


def build_general_search_query(query_terms: List[str]) -> str:
    """Build a general Cypher query from search terms."""
    valid_terms = [term.strip() for term in query_terms if term and term.strip()]
    
    if valid_terms:
        conditions = " OR ".join([
            f"(toLower(p.name) CONTAINS toLower('{term}') OR toLower(i.name) CONTAINS toLower('{term}') OR toLower(s.name) CONTAINS toLower('{term}') OR toLower(prop.name) CONTAINS toLower('{term}') OR toLower(chem.name) CONTAINS toLower('{term}') OR toLower(ln.name) CONTAINS toLower('{term}') OR toLower(bn.name) CONTAINS toLower('{term}'))"
            for term in valid_terms
        ])
        where_clause = f"WHERE {conditions}"
    else:
        where_clause = ""
    
    if where_clause:
        query = f"""MATCH (p:Plant)
OPTIONAL MATCH (p)-[:TREATS_ILLNESS]->(i:Illness)
OPTIONAL MATCH (i)-[:HAS_SYMPTOM]->(s:Symptom)
OPTIONAL MATCH (p)-[:PREPARED_AS]->(prep:Preparation)
OPTIONAL MATCH (p)-[:HAS_HEALING_PROPERTY]->(prop:CuringProperty)
OPTIONAL MATCH (p)-[:CONTAINS_COMPOUND]->(chem:ChemicalCompound)
OPTIONAL MATCH (p)-[:HAS_LOCAL_NAME]->(ln:LocalName)
OPTIONAL MATCH (p)-[:HAS_BIOLOGICAL_NAME]->(bn:BiologicalName)
{where_clause}
RETURN p.name as plant_name,
       collect(DISTINCT i.name) as illnesses,
       collect(DISTINCT s.name) as symptoms,
       collect(DISTINCT prep.name) as preparations,
       collect(DISTINCT prop.name) as properties,
       collect(DISTINCT chem.name) as compounds,
       collect(DISTINCT ln.name) as local_names,
       collect(DISTINCT bn.name) as biological_names
LIMIT 10"""
    else:
        query = """MATCH (p:Plant)
OPTIONAL MATCH (p)-[:TREATS_ILLNESS]->(i:Illness)
OPTIONAL MATCH (i)-[:HAS_SYMPTOM]->(s:Symptom)
OPTIONAL MATCH (p)-[:PREPARED_AS]->(prep:Preparation)
OPTIONAL MATCH (p)-[:HAS_HEALING_PROPERTY]->(prop:CuringProperty)
OPTIONAL MATCH (p)-[:CONTAINS_COMPOUND]->(chem:ChemicalCompound)
RETURN p.name as plant_name,
       collect(DISTINCT i.name) as illnesses,
       collect(DISTINCT s.name) as symptoms,
       collect(DISTINCT prep.name) as preparations,
       collect(DISTINCT prop.name) as properties,
       collect(DISTINCT chem.name) as compounds
LIMIT 10"""
    
    return query


def format_graph_context(results: List[Dict]) -> str:
    """Format Neo4j query results into context string for vector search enhancement."""
    if not results:
        return ""
    
    context_parts = []
    for result in results:
        plant_name = result.get("plant_name", "")
        illnesses = result.get("illnesses", [])
        symptoms = result.get("symptoms", [])
        preparations = result.get("preparations", [])
        properties = result.get("properties", [])
        compounds = result.get("compounds", [])
        local_names = result.get("local_names", [])
        biological_names = result.get("biological_names", [])
        plant_parts = result.get("plant_parts", [])
        regions = result.get("regions", [])
        soil_types = result.get("soil_types", [])
        
        context = f"Plant: {plant_name}"
        if local_names:
            context += f" (Local names: {', '.join(filter(None, local_names))})"
        if biological_names:
            context += f" (Scientific: {', '.join(filter(None, biological_names))})"
        if plant_parts:
            context += f" (Parts used: {', '.join(filter(None, plant_parts))})"
        if regions:
            context += f" (Region: {', '.join(filter(None, regions))})"
        if soil_types:
            context += f" (Soil: {', '.join(filter(None, soil_types))})"
        if compounds:
            context += f" (Compounds: {', '.join(filter(None, compounds))})"
        if properties:
            context += f" (Properties: {', '.join(filter(None, properties))})"
        if preparations:
            context += f" (Preparations: {', '.join(filter(None, preparations))})"
        if illnesses:
            context += f" (Treats: {', '.join(filter(None, illnesses))})"
        if symptoms:
            context += f" (Symptoms: {', '.join(filter(None, symptoms))})"
        
        context_parts.append(context)
    
    return " | ".join(context_parts)
