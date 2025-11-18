"""Schema reference and query helpers for Neo4j graph."""
from typing import Dict, List


# Expected Neo4j Node Labels
NODE_LABELS = {
    "Plant": "Plant",
    "Illness": "Illness",
    "Symptom": "Symptom",
    "PreparationMethod": "PreparationMethod"
}

# Expected Relationship Types
RELATIONSHIPS = {
    "TREATS": "TREATS",  # Plant -[TREATS]-> Illness
    "HAS_SYMPTOM": "HAS_SYMPTOM",  # Illness -[HAS_SYMPTOM]-> Symptom
    "USES_PREPARATION": "USES_PREPARATION"  # Plant -[USES_PREPARATION]-> PreparationMethod
}


def build_plant_query(plant_name: str = None, illness_name: str = None, symptom_name: str = None) -> str:
    """Build Cypher query to find plants based on illness or symptom."""
    if illness_name:
        return """MATCH (p:Plant)-[r:TREATS]->(i:Illness {name: $illness_name})
OPTIONAL MATCH (i)-[:HAS_SYMPTOM]->(s:Symptom)
OPTIONAL MATCH (p)-[:USES_PREPARATION]->(pm:PreparationMethod)
RETURN p.name as plant_name, 
       p.description as plant_description,
       i.name as illness_name,
       collect(DISTINCT s.name) as symptoms,
       collect(DISTINCT pm.name) as preparation_methods
LIMIT 10"""
    elif symptom_name:
        return """MATCH (i:Illness)-[:HAS_SYMPTOM]->(s:Symptom {name: $symptom_name})
MATCH (p:Plant)-[:TREATS]->(i)
OPTIONAL MATCH (p)-[:USES_PREPARATION]->(pm:PreparationMethod)
RETURN p.name as plant_name,
       p.description as plant_description,
       i.name as illness_name,
       s.name as symptom_name,
       collect(DISTINCT pm.name) as preparation_methods
LIMIT 10"""
    elif plant_name:
        return """MATCH (p:Plant {name: $plant_name})
OPTIONAL MATCH (p)-[:TREATS]->(i:Illness)
OPTIONAL MATCH (i)-[:HAS_SYMPTOM]->(s:Symptom)
OPTIONAL MATCH (p)-[:USES_PREPARATION]->(pm:PreparationMethod)
RETURN p.name as plant_name,
       p.description as plant_description,
       collect(DISTINCT i.name) as illnesses,
       collect(DISTINCT s.name) as symptoms,
       collect(DISTINCT pm.name) as preparation_methods
LIMIT 1"""
    else:
        return """MATCH (p:Plant)
OPTIONAL MATCH (p)-[:TREATS]->(i:Illness)
OPTIONAL MATCH (i)-[:HAS_SYMPTOM]->(s:Symptom)
RETURN p.name as plant_name,
       p.description as plant_description,
       collect(DISTINCT i.name) as illnesses
LIMIT 10"""


def build_general_search_query(query_terms: List[str]) -> str:
    """Build a general Cypher query from search terms."""
    # Filter out empty terms and escape special characters
    valid_terms = [term.strip() for term in query_terms if term and term.strip()]
    
    # Build WHERE clause if we have valid terms
    if valid_terms:
        conditions = " OR ".join([
            f"(toLower(p.name) CONTAINS toLower('{term}') OR toLower(i.name) CONTAINS toLower('{term}') OR toLower(s.name) CONTAINS toLower('{term}'))"
            for term in valid_terms
        ])
        where_clause = f"WHERE {conditions}"
    else:
        where_clause = ""
    
    if where_clause:
        query = f"""MATCH (p:Plant)
OPTIONAL MATCH (p)-[:TREATS]->(i:Illness)
OPTIONAL MATCH (i)-[:HAS_SYMPTOM]->(s:Symptom)
OPTIONAL MATCH (p)-[:USES_PREPARATION]->(pm:PreparationMethod)
{where_clause}
RETURN p.name as plant_name,
       p.description as plant_description,
       collect(DISTINCT i.name) as illnesses,
       collect(DISTINCT s.name) as symptoms,
       collect(DISTINCT pm.name) as preparation_methods
LIMIT 10"""
    else:
        query = """MATCH (p:Plant)
OPTIONAL MATCH (p)-[:TREATS]->(i:Illness)
OPTIONAL MATCH (i)-[:HAS_SYMPTOM]->(s:Symptom)
OPTIONAL MATCH (p)-[:USES_PREPARATION]->(pm:PreparationMethod)
RETURN p.name as plant_name,
       p.description as plant_description,
       collect(DISTINCT i.name) as illnesses,
       collect(DISTINCT s.name) as symptoms,
       collect(DISTINCT pm.name) as preparation_methods
LIMIT 10"""
    
    return query


def format_graph_context(results: List[Dict]) -> str:
    """Format Neo4j query results into context string for vector search enhancement."""
    if not results:
        return ""
    
    context_parts = []
    for result in results:
        plant_name = result.get("plant_name", "")
        plant_desc = result.get("plant_description", "")
        illnesses = result.get("illnesses", [])
        symptoms = result.get("symptoms", [])
        prep_methods = result.get("preparation_methods", [])
        
        context = f"Plant: {plant_name}"
        if plant_desc:
            context += f" ({plant_desc})"
        if illnesses:
            context += f" treats: {', '.join(filter(None, illnesses))}"
        if symptoms:
            context += f" symptoms: {', '.join(filter(None, symptoms))}"
        if prep_methods:
            context += f" preparation methods: {', '.join(filter(None, prep_methods))}"
        
        context_parts.append(context)
    
    return " | ".join(context_parts)

