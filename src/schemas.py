"""Schema reference and query helpers for Neo4j graph."""
from typing import Dict, List


# Plant Reference List (Plants 21-50)
# This is a reference list for documentation and potential matching logic.
# Actual plant data is stored in Neo4j, not in this file.
PLANT_NAMES_21_50 = [
    {
        "primary_name": "Kalmegh",
        "local_names": ["Green Chiretta"],
        "search_terms": ["kalmegh", "green chiretta", "chiretta"]
    },
    {
        "primary_name": "Vasaka",
        "local_names": ["Malabar Nut"],
        "search_terms": ["vasaka", "malabar nut"]
    },
    {
        "primary_name": "Bael",
        "local_names": ["Wood Apple"],
        "search_terms": ["bael", "wood apple"]
    },
    {
        "primary_name": "Jamun",
        "local_names": ["Java Plum"],
        "search_terms": ["jamun", "java plum"]
    },
    {
        "primary_name": "Ashoka",
        "local_names": [],
        "search_terms": ["ashoka"]
    },
    {
        "primary_name": "Moringa",
        "local_names": ["Drumstick Tree"],
        "search_terms": ["moringa", "drumstick tree", "drumstick"]
    },
    {
        "primary_name": "Sandalwood",
        "local_names": ["Chandan"],
        "search_terms": ["sandalwood", "chandan"]
    },
    {
        "primary_name": "Prishniparni",
        "local_names": [],
        "search_terms": ["prishniparni"]
    },
    {
        "primary_name": "Kokilaksha",
        "local_names": ["Marsh Barbel"],
        "search_terms": ["kokilaksha", "marsh barbel"]
    },
    {
        "primary_name": "Curry Tree",
        "local_names": ["Kadi Patta"],
        "search_terms": ["curry tree", "kadi patta", "curry leaf", "curry leaves"]
    },
    {
        "primary_name": "Betel Leaf",
        "local_names": ["Paan"],
        "search_terms": ["betel leaf", "paan", "betel"]
    },
    {
        "primary_name": "Indian Borage",
        "local_names": ["Mexican Mint", "Ajwain Leaf"],
        "search_terms": ["indian borage", "mexican mint", "ajwain leaf"]
    },
    {
        "primary_name": "Sacred Fig",
        "local_names": ["Peepal"],
        "search_terms": ["sacred fig", "peepal", "peepul"]
    },
    {
        "primary_name": "Parijat",
        "local_names": ["Night Flowering Jasmine"],
        "search_terms": ["parijat", "night flowering jasmine", "parijata"]
    },
    {
        "primary_name": "Hibiscus",
        "local_names": ["China Rose"],
        "search_terms": ["hibiscus", "china rose"]
    },
    {
        "primary_name": "Indian Beech",
        "local_names": ["Karanja"],
        "search_terms": ["indian beech", "karanja"]
    },
    {
        "primary_name": "Pomegranate",
        "local_names": ["Anar"],
        "search_terms": ["pomegranate", "anar"]
    },
    {
        "primary_name": "Guava",
        "local_names": ["Amrood"],
        "search_terms": ["guava", "amrood"]
    },
    {
        "primary_name": "Lemon",
        "local_names": ["Nimbu"],
        "search_terms": ["lemon", "nimbu"]
    },
    {
        "primary_name": "Greater Galangal",
        "local_names": ["Blue Ginger", "Kulanjan"],
        "search_terms": ["greater galangal", "blue ginger", "kulanjan", "galangal"]
    },
    {
        "primary_name": "Mint",
        "local_names": ["Pudina"],
        "search_terms": ["mint", "pudina"]
    },
    {
        "primary_name": "Indian Mustard",
        "local_names": ["Sarson"],
        "search_terms": ["indian mustard", "sarson", "mustard"]
    },
    {
        "primary_name": "Bamboo",
        "local_names": ["Vanshalochana"],
        "search_terms": ["bamboo", "vanshalochana"]
    },
    {
        "primary_name": "Jasmine",
        "local_names": ["Chameli"],
        "search_terms": ["jasmine", "chameli"]
    },
    {
        "primary_name": "Oleander",
        "local_names": ["Kaner"],
        "search_terms": ["oleander", "kaner"]
    },
    {
        "primary_name": "Crape Jasmine",
        "local_names": ["Chandni"],
        "search_terms": ["crape jasmine", "chandni", "chandani"]
    },
    {
        "primary_name": "Karimkurinji",
        "local_names": [],
        "search_terms": ["karimkurinji"]
    },
    {
        "primary_name": "Green Amaranth",
        "local_names": ["Chaulai"],
        "search_terms": ["green amaranth", "chaulai", "amaranth"]
    },
    {
        "primary_name": "Malabar Spinach",
        "local_names": ["Poi Saag"],
        "search_terms": ["malabar spinach", "poi saag", "poi"]
    }
]


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
        # Build conditions that check all name fields properly
        # For each term, check if it matches ANY field (plant name, local name, biological name, illness, symptom, etc.)
        # Structure: (term1 matches p.name OR ln.name OR bn.name OR ...) OR (term2 matches ...)
        conditions = " OR ".join([
            f"(toLower(p.name) CONTAINS toLower('{term}') OR " +
            f"EXISTS {{ (p)-[:HAS_LOCAL_NAME]->(ln_check:LocalName) WHERE toLower(ln_check.name) CONTAINS toLower('{term}') }} OR " +
            f"EXISTS {{ (p)-[:HAS_BIOLOGICAL_NAME]->(bn_check:BiologicalName) WHERE toLower(bn_check.name) CONTAINS toLower('{term}') }} OR " +
            f"toLower(i.name) CONTAINS toLower('{term}') OR " +
            f"toLower(s.name) CONTAINS toLower('{term}') OR " +
            f"toLower(prop.name) CONTAINS toLower('{term}') OR " +
            f"toLower(chem.name) CONTAINS toLower('{term}'))"
            for term in valid_terms
        ])
        where_clause = f"WHERE {conditions}"
        
        # Build relevance score calculation based on collected lists
        # Score weights: Plant name (10), Local/Bio names (8), Illness (5), Symptom (4), Property/Compound (3)
        # For multi-term queries, each term's matches are scored and summed
        score_parts = []
        for term in valid_terms:
            term_lower = term.lower()
            # Plant name match (highest priority) - check p.name directly
            score_parts.append(f"CASE WHEN toLower(p.name) CONTAINS '{term_lower}' THEN 10 ELSE 0 END")
            # Local name match - check if any local name contains term
            score_parts.append(f"CASE WHEN size([lname IN local_names WHERE toLower(lname) CONTAINS '{term_lower}']) > 0 THEN 8 ELSE 0 END")
            # Biological name match - check if any bio name contains term
            score_parts.append(f"CASE WHEN size([bname IN biological_names WHERE toLower(bname) CONTAINS '{term_lower}']) > 0 THEN 8 ELSE 0 END")
            # Illness match - check if any illness contains term
            score_parts.append(f"CASE WHEN size([ill IN illnesses WHERE toLower(ill) CONTAINS '{term_lower}']) > 0 THEN 5 ELSE 0 END")
            # Symptom match - check if any symptom contains term
            score_parts.append(f"CASE WHEN size([sym IN symptoms WHERE toLower(sym) CONTAINS '{term_lower}']) > 0 THEN 4 ELSE 0 END")
            # Property match - check if any property contains term
            score_parts.append(f"CASE WHEN size([pr IN properties WHERE toLower(pr) CONTAINS '{term_lower}']) > 0 THEN 3 ELSE 0 END")
            # Compound match - check if any compound contains term
            score_parts.append(f"CASE WHEN size([cm IN compounds WHERE toLower(cm) CONTAINS '{term_lower}']) > 0 THEN 3 ELSE 0 END")
        
        relevance_score = " + ".join(score_parts)
        order_by = f"ORDER BY relevance_score DESC, p.name ASC"
    else:
        where_clause = ""
        relevance_score = "0"
        order_by = "ORDER BY p.name ASC"
    
    if where_clause:
        # First collect all relationships, then calculate relevance score
        query = f"""MATCH (p:Plant)
OPTIONAL MATCH (p)-[:TREATS_ILLNESS]->(i:Illness)
OPTIONAL MATCH (i)-[:HAS_SYMPTOM]->(s:Symptom)
OPTIONAL MATCH (p)-[:PREPARED_AS]->(prep:Preparation)
OPTIONAL MATCH (p)-[:HAS_HEALING_PROPERTY]->(prop:CuringProperty)
OPTIONAL MATCH (p)-[:CONTAINS_COMPOUND]->(chem:ChemicalCompound)
OPTIONAL MATCH (p)-[:HAS_LOCAL_NAME]->(ln:LocalName)
OPTIONAL MATCH (p)-[:HAS_BIOLOGICAL_NAME]->(bn:BiologicalName)
{where_clause}
WITH p,
     collect(DISTINCT i.name) as illnesses,
     collect(DISTINCT s.name) as symptoms,
     collect(DISTINCT prep.name) as preparations,
     collect(DISTINCT prop.name) as properties,
     collect(DISTINCT chem.name) as compounds,
     collect(DISTINCT ln.name) as local_names,
     collect(DISTINCT bn.name) as biological_names
WITH p, illnesses, symptoms, preparations, properties, compounds, local_names, biological_names,
     ({relevance_score}) as relevance_score
RETURN p.name as plant_name,
       illnesses,
       symptoms,
       preparations,
       properties,
       compounds,
       local_names,
       biological_names,
       relevance_score
{order_by}
LIMIT 20"""
    else:
        query = f"""MATCH (p:Plant)
OPTIONAL MATCH (p)-[:TREATS_ILLNESS]->(i:Illness)
OPTIONAL MATCH (i)-[:HAS_SYMPTOM]->(s:Symptom)
OPTIONAL MATCH (p)-[:PREPARED_AS]->(prep:Preparation)
OPTIONAL MATCH (p)-[:HAS_HEALING_PROPERTY]->(prop:CuringProperty)
OPTIONAL MATCH (p)-[:CONTAINS_COMPOUND]->(chem:ChemicalCompound)
WITH p, i, s, prep, prop, chem
RETURN p.name as plant_name,
       collect(DISTINCT i.name) as illnesses,
       collect(DISTINCT s.name) as symptoms,
       collect(DISTINCT prep.name) as preparations,
       collect(DISTINCT prop.name) as properties,
       collect(DISTINCT chem.name) as compounds
{order_by}
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


def get_all_plant_search_terms() -> List[str]:
    """
    Get all searchable plant terms from the reference list (plants 21-50).
    Returns a flat list of all search terms (primary names, local names, aliases).
    Useful for matching logic or reference.
    """
    all_terms = []
    for plant in PLANT_NAMES_21_50:
        # Add primary name
        all_terms.append(plant["primary_name"].lower())
        # Add local names
        all_terms.extend([name.lower() for name in plant["local_names"]])
        # Add search terms
        all_terms.extend(plant["search_terms"])
    
    # Remove duplicates and return
    return list(set(all_terms))


def get_plant_by_name(search_term: str) -> Dict:
    """
    Find plant by searching in primary names, local names, or search terms.
    Returns the plant dict if found, None otherwise.
    Useful for matching user queries to plant names.
    """
    search_term_lower = search_term.lower()
    for plant in PLANT_NAMES_21_50:
        # Check primary name
        if plant["primary_name"].lower() == search_term_lower:
            return plant
        # Check local names
        if any(name.lower() == search_term_lower for name in plant["local_names"]):
            return plant
        # Check search terms
        if search_term_lower in plant["search_terms"]:
            return plant
    
    return None
