"""Neo4j client for querying the graph database."""
from typing import List, Dict, Optional
from neo4j import GraphDatabase
from src.config import Config
from src.schemas import build_plant_query, build_general_search_query, format_graph_context


class Neo4jClient:
    """Client for Neo4j graph database operations."""
    
    def __init__(self):
        try:
            if not Config.NEO4J_PASSWORD:
                raise ValueError("NEO4J_PASSWORD not set in .env file")
            
            self.driver = GraphDatabase.driver(
                Config.NEO4J_URI,
                auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD)
            )
            
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Neo4j at {Config.NEO4J_URI}: {str(e)}. Check your Neo4j Desktop connection and .env file settings.")
    
    def close(self):
        """Close the Neo4j driver connection."""
        if self.driver:
            self.driver.close()
    
    def _execute_query(self, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """Execute a Cypher query and return results."""
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]
    
    def search_plants_by_illness(self, illness_name: str) -> List[Dict]:
        """Search for plants that treat a specific illness."""
        query = build_plant_query(illness_name=illness_name)
        return self._execute_query(query, {"illness_name": illness_name})
    
    def search_plants_by_symptom(self, symptom_name: str) -> List[Dict]:
        """Search for plants that treat illnesses with a specific symptom."""
        query = build_plant_query(symptom_name=symptom_name)
        return self._execute_query(query, {"symptom_name": symptom_name})
    
    def search_plant_by_name(self, plant_name: str) -> Optional[Dict]:
        """Search for a specific plant by name."""
        query = build_plant_query(plant_name=plant_name)
        results = self._execute_query(query, {"plant_name": plant_name})
        return results[0] if results else None
    
    def general_search(self, query_terms: List[str]) -> List[Dict]:
        """General search across plants, illnesses, and symptoms."""
        query = build_general_search_query(query_terms)
        return self._execute_query(query)
    
    def get_graph_context(self, user_query: str) -> str:
        """Extract graph context from user query to enhance vector search."""
        query_lower = user_query.lower()
        
        # Try to identify key terms (simplified - could use NER in production)
        terms = [word for word in query_lower.split() if len(word) > 3]
        
        # Check for illness-related keywords
        if any(keyword in query_lower for keyword in ["treat", "cure", "help", "illness", "disease"]):
            # Try to extract illness or symptom name
            results = self.general_search(terms)
            if results:
                return format_graph_context(results)
        
        # Check for symptom-related keywords
        if any(keyword in query_lower for keyword in ["symptom", "pain", "headache", "fever", "ache"]):
            # Try symptoms first
            for term in terms:
                results = self.search_plants_by_symptom(term.capitalize())
                if results:
                    return format_graph_context(results)
        
        # Check for plant-related keywords
        if any(keyword in query_lower for keyword in ["plant", "herb", "medicine"]):
            # Try to find plant
            for term in terms:
                result = self.search_plant_by_name(term.capitalize())
                if result:
                    return format_graph_context([result])
        
        # Fallback: general search
        results = self.general_search(terms[:3])  # Limit to first 3 terms
        if results:
            return format_graph_context(results)
        
        return ""

