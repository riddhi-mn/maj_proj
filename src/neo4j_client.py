"""Neo4j client for querying the graph database."""
import logging
from typing import List, Dict, Optional
from neo4j import GraphDatabase
from src.config import Config
from src.schemas import build_plant_query, build_general_search_query, format_graph_context

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        try:
            logger.debug(f"[Neo4j] Executing query: {query[:200]}...")
            logger.debug(f"[Neo4j] Parameters: {parameters}")
            
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                records = [record.data() for record in result]
                
                logger.info(f"[Neo4j] Query returned {len(records)} results")
                if records:
                    logger.debug(f"[Neo4j] First result: {records[0]}")
                else:
                    logger.debug("[Neo4j] No results found")
                
                return records
        except Exception as e:
            logger.error(f"[Neo4j] Query execution failed: {str(e)}")
            logger.error(f"[Neo4j] Query: {query[:200]}...")
            logger.error(f"[Neo4j] Parameters: {parameters}")
            raise
    
    def search_plants_by_illness(self, illness_name: str) -> List[Dict]:
        """Search for plants that treat a specific illness."""
        logger.info(f"[Neo4j] Searching plants by illness: {illness_name}")
        query = build_plant_query(illness_name=illness_name)
        return self._execute_query(query, {"illness_name": illness_name})
    
    def search_plants_by_symptom(self, symptom_name: str) -> List[Dict]:
        """Search for plants that treat illnesses with a specific symptom."""
        logger.info(f"[Neo4j] Searching plants by symptom: {symptom_name}")
        query = build_plant_query(symptom_name=symptom_name)
        return self._execute_query(query, {"symptom_name": symptom_name})
    
    def search_plant_by_name(self, plant_name: str) -> Optional[Dict]:
        """Search for a specific plant by name."""
        logger.info(f"[Neo4j] Searching plant by name: {plant_name}")
        query = build_plant_query(plant_name=plant_name)
        results = self._execute_query(query, {"plant_name": plant_name})
        if results:
            logger.info(f"[Neo4j] Found plant: {results[0].get('plant_name', 'Unknown')}")
        else:
            logger.warning(f"[Neo4j] Plant not found: {plant_name}")
        return results[0] if results else None
    
    def general_search(self, query_terms: List[str]) -> List[Dict]:
        """General search across plants, illnesses, and symptoms."""
        logger.info(f"[Neo4j] General search with terms: {query_terms}")
        query = build_general_search_query(query_terms)
        return self._execute_query(query)
    
    def get_graph_context(self, user_query: str) -> str:
        """Extract graph context from user query to enhance vector search."""
        logger.info(f"[Neo4j] Getting graph context for query: {user_query}")
        query_lower = user_query.lower()
        
        # Try to identify key terms (simplified - could use NER in production)
        terms = [word for word in query_lower.split() if len(word) > 3]
        logger.debug(f"[Neo4j] Extracted search terms: {terms}")
        
        # Check for illness-related keywords
        if any(keyword in query_lower for keyword in ["treat", "cure", "help", "illness", "disease"]):
            logger.debug("[Neo4j] Detected illness-related query")
            results = self.general_search(terms)
            if results:
                context = format_graph_context(results)
                logger.info(f"[Neo4j] Found {len(results)} results via general search (illness)")
                return context
        
        # Check for symptom-related keywords
        if any(keyword in query_lower for keyword in ["symptom", "pain", "headache", "fever", "ache", "cough", "stress", "joint pain", "rash", "fatigue"]):
            logger.debug("[Neo4j] Detected symptom-related query")
            for term in terms:
                results = self.search_plants_by_symptom(term.capitalize())
                if results:
                    context = format_graph_context(results)
                    logger.info(f"[Neo4j] Found {len(results)} results via symptom search")
                    return context
        
        # Check for plant-related keywords or try direct plant search
        if any(keyword in query_lower for keyword in ["plant", "herb", "medicine", "medicinal", "what is", "tell me about"]):
            logger.debug("[Neo4j] Detected plant-related query")
            # Try each term as a potential plant name
            for term in terms:
                result = self.search_plant_by_name(term.capitalize())
                if result:
                    context = format_graph_context([result])
                    logger.info(f"[Neo4j] Found plant via name search: {term}")
                    return context
        
        # Check for chemical compound keywords
        if any(keyword in query_lower for keyword in ["compound", "chemical", "contains", "curcumin", "eugenol", "nimbin", "withanolides"]):
            logger.debug("[Neo4j] Detected compound-related query")
            results = self.general_search(terms)
            if results:
                context = format_graph_context(results)
                logger.info(f"[Neo4j] Found {len(results)} results via general search (compound)")
                return context
        
        # Check for property keywords
        if any(keyword in query_lower for keyword in ["property", "antimicrobial", "anti-inflammatory", "antioxidant", "adaptogenic", "detoxifying"]):
            logger.debug("[Neo4j] Detected property-related query")
            results = self.general_search(terms)
            if results:
                context = format_graph_context(results)
                logger.info(f"[Neo4j] Found {len(results)} results via general search (property)")
                return context
        
        # Fallback: general search
        logger.debug("[Neo4j] Using fallback general search")
        results = self.general_search(terms[:3])  # Limit to first 3 terms
        if results:
            context = format_graph_context(results)
            logger.info(f"[Neo4j] Found {len(results)} results via fallback search")
            return context
        
        logger.warning(f"[Neo4j] No graph context found for query: {user_query}")
        return ""

