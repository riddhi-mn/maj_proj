"""Graph-enhanced retrieval logic."""
import logging
from typing import List, Dict
from src.neo4j_client import Neo4jClient
from src.weaviate_client import WeaviateClient
from src.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphEnhancer:
    """Orchestrates graph-enhanced vector search."""
    
    def __init__(self, neo4j_client: Neo4jClient, weaviate_client: WeaviateClient):
        self.neo4j_client = neo4j_client
        self.weaviate_client = weaviate_client
    
    def retrieve(self, query: str) -> Dict[str, any]:
        """
        Perform graph-enhanced retrieval:
        1. Query Neo4j first to get graph context
        2. Enhance query with graph context
        3. Search Weaviate with enhanced query
        """
        logger.info(f"[GraphEnhancer] Starting retrieval for query: {query}")
        
        # Step 1: Get graph context from Neo4j
        try:
            logger.debug("[GraphEnhancer] Step 1: Querying Neo4j for graph context...")
            graph_context = self.neo4j_client.get_graph_context(query)
            logger.info(f"[GraphEnhancer] Graph context retrieved: {len(graph_context)} chars" if graph_context else "[GraphEnhancer] No graph context found")
            if graph_context:
                logger.debug(f"[GraphEnhancer] Graph context preview: {graph_context[:200]}...")
        except Exception as e:
            logger.error(f"[GraphEnhancer] Error getting graph context: {str(e)}")
            graph_context = ""
        
        # Step 2: Enhance query with graph context
        if graph_context:
            enhanced_query = f"{query} {graph_context}"
            logger.debug(f"[GraphEnhancer] Step 2: Enhanced query: {enhanced_query[:200]}...")
        else:
            enhanced_query = query
            logger.debug("[GraphEnhancer] Step 2: No enhancement applied (no graph context)")
        
        # Step 3: Search Weaviate with enhanced query
        try:
            logger.debug(f"[GraphEnhancer] Step 3: Searching Weaviate (limit: {Config.TOP_K_WEAVIATE})...")
            vector_results = self.weaviate_client.hybrid_search(
                enhanced_query,
                limit=Config.TOP_K_WEAVIATE
            )
            logger.info(f"[GraphEnhancer] Weaviate search returned {len(vector_results)} results")
            if vector_results:
                logger.debug(f"[GraphEnhancer] First Weaviate result: {vector_results[0].get('filename', 'unknown')}")
        except Exception as e:
            logger.error(f"[GraphEnhancer] Error searching Weaviate: {str(e)}")
            vector_results = []
        
        result = {
            "graph_context": graph_context,
            "enhanced_query": enhanced_query,
            "vector_results": vector_results,
            "query": query
        }
        
        logger.info(f"[GraphEnhancer] Retrieval complete - Graph: {len(graph_context)} chars, Vector: {len(vector_results)} results")
        return result
    
    def format_context_for_llm(self, retrieval_result: Dict[str, any]) -> str:
        """Format retrieval results for LLM context."""
        logger.debug("[GraphEnhancer] Formatting context for LLM...")
        context_parts = []
        
        # Add graph context if available
        if retrieval_result["graph_context"]:
            context_parts.append("Graph Context (from knowledge base):")
            context_parts.append(retrieval_result["graph_context"])
            context_parts.append("")
            logger.debug(f"[GraphEnhancer] Added graph context ({len(retrieval_result['graph_context'])} chars)")
        
        # Add vector search results
        if retrieval_result["vector_results"]:
            context_parts.append("Document Context (from PDFs):")
            for i, result in enumerate(retrieval_result["vector_results"], 1):
                context_parts.append(f"[{result['filename']}, Page {result['page_number']}]")
                context_parts.append(result["content"])
                context_parts.append("")
            logger.debug(f"[GraphEnhancer] Added {len(retrieval_result['vector_results'])} document chunks")
        
        # If no context available, indicate this to the LLM
        if not context_parts:
            logger.warning("[GraphEnhancer] No context available (neither graph nor vector results)")
            return "Context: No specific data available from knowledge base or documents. Use general knowledge if helpful."
        
        formatted_context = "Context:\n" + "\n".join(context_parts)
        logger.info(f"[GraphEnhancer] Formatted context: {len(formatted_context)} chars total")
        return formatted_context

