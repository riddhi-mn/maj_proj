"""Graph-enhanced retrieval logic."""
from typing import List, Dict
from src.neo4j_client import Neo4jClient
from src.weaviate_client import WeaviateClient
from src.config import Config


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
        # Step 1: Get graph context from Neo4j
        graph_context = self.neo4j_client.get_graph_context(query)
        
        # Step 2: Enhance query with graph context
        if graph_context:
            enhanced_query = f"{query} {graph_context}"
        else:
            enhanced_query = query
        
        # Step 3: Search Weaviate with enhanced query
        vector_results = self.weaviate_client.hybrid_search(
            enhanced_query,
            limit=Config.TOP_K_WEAVIATE
        )
        
        return {
            "graph_context": graph_context,
            "enhanced_query": enhanced_query,
            "vector_results": vector_results,
            "query": query
        }
    
    def format_context_for_llm(self, retrieval_result: Dict[str, any]) -> str:
        """Format retrieval results for LLM context."""
        context_parts = []
        
        # Add graph context if available
        if retrieval_result["graph_context"]:
            context_parts.append("Graph Context (from knowledge base):")
            context_parts.append(retrieval_result["graph_context"])
            context_parts.append("")
        
        # Add vector search results
        if retrieval_result["vector_results"]:
            context_parts.append("Document Context (from PDFs):")
            for i, result in enumerate(retrieval_result["vector_results"], 1):
                context_parts.append(f"[{result['filename']}, Page {result['page_number']}]")
                context_parts.append(result["content"])
                context_parts.append("")
        
        # If no context available, indicate this to the LLM
        if not context_parts:
            return "Context: No specific data available from knowledge base or documents. Use general knowledge if helpful."
        
        return "Context:\n" + "\n".join(context_parts)

