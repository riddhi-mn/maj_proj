"""RAG system evaluator - tests the hybrid RAG chatbot."""

import logging
from typing import Dict
from src.chatbot import HybridRAGChatbot
from src.graph_enhancer import GraphEnhancer
from src.neo4j_client import Neo4jClient
from src.weaviate_client import WeaviateClient

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Evaluator for the hybrid RAG system."""
    
    def __init__(self):
        """Initialize RAG evaluator with graph enhancer and chatbot."""
        try:
            # Initialize clients first
            logger.info("Initializing Neo4j client...")
            neo4j_client = Neo4jClient()
            
            logger.info("Initializing Weaviate client...")
            weaviate_client = WeaviateClient()
            
            # Initialize graph enhancer with clients
            logger.info("Initializing GraphEnhancer...")
            graph_enhancer = GraphEnhancer(neo4j_client, weaviate_client)
            
            # Initialize chatbot with graph enhancer
            logger.info("Initializing Chatbot...")
            self.chatbot = HybridRAGChatbot(graph_enhancer)
            
            logger.info("RAG evaluator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG evaluator: {str(e)}")
            raise
    
    def answer(self, question: str) -> Dict[str, any]:
        """
        Answer a question using the hybrid RAG system.
        
        Args:
            question: User question
            
        Returns:
            Dict with 'response', 'sources', 'graph_context', 'ranked_vector_chunks', etc.
        """
        try:
            # Clear memory for each test question (fresh start)
            self.chatbot.clear_memory()
            
            # Call chatbot.query() which handles the full retrieval and response flow
            # The retrieval_result inside query() has the ranked vector chunks we need
            # However, query() doesn't expose retrieval_result directly, so we need to
            # capture it from graph_enhancer.retrieve() separately
            # For testing, we use the query as-is (no history enhancement since memory is cleared)
            
            # Get retrieval results to capture ranked chunks (same query that will be used)
            retrieval_result = self.chatbot.graph_enhancer.retrieve(question)
            
            # Capture ranked vector chunks (already reranked by hybrid scoring)
            # These chunks are sorted by hybrid_score (descending) and may be filtered
            # Make a deep copy to preserve scoring info (needed for MRR/NDCG)
            ranked_vector_chunks = []
            for chunk in retrieval_result.get("vector_results", []):
                # Create a copy with all fields including _scoring
                chunk_copy = {
                    "content": chunk.get("content", ""),
                    "filename": chunk.get("filename", ""),
                    "page_number": chunk.get("page_number", 0),
                    "score": chunk.get("score", 0.0),
                    "hybrid_score": chunk.get("hybrid_score", chunk.get("_scoring", {}).get("hybrid_score", 0.0))
                }
                # Preserve _scoring if it exists
                if "_scoring" in chunk:
                    chunk_copy["_scoring"] = chunk["_scoring"]
                ranked_vector_chunks.append(chunk_copy)
            
            # Now call the chatbot query to get the final response
            # (This will do retrieval again, but that's okay for testing accuracy)
            result = self.chatbot.query(question)
            
            return {
                "response": result.get("response", ""),
                "sources": result.get("sources", []),
                "graph_context": result.get("graph_context", ""),
                "ranked_vector_chunks": ranked_vector_chunks,  # Include ranked chunks for MRR/NDCG
                "retrieval_used": True
            }
        except Exception as e:
            logger.error(f"Error in RAG evaluator: {str(e)}")
            return {
                "response": f"Error: {str(e)}",
                "sources": [],
                "graph_context": "",
                "ranked_vector_chunks": [],
                "retrieval_used": True
            }

