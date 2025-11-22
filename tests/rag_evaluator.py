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
            Dict with 'response', 'sources', 'graph_context', etc.
        """
        try:
            # Clear memory for each test question (fresh start)
            self.chatbot.clear_memory()
            
            result = self.chatbot.query(question)
            
            return {
                "response": result.get("response", ""),
                "sources": result.get("sources", []),
                "graph_context": result.get("graph_context", ""),
                "retrieval_used": True
            }
        except Exception as e:
            logger.error(f"Error in RAG evaluator: {str(e)}")
            return {
                "response": f"Error: {str(e)}",
                "sources": [],
                "graph_context": "",
                "retrieval_used": True
            }

