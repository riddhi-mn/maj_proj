"""Baseline LLM evaluator - tests LLM without RAG retrieval."""

import logging
from typing import Dict, List
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from src.config import Config

logger = logging.getLogger(__name__)


class BaselineLLM:
    """Baseline LLM that answers questions without retrieval."""
    
    def __init__(self):
        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in .env file")
        
        self.llm = ChatOpenAI(
            model_name=Config.LLM_MODEL,
            temperature=0.7,
            openai_api_key=Config.OPENAI_API_KEY
        )
        
        # Simple prompt without retrieval context
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a specialized assistant focused on medicinal plants, Ayurveda, Indian herbal medicine, 
            traditional treatments, and related health topics. Answer questions using your knowledge about:
            
            - Medicinal plants, herbal medicine, Ayurveda, Indian traditional medicine
            - Plant-based treatments, illnesses, symptoms, plant properties
            - Chemical compounds in plants, preparation methods
            - Plant parts, regional plant information, soil types, therapeutic applications
            
            Guidelines:
            - Keep responses concise: 200-250 words maximum, 2 paragraphs
            - Be accurate and informative
            - If you don't know something, say so
            - Focus on Indian medicinal plants and Ayurveda"""),
            ("human", "{question}"),
        ])
    
    def answer(self, question: str) -> Dict[str, any]:
        """
        Answer a question using only LLM knowledge (no retrieval).
        
        Args:
            question: User question
            
        Returns:
            Dict with 'response' and 'sources' (empty for baseline)
        """
        try:
            messages = self.prompt.format_messages(question=question)
            response = self.llm.invoke(messages)
            answer_text = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "response": answer_text,
                "sources": [],  # No sources for baseline
                "graph_context": "",
                "retrieval_used": False
            }
        except Exception as e:
            logger.error(f"Error in baseline LLM: {str(e)}")
            return {
                "response": f"Error: {str(e)}",
                "sources": [],
                "graph_context": "",
                "retrieval_used": False
            }

