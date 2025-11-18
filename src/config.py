"""Configuration management for the hybrid RAG chatbot."""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Centralized configuration."""
    
    # Neo4j Configuration
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
    
    # Weaviate Configuration
    WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    
    # LLM Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    
    # Processing Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    TOP_K_WEAVIATE = int(os.getenv("TOP_K_WEAVIATE", "5"))
    TOP_K_NEO4J = int(os.getenv("TOP_K_NEO4J", "10"))
    
    # PDF Directory
    PDF_DIR = os.getenv("PDF_DIR", "data/pdfs")

