"""Main chatbot orchestration using LangChain."""
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from src.graph_enhancer import GraphEnhancer
from src.config import Config


class HybridRAGChatbot:
    """Hybrid RAG chatbot combining Neo4j graph and Weaviate vector search."""
    
    def __init__(self, graph_enhancer: GraphEnhancer):
        self.graph_enhancer = graph_enhancer
        
        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in .env file")
        
        self.llm = ChatOpenAI(
            model_name=Config.LLM_MODEL,
            temperature=0.7,
            openai_api_key=Config.OPENAI_API_KEY
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Setup prompt template with single input variable for memory
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions about medicinal plants, 
            illnesses, symptoms, and treatment methods. You have access to:
            1. A structured knowledge graph with plants, illnesses, symptoms, and preparation methods
            2. Detailed PDF documents with comprehensive information
            
            Use the provided context to answer questions accurately. If the context is empty or doesn't 
            contain the answer, use your general knowledge but clearly state that you're providing general 
            information since no specific data is available. Be concise but informative."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory,
            verbose=False
        )
    
    def query(self, user_query: str) -> Dict[str, any]:
        """Process a user query and return a response."""
        # Retrieve context using graph-enhanced search
        retrieval_result = self.graph_enhancer.retrieve(user_query)
        
        # Format context for LLM
        context = self.graph_enhancer.format_context_for_llm(retrieval_result)
        
        # Combine query and context into single input string
        combined_input = f"{user_query}\n\n{context}" if context else user_query
        
        # Generate response using LangChain
        response = self.chain.run(input=combined_input)
        
        return {
            "response": response,
            "graph_context": retrieval_result["graph_context"],
            "sources": [
                {
                    "filename": r["filename"],
                    "page": r["page_number"]
                }
                for r in retrieval_result["vector_results"]
            ]
        }
    
    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.clear()

