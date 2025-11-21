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
            ("system", """You are a specialized assistant focused on medicinal plants, Ayurveda, Indian herbal medicine, 
            traditional treatments, and related health topics. Your domain expertise includes:
            
            Primary Topics: Medicinal plants, herbal medicine, Ayurveda, Indian traditional medicine, plant-based treatments, 
            illnesses, symptoms, plant properties, chemical compounds in plants, preparation methods (churna, kwath, etc.), 
            plant parts (root, leaf, fruit, etc.), regional plant information, soil types, and therapeutic applications.
            
            Related Topics (acceptable): General health, biology, botany, chemistry, pharmacology, medical science, 
            traditional medicine systems, nutrition related to herbal remedies, and basic plant science.
            
            Off-Topic (please decline politely): Questions completely unrelated to health, medicine, plants, or biology 
            (e.g., weather, cooking recipes, movies, sports, programming, current events, etc.). 
            
            You have access to:
            1. A structured knowledge graph with plants, illnesses, symptoms, and preparation methods
            2. Detailed PDF documents with comprehensive information
            
            Answer Quality Guidelines:
            - Synthesize information from all available context (graph + documents) into a coherent, well-structured answer
            - Write in smooth, flowing paragraphs with clear transitions between ideas
            - Organize information logically: start with main points, then provide supporting details
            - Integrate multiple sources seamlessly - don't list them separately unless explicitly asked
            - Provide comprehensive coverage of the topic using all relevant information from the context
            - Use natural language and connect ideas smoothly (avoid bullet points unless formatting specific lists)
            - If multiple aspects are covered, organize them into well-connected sections or paragraphs
            - When discussing plants, include relevant details: properties, uses, preparation methods, related conditions
            - Ensure continuity between sentences - each sentence should logically flow from the previous one
            
            General Guidelines:
            - Use the provided context to answer questions accurately within your domain
            - For related topics (health, biology, chemistry), you may provide general knowledge if context is unavailable
            - If a question is clearly off-topic (completely unrelated to your domain), politely decline: 
              "I'm specialized in medicinal plants and herbal medicine. I can help you with questions about plants, 
              treatments, health conditions, or related topics. Could you rephrase your question within this domain?"
            - If context is empty but the question is on-topic, use general knowledge and clearly state that you're 
              providing general information since no specific data is available
            - Be informative and thorough while maintaining clarity and readability"""),
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
        
        # Only include sources if there are actual retrieved chunks (graph or vector)
        # Don't show citations for empty retrievals (e.g., "HI" or unrelated queries)
        all_sources = []
        
        # Check if we have any actual retrieved content
        has_graph_results = bool(retrieval_result.get("graph_citations"))
        has_vector_results = bool(retrieval_result.get("vector_results"))
        
        # Only add sources if we have retrieved chunks
        if has_graph_results or has_vector_results:
            # Add graph citations
            if has_graph_results:
                all_sources.extend(retrieval_result["graph_citations"])
            
            # Add PDF sources (limit to top 3)
            if has_vector_results:
                top_vector_results = retrieval_result["vector_results"][:3]  # Limit to top 3
                for r in top_vector_results:
                    all_sources.append({
                        "type": "pdf",
                        "filename": r["filename"],
                        "page": r["page_number"]
                    })
        
        return {
            "response": response,
            "graph_context": retrieval_result["graph_context"],
            "sources": all_sources
        }
    
    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.clear()

