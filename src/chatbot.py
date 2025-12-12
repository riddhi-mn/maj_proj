"""Main chatbot orchestration using LangChain."""
import logging
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage
from src.graph_enhancer import GraphEnhancer
from src.config import Config

logger = logging.getLogger(__name__)


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
            - Keep responses concise and crisp: limit answers to 200-250 words maximum (approximately 2 paragraphs)
            - Structure answers in exactly 2 paragraphs:
              * First paragraph: Main answer with key information and primary points
              * Second paragraph: Supporting details, additional context, or related information
            - Synthesize information from all available context (graph + documents) into a coherent, well-structured answer
            - Write in smooth, flowing sentences with clear transitions between ideas
            - Prioritize the most relevant and important information - be selective, not exhaustive
            - Integrate multiple sources seamlessly - don't list them separately unless explicitly asked
            - Focus on quality over quantity - answer the question directly and concisely
            - Use natural language and connect ideas smoothly (avoid bullet points unless formatting specific lists)
            - When discussing plants, include only the most relevant details: key properties, primary uses, main preparation methods
            - Ensure continuity between sentences - each sentence should logically flow from the previous one
            
            General Guidelines:
            - ALWAYS use the conversation history to maintain context - if the user asks follow-up questions with 
              pronouns like "it", "that plant", "its properties", etc., refer back to what was discussed earlier
            - When answering follow-up questions, maintain continuity with previous messages in the conversation
            - If the user is asking about a plant mentioned earlier, continue discussing that same plant even if the 
              current context doesn't explicitly mention it
            - Prioritize conversation history over retrieved context when they conflict - conversation history takes precedence
            - Use the provided context to answer questions accurately within your domain
            - For related topics (health, biology, chemistry), you may provide general knowledge if context is unavailable
            - If a question is clearly off-topic (completely unrelated to your domain), politely decline: 
              "I'm specialized in medicinal plants and herbal medicine. I can help you with questions about plants, 
              treatments, health conditions, or related topics. Could you rephrase your question within this domain?"
            - If context is empty but the question is on-topic, use general knowledge and clearly state that you're 
              providing general information since no specific data is available
            - Be informative and concise - prioritize clarity and brevity over comprehensiveness
            - Always aim for 200-250 words and 2 paragraphs - do not exceed this limit"""),
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
        # Check for generic greetings/queries that shouldn't trigger retrieval
        query_lower = user_query.lower().strip()
        generic_greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", 
                            "greetings", "thanks", "thank you", "bye", "goodbye", "see you"]
        
        # If it's just a generic greeting, skip retrieval and return a greeting response
        if query_lower in generic_greetings or (len(query_lower.split()) <= 2 and query_lower in ["hi there", "hey there"]):
            logger.info(f"[Chatbot] Detected generic greeting: '{user_query}' - skipping retrieval")
            response = self.chain.run(input=user_query)
            return {
                "response": response,
                "graph_context": "",
                "sources": []  # No sources for generic greetings
            }
        
        # Get conversation history from memory BEFORE running chain (chain.run will add new messages)
        # Access memory messages directly
        try:
            conversation_history = self.memory.chat_memory.messages if hasattr(self.memory, 'chat_memory') else []
        except:
            conversation_history = []
        
        logger.info(f"[Chatbot] Conversation history BEFORE query: {len(conversation_history)} messages")
        
        # Debug: Print recent conversation history
        if conversation_history and len(conversation_history) > 0:
            logger.debug(f"[Chatbot] Recent messages:")
            for i, msg in enumerate(conversation_history[-4:], 1):
                try:
                    content = msg.content if hasattr(msg, 'content') else str(msg)[:100]
                    role = "Human" if "Human" in str(type(msg)) else "AI"
                    logger.debug(f"  {role}: {content[:100]}")
                except:
                    pass
        
        # Extract key entities/plants from conversation history to enhance query
        enhanced_query = self._enhance_query_with_history(user_query, conversation_history)
        if enhanced_query != user_query:
            logger.info(f"[Chatbot] Enhanced query with history: '{user_query}' -> '{enhanced_query}'")
        else:
            logger.debug(f"[Chatbot] Query not enhanced (no history terms or already in query)")
        
        # Retrieve context using graph-enhanced search (with conversation context)
        retrieval_result = self.graph_enhancer.retrieve(enhanced_query)
        
        # Format context for LLM
        context = self.graph_enhancer.format_context_for_llm(retrieval_result)
        
        # Add conversation context instruction to help LLM maintain continuity
        # Limit to last 3 exchanges (6 messages) to prevent token overflow
        if conversation_history and len(conversation_history) > 0:
            # Extract recent conversation context for LLM (last 3 exchanges = 6 messages max)
            recent_messages = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
            recent_context = self._format_recent_conversation(recent_messages)
            if recent_context:
                # Truncate recent context if too long (max 1200 chars for 3 exchanges)
                if len(recent_context) > 1200:
                    recent_context = recent_context[:1200] + "..."
                    logger.debug(f"[Chatbot] Truncated recent conversation context to 1200 chars")
                context = f"{recent_context}\n\n{context}" if context else recent_context
                logger.debug(f"[Chatbot] Added recent conversation context to prompt ({len(recent_context)} chars)")
        
        # Combine query and context into single input string
        combined_input = f"{user_query}\n\n{context}" if context else user_query
        
        # Generate response using LangChain (memory will automatically include chat_history via MessagesPlaceholder)
        # The chain.run() will automatically save user input and AI response to memory
        response = self.chain.run(input=combined_input)
        
        # Debug: Check memory after query
        try:
            history_after = self.memory.chat_memory.messages if hasattr(self.memory, 'chat_memory') else []
            logger.info(f"[Chatbot] Conversation history AFTER query: {len(history_after)} messages")
        except:
            pass
        
        # Only include sources if there are actual retrieved chunks (graph or vector)
        # Don't show citations for empty retrievals (e.g., "HI" or unrelated queries)
        all_sources = []
        
        # Check if we have any actual retrieved content
        graph_citations = retrieval_result.get("graph_citations", [])
        vector_results = retrieval_result.get("vector_results", [])
        
        # Filter vector results using hybrid_score (if available) or base score
        # Lower threshold to 0.01 to capture more results, but still filter completely irrelevant ones
        # Hybrid scores are normalized 0-1, so threshold of 0.01 is reasonable
        meaningful_vector_results = []
        for r in vector_results:
            # Prefer hybrid_score (from reranking), fallback to base score
            relevance_score = r.get("hybrid_score", r.get("_scoring", {}).get("hybrid_score", r.get("score", 0.0)))
            if relevance_score > 0.01:  # Lower threshold from 0.1 to 0.01
                meaningful_vector_results.append(r)
        
        # If we still have no results but there are vector_results, take top 3 anyway
        # (this handles cases where scores are very low but still valid)
        if not meaningful_vector_results and vector_results:
            # Sort by hybrid_score or score
            sorted_results = sorted(
                vector_results,
                key=lambda x: x.get("hybrid_score", x.get("_scoring", {}).get("hybrid_score", x.get("score", 0.0))),
                reverse=True
            )
            meaningful_vector_results = sorted_results[:3]
        
        has_graph_results = bool(graph_citations) and len(graph_citations) > 0
        has_vector_results = bool(meaningful_vector_results) and len(meaningful_vector_results) > 0
        
        # Only add sources if we have meaningful retrieved chunks
        if has_graph_results or has_vector_results:
            # Add graph citations
            if has_graph_results:
                all_sources.extend(graph_citations)
            
            # Add PDF sources (limit to top 3, only if they meet relevance threshold)
            if has_vector_results:
                top_vector_results = meaningful_vector_results[:3]  # Limit to top 3
                for r in top_vector_results:
                    all_sources.append({
                        "type": "pdf",
                        "filename": r["filename"],
                        "page": r["page_number"]
                    })
        
        # Log sources check for debugging
        logger.debug(f"[Chatbot] Sources check: graph={has_graph_results} ({len(graph_citations)} citations), vector={has_vector_results} ({len(meaningful_vector_results)}/{len(vector_results)} meaningful), total_sources={len(all_sources)}")
        
        logger.debug(f"[Chatbot] Sources check: graph={has_graph_results} ({len(graph_citations)} citations), vector={has_vector_results} ({len(meaningful_vector_results)}/{len(vector_results)} meaningful), total_sources={len(all_sources)}")
        
        return {
            "response": response,
            "graph_context": retrieval_result["graph_context"],
            "sources": all_sources
        }
    
    def _enhance_query_with_history(self, user_query: str, conversation_history: List[BaseMessage]) -> str:
        """Enhance user query with conversation history to maintain context."""
        if not conversation_history or len(conversation_history) == 0:
            return user_query
        
        # Extract plant names and key terms from recent conversation
        # Look for plant names in ALL messages (not just recent)
        extracted_terms = []
        
        # Check ALL messages for plant names/entities (prioritize most recent)
        for msg in conversation_history:
            # Try different ways to get content from message
            if hasattr(msg, 'content'):
                content = msg.content
            elif hasattr(msg, 'text'):
                content = msg.text
            elif hasattr(msg, '__str__'):
                content = str(msg)
            else:
                continue
            
            if not content:
                continue
            
            content_lower = content.lower()
            
            # Expanded plant names list (including common variants)
            plant_keywords = [
                'ashwagandha', 'ashwaganda', 'withania',
                'tulsi', 'basil', 'ocimum',
                'turmeric', 'haldi', 'curcuma',
                'neem', 'azadirachta', 'margosa',
                'amla', 'gooseberry', 'phyllanthus',
                'amrood', 'guava',
                'vasaka', 'adhatoda', 'malabar nut',
                'bael', 'wood apple', 'aegle',
                'jamun', 'java plum', 'syzygium',
                'ashoka', 'saraca',
                'moringa', 'drumstick',
                'sandalwood', 'chandan',
                'kalmegh', 'green chiretta', 'andrographis',
                'methi', 'fenugreek', 'trigonella',
                'aloe vera', 'aloe',
                'ginger', 'adrak',
                'cumin', 'jeera',
                'cardamom', 'elaichi',
                'coriander', 'dhania',
                'garlic', 'lahsun',
                'cinnamon', 'dalchini',
                'mint', 'pudina'
            ]
            
            # Extract mentioned plant names
            for plant in plant_keywords:
                if plant in content_lower and plant not in extracted_terms:
                    extracted_terms.append(plant)
        
        # Detect if this is a follow-up question (pronouns, vague references)
        query_lower = user_query.lower()
        is_followup = any(word in query_lower for word in [
            'it', 'its', 'they', 'them', 'that', 'this', 'which', 'what does',
            'what are', 'how is', 'how are', 'tell me more', 'more about'
        ])
        
        # If follow-up question and we found plant terms, aggressively add them
        if extracted_terms:
            missing_terms = [term for term in extracted_terms if term not in query_lower]
            if missing_terms:
                # For follow-ups, add the plant name(s) from history
                if is_followup:
                    # Add all missing plant terms for follow-ups
                    enhanced = f"{' '.join(missing_terms[:2])} {user_query}"
                    logger.info(f"[Chatbot] FOLLOW-UP detected: Enhanced query with history terms: '{user_query}' -> '{enhanced}'")
                    return enhanced
                else:
                    # For regular queries, add terms after
                    enhanced = f"{user_query} {' '.join(missing_terms[:2])}"
                    logger.debug(f"[Chatbot] Extracted terms from history: {extracted_terms}, missing: {missing_terms}")
                    return enhanced
        
        return user_query
    
    def _format_recent_conversation(self, recent_messages: List[BaseMessage]) -> str:
        """Format recent conversation messages for context."""
        if not recent_messages or len(recent_messages) == 0:
            return ""
        
        formatted = "Recent conversation context:\n"
        for i, msg in enumerate(recent_messages[-6:], 1):  # Last 3 exchanges (6 messages)
            # Try different ways to get content from message
            if hasattr(msg, 'content'):
                content = msg.content
            elif hasattr(msg, 'text'):
                content = msg.text
            elif hasattr(msg, '__str__'):
                content = str(msg)
            else:
                continue
            
            if not content:
                continue
            
            # Truncate each message to 200 chars to prevent token overflow
            if len(content) > 200:
                content = content[:200] + "..."
            
            # Determine role
            role = "User"
            if hasattr(msg, '__class__'):
                class_name = msg.__class__.__name__
                if 'AI' in class_name or 'Assistant' in class_name or 'System' in class_name:
                    role = "Assistant"
            
            # Truncate long messages
            content_short = content[:200] + "..." if len(content) > 200 else content
            formatted += f"{role}: {content_short}\n"
        
        formatted += "\nUse this conversation history to maintain context when answering. "
        formatted += "If the user asks about 'it' or 'that plant' or similar references, refer back to the conversation history.\n"
        
        return formatted
    
    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.clear()

