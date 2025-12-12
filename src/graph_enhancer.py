"""Graph-enhanced retrieval logic."""
import logging
import re
from typing import List, Dict, Set
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
        graph_results = []
        try:
            logger.debug("[GraphEnhancer] Step 1: Querying Neo4j for graph context...")
            graph_results = self.neo4j_client.get_graph_results(query)
            graph_context = self.neo4j_client.get_graph_context(query)
            logger.info(f"[GraphEnhancer] Graph context retrieved: {len(graph_context)} chars" if graph_context else "[GraphEnhancer] No graph context found")
            if graph_context:
                logger.debug(f"[GraphEnhancer] Graph context preview: {graph_context[:200]}...")
        except Exception as e:
            logger.error(f"[GraphEnhancer] Error getting graph context: {str(e)}")
            graph_context = ""
            graph_results = []
        
        # Step 2: Enhance query with graph context
        if graph_context:
            enhanced_query = f"{query} {graph_context}"
            logger.debug(f"[GraphEnhancer] Step 2: Enhanced query: {enhanced_query[:200]}...")
        else:
            enhanced_query = query
            logger.debug("[GraphEnhancer] Step 2: No enhancement applied (no graph context)")
        
        # Step 3: Search Weaviate with enhanced query
        # Retrieve more results initially for reranking (2x for better coverage)
        initial_limit = Config.TOP_K_WEAVIATE * 2 if graph_results else Config.TOP_K_WEAVIATE
        try:
            logger.debug(f"[GraphEnhancer] Step 3: Searching Weaviate (limit: {initial_limit})...")
            vector_results = self.weaviate_client.hybrid_search(
                enhanced_query,
                limit=initial_limit
            )
            logger.info(f"[GraphEnhancer] Weaviate search returned {len(vector_results)} results")
            if vector_results:
                logger.debug(f"[GraphEnhancer] First Weaviate result: {vector_results[0].get('filename', 'unknown')}")
        except Exception as e:
            logger.error(f"[GraphEnhancer] Error searching Weaviate: {str(e)}")
            vector_results = []
        
        # Step 4: Rerank vector results using entity-aware hybrid scoring (graph-first approach)
        if vector_results and graph_results:
            logger.debug(f"[GraphEnhancer] Step 4: Reranking {len(vector_results)} vector results using entity-aware hybrid scoring...")
            graph_entities = self._extract_graph_entities(graph_results)
            reranked_results = self._rerank_vector_results(vector_results, graph_entities, query)
            logger.info(f"[GraphEnhancer] Reranked to {len(reranked_results)} top results")
            vector_results = reranked_results
        elif vector_results:
            logger.debug(f"[GraphEnhancer] Step 4: No graph results - keeping original Weaviate ranking")
            # Still limit to TOP_K_WEAVIATE if we retrieved more
            vector_results = vector_results[:Config.TOP_K_WEAVIATE]
        
        # Extract graph citations
        graph_citations = self._extract_graph_citations(graph_results)
        
        result = {
            "graph_context": graph_context,
            "graph_results": graph_results,
            "graph_citations": graph_citations,
            "enhanced_query": enhanced_query,
            "vector_results": vector_results,
            "query": query
        }
        
        logger.info(f"[GraphEnhancer] Retrieval complete - Graph: {len(graph_context)} chars, Graph citations: {len(graph_citations)}, Vector: {len(vector_results)} reranked results")
        return result
    
    def _extract_graph_entities(self, graph_results: List[Dict]) -> Dict[str, Set[str]]:
        """
        Extract all entities from graph results for entity-aware reranking.
        Returns a dict with entity types as keys and sets of entity names as values.
        """
        entities = {
            "plants": set(),
            "illnesses": set(),
            "symptoms": set(),
            "compounds": set(),
            "properties": set(),
            "local_names": set(),
            "biological_names": set()
        }
        
        if not graph_results:
            return entities
        
        for result in graph_results:
            # Extract plant name
            if result.get("plant_name"):
                entities["plants"].add(result["plant_name"].lower())
            
            # Extract illnesses
            for illness in result.get("illnesses", []):
                if illness:
                    entities["illnesses"].add(str(illness).lower())
            
            # Extract symptoms
            for symptom in result.get("symptoms", []):
                if symptom:
                    entities["symptoms"].add(str(symptom).lower())
            
            # Extract compounds
            for compound in result.get("compounds", []):
                if compound:
                    entities["compounds"].add(str(compound).lower())
            
            # Extract properties
            for prop in result.get("properties", []):
                if prop:
                    entities["properties"].add(str(prop).lower())
            
            # Extract local names
            for local_name in result.get("local_names", []):
                if local_name:
                    entities["local_names"].add(str(local_name).lower())
            
            # Extract biological names
            for bio_name in result.get("biological_names", []):
                if bio_name:
                    entities["biological_names"].add(str(bio_name).lower())
        
        # Log extracted entities
        total_entities = sum(len(v) for v in entities.values())
        logger.debug(f"[GraphEnhancer] Extracted {total_entities} graph entities for reranking")
        if total_entities > 0:
            logger.debug(f"[GraphEnhancer] Entity counts: Plants={len(entities['plants'])}, Illnesses={len(entities['illnesses'])}, Symptoms={len(entities['symptoms'])}")
        
        return entities
    
    def _calculate_hybrid_score(self, chunk: Dict, graph_entities: Dict[str, Set[str]], base_score_weight: float = 0.35) -> float:
        """
        Calculate hybrid score combining Weaviate base score + graph entity match boost.
        Graph-first approach: 65% weight on entity matches, 35% on base score.
        
        Args:
            chunk: Vector search result with content and score
            graph_entities: Dict of entity types and their name sets
            base_score_weight: Weight for Weaviate base score (0-1)
        
        Returns:
            Hybrid score (higher = more relevant)
        """
        # Normalize base Weaviate score (assume 0-1 range, adjust if needed)
        base_score = chunk.get("score", 0.0)
        # Weaviate hybrid scores are typically 0-1, but may vary - normalize conservatively
        normalized_base = min(base_score / 1.0, 1.0) if base_score > 0 else 0.0
        
        # Count entity matches in chunk content
        content_lower = chunk.get("content", "").lower()
        entity_match_count = 0
        entity_match_details = []
        
        # Check each entity type (plants weighted MUCH higher, then illnesses, then others)
        # Increased plant weight significantly to improve plant precision
        entity_weights = {
            "plants": 10.0,  # Increased from 3.0 to heavily favor plant matches
            "illnesses": 3.0,  # Increased from 2.5
            "symptoms": 2.5,  # Increased from 2.0
            "compounds": 2.0,  # Increased from 1.5
            "properties": 2.0,  # Increased from 1.5
            "local_names": 5.0,  # Increased significantly (these are plant names too)
            "biological_names": 5.0  # Increased significantly (these are plant names too)
        }
        
        weighted_matches = 0.0
        for entity_type, entity_set in graph_entities.items():
            weight = entity_weights.get(entity_type, 1.0)
            for entity in entity_set:
                # Use word boundaries to avoid partial matches (e.g., "turmeric" not matching "curcumin")
                pattern = r'\b' + re.escape(entity) + r'\b'
                if re.search(pattern, content_lower):
                    entity_match_count += 1
                    weighted_matches += weight
                    entity_match_details.append(f"{entity_type}:{entity}")
        
        # Calculate entity boost score (0-1 normalized)
        # Cap at reasonable max to prevent single chunk from dominating
        max_possible_weight = sum(weight * len(entity_set) for entity_type, entity_set in graph_entities.items() 
                                 for weight in [entity_weights.get(entity_type, 1.0)])
        if max_possible_weight > 0:
            entity_boost = min(weighted_matches / max_possible_weight, 1.0)
        else:
            entity_boost = 0.0
        
        # Hybrid score: base_score_weight * base + (1 - base_score_weight) * entity_boost
        # Graph-first with stronger plant emphasis: 25% base, 75% entity matches
        # Reduced base weight to favor entity matches more (especially plants)
        base_score_weight_adjusted = 0.25  # Reduced from 0.35 to 0.25
        hybrid_score = (base_score_weight_adjusted * normalized_base) + ((1 - base_score_weight_adjusted) * entity_boost)
        
        # Extra boost if plant names match (double the plant contribution)
        if graph_entities.get("plants"):
            plant_match_boost = 0.0
            for plant in graph_entities["plants"]:
                pattern = r'\b' + re.escape(plant) + r'\b'
                if re.search(pattern, content_lower):
                    plant_match_boost += 0.1  # Additional boost per plant match
            hybrid_score = min(hybrid_score + plant_match_boost, 1.0)  # Cap at 1.0
        
        # Store scoring details for debugging
        chunk["_scoring"] = {
            "base_score": normalized_base,
            "entity_matches": entity_match_count,
            "weighted_matches": weighted_matches,
            "entity_boost": entity_boost,
            "hybrid_score": hybrid_score,
            "matched_entities": entity_match_details[:5],  # Store first 5 for debugging
            "has_plant_match": any("plants:" in detail or "local_names:" in detail or "biological_names:" in detail 
                                  for detail in entity_match_details)
        }
        
        return hybrid_score
    
    def _rerank_vector_results(self, vector_results: List[Dict], graph_entities: Dict[str, Set[str]], query: str) -> List[Dict]:
        """
        Rerank vector search results using entity-aware hybrid scoring.
        Graph-first approach: prioritizes chunks that mention graph entities.
        
        Args:
            vector_results: List of vector search results
            graph_entities: Dict of entity types and their name sets
            query: Original user query (for potential future query matching)
        
        Returns:
            Reranked and filtered list of top K results
        """
        if not vector_results:
            return []
        
        # Calculate hybrid scores for all chunks
        scored_chunks = []
        for chunk in vector_results:
            hybrid_score = self._calculate_hybrid_score(chunk, graph_entities)
            scored_chunks.append((chunk, hybrid_score))
        
        # Sort by hybrid score (descending)
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Filter: If graph exists, prioritize chunks with entity matches
        # But don't completely exclude non-matching chunks (keep at least some diversity)
        filtered_chunks = []
        chunks_with_entities = []
        chunks_without_entities = []
        
        for chunk, score in scored_chunks:
            entity_matches = chunk.get("_scoring", {}).get("entity_matches", 0)
            if entity_matches > 0:
                chunks_with_entities.append(chunk)
            else:
                chunks_without_entities.append(chunk)
        
        # STRICTER filtering: Prioritize chunks with plant matches, then other entity matches
        # Separate plant-matching chunks from other entity-matching chunks
        chunks_with_plant_matches = []
        chunks_with_other_entity_matches = []
        
        for chunk in chunks_with_entities:
            if chunk.get("_scoring", {}).get("has_plant_match", False):
                chunks_with_plant_matches.append(chunk)
            else:
                chunks_with_other_entity_matches.append(chunk)
        
        # Prefer chunks with plant matches first, then other entity matches, then base ranking
        if chunks_with_plant_matches:
            filtered_chunks.extend(chunks_with_plant_matches)
            remaining_slots = Config.TOP_K_WEAVIATE - len(filtered_chunks)
            if remaining_slots > 0:
                filtered_chunks.extend(chunks_with_other_entity_matches[:remaining_slots])
            remaining_slots = Config.TOP_K_WEAVIATE - len(filtered_chunks)
            if remaining_slots > 0:
                filtered_chunks.extend(chunks_without_entities[:remaining_slots])
        elif chunks_with_other_entity_matches:
            filtered_chunks.extend(chunks_with_other_entity_matches)
            remaining_slots = Config.TOP_K_WEAVIATE - len(filtered_chunks)
            if remaining_slots > 0:
                filtered_chunks.extend(chunks_without_entities[:remaining_slots])
        else:
            # No entity matches - fall back to base Weaviate ranking
            logger.debug("[GraphEnhancer] No chunks matched graph entities - using base Weaviate ranking")
            filtered_chunks = [chunk for chunk, _ in scored_chunks[:Config.TOP_K_WEAVIATE]]
        
        # Limit to TOP_K_WEAVIATE
        final_results = filtered_chunks[:Config.TOP_K_WEAVIATE]
        
        # Log reranking stats
        entity_match_count = sum(1 for c in final_results if c.get("_scoring", {}).get("entity_matches", 0) > 0)
        logger.info(f"[GraphEnhancer] Reranking complete: {entity_match_count}/{len(final_results)} chunks matched graph entities")
        if final_results:
            top_score = final_results[0].get("_scoring", {}).get("hybrid_score", 0.0)
            logger.debug(f"[GraphEnhancer] Top reranked score: {top_score:.3f}")
        
        # Store hybrid_score in the chunk for filtering/ranking (before removing _scoring)
        for chunk in final_results:
            if "_scoring" in chunk:
                # Store hybrid_score for later use in citation filtering
                chunk["hybrid_score"] = chunk["_scoring"].get("hybrid_score", chunk.get("score", 0.0))
                # Keep _scoring for metrics calculation (MRR/NDCG)
                # Don't remove it - needed for evaluation
    
    def _extract_graph_citations(self, graph_results: List[Dict]) -> List[Dict]:
        """Extract citation information from graph results. Returns top 4 citations."""
        citations = []
        seen_plants = set()
        
        if not graph_results:
            return citations
        
        for result in graph_results:
            plant_name = result.get("plant_name", "")
            if plant_name and plant_name not in seen_plants:
                seen_plants.add(plant_name)
                citations.append({
                    "type": "graph",
                    "source": "Knowledge Graph",
                    "name": plant_name,
                    "plants": [plant_name],
                    "illnesses": list(filter(None, result.get("illnesses", [])))[:3],  # Limit to 3
                    "symptoms": list(filter(None, result.get("symptoms", [])))[:3]  # Limit to 3
                })
        
        # Limit to top 4 graph citations
        return citations[:4]
    
    def format_context_for_llm(self, retrieval_result: Dict[str, any]) -> str:
        """Format retrieval results for LLM context."""
        logger.debug("[GraphEnhancer] Formatting context for LLM...")
        context_parts = []
        
        # Add graph context if available (truncate if too long)
        if retrieval_result["graph_context"]:
            graph_context = retrieval_result["graph_context"]
            # Limit graph context to 1500 chars to prevent token overflow
            if len(graph_context) > 1500:
                graph_context = graph_context[:1500] + "..."
                logger.debug(f"[GraphEnhancer] Truncated graph context from {len(retrieval_result['graph_context'])} to {len(graph_context)} chars")
            context_parts.append("Graph Context (from knowledge base):")
            context_parts.append(graph_context)
            context_parts.append("")
            logger.debug(f"[GraphEnhancer] Added graph context ({len(graph_context)} chars)")
        
        # Add vector search results (limit to top 2 and truncate each chunk to prevent token overflow)
        if retrieval_result["vector_results"]:
            top_vector_results = retrieval_result["vector_results"][:2]  # Limit to top 2 (reduced from 3)
            context_parts.append("Document Context (from PDFs):")
            context_parts.append("Use the following document excerpts to provide detailed information. Synthesize this content with the graph context above for a comprehensive answer.")
            context_parts.append("")
            for i, result in enumerate(top_vector_results, 1):
                content = result.get("content", "")
                # Truncate each chunk to 500 chars to prevent token overflow
                if len(content) > 500:
                    content = content[:500] + "..."
                context_parts.append(f"[Source {i}: {result['filename']}, Page {result['page_number']}]")
                context_parts.append(content)
                context_parts.append("")
            logger.debug(f"[GraphEnhancer] Added {len(top_vector_results)} document chunks (limited to top 2, max 500 chars each)")
        
        # If no context available, indicate this to the LLM
        if not context_parts:
            logger.warning("[GraphEnhancer] No context available (neither graph nor vector results)")
            return "Context: No specific data available from knowledge base or documents. If the question is within your domain (medicinal plants, herbal medicine, health), you may use general knowledge but clearly state it's general information. If the question is clearly off-topic, politely decline as per your guidelines."
        
        formatted_context = "Context:\n" + "\n".join(context_parts)
        logger.info(f"[GraphEnhancer] Formatted context: {len(formatted_context)} chars total")
        return formatted_context

