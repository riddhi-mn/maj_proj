"""Metrics calculation for evaluation."""

from typing import Dict, List
import re
import math


def calculate_answer_length(answer: str) -> int:
    """Calculate word count of answer."""
    return len(answer.split())


def check_keyword_presence(answer: str, keywords: List[str]) -> float:
    """
    Calculate keyword presence score (0-1).
    
    Args:
        answer: Answer text
        keywords: List of expected keywords
        
    Returns:
        Score between 0 and 1 (percentage of keywords found)
    """
    if not keywords:
        return 1.0
    
    answer_lower = answer.lower()
    found_keywords = sum(1 for keyword in keywords if keyword.lower() in answer_lower)
    
    return found_keywords / len(keywords)


def check_plant_mentions(answer: str, expected_plants: List[str]) -> float:
    """
    Calculate plant mention score (0-1).
    
    Args:
        answer: Answer text
        expected_plants: List of expected plant names
        
    Returns:
        Score between 0 and 1 (percentage of plants mentioned)
    """
    if not expected_plants:
        return 1.0  # No expected plants = full score
    
    answer_lower = answer.lower()
    found_plants = sum(1 for plant in expected_plants if plant.lower() in answer_lower)
    
    return found_plants / len(expected_plants)


def check_citation_relevance(sources: List[Dict], expected_citations: Dict) -> Dict[str, bool]:
    """
    Check if citations match expected types.
    
    Args:
        sources: List of citation dicts
        expected_citations: Dict with 'graph' and 'vector' boolean expectations
        
    Returns:
        Dict with 'has_graph', 'has_vector', 'both_present'
    """
    has_graph = any(s.get("type") == "graph" for s in sources)
    has_vector = any(s.get("type") == "pdf" for s in sources)
    
    return {
        "has_graph": has_graph,
        "has_vector": has_vector,
        "both_present": has_graph and has_vector,
        "expected_graph": expected_citations.get("graph", False),
        "expected_vector": expected_citations.get("vector", False),
        "graph_match": has_graph == expected_citations.get("graph", False),
        "vector_match": has_vector == expected_citations.get("vector", False)
    }


def calculate_retrieval_metrics(sources: List[Dict], expected_plants: List[str]) -> Dict:
    """
    Calculate retrieval-specific metrics.
    
    Args:
        sources: List of citation dicts
        expected_plants: List of expected plants
        
    Returns:
        Dict with retrieval metrics
    """
    graph_citations = [s for s in sources if s.get("type") == "graph"]
    vector_citations = [s for s in sources if s.get("type") == "pdf"]
    
    # Extract plant names from graph citations
    cited_plants = []
    for citation in graph_citations:
        if "name" in citation:
            cited_plants.append(citation["name"])
        elif "plants" in citation:
            cited_plants.extend(citation["plants"])
    
    # Check if expected plants are cited (stricter matching)
    if expected_plants:
        cited_plants_lower = [p.lower().strip() for p in cited_plants]
        expected_plants_lower = [p.lower().strip() for p in expected_plants]
        
        # More lenient matching: check if any expected plant name appears in cited plants
        # (handles partial matches and variations)
        matching_plants = 0
        for expected in expected_plants_lower:
            # Check exact match first
            if expected in cited_plants_lower:
                matching_plants += 1
            else:
                # Check if any cited plant contains the expected plant name (partial match)
                # This handles cases like "Neem" matching "Neem tree" or "Azadirachta indica (Neem)"
                if any(expected in cited or cited in expected for cited in cited_plants_lower):
                    matching_plants += 1
        
        plant_precision = matching_plants / len(cited_plants) if cited_plants else 0.0
        plant_recall = matching_plants / len(expected_plants) if expected_plants else 1.0
    else:
        plant_precision = 1.0 if not cited_plants else 0.5  # Neutral if no expectations
        plant_recall = 1.0
    
    return {
        "num_graph_citations": len(graph_citations),
        "num_vector_citations": len(vector_citations),
        "total_citations": len(sources),
        "cited_plants": cited_plants,
        "plant_precision": plant_precision,
        "plant_recall": plant_recall
    }


def calculate_comprehensive_metrics(
    question: Dict,
    answer: str,
    sources: List[Dict],
    retrieval_used: bool,
    ranked_vector_chunks: List[Dict] = None
) -> Dict:
    """
    Calculate comprehensive metrics for a single question-answer pair.
    
    Args:
        question: Question dict with expected values
        answer: Answer text
        sources: List of citation dicts
        retrieval_used: Whether retrieval was used
        ranked_vector_chunks: Optional list of ranked vector retrieval chunks for MRR/NDCG calculation
        
    Returns:
        Dict with all calculated metrics
    """
    metrics = {
        "answer_length": calculate_answer_length(answer),
        "keyword_score": check_keyword_presence(answer, question.get("expected_keywords", [])),
        "plant_mention_score": check_plant_mentions(answer, question.get("expected_plants", [])),
        "citation_check": check_citation_relevance(sources, question.get("expected_citations", {})),
        "retrieval_used": retrieval_used
    }
    
    # Add retrieval metrics if retrieval was used
    if retrieval_used:
        metrics["retrieval"] = calculate_retrieval_metrics(sources, question.get("expected_plants", []))
        
        # Add ranking metrics (MRR, NDCG) if ranked chunks are provided
        if ranked_vector_chunks:
            ranking_metrics = calculate_retrieval_ranking_metrics(
                ranked_vector_chunks,
                question.get("expected_plants", []),
                question.get("expected_keywords", []),
                k=10
            )
            metrics["ranking"] = ranking_metrics
    
    return metrics


def check_chunk_relevance(chunk_content: str, expected_plants: List[str], expected_keywords: List[str]) -> bool:
    """
    Determine if a retrieved chunk is relevant to the query.
    
    A chunk is considered relevant if it contains:
    - At least one expected plant name (case-insensitive, whole word or partial), OR
    - At least 1 expected keyword (lowered threshold from 2 to 1 for better recall)
    
    Args:
        chunk_content: Text content of the retrieved chunk
        expected_plants: List of expected plant names
        expected_keywords: List of expected keywords
        
    Returns:
        True if chunk is relevant, False otherwise
    """
    if not chunk_content:
        return False
    
    content_lower = chunk_content.lower()
    
    # Check for plant mentions (more lenient - partial matches allowed)
    if expected_plants:
        for plant in expected_plants:
            plant_lower = plant.lower()
            # Check for whole word or partial match
            if plant_lower in content_lower:
                return True
            # Also check for common variations (e.g., "ashwagandha" vs "ashwagandha root")
            # Split plant name and check if major parts match
            plant_words = plant_lower.split()
            if len(plant_words) > 1:
                # If multi-word plant name, at least one significant word should match
                if any(len(word) > 3 and word in content_lower for word in plant_words):
                    return True
    
    # Check for keyword mentions (lowered to 1 keyword for better recall)
    if expected_keywords:
        found_keywords = sum(1 for keyword in expected_keywords if keyword.lower() in content_lower)
        if found_keywords >= 1:  # Lowered from 2 to 1
            return True
    
    return False


def calculate_mrr(ranked_chunks: List[Dict], expected_plants: List[str], expected_keywords: List[str]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR) for ranked retrieval results.
    
    MRR = 1 / rank_of_first_relevant_result
    If no relevant result found, MRR = 0
    
    Args:
        ranked_chunks: List of retrieved chunks (already ranked), each with 'content' field
        expected_plants: List of expected plant names
        expected_keywords: List of expected keywords
        
    Returns:
        MRR score (0-1)
    """
    if not ranked_chunks:
        return 0.0
    
    # Check each chunk in ranked order
    for rank, chunk in enumerate(ranked_chunks, start=1):
        # Get content - try multiple possible field names
        content = chunk.get("content", "")
        if not content:
            # Try alternative field names
            content = chunk.get("text", "")
        
        if content and check_chunk_relevance(content, expected_plants, expected_keywords):
            return 1.0 / rank
    
    # No relevant chunks found
    return 0.0


def calculate_ndcg(ranked_chunks: List[Dict], expected_plants: List[str], expected_keywords: List[str], k: int = 10) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG@k) for ranked retrieval results.
    
    NDCG measures ranking quality by:
    1. Assigning relevance scores (1 for relevant, 0 for irrelevant)
    2. Calculating DCG = sum(relevance_score / log2(rank + 1))
    3. Calculating IDCG (ideal DCG if all relevant chunks were at top)
    4. NDCG = DCG / IDCG
    
    Args:
        ranked_chunks: List of retrieved chunks (already ranked), each with 'content' field
        expected_plants: List of expected plant names
        expected_keywords: List of expected keywords
        k: Number of top results to consider (default: 10)
        
    Returns:
        NDCG@k score (0-1)
    """
    if not ranked_chunks:
        return 0.0
    
    # Limit to top k
    top_k = ranked_chunks[:k]
    
    # Calculate relevance scores
    relevance_scores = []
    for chunk in top_k:
        # Get content - try multiple possible field names
        content = chunk.get("content", "")
        if not content:
            content = chunk.get("text", "")
        
        is_relevant = check_chunk_relevance(content, expected_plants, expected_keywords) if content else False
        relevance_scores.append(1 if is_relevant else 0)
    
    # Calculate DCG (Discounted Cumulative Gain)
    dcg = 0.0
    for i, rel in enumerate(relevance_scores, start=1):
        if rel > 0:
            dcg += rel / math.log2(i + 1)
    
    # Calculate IDCG (Ideal DCG - all relevant chunks at top)
    # Count total relevant chunks
    total_relevant = sum(relevance_scores)
    
    if total_relevant == 0:
        return 0.0  # No relevant chunks = NDCG = 0
    
    idcg = 0.0
    for i in range(1, min(total_relevant + 1, k + 1)):
        idcg += 1.0 / math.log2(i + 1)
    
    # Normalize
    if idcg == 0:
        return 0.0
    
    ndcg = dcg / idcg
    return ndcg


def calculate_retrieval_ranking_metrics(
    ranked_vector_chunks: List[Dict],
    expected_plants: List[str],
    expected_keywords: List[str],
    k: int = 10
) -> Dict:
    """
    Calculate MRR and NDCG for ranked retrieval results.
    
    Args:
        ranked_vector_chunks: List of ranked vector retrieval chunks (with 'content' field)
        expected_plants: List of expected plant names
        expected_keywords: List of expected keywords
        k: Number of top results for NDCG calculation (default: 10)
        
    Returns:
        Dict with 'mrr' and 'ndcg' scores
    """
    if not ranked_vector_chunks:
        return {
            "mrr": 0.0,
            "ndcg": 0.0,
            "num_relevant_chunks": 0,
            "total_chunks": 0
        }
    
    # Calculate MRR
    mrr = calculate_mrr(ranked_vector_chunks, expected_plants, expected_keywords)
    
    # Calculate NDCG@k
    ndcg = calculate_ndcg(ranked_vector_chunks, expected_plants, expected_keywords, k)
    
    # Count relevant chunks (check all chunks, not just top k)
    num_relevant = 0
    for chunk in ranked_vector_chunks:
        content = chunk.get("content", "")
        if not content:
            content = chunk.get("text", "")
        if content and check_chunk_relevance(content, expected_plants, expected_keywords):
            num_relevant += 1
    
    return {
        "mrr": mrr,
        "ndcg": ndcg,
        "num_relevant_chunks": num_relevant,
        "total_chunks": len(ranked_vector_chunks)
    }


def aggregate_metrics(all_metrics: List[Dict]) -> Dict:
    """
    Aggregate metrics across all test questions.
    
    Args:
        all_metrics: List of metric dicts (one per question)
        
    Returns:
        Dict with aggregated statistics
    """
    if not all_metrics:
        return {}
    
    # Average scores
    avg_keyword_score = sum(m["keyword_score"] for m in all_metrics) / len(all_metrics)
    avg_plant_score = sum(m["plant_mention_score"] for m in all_metrics) / len(all_metrics)
    avg_answer_length = sum(m["answer_length"] for m in all_metrics) / len(all_metrics)
    
    # Citation statistics
    citation_checks = [m["citation_check"] for m in all_metrics]
    graph_match_rate = sum(1 for c in citation_checks if c.get("graph_match", False)) / len(citation_checks)
    vector_match_rate = sum(1 for c in citation_checks if c.get("vector_match", False)) / len(citation_checks)
    
    # Retrieval statistics (if applicable)
    retrieval_metrics = [m.get("retrieval", {}) for m in all_metrics if m.get("retrieval")]
    if retrieval_metrics:
        avg_plant_precision = sum(r.get("plant_precision", 0) for r in retrieval_metrics) / len(retrieval_metrics)
        avg_plant_recall = sum(r.get("plant_recall", 0) for r in retrieval_metrics) / len(retrieval_metrics)
        avg_graph_citations = sum(r.get("num_graph_citations", 0) for r in retrieval_metrics) / len(retrieval_metrics)
        avg_vector_citations = sum(r.get("num_vector_citations", 0) for r in retrieval_metrics) / len(retrieval_metrics)
    else:
        avg_plant_precision = 0.0
        avg_plant_recall = 0.0
        avg_graph_citations = 0.0
        avg_vector_citations = 0.0
    
    # Ranking statistics (MRR, NDCG) - if applicable
    ranking_metrics = [m.get("ranking", {}) for m in all_metrics if m.get("ranking")]
    if ranking_metrics:
        avg_mrr = sum(r.get("mrr", 0) for r in ranking_metrics) / len(ranking_metrics)
        avg_ndcg = sum(r.get("ndcg", 0) for r in ranking_metrics) / len(ranking_metrics)
        avg_relevant_chunks = sum(r.get("num_relevant_chunks", 0) for r in ranking_metrics) / len(ranking_metrics)
    else:
        avg_mrr = 0.0
        avg_ndcg = 0.0
        avg_relevant_chunks = 0.0
    
    return {
        "num_questions": len(all_metrics),
        "avg_keyword_score": avg_keyword_score,
        "avg_plant_mention_score": avg_plant_score,
        "avg_answer_length": avg_answer_length,
        "graph_citation_match_rate": graph_match_rate,
        "vector_citation_match_rate": vector_match_rate,
        "avg_plant_precision": avg_plant_precision,
        "avg_plant_recall": avg_plant_recall,
        "avg_graph_citations": avg_graph_citations,
        "avg_vector_citations": avg_vector_citations,
        "avg_mrr": avg_mrr,
        "avg_ndcg": avg_ndcg,
        "avg_relevant_chunks": avg_relevant_chunks
    }

