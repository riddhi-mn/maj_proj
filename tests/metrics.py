"""Metrics calculation for evaluation."""

from typing import Dict, List
import re


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
    
    # Check if expected plants are cited
    if expected_plants:
        cited_plants_lower = [p.lower() for p in cited_plants]
        expected_plants_lower = [p.lower() for p in expected_plants]
        matching_plants = sum(1 for plant in expected_plants_lower if plant in cited_plants_lower)
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
    retrieval_used: bool
) -> Dict:
    """
    Calculate comprehensive metrics for a single question-answer pair.
    
    Args:
        question: Question dict with expected values
        answer: Answer text
        sources: List of citation dicts
        retrieval_used: Whether retrieval was used
        
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
    
    return metrics


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
        "avg_vector_citations": avg_vector_citations
    }

