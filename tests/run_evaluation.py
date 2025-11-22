"""CLI script to run evaluation comparing baseline LLM vs RAG system."""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from tests.baseline_llm import BaselineLLM
from tests.rag_evaluator import RAGEvaluator
from tests.test_dataset import get_test_questions, get_questions_by_category, get_questions_by_difficulty
from tests.metrics import calculate_comprehensive_metrics, aggregate_metrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_baseline_evaluation(questions: List[Dict]) -> List[Dict]:
    """Run evaluation on baseline LLM."""
    logger.info("=" * 60)
    logger.info("Running BASELINE LLM Evaluation")
    logger.info("=" * 60)
    
    baseline = BaselineLLM()
    results = []
    
    for i, question in enumerate(questions, 1):
        logger.info(f"\n[{i}/{len(questions)}] Question: {question['question']}")
        
        try:
            answer_result = baseline.answer(question["question"])
            metrics = calculate_comprehensive_metrics(
                question,
                answer_result["response"],
                answer_result["sources"],
                answer_result["retrieval_used"]
            )
            
            result = {
                "question_id": question["id"],
                "question": question["question"],
                "answer": answer_result["response"],
                "sources": answer_result["sources"],
                "metrics": metrics
            }
            results.append(result)
            
            logger.info(f"  ✓ Answer length: {metrics['answer_length']} words")
            logger.info(f"  ✓ Keyword score: {metrics['keyword_score']:.2f}")
            logger.info(f"  ✓ Plant mention score: {metrics['plant_mention_score']:.2f}")
            
        except Exception as e:
            logger.error(f"  ✗ Error: {str(e)}")
            results.append({
                "question_id": question["id"],
                "question": question["question"],
                "answer": f"ERROR: {str(e)}",
                "sources": [],
                "metrics": {}
            })
    
    return results


def run_rag_evaluation(questions: List[Dict]) -> List[Dict]:
    """Run evaluation on RAG system."""
    logger.info("=" * 60)
    logger.info("Running RAG SYSTEM Evaluation")
    logger.info("=" * 60)
    
    try:
        rag_evaluator = RAGEvaluator()
    except Exception as e:
        logger.error(f"Failed to initialize RAG evaluator: {str(e)}")
        logger.error("Make sure Neo4j and Weaviate are running!")
        sys.exit(1)
    
    results = []
    
    for i, question in enumerate(questions, 1):
        logger.info(f"\n[{i}/{len(questions)}] Question: {question['question']}")
        
        try:
            answer_result = rag_evaluator.answer(question["question"])
            metrics = calculate_comprehensive_metrics(
                question,
                answer_result["response"],
                answer_result["sources"],
                answer_result["retrieval_used"]
            )
            
            result = {
                "question_id": question["id"],
                "question": question["question"],
                "answer": answer_result["response"],
                "sources": answer_result["sources"],
                "graph_context": answer_result.get("graph_context", ""),
                "metrics": metrics
            }
            results.append(result)
            
            logger.info(f"  ✓ Answer length: {metrics['answer_length']} words")
            logger.info(f"  ✓ Keyword score: {metrics['keyword_score']:.3f} ({metrics['keyword_score']*100:.1f}%)")
            logger.info(f"  ✓ Plant mention score: {metrics['plant_mention_score']:.3f} ({metrics['plant_mention_score']*100:.1f}%)")
            if metrics.get("retrieval"):
                logger.info(f"  ✓ Graph citations: {metrics['retrieval']['num_graph_citations']}")
                logger.info(f"  ✓ Vector citations: {metrics['retrieval']['num_vector_citations']}")
                logger.info(f"  ✓ Plant precision: {metrics['retrieval']['plant_precision']:.3f}")
                logger.info(f"  ✓ Plant recall: {metrics['retrieval']['plant_recall']:.3f}")
            
        except Exception as e:
            logger.error(f"  ✗ Error: {str(e)}")
            results.append({
                "question_id": question["id"],
                "question": question["question"],
                "answer": f"ERROR: {str(e)}",
                "sources": [],
                "metrics": {}
            })
    
    return results


def save_results(baseline_results: List[Dict], rag_results: List[Dict], output_dir: Path):
    """Save evaluation results to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save individual results
    baseline_file = output_dir / f"baseline_results_{timestamp}.json"
    rag_file = output_dir / f"rag_results_{timestamp}.json"
    
    with open(baseline_file, 'w', encoding='utf-8') as f:
        json.dump(baseline_results, f, indent=2, ensure_ascii=False)
    
    with open(rag_file, 'w', encoding='utf-8') as f:
        json.dump(rag_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nResults saved:")
    logger.info(f"  Baseline: {baseline_file}")
    logger.info(f"  RAG: {rag_file}")
    
    return baseline_file, rag_file


def print_comparison(baseline_results: List[Dict], rag_results: List[Dict]):
    """Print comparison summary with neat formatting."""
    logger.info("\n" + "=" * 80)
    logger.info(" " * 25 + "COMPARISON SUMMARY")
    logger.info("=" * 80)
    
    # Aggregate metrics
    baseline_metrics = [r["metrics"] for r in baseline_results if r.get("metrics")]
    rag_metrics = [r["metrics"] for r in rag_results if r.get("metrics")]
    
    baseline_agg = aggregate_metrics(baseline_metrics)
    rag_agg = aggregate_metrics(rag_metrics)
    
    # Header
    logger.info("\n" + "─" * 80)
    logger.info(" " * 15 + "📊 ANSWER QUALITY METRICS")
    logger.info("─" * 80)
    logger.info(f"{'Metric':<40} {'Baseline':<18} {'RAG':<18} {'Improvement':<15}")
    logger.info("─" * 80)
    
    # Keyword score
    bl_kw = baseline_agg.get("avg_keyword_score", 0)
    rag_kw = rag_agg.get("avg_keyword_score", 0)
    diff_kw = rag_kw - bl_kw
    diff_kw_pct = (diff_kw / bl_kw * 100) if bl_kw > 0 else 0
    logger.info(f"{'Keyword Score (avg)':<40} {bl_kw:<18.3f} {rag_kw:<18.3f} {diff_kw:+.3f} ({diff_kw_pct:+.1f}%)")
    
    # Plant mention score
    bl_plant = baseline_agg.get("avg_plant_mention_score", 0)
    rag_plant = rag_agg.get("avg_plant_mention_score", 0)
    diff_plant = rag_plant - bl_plant
    diff_plant_pct = (diff_plant / bl_plant * 100) if bl_plant > 0 else 0
    logger.info(f"{'Plant Mention Score (avg)':<40} {bl_plant:<18.3f} {rag_plant:<18.3f} {diff_plant:+.3f} ({diff_plant_pct:+.1f}%)")
    
    # Answer length
    bl_len = baseline_agg.get("avg_answer_length", 0)
    rag_len = rag_agg.get("avg_answer_length", 0)
    diff_len = rag_len - bl_len
    logger.info(f"{'Answer Length (words, avg)':<40} {bl_len:<18.1f} {rag_len:<18.1f} {diff_len:+.1f}")
    
    # RAG-specific metrics
    if rag_agg.get("avg_graph_citations", 0) > 0:
        logger.info("\n" + "─" * 80)
        logger.info(" " * 15 + "📚 RETRIEVAL METRICS (RAG Only)")
        logger.info("─" * 80)
        logger.info(f"{'Metric':<50} {'Value':<25}")
        logger.info("─" * 80)
        
        logger.info(f"{'Average Graph Citations per Answer':<50} {rag_agg.get('avg_graph_citations', 0):<25.2f}")
        logger.info(f"{'Average Vector (PDF) Citations per Answer':<50} {rag_agg.get('avg_vector_citations', 0):<25.2f}")
        logger.info(f"{'Total Avg Citations per Answer':<50} {(rag_agg.get('avg_graph_citations', 0) + rag_agg.get('avg_vector_citations', 0)):<25.2f}")
        logger.info("")
        logger.info(f"{'Plant Precision (cited plants accuracy)':<50} {rag_agg.get('avg_plant_precision', 0):<25.3f}")
        logger.info(f"{'Plant Recall (coverage of expected plants)':<50} {rag_agg.get('avg_plant_recall', 0):<25.3f}")
        logger.info("")
        logger.info(f"{'Graph Citation Match Rate':<50} {rag_agg.get('graph_citation_match_rate', 0):<25.3f}")
        logger.info(f"{'Vector Citation Match Rate':<50} {rag_agg.get('vector_citation_match_rate', 0):<25.3f}")
    
    # Summary statistics
    logger.info("\n" + "─" * 80)
    logger.info(" " * 15 + "📈 SUMMARY STATISTICS")
    logger.info("─" * 80)
    logger.info(f"{'Total Questions Evaluated':<40} {baseline_agg.get('num_questions', 0):<40}")
    logger.info(f"{'Questions with Errors (Baseline)':<40} {sum(1 for r in baseline_results if 'ERROR' in r.get('answer', '')):<40}")
    logger.info(f"{'Questions with Errors (RAG)':<40} {sum(1 for r in rag_results if 'ERROR' in r.get('answer', '')):<40}")
    
    logger.info("\n" + "=" * 80)


def main():
    """Main evaluation runner."""
    parser = argparse.ArgumentParser(description="Evaluate RAG system vs Baseline LLM")
    parser.add_argument(
        "--mode",
        choices=["baseline", "rag", "both", "compare"],
        default="both",
        help="Evaluation mode: baseline, rag, both, or compare (default: both)"
    )
    parser.add_argument(
        "--category",
        type=str,
        help="Filter questions by category"
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=["easy", "medium", "hard"],
        help="Filter questions by difficulty"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tests/results",
        help="Output directory for results (default: tests/results)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to files"
    )
    
    args = parser.parse_args()
    
    # Get test questions
    questions = get_test_questions()
    if args.category:
        questions = get_questions_by_category(args.category)
    if args.difficulty:
        questions = get_questions_by_difficulty(args.difficulty)
    
    logger.info(f"Running evaluation on {len(questions)} questions")
    
    baseline_results = []
    rag_results = []
    
    # Run evaluations
    if args.mode in ["baseline", "both", "compare"]:
        baseline_results = run_baseline_evaluation(questions)
    
    if args.mode in ["rag", "both", "compare"]:
        rag_results = run_rag_evaluation(questions)
    
    # Save results
    if not args.no_save and (baseline_results or rag_results):
        output_dir = Path(args.output_dir)
        save_results(baseline_results, rag_results, output_dir)
    
    # Print comparison
    if args.mode == "compare" and baseline_results and rag_results:
        print_comparison(baseline_results, rag_results)
    
    logger.info("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()

