"""Test script for LLM-only evaluation (no retrieval)."""

import argparse
import logging
from typing import Dict, List
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

from tests.test_dataset import get_test_questions

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LLMOnlyTester:
    """Test the LLM model without any retrieval."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.7):
        """
        Initialize LLM-only tester.
        
        Args:
            model_name: OpenAI model name (e.g., "gpt-3.5-turbo", "gpt-4")
            temperature: Temperature for generation (0.0-1.0)
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        logger.info(f"Initializing LLM: {model_name} (temperature={temperature})")
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            openai_api_key=api_key
        )
        self.model_name = model_name
        
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
        
        logger.info(f"✓ LLM initialized: {model_name}")
    
    def answer(self, question: str) -> Dict[str, str]:
        """
        Get answer from LLM for a question.
        
        Args:
            question: User question
        
        Returns:
            Dict with 'response' and 'model' keys
        """
        try:
            messages = self.prompt.format_messages(question=question)
            response = self.llm.invoke(messages)
            answer_text = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "response": answer_text,
                "model": self.model_name
            }
        except Exception as e:
            logger.error(f"Error getting LLM response: {str(e)}")
            return {
                "response": f"ERROR: {str(e)}",
                "model": self.model_name
            }


def run_llm_test(questions: List[Dict], model_name: str = "gpt-3.5-turbo", temperature: float = 0.7):
    """
    Run LLM-only test on a set of questions.
    
    Args:
        questions: List of question dicts
        model_name: OpenAI model name
        temperature: Temperature for generation
    """
    logger.info("=" * 80)
    logger.info("LLM-ONLY TEST (No Retrieval)")
    logger.info("=" * 80)
    logger.info(f"Model: {model_name}")
    logger.info(f"Temperature: {temperature}")
    logger.info(f"Questions: {len(questions)}")
    logger.info("=" * 80)
    
    tester = LLMOnlyTester(model_name=model_name, temperature=temperature)
    
    results = []
    
    for i, question in enumerate(questions, 1):
        logger.info(f"\n[{i}/{len(questions)}] Question: {question['question']}")
        logger.info("-" * 80)
        
        try:
            answer_result = tester.answer(question["question"])
            
            response = answer_result["response"]
            logger.info(f"\nAnswer ({len(response)} chars):")
            logger.info(response[:200] + "..." if len(response) > 200 else response)
            
            results.append({
                "question_id": question["id"],
                "question": question["question"],
                "answer": response,
                "model": model_name,
                "expected_plants": question.get("expected_plants", []),
                "expected_keywords": question.get("expected_keywords", [])
            })
            
        except Exception as e:
            logger.error(f"  ✗ Error: {str(e)}")
            results.append({
                "question_id": question["id"],
                "question": question["question"],
                "answer": f"ERROR: {str(e)}",
                "model": model_name
            })
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total questions: {len(questions)}")
    logger.info(f"Successful answers: {sum(1 for r in results if 'ERROR' not in r.get('answer', ''))}")
    logger.info(f"Errors: {sum(1 for r in results if 'ERROR' in r.get('answer', ''))}")
    
    avg_length = sum(len(r.get("answer", "")) for r in results) / len(results) if results else 0
    logger.info(f"Average answer length: {avg_length:.0f} characters")
    
    return results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Test LLM model without retrieval")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo",
        help="OpenAI model name (default: gpt-3.5-turbo)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation (0.0-1.0, default: 0.7)"
    )
    parser.add_argument(
        "--question-id",
        type=int,
        help="Test only a specific question ID"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode - enter questions manually"
    )
    
    args = parser.parse_args()
    
    if args.interactive:
        # Interactive mode
        logger.info("Interactive LLM Test Mode")
        logger.info("Enter questions (type 'exit' to quit)")
        logger.info("-" * 80)
        
        tester = LLMOnlyTester(model_name=args.model, temperature=args.temperature)
        
        while True:
            try:
                question = input("\nQuestion: ").strip()
                if question.lower() in ['exit', 'quit', 'q']:
                    break
                if not question:
                    continue
                
                answer_result = tester.answer(question)
                print(f"\nAnswer:\n{answer_result['response']}\n")
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error: {str(e)}")
    else:
        # Batch test mode
        questions = get_test_questions()
        
        if args.question_id:
            questions = [q for q in questions if q["id"] == args.question_id]
            if not questions:
                logger.error(f"Question ID {args.question_id} not found!")
                return
        
        results = run_llm_test(questions, model_name=args.model, temperature=args.temperature)
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ Test complete!")
        logger.info("=" * 80)


if __name__ == "__main__":
    main()

