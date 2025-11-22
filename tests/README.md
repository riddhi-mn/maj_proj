# RAG Performance Evaluation Framework

This testing framework allows you to evaluate and compare the performance of your hybrid RAG system against a baseline LLM.

## Setup

Make sure your environment is configured:
- Neo4j is running (for RAG evaluation)
- Weaviate is running (for RAG evaluation)
- `.env` file has `OPENAI_API_KEY` set
- PDFs are ingested in Weaviate

## Usage

### Run Both Evaluations (Baseline + RAG)
```bash
python -m tests.run_evaluation
```

### Run Only Baseline LLM
```bash
python -m tests.run_evaluation --mode baseline
```

### Run Only RAG System
```bash
python -m tests.run_evaluation --mode rag
```

### Run and Compare
```bash
python -m tests.run_evaluation --mode compare
```

### Filter Questions
```bash
# By category
python -m tests.run_evaluation --category plant_identification

# By difficulty
python -m tests.run_evaluation --difficulty easy

# Both
python -m tests.run_evaluation --category treatment_query --difficulty medium
```

### Custom Output Directory
```bash
python -m tests.run_evaluation --output-dir my_results
```

### Don't Save Results
```bash
python -m tests.run_evaluation --no-save
```

## Test Dataset

The test dataset (`test_dataset.py`) contains 10 questions covering:
- Plant identification
- Plant properties
- Treatment queries
- Compound queries
- Comparison queries
- Preparation methods

Each question includes:
- Expected plants
- Expected keywords
- Expected citation types (graph/vector)

## Metrics Calculated

### Answer Quality Metrics
- **Keyword Score**: Percentage of expected keywords found in answer
- **Plant Mention Score**: Percentage of expected plants mentioned
- **Answer Length**: Word count

### Retrieval Metrics (RAG only)
- **Graph Citations**: Number of graph citations
- **Vector Citations**: Number of PDF citations
- **Plant Precision**: Accuracy of cited plants
- **Plant Recall**: Coverage of expected plants
- **Citation Match Rate**: Whether citations match expectations

## Results

Results are saved to `tests/results/` by default:
- `baseline_results_TIMESTAMP.json`
- `rag_results_TIMESTAMP.json`

Each result file contains:
- Question and answer
- Sources/citations
- Calculated metrics
- Graph context (RAG only)

## Adding Custom Questions

Edit `tests/test_dataset.py` and add questions to `TEST_QUESTIONS` list:

```python
{
    "id": 11,
    "question": "Your question here",
    "category": "your_category",
    "difficulty": "easy|medium|hard",
    "expected_plants": ["Plant1", "Plant2"],
    "expected_keywords": ["keyword1", "keyword2"],
    "expected_citations": {
        "graph": True,
        "vector": True
    }
}
```

## Notes

- Each question is evaluated independently (memory is cleared)
- Baseline LLM uses no retrieval (pure LLM knowledge)
- RAG system uses hybrid retrieval (Neo4j + Weaviate)
- Results can be compared side-by-side for analysis

