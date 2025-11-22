"""Test dataset with questions and ground truth for evaluation."""

TEST_QUESTIONS = [
    {
        "id": 1,
        "question": "What is Ashwagandha?",
        "category": "plant_identification",
        "difficulty": "easy",
        "expected_plants": ["Ashwagandha"],
        "expected_keywords": ["ashwagandha", "withania", "somnifera", "adaptogenic", "stress", "winter cherry"],
        "expected_citations": {
            "graph": True,  # Should have graph citation
            "vector": True  # Should have PDF citation
        }
    },
    {
        "id": 2,
        "question": "What are the medicinal properties of Tulsi?",
        "category": "plant_properties",
        "difficulty": "medium",
        "expected_plants": ["Tulsi"],
        "expected_keywords": ["tulsi", "ocimum", "antimicrobial", "respiratory", "eugenol"],
        "expected_citations": {
            "graph": True,
            "vector": True
        }
    },
    {
        "id": 3,
        "question": "What plants treat anxiety?",
        "category": "treatment_query",
        "difficulty": "medium",
        "expected_plants": ["Ashwagandha"],  # At least this one
        "expected_keywords": ["anxiety", "stress", "adaptogenic", "nervous"],
        "expected_citations": {
            "graph": True,
            "vector": True
        }
    },
    {
        "id": 4,
        "question": "What bioactive compounds does Neem contain?",
        "category": "compound_query",
        "difficulty": "medium",
        "expected_plants": ["Neem"],
        "expected_keywords": ["neem", "azadirachta", "nimbin", "compound", "chemical"],
        "expected_citations": {
            "graph": True,
            "vector": True
        }
    },
    {
        "id": 5,
        "question": "What plants treat skin conditions?",
        "category": "treatment_query",
        "difficulty": "hard",
        "expected_plants": ["Neem"],  # At least this one
        "expected_keywords": ["skin", "rash", "acne", "dermatological"],
        "expected_citations": {
            "graph": True,
            "vector": True
        }
    },
    {
        "id": 6,
        "question": "Tell me about Turmeric and its uses",
        "category": "plant_comprehensive",
        "difficulty": "medium",
        "expected_plants": ["Turmeric"],
        "expected_keywords": ["turmeric", "curcuma", "curcumin", "anti-inflammatory", "arthritis"],
        "expected_citations": {
            "graph": True,
            "vector": True
        }
    },
    {
        "id": 7,
        "question": "What is Amla used for?",
        "category": "plant_uses",
        "difficulty": "easy",
        "expected_plants": ["Amla"],
        "expected_keywords": ["amla", "gooseberry", "vitamin c", "antioxidant", "immune"],
        "expected_citations": {
            "graph": True,
            "vector": True
        }
    },
    {
        "id": 8,
        "question": "Compare Ashwagandha and Tulsi",
        "category": "comparison",
        "difficulty": "hard",
        "expected_plants": ["Ashwagandha", "Tulsi"],
        "expected_keywords": ["ashwagandha", "tulsi", "compare", "difference", "similar"],
        "expected_citations": {
            "graph": True,
            "vector": True
        }
    },
    {
        "id": 9,
        "question": "What preparation methods are used for medicinal plants?",
        "category": "preparation_methods",
        "difficulty": "medium",
        "expected_plants": [],  # General question
        "expected_keywords": ["preparation", "churna", "kwath", "method", "formulation"],
        "expected_citations": {
            "graph": True,
            "vector": True
        }
    },
    {
        "id": 10,
        "question": "What plants have anti-inflammatory properties?",
        "category": "property_query",
        "difficulty": "medium",
        "expected_plants": ["Turmeric"],  # At least this one
        "expected_keywords": ["anti-inflammatory", "inflammation", "turmeric", "curcumin"],
        "expected_citations": {
            "graph": True,
            "vector": True
        }
    },
    {
        "id": 11,
        "question": "Tell me about Neem - its properties, uses, and how it's prepared",
        "category": "plant_comprehensive",
        "difficulty": "medium",
        "expected_plants": ["Neem"],
        "expected_keywords": ["neem", "azadirachta", "indica", "nimbin", "detoxifying", "skin", "antimicrobial", "preparation", "churna", "kwath"],
        "expected_citations": {
            "graph": True,  # Should have graph citation
            "vector": True  # Should have PDF citation (comprehensive question)
        }
    },
    {
        "id": 12,
        "question": "What are the medicinal uses of Ashwagandha and what compounds does it contain?",
        "category": "plant_comprehensive",
        "difficulty": "hard",
        "expected_plants": ["Ashwagandha"],
        "expected_keywords": ["ashwagandha", "withania", "withanolides", "medicinal", "uses", "compound", "treatment", "anxiety", "stress"],
        "expected_citations": {
            "graph": True,  # Should have graph citation
            "vector": True  # Should have PDF citation (detailed compound info)
        }
    }
]


def get_test_questions():
    """Get all test questions."""
    return TEST_QUESTIONS


def get_questions_by_category(category: str = None):
    """Get questions filtered by category."""
    if category:
        return [q for q in TEST_QUESTIONS if q["category"] == category]
    return TEST_QUESTIONS


def get_questions_by_difficulty(difficulty: str = None):
    """Get questions filtered by difficulty."""
    if difficulty:
        return [q for q in TEST_QUESTIONS if q["difficulty"] == difficulty]
    return TEST_QUESTIONS

