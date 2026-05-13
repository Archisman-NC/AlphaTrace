import sys
import os
sys.path.append(os.getcwd())

from app.reasoning.intent_classifier import classify_intent
from app.reasoning.intent_validator import validate_and_route

def test_intent_pipeline():
    print("Testing End-to-End Intent Pipeline (Classifier + Validator)...\n")
    
    test_cases = [
        {
            "query": "why did my portfolio fall?",
            "current": "PORTFOLIO_001",
            "description": "Valid Reason Query"
        },
        {
            "query": "is this risky?",
            "current": "PORTFOLIO_002",
            "description": "Valid Risk Query"
        },
        {
            "query": "something something",
            "current": "PORTFOLIO_003",
            "description": "Ambiguous/Weak Query"
        }
    ]
    
    for case in test_cases:
        # Step 1: Classify
        classification = classify_intent(case["query"], case["current"])
        
        # Step 2: Validate
        validation = validate_and_route(case["query"], classification)
        
        print(f"Description: {case['description']}")
        print(f"Query: {case['query']}")
        print(f"Validation Result: {validation}")
        print("-" * 60)

if __name__ == "__main__":
    test_intent_pipeline()
