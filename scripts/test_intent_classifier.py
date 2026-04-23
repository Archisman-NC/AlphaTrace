import sys
import os
sys.path.append(os.getcwd())

from app.reasoning.intent_classifier import classify_intent

def test_high_precision_classifier():
    print("Testing High-Precision Intent Classifier (Multi-Intent + Confidence)...\n")
    
    test_cases = [
        {
            "query": "why did it fall and is it risky?",
            "current": "PORTFOLIO_002",
            "history": [],
            "description": "Multi-intent (Reason + Risk)"
        },
        {
            "query": "switch to PORTFOLIO_003 because this is losing money",
            "current": "PORTFOLIO_001",
            "history": [],
            "description": "Multi-intent (Switch + Reason)"
        },
        {
            "query": "give me a full checkup",
            "current": "PORTFOLIO_001",
            "history": [],
            "description": "Vague / Full Analysis"
        }
    ]
    
    for case in test_cases:
        result = classify_intent(case["query"], case["current"], case["history"])
        print(f"Description: {case['description']}")
        print(f"Query: {case['query']}")
        print(f"Result: {result}")
        print("-" * 50)

if __name__ == "__main__":
    test_high_precision_classifier()
