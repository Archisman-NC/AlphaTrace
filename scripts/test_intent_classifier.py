import sys
import os
sys.path.append(os.getcwd())

from app.reasoning.intent_classifier import classify_intent

def test_classifier():
    print("Testing Intent Classifier...\n")
    
    test_cases = [
        {
            "query": "Why did my portfolio drop today?",
            "current": "PORTFOLIO_001",
            "history": [],
            "expected": "reason"
        },
        {
            "query": "Is it safe to hold this?",
            "current": "PORTFOLIO_002",
            "history": [{"role": "user", "content": "Tell me about my tech stocks."}],
            "expected": "risk"
        },
        {
            "query": "Switch to PORTFOLIO_003",
            "current": "PORTFOLIO_001",
            "history": [],
            "expected": "switch_portfolio"
        }
    ]
    
    for case in test_cases:
        result = classify_intent(case["query"], case["current"], case["history"])
        print(f"Query: {case['query']}")
        print(f"Result: {result}")
        print("-" * 30)

if __name__ == "__main__":
    test_classifier()
