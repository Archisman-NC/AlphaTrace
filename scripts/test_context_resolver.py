import sys
import os
sys.path.append(os.getcwd())

from app.reasoning.context_resolver import resolve_context

def test_context_resolution():
    print("Testing Context Resolution Engine (Pronoun & Follow-up Resolution)...\n")
    
    session = {
        "current_portfolio": "PORTFOLIO_002",
        "last_analysis": {
            "summary": "Portfolio declined by 2.4% due to tech sector weakness following rate hikes."
        }
    }
    
    test_cases = [
        {
            "query": "why did it fall?",
            "desc": "Ambiguous Follow-up with Pronoun"
        },
        {
            "query": "show me PORTFOLIO_003",
            "desc": "Explicit New Command"
        },
        {
            "query": "is this risky now?",
            "desc": "Context-Dependent Risk Query"
        }
    ]
    
    for case in test_cases:
        result = resolve_context(case["query"], session)
        print(f"Description: {case['desc']}")
        print(f"Query: {case['query']}")
        print(f"Resolved: {result}")
        print("-" * 50)

if __name__ == "__main__":
    test_context_resolution()
