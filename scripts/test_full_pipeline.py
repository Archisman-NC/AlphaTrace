import sys
import os
sys.path.append(os.getcwd())

from app.reasoning.intent_classifier import classify_intent
from app.reasoning.intent_validator import validate_and_route
from app.reasoning.response_generator import generate_advisory_response

def test_full_response_pipeline():
    print("Testing Full Reasoning Pipeline (Classification -> Validation -> Synthesis)...\n")
    
    query = "why did my portfolio fall and is it safe?"
    current_portfolio = "PORTFOLIO_002"
    
    # Mock tool outputs to simulate successful analyzer runs
    mock_tool_outputs = {
        "reason": "Aggressive tech sector exposure led to a 2.4% dip following hawkish interest rate signaling.",
        "risk": "Standard deviation remains elevated, but maximum drawdown is within historical bounds for your aggressive profile.",
        "analysis": "Sector: Banking (+0.5%), Tech (-3.2%). Individual drivers: HDFCBANK (+1.2%), TCS (-4.5%)."
    }
    
    print(f"User Query: {query}\n")
    
    # 1. Classify
    print("[1] Classifying Intent...")
    classification = classify_intent(query, current_portfolio)
    print(f"Classification: {classification}\n")
    
    # 2. Validate
    print("[2] Validating and Routing...")
    validation = validate_and_route(query, classification)
    print(f"Validation: {validation}\n")
    
    # 3. Generate Response (Assume tools ran based on validated intents)
    if validation["action"] == "execute":
        print("[3] Synthesizing Advisory Response...")
        response = generate_advisory_response(
            query, 
            validation["validated_intent"], 
            validation["portfolio_id"], 
            mock_tool_outputs
        )
        print("Final Advisory Briefing:")
        print("-" * 50)
        print(response)
        print("-" * 50)
    else:
        print(f"[3] System Action: {validation['action'].upper()} - Reason: {validation['reason']}")

if __name__ == "__main__":
    test_full_response_pipeline()
