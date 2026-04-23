import sys
import os
sys.path.append(os.getcwd())

from app.reasoning.intent_classifier import classify_intent
from app.reasoning.intent_validator import validate_and_route
from app.reasoning.response_generator import generate_advisory_response

def test_personalized_pipeline():
    print("Testing Personalized Reasoning Pipeline (Copilot Mode)...\n")
    
    query = "why did my portfolio fall and is it safe?"
    current_portfolio = "PORTFOLIO_002"
    
    mock_tool_outputs = {
        "reason": "Aggressive tech sector exposure led to a 2.4% dip following hawkish interest rate signaling.",
        "risk": "Standard deviation remains elevated, but maximum drawdown is within historical bounds for your aggressive profile.",
        "analysis": "Sector: Tech (-3.2%). Individual drivers: TCS (-4.5%)."
    }
    
    profiles = [
        {"risk_tolerance": "low", "experience_level": "beginner", "desc": "Conservative Beginner"},
        {"risk_tolerance": "high", "experience_level": "advanced", "desc": "Aggressive Professional"}
    ]
    
    for profile in profiles:
        print(f"--- Running for: {profile['desc']} ---")
        
        # 1. Classify & Validate (Standard logic)
        classification = classify_intent(query, current_portfolio)
        validation = validate_and_route(query, classification)
        
        # 2. Personalized Response
        if validation["action"] == "execute":
            response = generate_advisory_response(
                query, 
                validation["validated_intent"], 
                validation["portfolio_id"], 
                mock_tool_outputs,
                profile
            )
            print(f"Personalized Advisory:\n")
            print(response)
        
        print("-" * 50 + "\n")

if __name__ == "__main__":
    test_personalized_pipeline()
