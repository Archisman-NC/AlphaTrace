import sys
import os
sys.path.append(os.getcwd())

from app.reasoning.intent_classifier import classify_intent
from app.reasoning.intent_validator import validate_and_route
from app.reasoning.router import execute_intents
from app.reasoning.response_generator import generate_advisory_response
from app.reasoning.response_polisher import polish_response

def run_integration_test():
    print("🚀 Running Full System Integration Test (Hybrid Reasoning)...\n")
    
    # Complex query to trigger OpenAI Polish (Multi-intent + Long Response potential)
    query = "why did my portfolio fall, is it safe, and should I switch to a more conservative strategy?"
    current_portfolio = "PORTFOLIO_002"
    
    user_profile = {"risk_tolerance": "medium", "experience_level": "intermediate"}
    
    mock_tool_outputs = {
        "reason": "Aggressive tech sector exposure led to a 2.4% dip following hawkish interest rate signaling from the Fed.",
        "risk": "Volatility metrics are currently 12% above your baseline, though historical drawdowns suggest this is a temporary cyclical correction.",
        "switch": "Switching to PORTFOLIO_004 (Balanced) would reduce tech exposure by 15% and increase fixed-income stability."
    }
    
    # 1. Intent & Validation
    print("[1] Logic Phase: Classifier & Validator...")
    classification = classify_intent(query, current_portfolio)
    validation = validate_and_route(query, classification)
    print(f"Intents detected: {validation.get('validated_intent', ['full_analysis'])}\n")
    
    if validation.get("action") != "execute":
        print(f"FAIL: Logic engine failed to authorize execution. Reason: {validation.get('reason', 'Unknown')}")
        return

    # 2. Execution Router
    print("[2] Execution Phase: Orchestrating Tools...")
    execution_results = execute_intents({
        "intent": validation.get("validated_intent", ["full_analysis"]),
        "portfolio_id": validation.get("portfolio_id", current_portfolio),
        "confidence": validation.get("confidence", 0.5)
    }, {"current_portfolio": current_portfolio})
    
    tool_data = {r["type"]: r["data"] for r in execution_results["results"]}

    # 3. Narrative Synthesis (Groq)
    print("[3] Synthesis Phase: Generating Raw Advisory (Groq)...")
    raw_response = generate_advisory_response(
        query,
        validation.get("validated_intent", ["full_analysis"]),
        execution_results.get("portfolio_id", current_portfolio),
        mock_tool_outputs,
        user_profile
    )
    
    # 4. Premium Polish (OpenAI)
    print("[4] Polish Phase: Selective OpenAI Refinement...")
    final_response = polish_response(
        raw_response,
        validation.get("validated_intent", ["full_analysis"]),
        user_profile,
        validation.get("confidence", 0.5)
    )
    
    print("\n" + "="*50)
    print("FINAL ALPHA-TRACE ADVISORY")
    print("="*50)
    print(final_response)
    print("="*50 + "\n")
    
    # Verification
    if "Follow-up" in final_response or "?" in final_response:
        print("✅ SUCCESS: Hybrid Intelligence is ACTIVE (OpenAI polish detected).")
    else:
        print("⚠️ NOTE: OpenAI was skipped or failed. Output is raw Groq synthesis.")

if __name__ == "__main__":
    run_integration_test()
