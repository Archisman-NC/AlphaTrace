import sys
import os
sys.path.append(os.getcwd())

from app.reasoning.text_synthesizer import synthesize_text_response

def test_ultima_strict_synthesis():
    print("Testing Ultima-Strict Synthesizer (No Bullet Points, No Inference)...\n")
    
    summary = "Your portfolio rose by 1.2% today."
    drivers = [
        "HDFC Bank gained 2.1% on strong quarterly outlook",
        "Index heavyweights supported the rally"
    ]
    risks = ["Rising oil prices remain a potential headwind for the banking sector"]
    
    print("--- Case 1: Full Data ---")
    response_1 = synthesize_text_response(summary, drivers, risks)
    print(response_1)
    print("-" * 50)
    
    # Check for formatting artifacts
    if "*" in response_1 or "-" in response_1 and "\n-" in response_1:
         print("WARNING: Formatting detected (bullet points or bolding).")

    print("\n--- Case 2: No Drivers/Risks ---")
    response_2 = synthesize_text_response(summary)
    print(response_2)
    print("-" * 50)

    print("\n--- Case 3: Missing Summary ---")
    response_3 = synthesize_text_response("")
    print(response_3)
    print("-" * 50)

if __name__ == "__main__":
    test_ultima_strict_synthesis()
