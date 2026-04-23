import sys
import os
sys.path.append(os.getcwd())

from app.reasoning.text_synthesizer import synthesize_text_response

def test_text_synthesis():
    print("Testing Plain-Text Synthesizer (No Hallucination)...\n")
    
    mock_data = {
        "direct_answer": "Your portfolio rose by 1.2% today.",
        "drivers": [
            "HDFC Bank gained 2.1% on strong quarterly outlook",
            "Index heavyweights supported the rally"
        ],
        "risks": "Rising oil prices remain a potential headwind for the banking sector"
    }
    
    response = synthesize_text_response(mock_data)
    print("Synthesized Response:")
    print("-" * 50)
    print(response)
    print("-" * 50)
    
    # Check for markdown formatting
    if "**" in response or "#" in response or "---" in response:
        print("WARNING: Markdown formatting detected in plain-text output.")
    else:
        print("SUCCESS: Plain-text schema adhered to.")

if __name__ == "__main__":
    test_text_synthesis()
